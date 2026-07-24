"""Direct regression tests for the bf16 GradScaler switch and the [GRAD-GUARD]
non-finite gradient guard + stop policy added to sam3/train/trainer.py.

These cover behavior the grad-accum numerics test does NOT:
  * GradScaler is disabled under bf16 and enabled only under fp16.
  * a disabled scaler ignores a poisoned (tiny) checkpoint scale.
  * the guard detects non-finite grads BEFORE clipping (first-bad-layer signal
    survives) and skips the optimizer step, leaving weights finite/unchanged.
  * the stop policy fails closed on consecutive skips and on window ratio.
"""

import json
import types
from collections import deque
from unittest import mock

import pytest
import torch

from sam3.train.trainer import (
    get_amp_type,
    GRAD_GUARD_MAX_CONSECUTIVE_SKIPS,
    GRAD_GUARD_WINDOW,
    GRAD_GUARD_WINDOW_MIN_SAMPLES,
    GRAD_GUARD_WINDOW_MAX_SKIP_RATIO,
    Trainer,
)


def _scaler_enabled(amp_enabled: bool, amp_dtype: str) -> bool:
    """Mirror of the enable expression in Trainer._setup_components."""
    return amp_enabled and get_amp_type(amp_dtype) == torch.float16


def test_scaler_enabled_only_for_fp16():
    assert _scaler_enabled(True, "float16") is True
    assert _scaler_enabled(True, "bfloat16") is False
    assert _scaler_enabled(False, "float16") is False


def test_disabled_scaler_ignores_poisoned_scale():
    # bf16 path: a disabled scaler must ignore a poisoned 2**-100 checkpoint scale
    sc = torch.amp.GradScaler("cpu", enabled=False)
    sc.load_state_dict(
        {
            "scale": 2.0**-100,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 2,
        }
    )
    assert sc.get_scale() == 1.0
    loss = torch.tensor(3.0)
    assert float(sc.scale(loss)) == float(loss)


def _bare_trainer():
    """A Trainer shell with only what the grad-guard helpers touch."""
    t = Trainer.__new__(Trainer)
    t._grad_skip_total = 0
    t._grad_skip_consecutive = 0
    t._grad_skip_window = deque(maxlen=GRAD_GUARD_WINDOW)
    t.steps = {"train": 0}
    t.logger = mock.Mock()
    return t


def _model_with_bad_grad(bad_layer_first=True):
    m = torch.nn.Sequential(
        torch.nn.Linear(4, 4),  # 0.*
        torch.nn.Linear(4, 4),  # 1.*
    )
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    # inject a NaN into the first layer's weight grad
    target = m[0] if bad_layer_first else m[1]
    target.weight.grad[0, 0] = float("nan")
    return m


def test_guard_detects_first_bad_layer_before_clip():
    t = _bare_trainer()
    t.model = _model_with_bad_grad(bad_layer_first=True)
    # grads are un-clipped here, so the first non-finite layer is the true origin
    with mock.patch("sam3.train.trainer.logging.warning") as warn:
        t._handle_nonfinite_grads("train")
    msg = warn.call_args.args[0]
    assert "first_nonfinite_param=0.weight" in msg  # layer 0, not a clip-spread NaN
    assert "hint, not proven origin" in msg
    assert "nan=1" in msg
    assert t._grad_skip_total == 1
    assert t._grad_skip_consecutive == 1


def test_consecutive_skips_stop_training():
    t = _bare_trainer()
    t.model = _model_with_bad_grad()
    # first (limit - 1) skips are tolerated
    for i in range(GRAD_GUARD_MAX_CONSECUTIVE_SKIPS - 1):
        t._handle_nonfinite_grads("train")
        assert t._grad_skip_consecutive == i + 1
    # the limit-th consecutive skip fails closed
    with pytest.raises(FloatingPointError, match="consecutive"):
        t._handle_nonfinite_grads("train")


def test_consecutive_counter_resets_on_good_step():
    t = _bare_trainer()
    t.model = _model_with_bad_grad()
    t._handle_nonfinite_grads("train")
    t._handle_nonfinite_grads("train")
    assert t._grad_skip_consecutive == 2
    # a good optimizer step resets the consecutive counter (as in train_epoch)
    t._grad_skip_consecutive = 0
    t._grad_skip_window.append(0)
    t._handle_nonfinite_grads("train")
    assert t._grad_skip_consecutive == 1  # no longer near the consecutive limit


def test_window_ratio_stop_training():
    t = _bare_trainer()
    t.model = _model_with_bad_grad()
    # fill the window with good steps, then dribble bad ones below the consecutive
    # limit but above the ratio threshold
    good = GRAD_GUARD_WINDOW_MIN_SAMPLES
    for _ in range(good):
        t._grad_skip_window.append(0)
    # interleave single bad steps (consecutive never reaches the limit) until the
    # window ratio exceeds the threshold -> must stop
    n_bad = int(good * GRAD_GUARD_WINDOW_MAX_SKIP_RATIO) + 2
    with pytest.raises(FloatingPointError, match="rate"):
        for _ in range(n_bad):
            t._handle_nonfinite_grads("train")
            t._grad_skip_consecutive = 0  # simulate a good step in between
            t._grad_skip_window.append(0)


def test_guard_skip_preserves_weights_end_to_end():
    # Replicate the train_epoch decision path with a disabled (bf16) scaler:
    # a non-finite grad must NOT reach AdamW and must leave weights finite.
    dev = "cpu"
    m = torch.nn.Linear(4, 4).to(dev)
    w0 = m.weight.detach().clone()
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(dev, enabled=False)

    m(torch.randn(2, 4)).sum().backward()
    m.weight.grad[0, 0] = float("nan")

    clip_params = [p for p in m.parameters() if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        [p.grad for p in clip_params], norm_type=2.0
    )
    grads_finite = bool(torch.isfinite(total_norm))
    assert grads_finite is False
    if grads_finite:
        scaler.step(opt)
    else:
        opt.zero_grad(set_to_none=True)
    scaler.update()

    assert bool(torch.isfinite(m.weight).all())
    assert bool((m.weight.detach() == w0).all())  # unchanged: step was skipped


def test_all_nonfinite_params_collected_and_hook_called():
    t = _bare_trainer()
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    m[0].weight.grad[0, 0] = float("nan")  # layer 0
    m[1].bias.grad[0] = float("inf")  # layer 1
    t.model = m

    captured = {}

    def hook(batch, step, nonfinite_params):
        captured["batch"] = batch
        captured["params"] = nonfinite_params

    t._on_nonfinite_grads_diag = hook  # override the no-op base hook
    sentinel_batch = [{"book_spine": object()}]
    with mock.patch("sam3.train.trainer.logging.warning") as warn:
        t._handle_nonfinite_grads("train", sentinel_batch)

    # the failing batch is handed to the diagnostic hook, with the FULL list
    assert captured["batch"] is sentinel_batch
    names = {p["name"] for p in captured["params"]}
    assert names == {"0.weight", "1.bias"}  # both, not just the first
    assert "nonfinite_param_tensors=2" in warn.call_args.args[0]


def _grad_diag_shell(tmp_path, accum=2):
    from core.training_loss_trace import LossTracingTrainer

    tr = LossTracingTrainer.__new__(LossTracingTrainer)
    tr.distributed_rank = 0
    tr._grad_nan_dump_count = 0
    tr._pending_step_datapoints = []
    tr.gradient_accumulation_steps = accum
    tr.epoch = 7
    tr.steps = {"train": 12345}
    tr.logging_conf = types.SimpleNamespace(log_dir=str(tmp_path))
    return tr


def _fake_datapoint():
    return types.SimpleNamespace(
        img_batch=torch.randn(1, 3, 8, 8),
        raw_images=[torch.randn(3, 16, 16)],
        find_text_batch=["book spine"],
        find_targets=[{"boxes": torch.randn(2, 4), "masks": torch.zeros(2, 8, 8)}],
        find_metadatas=[{"source_id": -1}],
    )


def test_grad_dump_writes_batch_targets_and_rng(tmp_path):
    from core.training_loss_trace import GRAD_NAN_DIAG_MAX_DUMPS

    tr = _grad_diag_shell(tmp_path, accum=2)
    dp = _fake_datapoint()
    # simulate _step capturing (key, datapoint) for each accum micro-batch
    tr._pending_step_datapoints = [("book_spine", dp), ("book_spine", dp)]
    nonfinite = [{"name": "backbone.vision_backbone.trunk.pos_embed", "nan": 576, "inf": 0}]

    tr._on_nonfinite_grads_diag(None, 12345, nonfinite)  # base passes emptied batch

    dumps = tmp_path / "nan_diagnostics"
    pts = list(dumps.glob("gradnan_*.pt"))
    jsons = list(dumps.glob("gradnan_*.json"))
    assert len(pts) == 1 and len(jsons) == 1

    blob = torch.load(pts[0], weights_only=False)
    assert len(blob["batch"]) == 2  # all accum micro-batches saved
    assert blob["batch"][0]["dataset_key"] == "book_spine"
    assert blob["batch"][0]["img_batch"] is not None  # <-- the popitem bug regressor
    assert blob["batch"][0]["find_text_batch"] == ["book spine"]
    assert blob["batch"][0]["find_targets"][0]["boxes"].device.type == "cpu"
    assert blob["nonfinite_grad_params"] == nonfinite
    assert "torch" in blob["rng"] and "numpy" in blob["rng"]
    meta = json.loads(jsons[0].read_text())
    assert meta["num_micro_batches"] == 2
    assert meta["img_batch_captured"] is True
    assert meta["num_nonfinite_grad_tensors"] == 1
    # buffer cleared after use (no leak across steps)
    assert tr._pending_step_datapoints == []

    # rate-limited: no more than GRAD_NAN_DIAG_MAX_DUMPS files
    for _ in range(GRAD_NAN_DIAG_MAX_DUMPS + 2):
        tr._pending_step_datapoints = [("book_spine", dp)]
        tr._on_nonfinite_grads_diag(None, 12345, nonfinite)
    assert len(list(dumps.glob("gradnan_*.pt"))) == GRAD_NAN_DIAG_MAX_DUMPS


def test_captured_datapoints_survive_popitem_consumption(tmp_path):
    """Regression for the v6 bug: the base _step does `batch.popitem()`, so the
    dict handed to the guard hook is empty. The datapoint must be captured BEFORE
    that, so the dump is not all-None."""
    tr = _grad_diag_shell(tmp_path, accum=2)
    dp = _fake_datapoint()

    # Simulate the two accum micro-batches as base _step sees them, capturing the
    # ref (as _step does) and THEN emptying the dict (as popitem does).
    for _ in range(tr.gradient_accumulation_steps):
        micro = {"book_spine": dp}
        key, datapoint = next(iter(micro.items()))  # captured before popitem
        if len(tr._pending_step_datapoints) >= tr.gradient_accumulation_steps:
            tr._pending_step_datapoints = []
        tr._pending_step_datapoints.append((key, datapoint))
        micro.popitem()  # base _step consumes it -> dict now empty
        assert micro == {}

    tr._on_nonfinite_grads_diag([{}, {}], 999, [])  # emptied batch handed in
    blob = torch.load(next((tmp_path / "nan_diagnostics").glob("gradnan_*.pt")), weights_only=False)
    assert len(blob["batch"]) == 2
    assert all(mb["img_batch"] is not None for mb in blob["batch"])
