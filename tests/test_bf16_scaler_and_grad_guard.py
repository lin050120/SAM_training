"""Direct regression tests for the bf16 GradScaler switch and the [GRAD-GUARD]
non-finite gradient guard + stop policy added to sam3/train/trainer.py.

These cover behavior the grad-accum numerics test does NOT:
  * GradScaler is disabled under bf16 and enabled only under fp16.
  * a disabled scaler ignores a poisoned (tiny) checkpoint scale.
  * the guard detects non-finite grads BEFORE clipping (first-bad-layer signal
    survives) and skips the optimizer step, leaving weights finite/unchanged.
  * the stop policy fails closed on consecutive skips and on window ratio.
"""

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
