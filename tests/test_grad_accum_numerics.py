"""Numerical tests for the SAM3 trainer gradient-accumulation loss scaling patch.

These drive the REAL (patched) sam3.train.trainer.Trainer._run_step — not a
re-implementation — via Trainer.__new__ plus the minimal attributes _run_step
touches. AMP autocast and GradScaler are constructed with enabled=False, so no
CUDA context is created; everything runs CPU-only in float64 for strict
tolerances. No SAM3 model is built and no training process is started.

Patch under test (sam3/train/trainer.py::_run_step):
    backward_loss = loss / accum_steps if accum_steps > 1 else loss
    self.scaler.scale(backward_loss).backward()
which makes accumulating accum_steps micro-batch gradients equal the gradient
of the MEAN loss over those micro-batches, while logged losses stay unscaled.
"""

from __future__ import annotations

import copy
import sys
import unittest
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import SAM301_ROOT  # noqa: E402

SAM301_TRAINER = SAM301_ROOT / "sam3" / "train" / "trainer.py"


class _RecordingMeter:
    def __init__(self) -> None:
        self.values: list[tuple[float, int]] = []

    def update(self, val: float, n: int) -> None:
        self.values.append((val, n))


def _make_trainer(optimizer, accum_steps):
    import torch
    from sam3.train.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.gradient_accumulation_steps = accum_steps
    trainer.optim = SimpleNamespace(
        zero_grad=lambda set_to_none=True: optimizer.zero_grad(set_to_none=set_to_none),
        optimizer=optimizer,
    )
    trainer.model = SimpleNamespace(no_sync=nullcontext)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.optim_conf = SimpleNamespace(amp=SimpleNamespace(enabled=False, amp_dtype="bfloat16"))
    return trainer


def _mse_step_fn(model):
    """A _step stand-in computing per-micro-batch MSE (mean over its samples)."""
    import torch

    def _step(batch, _model, _phase):
        x, y = batch
        loss = torch.nn.functional.mse_loss(model(x), y)
        return {"Losses/train_all_loss": loss}, x.shape[0], {}

    return _step


def _run(trainer, model, batch):
    trainer._step = _mse_step_fn(model)
    loss_mts = defaultdict(_RecordingMeter)
    trainer._run_step(batch, "train", loss_mts, {})
    return loss_mts


class GradAccumNumericsTest(unittest.TestCase):
    def _make_model_and_data(self):
        import torch

        torch.manual_seed(20260703)
        model = torch.nn.Linear(3, 1).double()
        x = torch.randn(4, 3, dtype=torch.float64)
        y = torch.randn(4, 1, dtype=torch.float64)
        return model, x, y

    def test_patch_is_present_in_sam301_trainer_source(self) -> None:
        source = SAM301_TRAINER.read_text(encoding="utf-8")
        self.assertIn("backward_loss = loss / accum_steps if accum_steps > 1 else loss", source)
        self.assertIn("self.scaler.scale(backward_loss).backward()", source)

    def test_batch4_accum1_equals_microbatch1_accum4_after_one_optimizer_step(self) -> None:
        """Same init weights + same 4 samples: one big batch (accum=1) and four
        1-sample micro-batches (accum=4) must produce the same parameter update."""
        import torch

        model_a, x, y = self._make_model_and_data()
        model_b = copy.deepcopy(model_a)
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)

        trainer_a = _make_trainer(opt_a, accum_steps=1)
        _run(trainer_a, model_a, (x, y))
        opt_a.step()

        trainer_b = _make_trainer(opt_b, accum_steps=4)
        micro_batches = [(x[i : i + 1], y[i : i + 1]) for i in range(4)]
        _run(trainer_b, model_b, micro_batches)
        opt_b.step()

        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(
                torch.allclose(p_a, p_b, rtol=1e-12, atol=1e-12),
                f"parameter mismatch after one step: max diff "
                f"{(p_a - p_b).abs().max().item():.3e}",
            )

    def test_accum1_behavior_is_unchanged_exact(self) -> None:
        """accum=1 must be bit-for-bit identical to a plain unscaled backward."""
        import torch

        model, x, y = self._make_model_and_data()
        reference = copy.deepcopy(model)

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = _make_trainer(opt, accum_steps=1)
        loss_mts = _run(trainer, model, (x, y))

        reference.zero_grad(set_to_none=True)
        ref_loss = torch.nn.functional.mse_loss(reference(x), y)
        ref_loss.backward()

        for p, p_ref in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(p.grad, p_ref.grad))
        # logged loss is the raw loss
        (logged, n), = loss_mts["Losses/train_all_loss"].values
        self.assertAlmostEqual(logged, ref_loss.item(), places=15)
        self.assertEqual(n, 4)

    def test_accum2_and_accum4_scale_backward_but_log_raw_loss(self) -> None:
        """N identical micro-batches: accumulated grad == grad of a single micro
        loss (mean semantics), and every logged loss value stays unscaled."""
        import torch

        for accum in (2, 4):
            with self.subTest(accum=accum):
                model, x, y = self._make_model_and_data()
                reference = copy.deepcopy(model)
                micro = (x[:1], y[:1])

                opt = torch.optim.SGD(model.parameters(), lr=0.1)
                trainer = _make_trainer(opt, accum_steps=accum)
                loss_mts = _run(trainer, model, [micro] * accum)

                reference.zero_grad(set_to_none=True)
                ref_loss = torch.nn.functional.mse_loss(reference(x[:1]), y[:1])
                ref_loss.backward()

                for p, p_ref in zip(model.parameters(), reference.parameters()):
                    self.assertTrue(
                        torch.allclose(p.grad, p_ref.grad, rtol=1e-12, atol=1e-12),
                        f"accum={accum}: grads are not mean-equivalent",
                    )
                logged_values = [v for v, _ in loss_mts["Losses/train_all_loss"].values]
                self.assertEqual(len(logged_values), accum)
                for value in logged_values:
                    self.assertAlmostEqual(value, ref_loss.item(), places=15)

    def test_batch_list_contract_still_enforced(self) -> None:
        """The pre-existing list/len asserts must still fire (patch didn't weaken them)."""
        import torch

        model, x, y = self._make_model_and_data()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = _make_trainer(opt, accum_steps=4)
        trainer._step = _mse_step_fn(model)
        with self.assertRaises(AssertionError):
            trainer._run_step((x, y), "train", defaultdict(_RecordingMeter), {})
        with self.assertRaises(AssertionError):
            trainer._run_step([(x[:1], y[:1])] * 3, "train", defaultdict(_RecordingMeter), {})


if __name__ == "__main__":
    unittest.main()
