from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.trainer import Trainer
from sam3.train.utils.train_utils import Phase

TRAIN_LOSS_TRACE_FILENAME = "train_optimizer_step_loss.jsonl"
NAN_DIAG_TAG = "[NAN-DIAG]"
NAN_DIAG_DIRNAME = "nan_diagnostics"
# Only dump for the first few non-finite steps: training stops right after, and a
# handful is plenty to diagnose while keeping logs/files small.
NAN_DIAG_MAX_DUMPS = 3

# Grad-time counterpart: GRAD-GUARD skips a step when the (accumulated) gradient
# is non-finite while the loss is still finite, so the loss-triggered dump above
# never fires. This captures the failing augmented batch + targets + RNG so the
# culprit sample/aug/op can be found offline.
GRAD_NAN_DIAG_TAG = "[GRAD-NAN-DIAG]"
GRAD_NAN_DIAG_MAX_DUMPS = 3


class ValidationMatchingSam3LossWrapper(Sam3LossWrapper):
    """Compute missing image-model matcher indices before validation loss."""

    @staticmethod
    def _loss_outputs(nested_out: dict[str, Any]):
        yield nested_out
        yield from nested_out.get("aux_outputs", [])
        first_stage = nested_out.get("first_stage")
        if first_stage is not None:
            yield first_stage

    def _ensure_validation_indices(
        self,
        nested_out: dict[str, Any],
        targets: Any,
    ) -> None:
        # SAM3Image computes these while training, but skips them in eval mode
        # when no interactive validation steps are configured.
        if torch.is_grad_enabled():
            return
        missing_outputs = [
            output for output in self._loss_outputs(nested_out) if "indices" not in output
        ]
        if not missing_outputs:
            return
        if self.matcher is None:
            raise RuntimeError(
                "validation loss requires matcher indices, but no matcher is configured"
            )
        for output in missing_outputs:
            output["indices"] = self.matcher(output, targets)

    def compute_loss(self, nested_out: dict[str, Any], targets: Any):
        self._ensure_validation_indices(nested_out, targets)
        return super().compute_loss(nested_out, targets)


class LossTracingTrainer(Trainer):
    """Record one train-loss moving average per optimizer-step window."""

    def __init__(
        self,
        *,
        train_loss_window_optimizer_steps: int = 20,
        **kwargs: Any,
    ) -> None:
        try:
            parsed_window = int(train_loss_window_optimizer_steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "train_loss_window_optimizer_steps must be a positive whole number"
            ) from exc
        if parsed_window != train_loss_window_optimizer_steps or parsed_window < 1:
            raise ValueError(
                "train_loss_window_optimizer_steps must be a positive whole number"
            )
        super().__init__(**kwargs)
        self.train_loss_window_optimizer_steps = parsed_window
        self._trace_micro_loss_sum = 0.0
        self._trace_micro_sample_count = 0
        self._trace_optimizer_losses: list[float] = []
        self._train_loss_trace_path = (
            Path(self.logging_conf.log_dir) / TRAIN_LOSS_TRACE_FILENAME
        )
        self._nonfinite_dump_count = 0
        self._grad_nan_dump_count = 0

    def _record_training_microbatch(self, loss: float, batch_size: int) -> None:
        if self.distributed_rank != 0:
            return
        if not math.isfinite(loss) or batch_size < 1:
            return
        self._trace_micro_loss_sum += loss * batch_size
        self._trace_micro_sample_count += batch_size

        accumulation = int(self.gradient_accumulation_steps)
        completed_micro_steps = int(self.steps[Phase.TRAIN])
        if completed_micro_steps % accumulation != 0:
            return

        optimizer_loss = (
            self._trace_micro_loss_sum / self._trace_micro_sample_count
        )
        self._trace_micro_loss_sum = 0.0
        self._trace_micro_sample_count = 0
        self._trace_optimizer_losses.append(optimizer_loss)
        window = self.train_loss_window_optimizer_steps
        if len(self._trace_optimizer_losses) < window:
            return

        optimizer_step = completed_micro_steps // accumulation
        record = {
            "optimizer_step": optimizer_step,
            "window_optimizer_steps": window,
            "loss": sum(self._trace_optimizer_losses[-window:]) / window,
        }
        self._train_loss_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._train_loss_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        self._trace_optimizer_losses.clear()

    @staticmethod
    def _tensor_report(value: Any) -> dict[str, Any]:
        if not isinstance(value, torch.Tensor):
            return {"type": type(value).__name__}
        finite = torch.isfinite(value)
        all_finite = bool(finite.all().item())
        report: dict[str, Any] = {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "all_finite": all_finite,
        }
        if not all_finite:
            report["nan"] = int(torch.isnan(value).sum().item())
            report["posinf"] = int(torch.isposinf(value).sum().item())
            report["neginf"] = int(torch.isneginf(value).sum().item())
        if bool(finite.any().item()):
            finite_values = value[finite].float()
            report["finite_min"] = float(finite_values.min().item())
            report["finite_max"] = float(finite_values.max().item())
        return report

    def _dump_nonfinite_diagnostic(self, datapoint: Any, model: Any, loss_value: float) -> None:
        """On a non-finite train loss, record what actually went wrong.

        The two questions this answers, which the current logs cannot:
        * are the model PARAMETERS non-finite? -> weights were corrupted by a
          prior optimizer step (an optimization problem), vs
        * are params + the input image finite but the output NaN? -> the NaN is
          generated inside this forward on a finite augmented sample (a per-batch
          numerical instability; the weights are still fine and skipping this one
          batch would be safe).
        It also snapshots the failing batch for offline replay. Fully guarded so
        the diagnostic can never take the training process down with it.
        """
        if getattr(self, "distributed_rank", 0) != 0:
            return
        if self._nonfinite_dump_count >= NAN_DIAG_MAX_DUMPS:
            return
        self._nonfinite_dump_count += 1
        index = self._nonfinite_dump_count
        try:
            net = getattr(model, "module", model)
            bad_params: list[str] = []
            total_bad = 0
            for name, param in net.named_parameters():
                if not bool(torch.isfinite(param).all().item()):
                    total_bad += 1
                    if len(bad_params) < 10:
                        nan_n = int(torch.isnan(param).sum().item())
                        inf_n = int((~torch.isfinite(param)).sum().item()) - nan_n
                        bad_params.append(f"{name}(nan={nan_n},inf={inf_n})")
            weights_finite = total_bad == 0

            img = getattr(datapoint, "img_batch", None)
            img_report = self._tensor_report(img)
            input_finite = bool(img_report.get("all_finite", False))

            if weights_finite and input_finite:
                verdict = (
                    "WEIGHTS FINITE + INPUT FINITE -> NaN is generated inside this forward "
                    "on a finite augmented sample; weights are NOT corrupted, so skipping "
                    "this single batch would be safe. Look for a model-side instability "
                    "(norm/softmax/division) triggered by this input."
                )
            elif weights_finite:
                verdict = (
                    "WEIGHTS FINITE but INPUT non-finite -> the data/augmentation pipeline "
                    "produced a NaN/Inf input image; fix the offending transform."
                )
            else:
                verdict = (
                    f"WEIGHTS non-finite ({total_bad} param tensors) -> parameters were "
                    "already corrupted by a prior optimizer step; not a single-batch forward "
                    "issue. Skipping batches would only hide it."
                )

            payload = {
                "diag_index": index,
                "epoch": int(getattr(self, "epoch", -1)),
                "steps_train": int(self.steps[Phase.TRAIN]),
                "loss_value": loss_value,
                "weights_finite": weights_finite,
                "num_nonfinite_param_tensors": total_bad,
                "example_bad_params": bad_params,
                "input_img_batch": img_report,
                "find_text_batch": list(getattr(datapoint, "find_text_batch", []) or [])[:16],
                "verdict": verdict,
            }
            print(f"{NAN_DIAG_TAG} {json.dumps(payload, ensure_ascii=True)}", file=sys.stderr, flush=True)

            dump_dir = Path(self.logging_conf.log_dir) / NAN_DIAG_DIRNAME
            dump_dir.mkdir(parents=True, exist_ok=True)
            stem = dump_dir / f"nan_{index}_ep{payload['epoch']}_step{payload['steps_train']}"
            stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            snapshot = {
                "img_batch": img.detach().cpu() if isinstance(img, torch.Tensor) else None,
                "find_text_batch": getattr(datapoint, "find_text_batch", None),
            }
            torch.save(snapshot, stem.with_suffix(".pt"))
        except Exception as exc:  # diagnostics must never crash training
            print(f"{NAN_DIAG_TAG} failed to record diagnostic: {exc!r}", file=sys.stderr, flush=True)

    @classmethod
    def _to_cpu(cls, value: Any) -> Any:
        """Best-effort move of (possibly nested) tensors to CPU for a picklable
        snapshot; leaves non-tensor objects as-is."""
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu")
        if isinstance(value, dict):
            return {k: cls._to_cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            seq = [cls._to_cpu(v) for v in value]
            return type(value)(seq) if isinstance(value, tuple) else seq
        return value

    def _on_nonfinite_grads_diag(self, batch, step, nonfinite_params) -> None:
        """Grad-time counterpart of _dump_nonfinite_diagnostic. GRAD-GUARD calls
        this when the (accumulated) gradient is non-finite while the loss was
        still finite -- so the loss-triggered dump never fires. Snapshot every
        augmented micro-batch (img + raw image + targets/masks + metadata/source
        id) plus RNG and the full non-finite-param list, so the culprit sample /
        augmentation / op can be found offline. Fully guarded upstream."""
        if getattr(self, "distributed_rank", 0) != 0:
            return
        if self._grad_nan_dump_count >= GRAD_NAN_DIAG_MAX_DUMPS:
            return
        self._grad_nan_dump_count += 1
        index = self._grad_nan_dump_count
        try:
            # batch is the list of accum micro-batches (or a single one); the NaN
            # grad is their SUM, so save them all -- the culprit is among them.
            micro = batch if isinstance(batch, list) else [batch]
            saved: list[dict[str, Any]] = []
            for mb in micro:
                dp = next(iter(mb.values())) if isinstance(mb, dict) and mb else mb
                saved.append(
                    {
                        "img_batch": self._to_cpu(getattr(dp, "img_batch", None)),
                        "raw_images": self._to_cpu(getattr(dp, "raw_images", None)),
                        "find_text_batch": getattr(dp, "find_text_batch", None),
                        "find_targets": self._to_cpu(getattr(dp, "find_targets", None)),
                        "find_metadatas": self._to_cpu(
                            getattr(dp, "find_metadatas", None)
                        ),
                    }
                )
            rng = {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            }
            epoch = int(getattr(self, "epoch", -1))
            meta = {
                "diag_index": index,
                "kind": "grad_nan",
                "epoch": epoch,
                "steps_train": int(step),
                "num_micro_batches": len(micro),
                "num_nonfinite_grad_tensors": len(nonfinite_params),
                "nonfinite_grad_params": nonfinite_params[:500],
                "note": (
                    "Loss was finite; only the ACCUMULATED gradient was non-finite. "
                    "The first param is a HINT only (backprop can spread NaN, e.g. "
                    "attention mixes all tokens) -- inspect the whole param list and "
                    "re-run each saved micro-batch to find the true origin. RNG is the "
                    "main-process state; augmentation ran in dataloader workers, so the "
                    "saved augmented tensors (not RNG) are the ground truth."
                ),
            }
            print(
                f"{GRAD_NAN_DIAG_TAG} {json.dumps(meta, ensure_ascii=True)}",
                file=sys.stderr,
                flush=True,
            )
            dump_dir = Path(self.logging_conf.log_dir) / NAN_DIAG_DIRNAME
            dump_dir.mkdir(parents=True, exist_ok=True)
            stem = dump_dir / f"gradnan_{index}_ep{epoch}_step{int(step)}"
            stem.with_suffix(".json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            torch.save(
                {
                    "batch": saved,
                    "rng": rng,
                    "nonfinite_grad_params": nonfinite_params,
                    "meta": meta,
                },
                stem.with_suffix(".pt"),
            )
        except Exception as exc:  # diagnostics must never crash training
            print(
                f"{GRAD_NAN_DIAG_TAG} failed to record grad diagnostic: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    def _step(self, batch: Any, model: Any, phase: str):
        # The base _step pops the datapoint out of this dict, so grab a reference
        # first in case we need it for a non-finite diagnostic below.
        captured_datapoint = None
        if phase == Phase.TRAIN and isinstance(batch, dict) and batch:
            captured_datapoint = next(iter(batch.values()))
        result = super()._step(batch, model, phase)
        if phase == Phase.TRAIN:
            loss_dict, batch_size, _extra_losses = result
            loss = next(iter(loss_dict.values()))
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value) and captured_datapoint is not None:
                self._dump_nonfinite_diagnostic(captured_datapoint, model, loss_value)
            self._record_training_microbatch(loss_value, int(batch_size))
        return result
