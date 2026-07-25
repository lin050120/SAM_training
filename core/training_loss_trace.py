from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.trainer import Trainer
from sam3.train.utils.distributed import unwrap_ddp_if_wrapped
from sam3.train.utils.train_utils import Phase, get_amp_type

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

# Per-epoch test-split loss, computed right after the normal validation pass so
# the already-resident model/criterion are reused: no extra GPU memory beyond one
# no-grad forward. Written one JSON object per line, same shape as SAM3's
# val_stats.json, so the UI can plot it exactly like the val curve.
TEST_LOSS_STATS_FILENAME = "test_stats.json"
TEST_LOSS_TAG = "[TEST-LOSS]"
VAL_SPLIT_DIRNAME = "val"
TEST_SPLIT_DIRNAME = "test"


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
        test_loss_data: Any = None,
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
        # (key, datapoint) refs captured BEFORE base _step's popitem consumes the
        # batch, accumulated across the accum micro-batches of one optimizer step,
        # so the grad-time guard hook can snapshot the actual failing batch.
        self._pending_step_datapoints: list[tuple[Any, Any]] = []
        # Per-epoch test loss. `test_loss_data` is an optional explicit loader
        # config (same shape as data.val); without it the test split is derived
        # from data.val, so existing runs pick this up on resume with no config
        # change. Instantiated lazily once, then reused every epoch.
        self._test_loss_data_conf = test_loss_data
        self._test_dataset: Any = None
        self._test_dataset_resolved = False
        # Identity of the annotations actually loaded. The dataset parses the
        # file once at build time, so later edits to it do not change the
        # numbers -- recording the digest here is what makes points from
        # different runs (or a mid-project test-set change) comparable.
        self._test_annotations_path: str | None = None
        self._test_annotations_sha256: str | None = None
        self._test_stats_path = (
            Path(self.logging_conf.log_dir) / TEST_LOSS_STATS_FILENAME
        )

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
        augmentation / op can be found offline. Fully guarded upstream.

        NOTE: the base guard passes `batch`, but by now base _step's popitem has
        emptied those dicts, so we use the (key, datapoint) refs captured in
        _step BEFORE popitem (self._pending_step_datapoints)."""
        if getattr(self, "distributed_rank", 0) != 0:
            self._pending_step_datapoints = []
            return
        if self._grad_nan_dump_count >= GRAD_NAN_DIAG_MAX_DUMPS:
            self._pending_step_datapoints = []
            return
        self._grad_nan_dump_count += 1
        index = self._grad_nan_dump_count
        try:
            # The captured datapoints are this optimizer step's accum micro-
            # batches; the NaN grad is their SUM, so save them all -- the culprit
            # is among them (re-run each offline to find which one + BF16-vs-FP32).
            captured = list(self._pending_step_datapoints)
            saved: list[dict[str, Any]] = []
            for key, dp in captured:
                saved.append(
                    {
                        "dataset_key": key,
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
                "num_micro_batches": len(captured),
                "img_batch_captured": bool(
                    saved and any(s["img_batch"] is not None for s in saved)
                ),
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
        finally:
            # Release the captured micro-batch refs for this optimizer step.
            self._pending_step_datapoints = []

    # ------------------------------------------------------------------
    # Per-epoch test-split loss
    # ------------------------------------------------------------------
    @staticmethod
    def _swap_split_dir(path_value: Any, source: str, target: str) -> str | None:
        """Rewrite `.../<source>/...` as `.../<target>/...` on the last match."""
        if not path_value:
            return None
        parts = list(Path(str(path_value)).parts)
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == source:
                parts[index] = target
                return str(Path(*parts))
        return None

    def _resolve_test_loader_conf(self):
        """Explicit `test_loss_data` if configured, else `data.val` repointed at
        the sibling test split. Returns None when there is no usable test set."""
        if self._test_loss_data_conf is not None:
            return OmegaConf.create(
                OmegaConf.to_container(self._test_loss_data_conf, resolve=True)
            )
        val_conf = self.data_conf.get(Phase.VAL, None) if self.data_conf else None
        if val_conf is None:
            return None
        conf = OmegaConf.create(OmegaConf.to_container(val_conf, resolve=True))
        dataset = conf.get("dataset")
        if dataset is None:
            return None
        annotations = self._swap_split_dir(
            dataset.get("ann_file"), VAL_SPLIT_DIRNAME, TEST_SPLIT_DIRNAME
        )
        images = self._swap_split_dir(
            dataset.get("img_folder"), VAL_SPLIT_DIRNAME, TEST_SPLIT_DIRNAME
        )
        if annotations is None or images is None:
            return None
        if not Path(annotations).is_file() or not Path(images).is_dir():
            return None
        dataset.ann_file = annotations
        dataset.img_folder = images
        return conf

    def _get_test_dataset(self):
        """Build the test loader once; None disables the feature for this run."""
        if self._test_dataset_resolved:
            return self._test_dataset
        self._test_dataset_resolved = True
        try:
            conf = self._resolve_test_loader_conf()
            if conf is None:
                print(
                    f"{TEST_LOSS_TAG} no usable test split; per-epoch test loss disabled",
                    flush=True,
                )
                return None
            self._test_dataset = instantiate(conf)
            self._test_annotations_path = str(conf.dataset.ann_file)
            self._test_annotations_sha256 = self._sha256_of_file(
                self._test_annotations_path
            )
            print(
                f"{TEST_LOSS_TAG} enabled, annotations={self._test_annotations_path} "
                f"sha256={self._test_annotations_sha256}",
                flush=True,
            )
        except Exception as exc:  # a broken test split must not stop training
            self._test_dataset = None
            print(
                f"{TEST_LOSS_TAG} could not build the test loader: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        return self._test_dataset

    @staticmethod
    def _sha256_of_file(path: str | Path) -> str | None:
        """Digest of the annotations actually loaded; None if unreadable."""
        try:
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        return {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    @classmethod
    def _restore_rng_state(cls, state: dict[str, Any]) -> None:
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])

    def _run_test_loss_eval(self) -> None:
        """Sample-weighted mean loss over the whole test split, reusing the model
        and criterion the validation pass just used -- so it costs one extra
        no-grad forward and no extra GPU memory.

        Deliberately does NOT go through the base `_step`: that bumps
        `self.steps[phase]` and updates the phase meters, which are only reset at
        the end of `val_epoch`, so reusing it would corrupt the validation
        metrics with test data. The forward/loss below mirrors `_step` exactly.

        Never raises, and restores RNG state, so a broken or missing test split
        can neither stop training nor perturb the online augmentation stream.
        """
        dataset = self._get_test_dataset()
        if dataset is None:
            return
        rng_state = self._capture_rng_state()
        model = self.model
        inner = unwrap_ddp_if_wrapped(model)
        was_training = model.training
        try:
            amp_enabled = bool(self.optim_conf.amp.enabled) if self.optim_conf else False
            amp_dtype = (
                get_amp_type(self.optim_conf.amp.amp_dtype) if self.optim_conf else None
            )
            loader = dataset.get_loader(epoch=int(self.epoch))
            model.eval()
            if hasattr(inner, "on_validation_epoch_start"):
                inner.on_validation_epoch_start()

            weighted: dict[str, float] = {}
            samples = 0
            dataset_key: str | None = None
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=amp_enabled,
                    dtype=amp_dtype,
                ):
                    for batch in loader:
                        dataset_key, datapoint = batch.popitem()
                        datapoint = copy_data_to_device(
                            datapoint, self.device, non_blocking=True
                        )
                        find_stages = model(datapoint)
                        find_targets = [
                            inner.back_convert(x) for x in datapoint.find_targets
                        ]
                        loss = self._find_loss(dataset_key)(find_stages, find_targets)
                        batch_size = len(datapoint.img_batch)
                        samples += batch_size
                        components = (
                            loss.items()
                            if isinstance(loss, dict)
                            else {"core_loss": loss}.items()
                        )
                        for name, value in components:
                            weighted[name] = (
                                weighted.get(name, 0.0)
                                + float(value.detach().float().item()) * batch_size
                            )

            if hasattr(inner, "on_validation_epoch_end"):
                inner.on_validation_epoch_end()
            del loader
            gc.collect()

            if samples < 1 or dataset_key is None:
                print(
                    f"{TEST_LOSS_TAG} test split yielded no samples; skipping epoch "
                    f"{self.epoch}",
                    file=sys.stderr,
                    flush=True,
                )
                return

            averages = {name: total / samples for name, total in weighted.items()}
            record: dict[str, Any] = {
                f"Losses/test_{dataset_key}_{name}": value
                for name, value in sorted(averages.items())
            }
            # Headline number, named like SAM3's own `Losses/val_<key>_loss`.
            record[f"Losses/test_{dataset_key}_loss"] = averages.get(
                "core_loss", float("nan")
            )
            record["Trainer/epoch"] = self.epoch
            record["Trainer/test_samples"] = samples
            # Pins each point to the exact test set it was measured on, so a
            # mid-project change to the split is visible instead of silently
            # mixing two incomparable curves.
            record["Data/test_annotations"] = self._test_annotations_path
            record["Data/test_annotations_sha256"] = self._test_annotations_sha256
            self._write_test_stats(record, dataset_key)
        except Exception as exc:  # test loss is diagnostic; never stop training
            print(
                f"{TEST_LOSS_TAG} failed for epoch {self.epoch}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            self._restore_rng_state(rng_state)
            if was_training:
                model.train()

    def _write_test_stats(self, record: dict[str, Any], dataset_key: str) -> None:
        if self.distributed_rank != 0:
            return
        headline = record.get(f"Losses/test_{dataset_key}_loss")
        print(
            f"{TEST_LOSS_TAG} epoch={self.epoch} loss={headline} "
            f"samples={record.get('Trainer/test_samples')}",
            flush=True,
        )
        try:
            self.logger.log_dict(record, self.epoch)
        except Exception as exc:
            print(
                f"{TEST_LOSS_TAG} tensorboard log failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        self._test_stats_path.parent.mkdir(parents=True, exist_ok=True)
        with self._test_stats_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def run_val(self):
        """Normal validation, then the test-split loss for the same weights."""
        super().run_val()
        self._run_test_loss_eval()

    def _step(self, batch: Any, model: Any, phase: str):
        # The base _step pops (key, datapoint) out of this dict, so grab a
        # reference first -- both for the loss diagnostic below and, accumulated
        # across the accum micro-batches, for the grad-guard batch snapshot.
        captured_key = None
        captured_datapoint = None
        if phase == Phase.TRAIN and isinstance(batch, dict) and batch:
            captured_key, captured_datapoint = next(iter(batch.items()))
            # A finished optimizer step leaves a full buffer that no bad-grad hook
            # cleared (the step was good): reset before the new step accumulates,
            # so we hold at most one optimizer step's micro-batches (no leak).
            if len(self._pending_step_datapoints) >= self.gradient_accumulation_steps:
                self._pending_step_datapoints = []
            self._pending_step_datapoints.append((captured_key, captured_datapoint))
        result = super()._step(batch, model, phase)
        if phase == Phase.TRAIN:
            loss_dict, batch_size, _extra_losses = result
            loss = next(iter(loss_dict.values()))
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value) and captured_datapoint is not None:
                self._dump_nonfinite_diagnostic(captured_datapoint, model, loss_value)
            self._record_training_microbatch(loss_value, int(batch_size))
        return result
