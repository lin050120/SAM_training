from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from sam3.train.trainer import Trainer
from sam3.train.utils.train_utils import Phase

TRAIN_LOSS_TRACE_FILENAME = "train_optimizer_step_loss.jsonl"


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

    def _step(self, batch: Any, model: Any, phase: str):
        result = super()._step(batch, model, phase)
        if phase == Phase.TRAIN:
            loss_dict, batch_size, _extra_losses = result
            loss = next(iter(loss_dict.values()))
            loss_value = float(loss.detach().item())
            self._record_training_microbatch(loss_value, int(batch_size))
        return result
