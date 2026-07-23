from __future__ import annotations

import math
import sys
from typing import Any

import torch
from sam3.train.matcher import BinaryHungarianMatcherV2

# Prefix so the guard's diagnostics are trivially greppable in a run's captured
# stdout/stderr (the same stream the crash traceback landed in).
GUARD_TAG = "[NUMERIC-GUARD]"


class NumericallyStableBinaryHungarianMatcherV2(BinaryHungarianMatcherV2):
    """Hungarian matching that never feeds NaN/Inf to scipy, and says why.

    The stock matcher builds its cost matrix from ``pred_logits``, ``pred_boxes``
    and the (padded) target boxes and hands it to
    ``scipy.optimize.linear_sum_assignment``, which aborts on any non-finite
    entry (``matrix contains invalid numeric entries``). Two things can make the
    cost non-finite:

    * a genuinely non-finite model/target value (NaN/Inf), or
    * a degenerate zero-area box, which makes ``generalized_box_iou`` divide by
      zero even though every input coordinate is finite.

    This subclass sanitizes both cases in an FP32 *copy* used only for matching —
    the real model outputs and the loss are untouched — so training keeps
    running, and it reports exactly which tensor was bad (pred_logits /
    pred_boxes / target boxes), with counts and box statistics, so the root cause
    can finally be pinned instead of guessed at. Matching math is otherwise
    identical to the base class, so results on clean inputs are unchanged.
    """

    def __init__(
        self,
        *args: Any,
        min_box_side: float = 1e-6,
        max_abs_logit: float = 80.0,
        degenerate_side: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        min_side = float(min_box_side)
        max_logit = float(max_abs_logit)
        degen = float(degenerate_side)
        if not math.isfinite(min_side) or min_side <= 0.0 or min_side >= 1.0:
            raise ValueError("min_box_side must be finite and in the open interval (0, 1)")
        if not math.isfinite(max_logit) or max_logit <= 0.0:
            raise ValueError("max_abs_logit must be finite and positive")
        if not math.isfinite(degen) or degen < min_side or degen >= 1.0:
            raise ValueError("degenerate_side must be finite and in [min_box_side, 1)")
        super().__init__(*args, **kwargs)
        self.min_box_side = min_side
        self.max_abs_logit = max_logit
        self.degenerate_side = degen
        self._report_counts: dict[str, int] = {}

    def _report(self, category: str, message: str) -> None:
        # Rate-limited so a fully diverged model cannot flood the log: every one
        # of the first 50 events, then one in 200.
        count = self._report_counts.get(category, 0) + 1
        self._report_counts[category] = count
        if count <= 50 or count % 200 == 0:
            print(f"{GUARD_TAG} {message} (occurrence #{count})", file=sys.stderr, flush=True)

    def _sanitize(self, value: torch.Tensor, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"matcher {label} must be a torch.Tensor")
        value = value.to(dtype=torch.float32)
        finite = torch.isfinite(value)
        if not bool(finite.all().item()):
            nan_n = int(torch.isnan(value).sum().item())
            posinf_n = int(torch.isposinf(value).sum().item())
            neginf_n = int(torch.isneginf(value).sum().item())
            if bool(finite.any().item()):
                fin = value[finite]
                span = f"finite range=[{fin.min().item():.4g}, {fin.max().item():.4g}]"
            else:
                span = "no finite entries"
            self._report(
                f"nonfinite:{label}",
                f"non-finite {label} from the model/targets: nan={nan_n} "
                f"+inf={posinf_n} -inf={neginf_n} shape={tuple(value.shape)} {span} "
                "-> sanitized so matching can continue; this points at a model "
                "output problem, not a degenerate box",
            )
            value = torch.nan_to_num(
                value, nan=0.0, posinf=self.max_abs_logit, neginf=-self.max_abs_logit
            )
        return value

    def _stable_boxes(self, value: torch.Tensor, label: str) -> torch.Tensor:
        boxes = self._sanitize(value, label).clamp(min=0.0, max=1.0)
        sides = boxes[..., 2:]
        degenerate = sides < self.degenerate_side
        if bool(degenerate.any().item()):
            n = int(degenerate.any(dim=-1).sum().item())
            min_side = float(sides.min().item())
            # One representative offending box, in cxcywh, for the log.
            flat = boxes.reshape(-1, boxes.shape[-1])
            bad_row = (flat[:, 2:] < self.degenerate_side).any(dim=-1).nonzero()
            example = flat[bad_row[0, 0]].tolist() if bad_row.numel() else None
            self._report(
                f"degenerate:{label}",
                f"degenerate box in {label}: {n} box(es) with a side < "
                f"{self.degenerate_side:g} (min side={min_side:.3g}), example "
                f"cxcywh={example} -> floored to {self.min_box_side:g} so GIoU "
                "stays finite",
            )
        boxes = boxes.clone()
        boxes[..., 2:] = sides.clamp_min(self.min_box_side)
        return boxes

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Any],
        batched_targets: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ):
        safe_outputs = dict(outputs)
        safe_outputs["pred_logits"] = self._sanitize(
            outputs["pred_logits"], "pred_logits"
        ).clamp(min=-self.max_abs_logit, max=self.max_abs_logit)
        safe_outputs["pred_boxes"] = self._stable_boxes(outputs["pred_boxes"], "pred_boxes")

        safe_targets = dict(batched_targets)
        for key in ("boxes", "boxes_padded"):
            if isinstance(batched_targets.get(key), torch.Tensor):
                safe_targets[key] = self._stable_boxes(
                    batched_targets[key], f"target.{key}"
                )

        try:
            return super().forward(safe_outputs, safe_targets, *args, **kwargs)
        except ValueError as exc:
            if "invalid numeric entries" not in str(exc):
                raise
            # Inputs were made finite and non-degenerate above, so the base cost
            # cannot be non-finite from a known path. Reaching here means an
            # unmodeled source; surface it loudly rather than masking it.
            self._report(
                "postcost",
                "cost matrix still non-finite AFTER input stabilization -- "
                "unmodeled source in the matcher cost; re-raising",
            )
            raise
