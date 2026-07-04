"""Best-checkpoint selection: explicit hierarchical rules, no ad-hoc weighted score.

Rules (in order; all recorded into best_checkpoint.json):

1. Exclude candidates whose evaluation_status is not "completed", or whose
   mean_iou_all_gt is missing/NaN, and any candidate flagged is_baseline (the
   original sam3.pt baseline is reported separately, never auto-selected).
2. Highest mean_iou_all_gt wins.
3. If two candidates' mean_iou_all_gt differ by <= TIE_TOLERANCE (0.005):
   higher mean_boundary_f1_all_gt wins.
4. Still tied (same tolerance on boundary F1): lower miss_rate_iou_50 wins.
5. Still tied: lower false_positive_per_image wins.
6. Still tied: EARLIER epoch wins (no unjustified bias toward longer training).
"""

from __future__ import annotations

import math
from typing import Any

TIE_TOLERANCE = 0.005
SELECTION_METRIC = "mean_iou_all_gt"
SELECTION_RULE = (
    "max mean_iou_all_gt; ties within {tol} broken by higher mean_boundary_f1_all_gt, "
    "then lower miss_rate_iou_50, then lower false_positive_per_image, then earlier epoch"
).format(tol=TIE_TOLERANCE)


def _valid(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def eligible_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in candidates:
        if c.get("is_baseline"):
            continue
        if c.get("evaluation_status") != "completed":
            continue
        if not _valid(c.get("mean_iou_all_gt")):
            continue
        out.append(c)
    return out


def select_best(candidates: list[dict[str, Any]], tie_tolerance: float = TIE_TOLERANCE) -> dict[str, Any]:
    """Return {"status": "completed"|"blocked", "best": <candidate>|None, ...}.

    Never fabricates a best checkpoint: with zero eligible candidates the result is
    status="blocked" with a reason.
    """
    pool = eligible_candidates(candidates)
    if not pool:
        return {
            "status": "blocked",
            "best": None,
            "reason": (
                "no eligible checkpoint: all candidates failed evaluation, had no valid "
                "mean_iou_all_gt, or were baseline-only"
            ),
            "selection_metric": SELECTION_METRIC,
            "selection_rule": SELECTION_RULE,
            "tie_tolerance": tie_tolerance,
        }

    top_iou = max(c["mean_iou_all_gt"] for c in pool)
    contenders = [c for c in pool if top_iou - c["mean_iou_all_gt"] <= tie_tolerance]
    trace: list[str] = [
        f"{len(pool)} eligible checkpoint(s); top mean_iou_all_gt={top_iou:.6f}; "
        f"{len(contenders)} within tie tolerance {tie_tolerance}"
    ]

    if len(contenders) > 1:
        best_bf1 = max((c.get("mean_boundary_f1_all_gt") or float("-inf")) for c in contenders)
        if _valid(best_bf1):
            contenders = [
                c
                for c in contenders
                if _valid(c.get("mean_boundary_f1_all_gt"))
                and best_bf1 - c["mean_boundary_f1_all_gt"] <= tie_tolerance
            ] or contenders
            trace.append(
                f"tie-break 1 (mean_boundary_f1_all_gt, best={best_bf1:.6f}): {len(contenders)} remain"
            )

    if len(contenders) > 1:
        best_miss = min((c.get("miss_rate_iou_50") if _valid(c.get("miss_rate_iou_50")) else float("inf")) for c in contenders)
        if _valid(best_miss):
            contenders = [
                c for c in contenders if _valid(c.get("miss_rate_iou_50")) and c["miss_rate_iou_50"] == best_miss
            ] or contenders
            trace.append(f"tie-break 2 (miss_rate_iou_50, best={best_miss:.6f}): {len(contenders)} remain")

    if len(contenders) > 1:
        best_fp = min(
            (c.get("false_positive_per_image") if _valid(c.get("false_positive_per_image")) else float("inf"))
            for c in contenders
        )
        if _valid(best_fp):
            contenders = [
                c
                for c in contenders
                if _valid(c.get("false_positive_per_image")) and c["false_positive_per_image"] == best_fp
            ] or contenders
            trace.append(f"tie-break 3 (false_positive_per_image, best={best_fp:.6f}): {len(contenders)} remain")

    if len(contenders) > 1:
        contenders = sorted(contenders, key=lambda c: (c.get("epoch") if c.get("epoch") is not None else float("inf")))
        trace.append(f"tie-break 4 (earlier epoch): chose epoch {contenders[0].get('epoch')}")

    best = contenders[0]
    trace.append(f"selected {best.get('checkpoint_name')} (epoch {best.get('epoch')})")
    return {
        "status": "completed",
        "best": best,
        "reason": "; ".join(trace),
        "selection_metric": SELECTION_METRIC,
        "selection_rule": SELECTION_RULE,
        "tie_tolerance": tie_tolerance,
    }
