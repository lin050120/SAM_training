"""One-to-one instance mask matching between GT and predictions.

Instance segmentation evaluation: a prediction may match at most one GT and vice
versa. Matching maximizes total IoU via the Hungarian algorithm
(scipy.optimize.linear_sum_assignment — scipy is already a project dependency via
SAM3's own matcher). Pairs whose IoU is exactly 0 are discarded from the assignment
result: a zero-overlap "match" is not a match, the GT counts as missed and the
prediction as a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MATCHING_METHOD = "hungarian_max_total_iou"


def mask_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    intersection = np.logical_and(gt, pred).sum(dtype=np.int64)
    union = np.logical_or(gt, pred).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def iou_matrix(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix. Vectorized over flattened bool masks."""
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    if n_gt == 0 or n_pred == 0:
        return matrix
    gt_flat = np.stack([m.astype(bool).ravel() for m in gt_masks])          # (G, HW)
    pred_flat = np.stack([m.astype(bool).ravel() for m in pred_masks])      # (P, HW)
    gt_areas = gt_flat.sum(axis=1, dtype=np.int64)                          # (G,)
    pred_areas = pred_flat.sum(axis=1, dtype=np.int64)                      # (P,)
    intersections = gt_flat.astype(np.int64) @ pred_flat.T.astype(np.int64)  # (G, P)
    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    np.divide(intersections, unions, out=matrix, where=unions > 0)
    return matrix


@dataclass(frozen=True)
class MatchResult:
    method: str
    # (gt_index, pred_index, iou) for every one-to-one match with iou > 0
    matches: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)

    @property
    def matched_gt_ious(self) -> dict[int, float]:
        return {gi: iou for gi, _, iou in self.matches}


def match_instances(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> MatchResult:
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 or n_pred == 0:
        return MatchResult(
            method=MATCHING_METHOD,
            matches=[],
            unmatched_gt=list(range(n_gt)),
            unmatched_pred=list(range(n_pred)),
        )
    matrix = iou_matrix(gt_masks, pred_masks)

    from scipy.optimize import linear_sum_assignment

    gt_idx, pred_idx = linear_sum_assignment(-matrix)
    matches: list[tuple[int, int, float]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for gi, pi in zip(gt_idx, pred_idx):
        iou = float(matrix[gi, pi])
        if iou <= 0.0:
            continue  # zero-overlap assignment is not a match
        matches.append((int(gi), int(pi), iou))
        matched_gt.add(int(gi))
        matched_pred.add(int(pi))
    return MatchResult(
        method=MATCHING_METHOD,
        matches=matches,
        unmatched_gt=[i for i in range(n_gt) if i not in matched_gt],
        unmatched_pred=[i for i in range(n_pred) if i not in matched_pred],
    )
