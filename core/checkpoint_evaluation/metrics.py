"""Mask-level metrics for checkpoint evaluation.

Definitions (all recorded in the evaluation report; see docs/CHECKPOINT_EVALUATION_CN.md):

- mean_iou_all_gt: every GT instance in the validation set contributes exactly one
  IoU value; an unmatched (missed) GT contributes 0. Averaged over ALL GT instances.
  This is the primary selection metric — matched-only averages are reported too but
  are inflated by construction and never used for selection.
- recall_iou_T = |matched pairs with IoU >= T| / gt_count
- precision_iou_T = |matched pairs with IoU >= T| / prediction_count
- miss_rate_iou_50 = 1 - recall_iou_50; missed_gt_count = gt_count - matched_count_iou_50
- false_positive_count = prediction_count - matched_count_iou_50 (a prediction that
  did not achieve an IoU>=0.5 one-to-one match is a false positive)
- boundary F1: computed per matched pair at ORIGINAL image resolution (prediction
  masks are already restored to original size by Sam3Adapter). Boundary pixels are
  the mask XOR its 1-px erosion; a boundary pixel counts as correct if it lies
  within `boundary_tolerance_px` of the other mask's boundary (implemented by
  dilating the reference boundary with a (2*tol+1) square kernel). Unmatched GT
  contributes 0 to mean_boundary_f1_all_gt.
- area ratio: prediction_area / gt_area over pairs matched at IoU >= 0.5 (garbage
  low-IoU pairings would make the ratio meaningless); checks SAM3's known
  systematic mask over-expansion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.checkpoint_evaluation.mask_matching import MatchResult

DEFAULT_BOUNDARY_TOLERANCE_PX = 2
IOU_THRESHOLDS = (0.5, 0.75, 0.9)


def boundary_f1(gt_mask: np.ndarray, pred_mask: np.ndarray, tolerance_px: int = DEFAULT_BOUNDARY_TOLERANCE_PX) -> float:
    """Boundary F1 with pixel tolerance, at the masks' native (original) resolution."""
    gt = gt_mask.astype(np.uint8)
    pred = pred_mask.astype(np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    gt_boundary = gt - cv2.erode(gt, kernel3, iterations=1)
    pred_boundary = pred - cv2.erode(pred, kernel3, iterations=1)
    n_gt_b = int(gt_boundary.sum())
    n_pred_b = int(pred_boundary.sum())
    if n_gt_b == 0 and n_pred_b == 0:
        return 1.0
    if n_gt_b == 0 or n_pred_b == 0:
        return 0.0
    tol_kernel = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), np.uint8)
    gt_dilated = cv2.dilate(gt_boundary, tol_kernel, iterations=1)
    pred_dilated = cv2.dilate(pred_boundary, tol_kernel, iterations=1)
    precision = float((pred_boundary & gt_dilated).sum()) / n_pred_b
    recall = float((gt_boundary & pred_dilated).sum()) / n_gt_b
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class ImageMetrics:
    image_name: str
    gt_count: int
    prediction_count: int
    matching_method: str
    # one IoU per GT instance (unmatched -> 0.0)
    gt_ious: list[float] = field(default_factory=list)
    # one boundary F1 per GT instance (unmatched -> 0.0)
    gt_boundary_f1s: list[float] = field(default_factory=list)
    matched_count_iou_50: int = 0
    matched_counts_by_threshold: dict[float, int] = field(default_factory=dict)
    false_positive_count: int = 0
    # prediction_area / gt_area for pairs matched at IoU >= 0.5
    area_ratios: list[float] = field(default_factory=list)
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "gt_count": self.gt_count,
            "prediction_count": self.prediction_count,
            "matched_count_iou_50": self.matched_count_iou_50,
            "false_positive_count": self.false_positive_count,
            "mean_iou_all_gt": float(np.mean(self.gt_ious)) if self.gt_ious else math.nan,
            "mean_boundary_f1_all_gt": float(np.mean(self.gt_boundary_f1s)) if self.gt_boundary_f1s else math.nan,
            "error": self.error or "",
        }


def compute_image_metrics(
    image_name: str,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    match: MatchResult,
    boundary_tolerance_px: int = DEFAULT_BOUNDARY_TOLERANCE_PX,
) -> ImageMetrics:
    gt_ious = [0.0] * len(gt_masks)
    gt_bf1 = [0.0] * len(gt_masks)
    matched_by_threshold = {t: 0 for t in IOU_THRESHOLDS}
    area_ratios: list[float] = []

    for gi, pi, iou in match.matches:
        gt_ious[gi] = iou
        gt_bf1[gi] = boundary_f1(gt_masks[gi], pred_masks[pi], boundary_tolerance_px)
        for t in IOU_THRESHOLDS:
            if iou >= t:
                matched_by_threshold[t] += 1
        if iou >= 0.5:
            gt_area = float(np.count_nonzero(gt_masks[gi]))
            pred_area = float(np.count_nonzero(pred_masks[pi]))
            if gt_area > 0:
                area_ratios.append(pred_area / gt_area)

    matched_50 = matched_by_threshold[0.5]
    return ImageMetrics(
        image_name=image_name,
        gt_count=len(gt_masks),
        prediction_count=len(pred_masks),
        matching_method=match.method,
        gt_ious=gt_ious,
        gt_boundary_f1s=gt_bf1,
        matched_count_iou_50=matched_50,
        matched_counts_by_threshold=matched_by_threshold,
        false_positive_count=len(pred_masks) - matched_50,
        area_ratios=area_ratios,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def aggregate_metrics(per_image: list[ImageMetrics], boundary_tolerance_px: int) -> dict[str, Any]:
    """Aggregate image-level results into the checkpoint-level metric dict.

    GT-averaged metrics pool ALL GT instances across the whole validation set (an
    image with more spines contributes proportionally more instances — this is
    instance-level averaging, recorded as such).
    """
    ok_images = [m for m in per_image if m.error is None]
    all_gt_ious: list[float] = [iou for m in ok_images for iou in m.gt_ious]
    all_gt_bf1: list[float] = [b for m in ok_images for b in m.gt_boundary_f1s]
    matched_ious: list[float] = [iou for m in ok_images for iou in m.gt_ious if iou > 0.0]
    matched_bf1: list[float] = [
        b for m in ok_images for iou, b in zip(m.gt_ious, m.gt_boundary_f1s) if iou > 0.0
    ]
    area_ratios: list[float] = [r for m in ok_images for r in m.area_ratios]

    gt_count = sum(m.gt_count for m in ok_images)
    pred_count = sum(m.prediction_count for m in ok_images)
    matched_by_threshold = {
        t: sum(m.matched_counts_by_threshold.get(t, 0) for m in ok_images) for t in IOU_THRESHOLDS
    }
    matched_50 = matched_by_threshold[0.5]
    fp_count = sum(m.false_positive_count for m in ok_images)
    n_images = len(ok_images)

    def _ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator > 0 else None

    metrics: dict[str, Any] = {
        "image_count": n_images,
        "failed_image_count": len(per_image) - n_images,
        "gt_count": gt_count,
        "prediction_count": pred_count,
        "matched_count_iou_50": matched_50,
        "missed_gt_count": gt_count - matched_50,
        "miss_rate_iou_50": _ratio(gt_count - matched_50, gt_count),
        "false_positive_count": fp_count,
        "false_positive_per_image": fp_count / n_images if n_images > 0 else None,
        "mean_iou_all_gt": _finite_or_none(np.mean(all_gt_ious)) if all_gt_ious else None,
        "median_iou_all_gt": _finite_or_none(np.median(all_gt_ious)) if all_gt_ious else None,
        "mean_iou_matched_only": _finite_or_none(np.mean(matched_ious)) if matched_ious else None,
        "median_iou_matched_only": _finite_or_none(np.median(matched_ious)) if matched_ious else None,
        "mean_boundary_f1_all_gt": _finite_or_none(np.mean(all_gt_bf1)) if all_gt_bf1 else None,
        "median_boundary_f1_all_gt": _finite_or_none(np.median(all_gt_bf1)) if all_gt_bf1 else None,
        "mean_boundary_f1_matched_only": _finite_or_none(np.mean(matched_bf1)) if matched_bf1 else None,
        "boundary_tolerance_px": boundary_tolerance_px,
        "boundary_resolution": "original_image_resolution",
        "mean_area_ratio": _finite_or_none(np.mean(area_ratios)) if area_ratios else None,
        "median_area_ratio": _finite_or_none(np.median(area_ratios)) if area_ratios else None,
        "area_ratio_p10": _finite_or_none(np.percentile(area_ratios, 10)) if area_ratios else None,
        "area_ratio_p90": _finite_or_none(np.percentile(area_ratios, 90)) if area_ratios else None,
        "pred_larger_than_gt_rate": (
            float(np.mean([r > 1.0 for r in area_ratios])) if area_ratios else None
        ),
    }
    for t in IOU_THRESHOLDS:
        suffix = str(int(t * 100))
        metrics[f"recall_iou_{suffix}"] = _ratio(matched_by_threshold[t], gt_count)
        metrics[f"precision_iou_{suffix}"] = _ratio(matched_by_threshold[t], pred_count)
    return metrics
