"""Raw prediction, matching, and visualization artifacts for checkpoint evaluation."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.checkpoint_evaluation.mask_matching import MATCHING_METHOD, iou_matrix
from core.checkpoint_evaluation.metrics import boundary_f1
from core.checkpoint_evaluation.report_writer import atomic_write_json, atomic_write_text
from core.npz_io import InstanceSet


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("._") or "item"


def checkpoint_key(candidate_name: str, is_baseline: bool = False) -> str:
    if is_baseline:
        return "baseline"
    return safe_name(Path(candidate_name).stem)


def _centroid(mask: np.ndarray) -> tuple[float | None, float | None]:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return None, None
    return float(xs.mean()), float(ys.mean())


def _bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def write_gt_snapshot(split_dir: Path, split: str, ground_truth: list, annotations_path: Path, annotations_sha256: str) -> None:
    items: list[dict[str, Any]] = []
    for gt in ground_truth:
        image_sha = None
        try:
            from core.checkpoint_export import sha256_of_file

            image_sha = sha256_of_file(gt.path)
        except Exception:  # noqa: BLE001 - snapshot remains useful without it
            image_sha = None
        for idx, mask in enumerate(gt.masks):
            cx, cy = _centroid(mask)
            items.append(
                {
                    "split": split,
                    "image_id": gt.image_id,
                    "file_name": gt.file_name,
                    "image_path": str(gt.path),
                    "image_sha256": image_sha,
                    "width": gt.width,
                    "height": gt.height,
                    "gt_instance_index": idx,
                    "gt_instance_id": gt.annotation_ids[idx] if idx < len(gt.annotation_ids) else idx,
                    "area": int(np.count_nonzero(mask)),
                    "bbox_xywh": _bbox_xywh(mask),
                    "centroid_x": cx,
                    "centroid_y": cy,
                }
            )
    atomic_write_json(
        split_dir / "gt_snapshot.json",
        {
            "split": split,
            "annotations_path": str(annotations_path),
            "annotations_sha256": annotations_sha256,
            "instances": items,
        },
    )


def write_raw_predictions(
    split_dir: Path,
    split: str,
    ckpt_key: str,
    gt,
    instances: InstanceSet,
    inference_parameters: dict[str, Any],
    inference_seconds: float,
) -> tuple[Path, Path]:
    raw_dir = split_dir / "raw_predictions" / ckpt_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(gt.image_id):06d}_{safe_name(Path(gt.file_name).stem)}"
    npz_path = raw_dir / f"{stem}.npz"
    meta_path = raw_dir / f"{stem}.json"
    np.savez_compressed(
        npz_path,
        masks=instances.masks.astype(bool),
        scores=instances.scores.astype(np.float32),
        bboxes=instances.bboxes.astype(np.int32),
        instance_ids=instances.instance_ids.astype(np.int32),
    )
    predictions = []
    for idx in range(instances.count):
        mask = instances.masks[idx]
        cx, cy = _centroid(mask)
        predictions.append(
            {
                "prediction_index": idx,
                "prediction_id": int(instances.instance_ids[idx]) if idx < len(instances.instance_ids) else idx + 1,
                "score": float(instances.scores[idx]) if idx < len(instances.scores) else None,
                "box_xywh": [int(v) for v in instances.bboxes[idx].tolist()] if idx < len(instances.bboxes) else _bbox_xywh(mask),
                "area": int(np.count_nonzero(mask)),
                "centroid_x": cx,
                "centroid_y": cy,
            }
        )
    atomic_write_json(
        meta_path,
        {
            "split": split,
            "image_id": int(gt.image_id),
            "image_path": str(gt.path),
            "file_name": gt.file_name,
            "width": int(gt.width),
            "height": int(gt.height),
            "raw_prediction_npz": str(npz_path),
            "prediction_count": int(instances.count),
            "predictions": predictions,
            "inference_parameters": inference_parameters,
            "inference_time_seconds": inference_seconds,
        },
    )
    return npz_path, meta_path


def load_raw_prediction_masks(npz_path: Path) -> list[np.ndarray]:
    with np.load(npz_path) as data:
        return [m.astype(bool) for m in data["masks"]]


def _hungarian_assignments(matrix: np.ndarray) -> list[dict[str, Any]]:
    if matrix.size == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-matrix)
    return [
        {"gt_index": int(g), "pred_index": int(p), "iou": float(matrix[g, p]), "accepted_iou_gt_0": bool(matrix[g, p] > 0)}
        for g, p in zip(rows, cols)
    ]


def build_instance_rows(split: str, checkpoint_name: str, gt, pred_masks: list[np.ndarray], match, match_threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matched_gt = {gi for gi, _, _ in match.matches}
    matched_pred = {pi for _, pi, _ in match.matches}
    for gi, pi, iou in match.matches:
        gt_mask = gt.masks[gi]
        pred_mask = pred_masks[pi]
        gt_area = int(np.count_nonzero(gt_mask))
        pred_area = int(np.count_nonzero(pred_mask))
        gt_cx, gt_cy = _centroid(gt_mask)
        pred_cx, pred_cy = _centroid(pred_mask)
        distance = None
        if gt_cx is not None and pred_cx is not None:
            distance = math.hypot(float(gt_cx) - float(pred_cx), float(gt_cy) - float(pred_cy))
        bf1 = boundary_f1(gt_mask, pred_mask)
        rows.append(
            {
                "split": split,
                "checkpoint": checkpoint_name,
                "image_id": gt.image_id,
                "image_path": str(gt.path),
                "gt_instance_id": gt.annotation_ids[gi] if gi < len(gt.annotation_ids) else gi,
                "pred_instance_id": pi + 1,
                "matched": True,
                "match_method": MATCHING_METHOD,
                "iou": float(iou),
                "boundary_f1": float(bf1),
                "gt_area": gt_area,
                "prediction_area": pred_area,
                "area_ratio": (pred_area / gt_area) if gt_area > 0 else None,
                "gt_centroid_x": gt_cx,
                "gt_centroid_y": gt_cy,
                "prediction_centroid_x": pred_cx,
                "prediction_centroid_y": pred_cy,
                "centroid_distance": distance,
                "miss": bool(iou < match_threshold),
                "false_positive": bool(iou < match_threshold),
                "match_threshold": match_threshold,
                "notes": "" if iou >= match_threshold else "hungarian_match_below_iou_threshold",
            }
        )
    for gi in match.unmatched_gt:
        gt_mask = gt.masks[gi]
        gt_cx, gt_cy = _centroid(gt_mask)
        rows.append(
            {
                "split": split,
                "checkpoint": checkpoint_name,
                "image_id": gt.image_id,
                "image_path": str(gt.path),
                "gt_instance_id": gt.annotation_ids[gi] if gi < len(gt.annotation_ids) else gi,
                "pred_instance_id": "",
                "matched": False,
                "match_method": MATCHING_METHOD,
                "iou": 0.0,
                "boundary_f1": 0.0,
                "gt_area": int(np.count_nonzero(gt_mask)),
                "prediction_area": "",
                "area_ratio": "",
                "gt_centroid_x": gt_cx,
                "gt_centroid_y": gt_cy,
                "prediction_centroid_x": "",
                "prediction_centroid_y": "",
                "centroid_distance": "",
                "miss": True,
                "false_positive": False,
                "match_threshold": match_threshold,
                "notes": "unmatched_gt",
            }
        )
    for pi in match.unmatched_pred:
        pred_mask = pred_masks[pi]
        pred_cx, pred_cy = _centroid(pred_mask)
        rows.append(
            {
                "split": split,
                "checkpoint": checkpoint_name,
                "image_id": gt.image_id,
                "image_path": str(gt.path),
                "gt_instance_id": "",
                "pred_instance_id": pi + 1,
                "matched": False,
                "match_method": MATCHING_METHOD,
                "iou": "",
                "boundary_f1": "",
                "gt_area": "",
                "prediction_area": int(np.count_nonzero(pred_mask)),
                "area_ratio": "",
                "gt_centroid_x": "",
                "gt_centroid_y": "",
                "prediction_centroid_x": pred_cx,
                "prediction_centroid_y": pred_cy,
                "centroid_distance": "",
                "miss": False,
                "false_positive": True,
                "match_threshold": match_threshold,
                "notes": "unmatched_prediction",
            }
        )
    return rows


def write_match_record(split_dir: Path, split: str, ckpt_key: str, checkpoint_name: str, gt, pred_masks: list[np.ndarray], match, raw_npz_path: Path) -> Path:
    record_dir = split_dir / "match_records" / ckpt_key
    record_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(gt.image_id):06d}_{safe_name(Path(gt.file_name).stem)}"
    path = record_dir / f"{stem}.json"
    matrix = iou_matrix(gt.masks, pred_masks)
    atomic_write_json(
        path,
        {
            "split": split,
            "checkpoint": checkpoint_name,
            "image_id": int(gt.image_id),
            "image_path": str(gt.path),
            "raw_prediction_path": str(raw_npz_path),
            "matching_method": MATCHING_METHOD,
            "iou_matrix": matrix.tolist(),
            "hungarian_assignment": _hungarian_assignments(matrix),
            "accepted_matches": [
                {"gt_index": int(gi), "pred_index": int(pi), "iou": float(iou)} for gi, pi, iou in match.matches
            ],
            "unmatched_gt": [int(i) for i in match.unmatched_gt],
            "unmatched_prediction": [int(i) for i in match.unmatched_pred],
        },
    )
    return path


_INSTANCE_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (128, 0, 255),
    (128, 255, 0),
    (255, 0, 128),
    (0, 255, 128),
)


def _instance_color(index: int, offset: int = 0) -> tuple[int, int, int]:
    return _INSTANCE_COLORS_BGR[(index + offset) % len(_INSTANCE_COLORS_BGR)]


def _overlay_masks(
    image: np.ndarray,
    masks: list[np.ndarray],
    color: tuple[int, int, int] | None = None,
    alpha: float = 0.45,
    color_offset: int = 0,
) -> np.ndarray:
    out = image.copy()
    for idx, mask in enumerate(masks):
        m = mask.astype(bool)
        mask_color = color or _instance_color(idx, color_offset)
        out[m] = ((1 - alpha) * out[m] + alpha * np.array(mask_color, dtype=np.uint8)).astype(np.uint8)
    return out


def write_visualizations(split_dir: Path, split: str, ckpt_key: str, gt, pred_masks: list[np.ndarray], match) -> dict[str, str]:
    image = cv2.imread(str(gt.path), cv2.IMREAD_COLOR)
    if image is None:
        return {}
    vis_dir = split_dir / "visualizations" / ckpt_key
    vis_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{int(gt.image_id):06d}_{safe_name(Path(gt.file_name).stem)}"
    paths = {
        "original": vis_dir / f"{stem}_original.png",
        "gt_overlay": vis_dir / f"{stem}_gt_overlay.png",
        "prediction_overlay": vis_dir / f"{stem}_prediction_overlay.png",
        "matched_overlay": vis_dir / f"{stem}_matched_overlay.png",
        "fp_miss_overlay": vis_dir / f"{stem}_fp_miss_overlay.png",
        "boundary_overlay": vis_dir / f"{stem}_boundary_overlay.png",
    }
    cv2.imwrite(str(paths["original"]), image)
    cv2.imwrite(str(paths["gt_overlay"]), _overlay_masks(image, gt.masks))
    cv2.imwrite(str(paths["prediction_overlay"]), _overlay_masks(image, pred_masks, color_offset=3))
    matched = image.copy()
    for gi, pi, iou in match.matches:
        matched = _overlay_masks(matched, [gt.masks[gi]], _instance_color(gi), 0.35)
        matched = _overlay_masks(matched, [pred_masks[pi]], _instance_color(pi, 3), 0.35)
        x, y, w, h = _bbox_xywh(gt.masks[gi])
        cv2.putText(matched, f"gt{gi}/p{pi} IoU={iou:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(paths["matched_overlay"]), matched)
    fp_miss = image.copy()
    fp_miss = _overlay_masks(fp_miss, [gt.masks[i] for i in match.unmatched_gt], alpha=0.55)
    fp_miss = _overlay_masks(fp_miss, [pred_masks[i] for i in match.unmatched_pred], alpha=0.55, color_offset=3)
    cv2.imwrite(str(paths["fp_miss_overlay"]), fp_miss)
    boundary = image.copy()
    for idx, mask in enumerate(gt.masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary, contours, -1, _instance_color(idx), 1)
    for idx, mask in enumerate(pred_masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary, contours, -1, _instance_color(idx, 3), 1)
    cv2.imwrite(str(paths["boundary_overlay"]), boundary)
    return {k: str(v) for k, v in paths.items()}


INSTANCE_COLUMNS = [
    "split", "checkpoint", "image_id", "image_path", "gt_instance_id", "pred_instance_id",
    "matched", "match_method", "iou", "boundary_f1", "gt_area", "prediction_area",
    "area_ratio", "gt_centroid_x", "gt_centroid_y", "prediction_centroid_x",
    "prediction_centroid_y", "centroid_distance", "miss", "false_positive",
    "match_threshold", "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in columns})
    atomic_write_text(path, buffer.getvalue())
