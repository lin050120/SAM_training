"""Evaluation output files + atomic training_summary.json integration."""

from __future__ import annotations

import csv
import json
import os
import uuid
from pathlib import Path
from typing import Any

CHECKPOINT_CSV_COLUMNS = [
    "split",
    "checkpoint_name",
    "checkpoint_path",
    "epoch",
    "sha256",
    "size_bytes",
    "checkpoint_type",
    "is_baseline",
    "evaluation_status",
    "mean_iou_all_gt",
    "median_iou_all_gt",
    "mean_iou_matched_only",
    "median_iou_matched_only",
    "recall_iou_50",
    "recall_iou_75",
    "recall_iou_90",
    "precision_iou_50",
    "precision_iou_75",
    "precision_iou_90",
    "mean_boundary_f1_all_gt",
    "median_boundary_f1_all_gt",
    "gt_count",
    "prediction_count",
    "matched_count_iou_50",
    "missed_gt_count",
    "miss_rate_iou_50",
    "false_positive_count",
    "false_positive_per_image",
    "mean_area_ratio",
    "median_area_ratio",
    "area_ratio_p10",
    "area_ratio_p90",
    "pred_larger_than_gt_rate",
    "inference_time_seconds",
    "cache_hit",
    "error_message",
]

PER_IMAGE_CSV_COLUMNS = [
    "split",
    "checkpoint_name",
    "image_id",
    "image_name",
    "gt_count",
    "prediction_count",
    "matched_count_iou_50",
    "false_positive_count",
    "mean_iou_all_gt",
    "mean_boundary_f1_all_gt",
    "raw_prediction_path",
    "match_record_path",
    "error",
]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_checkpoint_metrics(evaluation_dir: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_json(evaluation_dir / "checkpoint_metrics.json", rows)
    csv_path = evaluation_dir / "checkpoint_metrics.csv"
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CHECKPOINT_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CHECKPOINT_CSV_COLUMNS})
    atomic_write_text(csv_path, buffer.getvalue())


def write_per_image_metrics(evaluation_dir: Path, rows: list[dict[str, Any]]) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=PER_IMAGE_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in PER_IMAGE_CSV_COLUMNS})
    atomic_write_text(evaluation_dir / "per_image_metrics.csv", buffer.getvalue())


def update_training_summary(run_dir: Path, evaluation_block: dict[str, Any]) -> str | None:
    """Insert/replace training_summary.json's `checkpoint_evaluation` block.

    Atomic (tmp + os.replace); all existing fields preserved; on any failure the
    original file is left untouched and the error string is returned instead of
    raising (a summary-update failure must not invalidate the evaluation itself).
    """
    summary_path = Path(run_dir) / "training_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            return f"training_summary.json is not a JSON object: {summary_path}"
        summary["checkpoint_evaluation"] = evaluation_block
        atomic_write_json(summary_path, summary)
        return None
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return f"{type(exc).__name__}: {exc}"
