from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ui.ui_utils import safe_read_json


@dataclass
class RunSummary:
    run_id: str
    root: str
    created_at: str | None
    mode: str | None
    status: str
    prompt: str | None
    checkpoint: str | None
    device: str | None
    input_images: int
    success_images: int
    failed_images: int
    raw_instances: int
    nms_instances: int
    coco_annotations: int
    total_run_seconds: float | None
    errors_count: int
    segmentation_formats: list[str]
    cvat_export_status: str
    warnings: list[str] = field(default_factory=list)


def _list_run_dirs(inference_root: Path) -> list[Path]:
    if not inference_root.exists():
        return []
    return sorted(
        (p for p in inference_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )


def summarize_run(run_dir: Path) -> RunSummary:
    warnings: list[str] = []

    run_config, err = safe_read_json(run_dir / "run_config.json")
    if err:
        warnings.append(err)
    run_config = run_config or {}

    manifest, err = safe_read_json(run_dir / "manifest.json")
    if err:
        warnings.append(err)
    manifest = manifest or []

    errors_data, err = safe_read_json(run_dir / "errors.json")
    if err:
        warnings.append(err)
    errors_data = errors_data or []

    validation, err = safe_read_json(run_dir / "validation_report.json")
    if err:
        warnings.append(err)
    validation = validation or {}

    run_summary, _ = safe_read_json(run_dir / "run_summary.json")

    success = sum(1 for row in manifest if row.get("status") == "ok")
    failed = sum(1 for row in manifest if row.get("status") == "error")
    raw_instances = sum(int(row.get("raw_prediction_count", 0) or 0) for row in manifest)
    nms_instances = sum(int(row.get("nms_instance_count", 0) or 0) for row in manifest)

    segmentation_formats: list[str] = []
    cvat_export_dir = run_dir / "cvat_export"
    for fmt in ["polygon", "rle"]:
        if (cvat_export_dir / fmt / "instances_default.json").exists():
            segmentation_formats.append(fmt)
    if not segmentation_formats and (cvat_export_dir / "annotations" / "instances_default.json").exists():
        segmentation_formats.append("polygon (legacy path)")

    cvat_validation, _ = safe_read_json(cvat_export_dir / "validation_report.json")
    if cvat_validation is None:
        cvat_export_status = "not exported"
    elif cvat_validation.get("ok"):
        cvat_export_status = "validated ok"
    else:
        cvat_export_status = "validated with errors"

    status = "unknown"
    if manifest:
        status = "ok" if failed == 0 and not errors_data else "has errors"
    elif not run_config:
        status = "unavailable (missing run_config.json)"
        warnings.append("run_config.json missing or unreadable")

    total_run_seconds = None
    if run_summary and "total_run_seconds" in run_summary:
        total_run_seconds = run_summary.get("total_run_seconds")
    elif isinstance(run_config.get("timings"), dict):
        total_run_seconds = run_config["timings"].get("total_run_seconds")

    return RunSummary(
        run_id=run_dir.name,
        root=str(run_dir),
        created_at=run_config.get("created_at"),
        mode=run_config.get("mode"),
        status=status,
        prompt=run_config.get("prompt"),
        checkpoint=run_config.get("sam3_checkpoint"),
        device=run_config.get("actual_device") or run_config.get("requested_device"),
        input_images=len(manifest),
        success_images=success,
        failed_images=failed,
        raw_instances=raw_instances,
        nms_instances=nms_instances,
        coco_annotations=sum(len(row.get("annotation_ids", []) or []) for row in manifest),
        total_run_seconds=total_run_seconds,
        errors_count=len(errors_data) if isinstance(errors_data, list) else 0,
        segmentation_formats=segmentation_formats,
        cvat_export_status=cvat_export_status,
        warnings=warnings,
    )


def list_inference_runs(inference_root: Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for run_dir in _list_run_dirs(inference_root):
        try:
            summaries.append(summarize_run(run_dir))
        except Exception as exc:  # pragma: no cover - defensive, one bad run must not break the page
            summaries.append(
                RunSummary(
                    run_id=run_dir.name,
                    root=str(run_dir),
                    created_at=None,
                    mode=None,
                    status="unavailable",
                    prompt=None,
                    checkpoint=None,
                    device=None,
                    input_images=0,
                    success_images=0,
                    failed_images=0,
                    raw_instances=0,
                    nms_instances=0,
                    coco_annotations=0,
                    total_run_seconds=None,
                    errors_count=0,
                    segmentation_formats=[],
                    cvat_export_status="unknown",
                    warnings=[f"failed to summarize run: {exc!r}"],
                )
            )
    return summaries


def summary_table_rows(summaries: list[RunSummary]) -> list[list[Any]]:
    return [
        [
            s.run_id,
            s.created_at or "unknown",
            s.status,
            s.input_images,
            s.success_images,
            s.failed_images,
            s.raw_instances,
            s.nms_instances,
            s.coco_annotations,
            s.prompt or "unknown",
            s.device or "unknown",
            f"{s.total_run_seconds:.2f}" if s.total_run_seconds is not None else "unavailable",
            s.errors_count,
            ",".join(s.segmentation_formats) or "none",
            s.cvat_export_status,
        ]
        for s in summaries
    ]


SUMMARY_TABLE_HEADERS = [
    "run_id",
    "created_at",
    "status",
    "input_images",
    "success",
    "failed",
    "raw_instances",
    "nms_instances",
    "coco_annotations",
    "prompt",
    "device",
    "total_time_s",
    "errors",
    "segmentation_formats",
    "cvat_export",
]


def read_manifest_row(run_dir: Path, file_name: str) -> dict[str, Any] | None:
    manifest, _ = safe_read_json(run_dir / "manifest.json")
    if not manifest:
        return None
    for row in manifest:
        if row.get("file_name") == file_name:
            return row
    return None


def list_image_names(run_dir: Path) -> list[str]:
    manifest, _ = safe_read_json(run_dir / "manifest.json")
    if not manifest:
        return []
    return [row.get("file_name") for row in manifest if row.get("file_name")]


def load_instance_table(run_dir: Path, file_name: str) -> tuple[list[dict[str, Any]], str | None]:
    """Join COCO annotations for an image with NMS npz source_instance_ids."""
    row = read_manifest_row(run_dir, file_name)
    if row is None:
        return [], f"no manifest entry for {file_name}"
    if row.get("status") != "ok":
        return [], f"image status is {row.get('status')}: {row.get('error')}"

    coco, err = safe_read_json(run_dir / "coco" / "instances_default.json")
    if err or coco is None:
        return [], err or "coco/instances_default.json missing"

    image_id = row.get("coco_image_id")
    annotations = [ann for ann in coco.get("annotations", []) if ann.get("image_id") == image_id]

    source_ids: list[int] = []
    nms_npz_rel = row.get("nms_npz")
    if nms_npz_rel:
        try:
            from core.npz_io import load_npz

            instances = load_npz(run_dir / nms_npz_rel)
            raw_source_ids = instances.extra.get("source_instance_ids")
            if raw_source_ids is not None:
                source_ids = [int(v) for v in raw_source_ids]
        except Exception:
            source_ids = []

    table: list[dict[str, Any]] = []
    for idx, ann in enumerate(annotations):
        table.append(
            {
                "annotation_id": ann.get("id"),
                "source_instance_id": source_ids[idx] if idx < len(source_ids) else None,
                "score": ann.get("score"),
                "bbox": ann.get("bbox"),
                "area": ann.get("area"),
            }
        )
    return table, None


def list_nms_removed_pairs(run_dir: Path) -> list[dict[str, Any]]:
    """List every kept/removed NMS pair recorded in the manifest, across all images.

    Lets pages offer the known pairs for selection instead of making the user
    re-type instance IDs that the run already recorded.
    """
    manifest, _ = safe_read_json(run_dir / "manifest.json")
    pairs: list[dict[str, Any]] = []
    for row in manifest or []:
        for removed in row.get("nms_removed", []) or []:
            pairs.append(
                {
                    "file_name": row.get("file_name"),
                    "kept_source_id": removed.get("kept_instance_id"),
                    "removed_source_id": removed.get("source_instance_id"),
                    "score": removed.get("score"),
                    "overlap": removed.get("overlap"),
                    "reason": removed.get("reason"),
                }
            )
    return pairs


def find_nms_review_images(run_dir: Path, file_name: str) -> list[dict[str, Any]]:
    row = read_manifest_row(run_dir, file_name)
    if row is None:
        return []
    reviews = []
    for removed in row.get("nms_removed", []) or []:
        kept = removed.get("kept_instance_id")
        source = removed.get("source_instance_id")
        png = run_dir / "visualizations" / "nms_review" / f"instance_{kept}_vs_{source}.png"
        json_path = run_dir / "visualizations" / "nms_review" / f"instance_{kept}_vs_{source}.json"
        data, _ = safe_read_json(json_path)
        reviews.append(
            {
                "kept_instance_id": kept,
                "removed_instance_id": source,
                "image_path": str(png) if png.exists() else None,
                "detail": data,
            }
        )
    return reviews


def list_training_runs(training_root: Path) -> list[dict[str, Any]]:
    if not training_root.exists():
        return []
    rows = []
    for run_dir in sorted((p for p in training_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        info, _ = safe_read_json(run_dir / "dataset_info.json")
        command_path = run_dir / "command.txt"
        # training_summary.json only exists once a training run has actually stopped
        # (completed/failed/cancelled); preflight-only runs (stage D1/E1 preflight
        # without launching training) simply have no summary yet, not an error.
        summary, _ = safe_read_json(run_dir / "training_summary.json")
        rows.append(
            {
                "run_id": run_dir.name,
                "root": str(run_dir),
                "dataset_info": info,
                "command": command_path.read_text(encoding="utf-8").strip() if command_path.exists() else None,
                "training_summary": summary,
                "status": summary.get("status") if summary else "preflight_only_or_unknown",
            }
        )
    return rows
