from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import torch
from PIL import Image, ImageOps

from core.coco_export import build_coco, validate_coco, write_coco
from core.config import DEFAULT_CATEGORY_NAME, DEFAULT_TRAINING_PROMPT
from core.cvat_export import export_cvat_package
from core.mask_nms import apply_mask_nms
from core.npz_io import InstanceSet, load_npz, save_npz
from core.run_manager import create_inference_run, runtime_info, setup_file_logger, write_json
from core.sam3_adapter import Sam3Adapter
from core.sam_model_registry import model_provenance
from core.visualization import write_overlay


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _sync_cuda_if_needed(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _manifest_row(source_image: Path | str | None, file_name: str, image_id: int, width: int = 0, height: int = 0) -> dict[str, Any]:
    return {
        "source_image": str(source_image) if source_image is not None else None,
        "copied_image": None,
        "raw_npz": None,
        "nms_npz": None,
        "raw_visualization": None,
        "nms_visualization": None,
        "coco_image_id": image_id,
        "annotation_ids": [],
        "raw_prediction_count": 0,
        "nms_instance_count": 0,
        "nms_removed_count": 0,
        "nms_removed": [],
        "scores": [],
        "width": int(width),
        "height": int(height),
        "status": "pending",
        "error": None,
        "file_name": file_name,
    }


def _copy_image(
    source_image: Path,
    destination: Path,
    cvat_destination: Path,
    normalize_exif_orientation: bool = True,
) -> tuple[int, int, int | None]:
    orientation: int | None = None
    try:
        with Image.open(source_image) as image:
            orientation = image.getexif().get(274)
            if normalize_exif_orientation and orientation and orientation != 1:
                normalized = ImageOps.exif_transpose(image)
                save_kwargs: dict[str, Any] = {}
                if destination.suffix.lower() in {".jpg", ".jpeg"} and normalized.mode != "RGB":
                    normalized = normalized.convert("RGB")
                    save_kwargs["quality"] = 95
                elif destination.suffix.lower() in {".jpg", ".jpeg"}:
                    save_kwargs["quality"] = 95
                normalized.save(destination, **save_kwargs)
            else:
                shutil.copy2(source_image, destination)
    except Exception:
        orientation = None
        shutil.copy2(source_image, destination)
    shutil.copy2(destination, cvat_destination)
    image = cv2.imread(str(destination))
    if image is None:
        raise ValueError(f"Cannot read copied image: {destination}")
    height, width = image.shape[:2]
    return width, height, orientation


def _write_image_outputs(
    paths,
    row: dict[str, Any],
    copied_image: Path,
    raw_instances: InstanceSet,
    image_stem: str,
    nms_iou_thresh: float,
    nms_metric: str,
    nms_mode: str,
) -> tuple[InstanceSet, Any, dict[str, float]]:
    timings: dict[str, float] = {}
    raw_npz_dst = paths.npz_raw / f"{image_stem}.npz"
    start = time.perf_counter()
    save_npz(raw_npz_dst, raw_instances)
    timings["raw_npz_write_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    nms = apply_mask_nms(raw_instances, iou_thresh=nms_iou_thresh, metric=nms_metric, mode=nms_mode)
    timings["nms_seconds"] = time.perf_counter() - start
    nms_npz = paths.npz_nms / f"{image_stem}.npz"
    start = time.perf_counter()
    save_npz(nms_npz, nms.instances)
    timings["nms_npz_write_seconds"] = time.perf_counter() - start
    raw_visual = paths.visual_raw / f"{image_stem}.jpg"
    nms_visual = paths.visual_nms / f"{image_stem}.jpg"
    start = time.perf_counter()
    write_overlay(copied_image, raw_instances, raw_visual)
    write_overlay(copied_image, nms.instances, nms_visual)
    timings["visualization_seconds"] = time.perf_counter() - start
    row.update(
        {
            "copied_image": str(copied_image.relative_to(paths.root)),
            "raw_npz": str(raw_npz_dst.relative_to(paths.root)),
            "nms_npz": str(nms_npz.relative_to(paths.root)),
            "raw_visualization": str(raw_visual.relative_to(paths.root)),
            "nms_visualization": str(nms_visual.relative_to(paths.root)),
            "raw_prediction_count": raw_instances.count,
            "nms_instance_count": nms.instances.count,
            "nms_removed_count": raw_instances.count - nms.instances.count,
            "nms_removed": [asdict(item) for item in nms.suppressed],
            "scores": [float(score) for score in raw_instances.scores],
            "status": "ok",
        }
    )
    return nms.instances, nms, timings


def _finalize_run(paths, manifest, errors, image_records, instances_by_image_id, category_name: str, min_area: int) -> dict[str, float]:
    timings: dict[str, float] = {}
    start = time.perf_counter()
    coco, ann_ids_by_image, coco_errors = build_coco(
        image_records,
        instances_by_image_id,
        category_name=category_name,
        min_area=min_area,
    )
    for row in manifest:
        row["annotation_ids"] = ann_ids_by_image.get(int(row["coco_image_id"]), [])
    coco_path = paths.coco_dir / "instances_default.json"
    write_coco(coco_path, coco)
    validation = validate_coco(coco, paths.input_images)
    validation["coco_export_errors"] = coco_errors
    write_json(paths.validation_report, validation)
    timings["coco_export_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    write_json(paths.manifest, manifest)
    write_json(paths.errors, errors)
    cvat_report = export_cvat_package(paths.root)
    write_json(paths.root / "cvat_export" / "validation_report.json", cvat_report)
    timings["cvat_export_seconds"] = time.perf_counter() - start
    return timings


def migrate_legacy_raw_run(
    legacy_run_dir: Path,
    output_root: Path,
    book_root: Path,
    sam3_root: Path,
    limit: int | None = None,
    nms_iou_thresh: float = 0.5,
    nms_metric: str = "iou",
    nms_mode: str = "suppress",
    min_area: int = 200,
    category_name: str = DEFAULT_CATEGORY_NAME,
) -> Path:
    """Create a new unified inference run from an existing ft_01 raw run."""
    paths = create_inference_run(output_root)
    logger = setup_file_logger(paths.logs / "run.log")
    logger.info("Migrating legacy raw run %s to %s", legacy_run_dir, paths.root)

    predictions_path = legacy_run_dir / "sam3_predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    import json

    records = json.loads(predictions_path.read_text(encoding="utf-8"))
    if limit is not None:
        records = records[:limit]

    manifest: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    image_records: list[dict[str, Any]] = []
    instances_by_image_id: dict[int, InstanceSet] = {}

    run_config = {
        "run_id": paths.run_id,
        "created_at": datetime.now().isoformat(),
        "mode": "migrate_legacy_raw_run",
        "legacy_run_dir": str(legacy_run_dir),
        "output_dir": str(paths.root),
        "sam3_checkpoint": None,
        "sam3_model_config": None,
        "prompt": records[0].get("prompt") if records else None,
        "inference_threshold": None,
        "nms_type": nms_metric,
        "nms_threshold": nms_iou_thresh,
        "nms_mode": nms_mode,
        "min_area": min_area,
        "random_seed": 0,
        "runtime": runtime_info(book_root, sam3_root),
    }
    write_json(paths.run_config, run_config)

    for image_index, rec in enumerate(records, 1):
        file_name = rec["file_name"]
        row = _manifest_row(rec.get("orig_path"), file_name, image_index, rec["width"], rec["height"])
        try:
            source_image = legacy_run_dir / "images" / file_name
            if not source_image.exists() and rec.get("orig_path"):
                source_image = Path(rec["orig_path"])
            copied_image = paths.input_images / file_name
            width, height, orientation = _copy_image(
                source_image,
                copied_image,
                paths.cvat_images / file_name,
                normalize_exif_orientation=False,
            )
            raw_npz_src = legacy_run_dir / rec["npz_path"]
            raw_instances = load_npz(raw_npz_src)
            nms_instances, _, image_timings = _write_image_outputs(
                paths, row, copied_image, raw_instances, Path(file_name).stem, nms_iou_thresh, nms_metric, nms_mode
            )
            row["timings"] = image_timings
            row["width"] = width
            row["height"] = height
            row["exif_orientation"] = orientation
            row["image_orientation_normalized"] = False
            image_records.append({"coco_image_id": image_index, "file_name": file_name, "width": width, "height": height})
            instances_by_image_id[image_index] = nms_instances
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            errors.append({"file_name": file_name, "error": repr(exc)})
            logger.exception("Failed processing %s", file_name)
        manifest.append(row)
        write_json(paths.manifest, manifest)
        write_json(paths.errors, errors)

    _finalize_run(paths, manifest, errors, image_records, instances_by_image_id, category_name, min_area)
    logger.info("Completed unified run: %s", paths.root)
    return paths.root


def run_sam3_image_directory(
    input_dir: Path,
    output_root: Path,
    book_root: Path,
    sam3_root: Path,
    checkpoint: Path,
    prompt: str = DEFAULT_TRAINING_PROMPT,
    score_threshold: float = 0.3,
    confidence_threshold: float = 0.05,
    dtype_mode: str = "bf16",
    device: str = "cuda",
    limit: int | None = None,
    nms_iou_thresh: float = 0.5,
    nms_metric: str = "iou",
    nms_mode: str = "suppress",
    min_area: int = 200,
    category_name: str = DEFAULT_CATEGORY_NAME,
    adapter_factory: Callable[..., Sam3Adapter] = Sam3Adapter,
) -> Path:
    """Run real SAM3 inference from images and write the unified inference run."""
    total_run_start = time.perf_counter()
    paths = create_inference_run(output_root)
    logger = setup_file_logger(paths.logs / "run.log")
    environment_start = time.perf_counter()
    runtime = runtime_info(book_root, sam3_root)
    environment_setup_seconds = time.perf_counter() - environment_start
    image_paths = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None:
        image_paths = image_paths[:limit]
    logger.info("Running SAM3 inference on %d image(s) from %s", len(image_paths), input_dir)
    model_load_start = time.perf_counter()
    adapter = adapter_factory(
        checkpoint=checkpoint,
        sam3_root=sam3_root,
        confidence_threshold=confidence_threshold,
        dtype_mode=dtype_mode,
        device=device,
    )
    model_load_seconds = time.perf_counter() - model_load_start
    sam_model_info = model_provenance(checkpoint, validate_load=False)
    sam_model_info.update(
        {
            "resolved_model_path": str(Path(checkpoint).expanduser().resolve(strict=False)),
            "model_sha256": getattr(adapter, "checkpoint_sha256", sam_model_info.get("sam_model_sha256")),
            "model_type": getattr(adapter, "checkpoint_type", sam_model_info.get("sam_model_type")),
            "cache_hit": getattr(adapter, "cache_hit", False),
            "load_timestamp": getattr(adapter, "loaded_at", datetime.now().isoformat()),
            "inference_parameters": {
                "prompt": prompt,
                "score_threshold": score_threshold,
                "processor_confidence_threshold": confidence_threshold,
                "dtype_mode": dtype_mode,
                "requested_device": device,
                "actual_device": adapter.device,
                "nms_iou_thresh": nms_iou_thresh,
                "nms_metric": nms_metric,
                "nms_mode": nms_mode,
                "min_area": min_area,
                "category_name": category_name,
            },
            "prompt": prompt,
            "threshold": score_threshold,
            "nms_settings": {
                "iou_thresh": nms_iou_thresh,
                "metric": nms_metric,
                "mode": nms_mode,
                "min_area": min_area,
            },
            "generation_timestamp": datetime.now().isoformat(),
        }
    )
    write_json(paths.root / "sam_model_provenance.json", sam_model_info)
    logger.info(
        "Loaded SAM model path=%s sha256=%s type=%s cache_hit=%s",
        sam_model_info.get("resolved_model_path"),
        sam_model_info.get("model_sha256"),
        sam_model_info.get("model_type"),
        sam_model_info.get("cache_hit"),
    )
    run_config = {
        "run_id": paths.run_id,
        "created_at": datetime.now().isoformat(),
        "mode": "sam3_image_directory",
        "input_dir": str(input_dir),
        "output_dir": str(paths.root),
        "sam3_checkpoint": str(checkpoint),
        "sam_model": sam_model_info,
        "sam3_model_config": "build_sam3_image_model",
        "prompt": prompt,
        "inference_threshold": score_threshold,
        "processor_confidence_threshold": confidence_threshold,
        "dtype_mode": dtype_mode,
        "requested_device": device,
        "actual_device": adapter.device,
        "environment_setup_seconds": environment_setup_seconds,
        "model_load_seconds": model_load_seconds,
        "nms_type": nms_metric,
        "nms_threshold": nms_iou_thresh,
        "nms_mode": nms_mode,
        "min_area": min_area,
        "random_seed": 0,
        "runtime": runtime,
    }
    write_json(paths.run_config, run_config)

    manifest: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    image_records: list[dict[str, Any]] = []
    instances_by_image_id: dict[int, InstanceSet] = {}
    for image_index, source_image in enumerate(image_paths, 1):
        file_name = source_image.name
        row = _manifest_row(source_image, file_name, image_index)
        image_total_start = time.perf_counter()
        try:
            copied_image = paths.input_images / file_name
            image_read_start = time.perf_counter()
            width, height, orientation = _copy_image(source_image, copied_image, paths.cvat_images / file_name)
            image_read_seconds = time.perf_counter() - image_read_start
            infer_start = time.perf_counter()
            raw_instances = adapter.predict(
                copied_image,
                prompt=prompt,
                score_threshold=score_threshold,
                min_area=min_area,
            )
            _sync_cuda_if_needed(adapter.device)
            inference_seconds = time.perf_counter() - infer_start
            nms_instances, _, output_timings = _write_image_outputs(
                paths, row, copied_image, raw_instances, source_image.stem, nms_iou_thresh, nms_metric, nms_mode
            )
            output_write_seconds = (
                output_timings["raw_npz_write_seconds"]
                + output_timings["nms_npz_write_seconds"]
                + output_timings["visualization_seconds"]
            )
            row["timings"] = {
                "image_read_seconds": image_read_seconds,
                "inference_seconds": inference_seconds,
                "sam3_inference_seconds": inference_seconds,
                "nms_seconds": output_timings["nms_seconds"],
                "raw_npz_write_seconds": output_timings["raw_npz_write_seconds"],
                "nms_npz_write_seconds": output_timings["nms_npz_write_seconds"],
                "visualization_seconds": output_timings["visualization_seconds"],
                "output_write_seconds": output_write_seconds,
                "total_image_seconds": time.perf_counter() - image_total_start,
            }
            row["width"] = width
            row["height"] = height
            row["exif_orientation"] = orientation
            row["image_orientation_normalized"] = bool(orientation and orientation != 1)
            image_records.append({"coco_image_id": image_index, "file_name": file_name, "width": width, "height": height})
            instances_by_image_id[image_index] = nms_instances
            logger.info(
                "%s raw=%d nms=%d image_read=%.4fs inference=%.4fs nms=%.4fs output_write=%.4fs total_image=%.4fs",
                file_name,
                raw_instances.count,
                nms_instances.count,
                row["timings"]["image_read_seconds"],
                row["timings"]["inference_seconds"],
                row["timings"]["nms_seconds"],
                row["timings"]["output_write_seconds"],
                row["timings"]["total_image_seconds"],
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
            errors.append({"file_name": file_name, "error": repr(exc)})
            logger.exception("Failed processing %s", file_name)
        manifest.append(row)
        write_json(paths.manifest, manifest)
        write_json(paths.errors, errors)

    finalize_timings = _finalize_run(paths, manifest, errors, image_records, instances_by_image_id, category_name, min_area)
    total_run_seconds = time.perf_counter() - total_run_start
    run_summary = {
        "run_id": paths.run_id,
        "environment_setup_seconds": environment_setup_seconds,
        "model_load_seconds": model_load_seconds,
        "image_read_seconds": sum(row.get("timings", {}).get("image_read_seconds", 0.0) for row in manifest),
        "sam3_inference_seconds": sum(row.get("timings", {}).get("sam3_inference_seconds", 0.0) for row in manifest),
        "raw_npz_write_seconds": sum(row.get("timings", {}).get("raw_npz_write_seconds", 0.0) for row in manifest),
        "nms_seconds": sum(row.get("timings", {}).get("nms_seconds", 0.0) for row in manifest),
        "nms_npz_write_seconds": sum(row.get("timings", {}).get("nms_npz_write_seconds", 0.0) for row in manifest),
        "coco_export_seconds": finalize_timings["coco_export_seconds"],
        "visualization_seconds": sum(row.get("timings", {}).get("visualization_seconds", 0.0) for row in manifest),
        "cvat_export_seconds": finalize_timings["cvat_export_seconds"],
        "total_run_seconds": total_run_seconds,
        "sam_model": sam_model_info,
    }
    write_json(paths.root / "run_summary.json", run_summary)
    run_config["timings"] = run_summary
    write_json(paths.run_config, run_config)
    logger.info(
        "Timing summary environment=%.4fs model_load=%.4fs inference=%.4fs nms=%.4fs coco=%.4fs cvat=%.4fs total=%.4fs",
        environment_setup_seconds,
        model_load_seconds,
        run_summary["sam3_inference_seconds"],
        run_summary["nms_seconds"],
        run_summary["coco_export_seconds"],
        run_summary["cvat_export_seconds"],
        total_run_seconds,
    )
    logger.info("Completed SAM3 inference run: %s", paths.root)
    return paths.root
