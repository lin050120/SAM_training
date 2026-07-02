from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as cocomask

from core.coco_export import build_coco, write_coco
from core.mask_nms import mask_bbox_xywh
from core.npz_io import load_npz
from core.run_manager import write_json


def _decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = cocomask.frPyObjects(segmentation, height, width)
        rle = cocomask.merge(rles)
    else:
        rle = segmentation
        if isinstance(rle.get("counts"), str):
            rle = dict(rle)
            rle["counts"] = rle["counts"].encode("ascii")
    return cocomask.decode(rle).astype(bool)


def validate_cvat_package(
    run_dir: Path,
    annotation_path: Path | None = None,
    image_dir: Path | None = None,
    require_exact_masks: bool = False,
) -> dict[str, Any]:
    image_dir = image_dir or run_dir / "cvat_export" / "images"
    annotation_path = annotation_path or run_dir / "cvat_export" / "annotations" / "instances_default.json"
    manifest_path = run_dir / "manifest.json"
    errors: list[str] = []
    warnings: list[str] = []
    per_image_counts: dict[str, int] = {}
    exact_mask_matches = 0
    exact_mask_total = 0

    if not annotation_path.exists():
        return {"ok": False, "errors": [f"missing annotations: {annotation_path}"], "warnings": warnings}
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    category_ids = {cat.get("id") for cat in categories}
    category_names = [cat.get("name") for cat in categories]
    if "book_spine" not in category_names:
        warnings.append(f"category name is not exact book_spine: {category_names}")

    image_ids = [img.get("id") for img in images]
    ann_ids = [ann.get("id") for ann in annotations]
    for image_id, count in Counter(image_ids).items():
        if count > 1:
            errors.append(f"duplicate image_id: {image_id}")
    for ann_id, count in Counter(ann_ids).items():
        if count > 1:
            errors.append(f"duplicate annotation_id: {ann_id}")

    images_by_id = {img["id"]: img for img in images}
    registered_files = {img["file_name"] for img in images}
    actual_files = {p.name for p in image_dir.iterdir() if p.is_file()} if image_dir.exists() else set()
    for file_name in sorted(registered_files - actual_files):
        errors.append(f"COCO image missing from cvat images dir: {file_name}")
    for file_name in sorted(actual_files - registered_files):
        warnings.append(f"unregistered image file in cvat images dir: {file_name}")

    for img in images:
        image_path = image_dir / img["file_name"]
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            errors.append(f"cannot read image: {img['file_name']}")
            continue
        height, width = image.shape[:2]
        if int(img.get("width", -1)) != width or int(img.get("height", -1)) != height:
            errors.append(f"image size mismatch: {img['file_name']}")

    anns_by_image = defaultdict(list)
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id not in images_by_id:
            errors.append(f"annotation {ann.get('id')} references missing image_id: {image_id}")
            continue
        if ann.get("category_id") not in category_ids:
            errors.append(f"annotation {ann.get('id')} references invalid category_id: {ann.get('category_id')}")
        image = images_by_id[image_id]
        height, width = int(image["height"]), int(image["width"])
        try:
            decoded = _decode_segmentation(ann.get("segmentation"), height, width)
        except Exception as exc:
            errors.append(f"annotation {ann.get('id')} segmentation decode failed: {exc!r}")
            continue
        if decoded.sum() == 0:
            errors.append(f"annotation {ann.get('id')} empty mask")
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            errors.append(f"annotation {ann.get('id')} invalid bbox length")
        else:
            x, y, w, h = [float(v) for v in bbox]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 1 or y + h > height + 1:
                errors.append(f"annotation {ann.get('id')} bbox out of bounds")
        if float(ann.get("area", 0)) <= 0:
            errors.append(f"annotation {ann.get('id')} invalid area")
        anns_by_image[image["file_name"]].append(ann)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    for row in manifest:
        if row.get("status") != "ok" or not row.get("nms_npz"):
            continue
        npz_path = run_dir / row["nms_npz"]
        if not npz_path.exists():
            errors.append(f"manifest nms_npz missing: {row['nms_npz']}")
            continue
        instances = load_npz(npz_path)
        ann_list = anns_by_image.get(row["file_name"], [])
        if len(ann_list) != instances.count:
            errors.append(f"annotation count mismatch for {row['file_name']}: coco={len(ann_list)} npz={instances.count}")
        for ann, mask in zip(ann_list, instances.masks):
            exact_mask_total += 1
            if int(round(float(ann.get("area", 0)))) != int(mask.sum()):
                errors.append(f"area does not equal final mask pixels for annotation {ann.get('id')}")
            try:
                decoded = _decode_segmentation(ann.get("segmentation"), int(row["height"]), int(row["width"]))
            except Exception:
                continue
            if decoded.shape == mask.shape and np.array_equal(decoded, mask):
                exact_mask_matches += 1
            elif require_exact_masks:
                errors.append(f"decoded mask does not exactly match final mask for annotation {ann.get('id')}")

    for img in images:
        per_image_counts[img["file_name"]] = len(anns_by_image.get(img["file_name"], []))
    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "images": len(images),
        "annotations": len(annotations),
        "categories": categories,
        "per_image_annotation_counts": per_image_counts,
        "exact_mask_matches": exact_mask_matches,
        "exact_mask_total": exact_mask_total,
        "require_exact_masks": require_exact_masks,
    }
    return report


def _run_coco_inputs(run_dir: Path) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    image_records: list[dict[str, Any]] = []
    instances_by_image_id: dict[int, Any] = {}
    for row in manifest:
        if row.get("status") != "ok" or not row.get("nms_npz"):
            continue
        image_id = int(row["coco_image_id"])
        image_records.append(
            {
                "coco_image_id": image_id,
                "file_name": row["file_name"],
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
        )
        instances_by_image_id[image_id] = load_npz(run_dir / row["nms_npz"])
    return image_records, instances_by_image_id


def _copy_cvat_images(run_dir: Path, cvat_image_dir: Path) -> None:
    cvat_image_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest:
        if row.get("copied_image"):
            src = run_dir / row["copied_image"]
            if src.exists():
                shutil.copy2(src, cvat_image_dir / Path(row["file_name"]).name)


def _write_mode_package(run_dir: Path, mode: str, category_name: str, min_area: int) -> dict[str, Any]:
    image_records, instances_by_image_id = _run_coco_inputs(run_dir)
    coco, _, errors = build_coco(
        image_records,
        instances_by_image_id,
        category_name=category_name,
        min_area=min_area,
        segmentation_format=mode,
    )
    mode_dir = run_dir / "cvat_export" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = mode_dir / "instances_default.json"
    write_coco(annotation_path, coco)
    report = validate_cvat_package(
        run_dir,
        annotation_path=annotation_path,
        image_dir=run_dir / "cvat_export" / "images",
        require_exact_masks=(mode == "rle"),
    )
    report["coco_export_errors"] = errors
    write_json(mode_dir / "validation_report.json", report)
    return report


def export_cvat_package(
    run_dir: Path,
    make_zip: bool = False,
    segmentation_format: str = "polygon",
    category_name: str = "book_spine",
    min_area: int = 200,
) -> dict[str, Any]:
    if segmentation_format not in {"polygon", "rle", "both"}:
        raise ValueError(f"Unsupported segmentation_format: {segmentation_format}")
    coco_src = run_dir / "coco" / "instances_default.json"
    cvat_image_dir = run_dir / "cvat_export" / "images"
    cvat_ann_dir = run_dir / "cvat_export" / "annotations"
    cvat_ann_dir.mkdir(parents=True, exist_ok=True)
    _copy_cvat_images(run_dir, cvat_image_dir)
    if not coco_src.exists() and segmentation_format == "polygon":
        raise FileNotFoundError(coco_src)

    reports: dict[str, Any] = {}
    if segmentation_format in {"polygon", "both"}:
        if coco_src.exists():
            shutil.copy2(coco_src, cvat_ann_dir / "instances_default.json")
            legacy_report = validate_cvat_package(run_dir)
            write_json(run_dir / "cvat_export" / "validation_report.json", legacy_report)
        reports["polygon"] = _write_mode_package(run_dir, "polygon", category_name, min_area)
        if segmentation_format == "polygon":
            report = reports["polygon"]
        else:
            report = {"ok": reports["polygon"]["ok"], "modes": reports}
    if segmentation_format in {"rle", "both"}:
        reports["rle"] = _write_mode_package(run_dir, "rle", category_name, min_area)
        if segmentation_format == "rle":
            report = reports["rle"]
        else:
            report = {"ok": all(mode_report["ok"] for mode_report in reports.values()), "modes": reports}
    if make_zip:
        zip_path = run_dir / f"cvat_export_{segmentation_format}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in (run_dir / "cvat_export").rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(run_dir / "cvat_export"))
        report["zip_path"] = str(zip_path)
    return report


def polygon_fidelity_report(run_dir: Path) -> dict[str, Any]:
    annotation_path = run_dir / "cvat_export" / "polygon" / "instances_default.json"
    if not annotation_path.exists():
        annotation_path = run_dir / "cvat_export" / "annotations" / "instances_default.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest:
        if manifest_row.get("status") != "ok" or not manifest_row.get("nms_npz"):
            continue
        image_id = int(manifest_row["coco_image_id"])
        image = images_by_id[image_id]
        height, width = int(image["height"]), int(image["width"])
        instances = load_npz(run_dir / manifest_row["nms_npz"])
        source_ids = instances.extra.get("source_instance_ids", instances.instance_ids)
        for ann, mask, source_id in zip(anns_by_image[image_id], instances.masks, source_ids):
            decoded = _decode_segmentation(ann["segmentation"], height, width)
            intersection = int(np.logical_and(mask, decoded).sum())
            union = int(np.logical_or(mask, decoded).sum())
            original_area = int(mask.sum())
            decoded_area = int(decoded.sum())
            false_negative = int(np.logical_and(mask, ~decoded).sum())
            false_positive = int(np.logical_and(~mask, decoded).sum())
            iou = intersection / union if union else 0.0
            dice = (2 * intersection) / (original_area + decoded_area) if original_area + decoded_area else 0.0
            rows.append(
                {
                    "source_instance_id": int(source_id),
                    "annotation_id": int(ann["id"]),
                    "original_area": original_area,
                    "decoded_polygon_area": decoded_area,
                    "area_ratio": decoded_area / original_area if original_area else 0.0,
                    "intersection": intersection,
                    "union": union,
                    "iou": iou,
                    "dice": dice,
                    "false_negative_pixels": false_negative,
                    "false_positive_pixels": false_positive,
                    "exact_match": bool(np.array_equal(mask, decoded)),
                }
            )

    ious = np.asarray([row["iou"] for row in rows], dtype=float)
    dices = np.asarray([row["dice"] for row in rows], dtype=float)
    area_ratios = np.asarray([row["area_ratio"] for row in rows], dtype=float)
    report = {
        "note": "Polygon fidelity measures conversion loss between NMS masks and polygon COCO masks; it is not SAM3 segmentation accuracy.",
        "instance_count": len(rows),
        "mean_iou": float(ious.mean()) if len(ious) else None,
        "median_iou": float(np.median(ious)) if len(ious) else None,
        "minimum_iou": float(ious.min()) if len(ious) else None,
        "iou_quantiles": {str(q): float(np.quantile(ious, q)) for q in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]} if len(ious) else {},
        "mean_dice": float(dices.mean()) if len(dices) else None,
        "median_area_ratio": float(np.median(area_ratios)) if len(area_ratios) else None,
        "max_area_deviation": float(np.max(np.abs(area_ratios - 1.0))) if len(area_ratios) else None,
        "exact_match_count": int(sum(1 for row in rows if row["exact_match"])),
        "per_instance": rows,
    }
    polygon_dir = run_dir / "cvat_export" / "polygon"
    polygon_dir.mkdir(parents=True, exist_ok=True)
    for report_path in [run_dir / "cvat_export" / "polygon_fidelity_report.json", polygon_dir / "polygon_fidelity_report.json"]:
        write_json(report_path, report)
    csv_path = run_dir / "cvat_export" / "polygon_fidelity_per_instance.csv"
    header = [
        "source_instance_id",
        "annotation_id",
        "original_area",
        "decoded_polygon_area",
        "area_ratio",
        "intersection",
        "union",
        "iou",
        "dice",
        "false_negative_pixels",
        "false_positive_pixels",
        "exact_match",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (polygon_dir / "polygon_fidelity_per_instance.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    return report


def write_nms_pair_review(run_dir: Path, image_name: str, kept_source_id: int, removed_source_id: int) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    row = next(item for item in manifest if item["file_name"] == image_name)
    raw = load_npz(run_dir / row["raw_npz"])
    raw_source_to_index = {int(instance_id): idx for idx, instance_id in enumerate(raw.instance_ids)}
    kept_idx = raw_source_to_index[kept_source_id]
    removed_idx = raw_source_to_index[removed_source_id]
    kept = raw.masks[kept_idx]
    removed = raw.masks[removed_idx]
    intersection_mask = np.logical_and(kept, removed)
    intersection = int(intersection_mask.sum())
    kept_area = int(kept.sum())
    removed_area = int(removed.sum())
    union = int(np.logical_or(kept, removed).sum())
    iou = intersection / union if union else 0.0
    kept_containment = intersection / kept_area if kept_area else 0.0
    removed_containment = intersection / removed_area if removed_area else 0.0

    image = cv2.imread(str(run_dir / row["copied_image"]))
    if image is None:
        raise ValueError(f"Cannot read image for NMS review: {row['copied_image']}")
    overlay = image.copy()
    overlay[kept] = (0.45 * overlay[kept] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[removed] = (0.45 * overlay[removed] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    overlay[intersection_mask] = (0, 255, 255)
    for mask, color, label, score, area in [
        (kept, (0, 255, 0), f"kept source {kept_source_id}", float(raw.scores[kept_idx]), kept_area),
        (removed, (0, 0, 255), f"removed source {removed_source_id}", float(raw.scores[removed_idx]), removed_area),
    ]:
        x, y, w, h = mask_bbox_xywh(mask)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, f"{label} score={score:.3f} area={area}", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(
        overlay,
        f"IoU={iou:.4f} kept_cont={kept_containment:.4f} removed_cont={removed_containment:.4f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    output_path = run_dir / "visualizations" / "nms_review" / f"instance_{kept_source_id}_vs_{removed_source_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    report = {
        "image_name": image_name,
        "kept_source_instance_id": kept_source_id,
        "removed_source_instance_id": removed_source_id,
        "kept_score": float(raw.scores[kept_idx]),
        "removed_score": float(raw.scores[removed_idx]),
        "kept_area": kept_area,
        "removed_area": removed_area,
        "intersection": intersection,
        "union": union,
        "iou": iou,
        "containment_intersection_over_kept": kept_containment,
        "containment_intersection_over_removed": removed_containment,
        "semantic_correctness": "manual review required",
        "visualization": str(output_path.relative_to(run_dir)),
        "diagnostic": "Mask IoU NMS can suppress nearby or overlapping thin instances; semantic correctness requires manual review for adjacent spines, contained masks, and large/small mask conflicts.",
    }
    write_json(run_dir / "visualizations" / "nms_review" / f"instance_{kept_source_id}_vs_{removed_source_id}.json", report)
    return report
