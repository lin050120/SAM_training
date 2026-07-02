from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as cocomask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.npz_io import load_npz
from core.run_manager import write_json


def _decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    rles = cocomask.frPyObjects(segmentation, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle).astype(bool)


def _path_status(run_dir: Path, relative: str) -> dict[str, Any]:
    path = run_dir / relative
    return {"path": relative, "exists": path.exists(), "is_dir": path.is_dir(), "is_file": path.is_file()}


def accept_run(run_dir: Path, image_name: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "input_images",
        "npz_raw",
        "npz_nms",
        "visualizations/raw",
        "visualizations/nms",
        "coco/instances_default.json",
        "run_config.json",
        "manifest.json",
        "errors.json",
        "logs",
    ]
    required_status = [_path_status(run_dir, item) for item in required]
    for item in required_status:
        if not item["exists"]:
            errors.append(f"missing required path: {item['path']}")

    image_path = run_dir / "input_images" / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        errors.append(f"cannot read input image: {image_path}")
        height = width = 0
    else:
        height, width = image.shape[:2]

    raw_npz_path = run_dir / "npz_raw" / f"{Path(image_name).stem}.npz"
    nms_npz_path = run_dir / "npz_nms" / f"{Path(image_name).stem}.npz"
    raw_keys: list[str] = []
    raw = None
    nms = None
    if raw_npz_path.exists():
        with np.load(raw_npz_path, allow_pickle=False) as data:
            raw_keys = list(data.files)
        raw = load_npz(raw_npz_path)
    else:
        errors.append(f"missing raw npz: {raw_npz_path}")
    if nms_npz_path.exists():
        nms = load_npz(nms_npz_path)
    else:
        errors.append(f"missing nms npz: {nms_npz_path}")

    if raw is not None:
        if raw.count != 20:
            errors.append(f"raw mask count expected 20, got {raw.count}")
        if len(raw.scores) != raw.count:
            errors.append(f"raw score count mismatch: scores={len(raw.scores)} masks={raw.count}")
        if raw.masks.shape[1:] != (height, width):
            errors.append(f"raw mask shape mismatch: {raw.masks.shape[1:]} vs image {(height, width)}")
    if nms is not None and nms.count != 19:
        errors.append(f"nms mask count expected 19, got {nms.count}")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) if (run_dir / "manifest.json").exists() else []
    row = next((item for item in manifest if item.get("file_name") == image_name), {})
    for key in ["copied_image", "raw_npz", "nms_npz", "raw_visualization", "nms_visualization"]:
        value = row.get(key)
        if not value or not (run_dir / value).exists():
            errors.append(f"manifest path missing for {key}: {value}")

    run_errors = json.loads((run_dir / "errors.json").read_text(encoding="utf-8")) if (run_dir / "errors.json").exists() else None
    if run_errors:
        errors.append(f"errors.json is not empty: {run_errors}")

    for relative in [f"visualizations/raw/{Path(image_name).stem}.jpg", f"visualizations/nms/{Path(image_name).stem}.jpg"]:
        visual = cv2.imread(str(run_dir / relative))
        if visual is None:
            errors.append(f"cannot read visualization: {relative}")

    coco = json.loads((run_dir / "coco" / "instances_default.json").read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    image_ids = [img.get("id") for img in images]
    ann_ids = [ann.get("id") for ann in annotations]
    if len(ann_ids) != len(set(ann_ids)):
        errors.append("duplicate annotation_id")
    if len(image_ids) != len(set(image_ids)):
        errors.append("duplicate image_id")
    image_record = next((img for img in images if img.get("file_name") == image_name), None)
    if image_record is None:
        errors.append(f"COCO image missing: {image_name}")
        image_id = None
    else:
        image_id = image_record["id"]
        if int(image_record.get("width", -1)) != width or int(image_record.get("height", -1)) != height:
            errors.append("COCO image dimensions mismatch")
    image_annotations = [ann for ann in annotations if ann.get("image_id") == image_id]
    if len(image_annotations) != 19:
        errors.append(f"COCO annotation count expected 19, got {len(image_annotations)}")

    segmentation_exact_matches = 0
    segmentation_decodable = True
    segmentation_mismatches: list[dict[str, int]] = []
    area_matches = 0
    bbox_valid = True
    if nms is not None:
        for ann, mask in zip(image_annotations, nms.masks):
            x, y, w, h = [float(v) for v in ann.get("bbox", [])]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 1 or y + h > height + 1:
                bbox_valid = False
            if int(round(float(ann.get("area", 0)))) == int(mask.sum()):
                area_matches += 1
            try:
                decoded = _decode_segmentation(ann.get("segmentation"), height, width)
            except Exception as exc:
                segmentation_decodable = False
                errors.append(f"segmentation decode failed for annotation {ann.get('id')}: {exc!r}")
                continue
            if decoded.shape == mask.shape and np.array_equal(decoded, mask):
                segmentation_exact_matches += 1
            else:
                mismatch_pixels = int(np.logical_xor(decoded, mask).sum())
                segmentation_mismatches.append({"annotation_id": int(ann.get("id")), "mismatch_pixels": mismatch_pixels})
                warnings.append(
                    f"segmentation polygon is decodable but not pixel-identical for annotation {ann.get('id')}"
                )
        if not bbox_valid:
            errors.append("one or more COCO bboxes are out of bounds")
        if area_matches != len(image_annotations):
            errors.append(f"area mismatch count: {len(image_annotations) - area_matches}")
        if segmentation_exact_matches != len(image_annotations):
            errors.append(
                f"segmentation exact mask mismatch count: {len(image_annotations) - segmentation_exact_matches}"
            )

    cvat_report_path = run_dir / "cvat_export" / "validation_report.json"
    cvat_report = json.loads(cvat_report_path.read_text(encoding="utf-8")) if cvat_report_path.exists() else None
    if not cvat_report or not cvat_report.get("ok") or cvat_report.get("errors"):
        errors.append(f"CVAT validation failed: {cvat_report}")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "run_dir": str(run_dir),
        "image_name": image_name,
        "required_paths": required_status,
        "image": {"width": width, "height": height},
        "raw_npz": {
            "path": str(raw_npz_path),
            "keys": raw_keys,
            "mask_count": raw.count if raw is not None else None,
            "score_count": int(len(raw.scores)) if raw is not None else None,
            "mask_shape": list(raw.masks.shape) if raw is not None else None,
            "mask_dtype": str(raw.masks.dtype) if raw is not None else None,
        },
        "nms_npz": {
            "path": str(nms_npz_path),
            "mask_count": nms.count if nms is not None else None,
            "source_instance_ids": nms.extra.get("source_instance_ids", np.array([], dtype=np.int32)).astype(int).tolist()
            if nms is not None
            else [],
        },
        "nms": {
            "threshold": json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")).get("nms_threshold")
            if (run_dir / "run_config.json").exists()
            else None,
            "removed": row.get("nms_removed", []),
        },
        "coco": {
            "annotation_count": len(annotations),
            "image_annotation_count": len(image_annotations),
            "annotation_ids_unique": len(ann_ids) == len(set(ann_ids)),
            "image_ids_unique": len(image_ids) == len(set(image_ids)),
            "image_id_valid": image_id in set(image_ids) if image_id is not None else False,
            "bbox_valid": bbox_valid,
            "area_matches": area_matches,
            "segmentation_decodable": segmentation_decodable,
            "segmentation_exact_matches": segmentation_exact_matches,
            "segmentation_mismatches": segmentation_mismatches,
        },
        "manifest": row,
        "errors_json": run_errors,
        "cvat_validation": cvat_report,
    }
    write_json(run_dir / "acceptance_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a unified inference run without running SAM3.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-name", default="im_000001.png")
    args = parser.parse_args()
    report = accept_run(args.run_dir, args.image_name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
