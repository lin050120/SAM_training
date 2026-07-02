from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as cocomask

from core.mask_nms import mask_bbox_xywh
from core.npz_io import InstanceSet


def mask_to_polygon(mask: np.ndarray, min_area: int = 1) -> tuple[list[list[float]], list[float] | None, float]:
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    area = 0.0
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area or len(contour) < 3:
            continue
        points = contour.reshape(-1, 2).astype(float)
        if len(points) < 3:
            continue
        polygons.append(points.reshape(-1).tolist())
        xs_all.append(points[:, 0])
        ys_all.append(points[:, 1])
        area += float(mask.sum()) if not area else 0.0
    if not polygons:
        return [], None, 0.0
    bbox = [float(v) for v in mask_bbox_xywh(mask)]
    return polygons, bbox, float(mask.sum())


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    encoded = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in encoded["size"]], "counts": counts}


def build_coco(
    image_records: list[dict[str, Any]],
    instances_by_image_id: dict[int, InstanceSet],
    category_name: str = "book_spine",
    min_area: int = 1,
    segmentation_format: str = "polygon",
) -> tuple[dict[str, Any], dict[int, list[int]], list[str]]:
    if segmentation_format not in {"polygon", "rle"}:
        raise ValueError(f"Unsupported segmentation_format: {segmentation_format}")
    coco: dict[str, Any] = {
        "info": {"description": "book spine SAM3 predictions"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name, "supercategory": ""}],
    }
    ann_ids_by_image: dict[int, list[int]] = {}
    errors: list[str] = []
    ann_id = 1
    for record in image_records:
        image_id = int(record["coco_image_id"])
        width = int(record["width"])
        height = int(record["height"])
        file_name = str(record["file_name"])
        coco["images"].append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        ann_ids_by_image[image_id] = []
        instances = instances_by_image_id.get(image_id)
        if instances is None:
            continue
        for idx, mask in enumerate(instances.masks):
            if segmentation_format == "rle":
                area = float(mask.sum())
                bbox = [float(v) for v in mask_bbox_xywh(mask)]
                segmentation: Any = mask_to_rle(mask)
                if area <= 0:
                    continue
            else:
                polygons, bbox, area = mask_to_polygon(mask, min_area=min_area)
                if not polygons or bbox is None or area <= 0:
                    continue
                segmentation = polygons
            x, y, w, h = bbox
            if x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
                errors.append(f"annotation {ann_id} bbox outside image bounds: {file_name}")
                continue
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "score": float(instances.scores[idx]) if idx < len(instances.scores) else None,
                }
            )
            ann_ids_by_image[image_id].append(ann_id)
            ann_id += 1
    return coco, ann_ids_by_image, errors


def write_coco(path: Path, coco: dict[str, Any]) -> None:
    path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_coco(coco: dict[str, Any], image_dir: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    image_ids = [img["id"] for img in coco.get("images", [])]
    ann_ids = [ann["id"] for ann in coco.get("annotations", [])]
    if len(image_ids) != len(set(image_ids)):
        errors.append("duplicate image_id")
    if len(ann_ids) != len(set(ann_ids)):
        errors.append("duplicate annotation_id")
    valid_images = set(image_ids)
    for image in coco.get("images", []):
        if image_dir and not (image_dir / image["file_name"]).exists():
            errors.append(f"missing image file: {image['file_name']}")
    for ann in coco.get("annotations", []):
        if ann.get("image_id") not in valid_images:
            errors.append(f"annotation {ann.get('id')} references missing image_id")
        if not ann.get("segmentation"):
            errors.append(f"annotation {ann.get('id')} empty segmentation")
        bbox = ann.get("bbox") or []
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            errors.append(f"annotation {ann.get('id')} invalid bbox")
        if ann.get("area", 0) <= 0:
            errors.append(f"annotation {ann.get('id')} invalid area")
    return {"ok": not errors, "errors": errors, "images": len(image_ids), "annotations": len(ann_ids)}
