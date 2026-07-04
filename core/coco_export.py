from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as cocomask

from core.mask_nms import mask_bbox_xywh
from core.npz_io import InstanceSet

POLYGON_TARGET_POINTS = 8
POLYGON_MAX_POINTS = 12
POLYGON_USE_MINRECT_FALLBACK = True


def simplify_contour_to_target(
    contour: np.ndarray,
    target_points: int = POLYGON_TARGET_POINTS,
    max_points: int = POLYGON_MAX_POINTS,
    use_minrect_fallback: bool = POLYGON_USE_MINRECT_FALLBACK,
) -> np.ndarray:
    """Approximate one contour to a small CVAT-editable polygon.

    Mirrors the legacy ft_02_1 flow: binary-search approxPolyDP epsilon until the
    contour has at most target_points while staying as tight as possible. Rectangular
    spines naturally remain 4 points; irregular masks usually land near 8 points.
    """
    if len(contour) < 3:
        return contour.reshape(-1, 2)
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return contour.reshape(-1, 2)
    lo, hi = 0.0, peri
    best = cv2.approxPolyDP(contour, peri * 0.02, True)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(contour, mid, True)
        if len(approx) > target_points:
            lo = mid
        else:
            best = approx
            hi = mid
    points = best.reshape(-1, 2)
    if len(points) < 3:
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect).astype(np.int32)
    if use_minrect_fallback and len(points) > max_points:
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect).astype(np.int32)
    return points


def mask_to_polygon(
    mask: np.ndarray,
    min_area: int = 1,
    target_points: int = POLYGON_TARGET_POINTS,
    max_points: int = POLYGON_MAX_POINTS,
) -> tuple[list[list[float]], list[float] | None, float]:
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None, 0.0
    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < min_area or len(contour) < 3:
        return [], None, 0.0
    points = simplify_contour_to_target(contour, target_points=target_points, max_points=max_points).astype(float)
    if len(points) < 3:
        return [], None, 0.0
    bbox = [float(v) for v in mask_bbox_xywh(mask)]
    return [points.reshape(-1).tolist()], bbox, float(mask.sum())


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
    polygon_target_points: int = POLYGON_TARGET_POINTS,
    polygon_max_points: int = POLYGON_MAX_POINTS,
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
                polygons, bbox, area = mask_to_polygon(
                    mask,
                    min_area=min_area,
                    target_points=polygon_target_points,
                    max_points=polygon_max_points,
                )
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
