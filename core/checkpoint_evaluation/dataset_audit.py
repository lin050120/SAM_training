"""Dataset split audit for checkpoint evaluation.

The audit is deliberately file-based and split-aware: it records COCO counts,
annotation/image hashes, missing images, duplicate identifiers, and cross-split
overlap by normalized file name, byte size, file SHA256, and decoded pixel hash.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from core.checkpoint_evaluation.report_writer import atomic_write_json, atomic_write_text

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_of_path(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha256(path: Path) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    h = hashlib.sha256()
    h.update(str(image.shape).encode("utf-8"))
    h.update(str(image.dtype).encode("utf-8"))
    h.update(image.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class SplitPaths:
    name: str
    annotations: Path
    images_dir: Path


def _load_coco(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_record(split: SplitPaths) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotations = Path(split.annotations).expanduser().resolve(strict=False)
    images_dir = Path(split.images_dir).expanduser().resolve(strict=False)
    data = _load_coco(annotations)
    images = data.get("images", [])
    annotations_list = data.get("annotations", [])
    ann_by_image: dict[int, int] = {}
    ann_ids: list[int] = []
    for ann in annotations_list:
        image_id = int(ann.get("image_id", -1))
        ann_by_image[image_id] = ann_by_image.get(image_id, 0) + 1
        if "id" in ann:
            ann_ids.append(int(ann["id"]))
    image_ids = [int(im["id"]) for im in images]
    file_names = [str(im["file_name"]) for im in images]
    disk_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ) if images_dir.is_dir() else []
    image_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for im in images:
        file_name = str(im["file_name"])
        path = images_dir / file_name
        if not path.is_file():
            missing_images.append(file_name)
            file_sha = None
            px_sha = None
            size = None
        else:
            file_sha = sha256_of_path(path)
            px_sha = pixel_sha256(path)
            size = path.stat().st_size
        image_rows.append(
            {
                "split": split.name,
                "image_id": int(im["id"]),
                "file_name": file_name,
                "normalized_file_name": Path(file_name).name.lower(),
                "image_path": str(path),
                "width": int(im.get("width", 0)),
                "height": int(im.get("height", 0)),
                "file_size": size,
                "image_sha256": file_sha,
                "pixel_sha256": px_sha,
                "annotation_count": ann_by_image.get(int(im["id"]), 0),
            }
        )
    duplicate_image_ids = sorted({i for i in image_ids if image_ids.count(i) > 1})
    duplicate_file_names = sorted({n for n in file_names if file_names.count(n) > 1})
    duplicate_annotation_ids = sorted({i for i in ann_ids if ann_ids.count(i) > 1})
    record = {
        "split": split.name,
        "annotations_path": str(annotations),
        "annotations_sha256": sha256_of_path(annotations),
        "images_dir": str(images_dir),
        "image_file_count": len(disk_files),
        "coco_image_count": len(images),
        "unique_image_id_count": len(set(image_ids)),
        "unique_file_name_count": len(set(file_names)),
        "annotation_count": len(annotations_list),
        "gt_instance_count": len(annotations_list),
        "category_ids": [c.get("id") for c in data.get("categories", [])],
        "category_names": [c.get("name") for c in data.get("categories", [])],
        "missing_images": missing_images,
        "unannotated_images": [r["file_name"] for r in image_rows if r["annotation_count"] == 0],
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_file_names": duplicate_file_names,
        "duplicate_annotation_ids": duplicate_annotation_ids,
    }
    return record, image_rows


def _overlap_count(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], key: str) -> int:
    a = {r.get(key) for r in rows_a if r.get(key)}
    b = {r.get(key) for r in rows_b if r.get(key)}
    return len(a & b)


def audit_dataset_splits(splits: list[SplitPaths], output_dir: Path | None = None) -> dict[str, Any]:
    split_records: dict[str, dict[str, Any]] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        record, rows = _split_record(split)
        split_records[split.name] = record
        rows_by_split[split.name] = rows

    overlap: dict[str, dict[str, int]] = {}
    names = [s.name for s in splits]
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            key = f"{left}_vs_{right}"
            overlap[key] = {
                "normalized_file_name": _overlap_count(rows_by_split[left], rows_by_split[right], "normalized_file_name"),
                "file_size": _overlap_count(rows_by_split[left], rows_by_split[right], "file_size"),
                "image_sha256": _overlap_count(rows_by_split[left], rows_by_split[right], "image_sha256"),
                "pixel_sha256": _overlap_count(rows_by_split[left], rows_by_split[right], "pixel_sha256"),
            }

    blocking_reasons: list[str] = []
    for name, record in split_records.items():
        if record["missing_images"]:
            blocking_reasons.append(f"{name}: missing images ({len(record['missing_images'])})")
        if record["duplicate_image_ids"]:
            blocking_reasons.append(f"{name}: duplicate image ids ({len(record['duplicate_image_ids'])})")
        if record["duplicate_annotation_ids"]:
            blocking_reasons.append(f"{name}: duplicate annotation ids ({len(record['duplicate_annotation_ids'])})")
    audit = {
        "splits": split_records,
        "cross_split_overlap": overlap,
        "allowed_for_formal_evaluation": False,
        "formal_evaluation_blocking_reasons": [
            "dataset identity registry has allowed_for_model_evaluation=false; test results are diagnostic only"
        ] + blocking_reasons,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "dataset_split_audit.json", audit)
        from io import StringIO

        buffer = StringIO()
        fieldnames = [
            "split", "annotations_path", "annotations_sha256", "image_file_count",
            "coco_image_count", "unique_image_id_count", "unique_file_name_count",
            "annotation_count", "gt_instance_count", "missing_image_count",
            "unannotated_image_count", "duplicate_image_id_count",
            "duplicate_annotation_id_count",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for record in split_records.values():
            writer.writerow(
                {
                    **{k: record.get(k) for k in fieldnames if k in record},
                    "missing_image_count": len(record["missing_images"]),
                    "unannotated_image_count": len(record["unannotated_images"]),
                    "duplicate_image_id_count": len(record["duplicate_image_ids"]),
                    "duplicate_annotation_id_count": len(record["duplicate_annotation_ids"]),
                }
            )
        atomic_write_text(output_dir / "dataset_split_audit.csv", buffer.getvalue())
    return audit
