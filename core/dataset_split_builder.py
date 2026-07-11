from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetBuildConfig:
    annotation_pool_dir: Path
    test_dir: Path
    output_dir: Path
    category_name: str = "book spine"
    val_ratio: float = 0.10
    seed: int = 42
    filename_prefix: str = "im_"
    filename_pad: int = 6
    overwrite: bool = False


@dataclass
class DatasetRecord:
    new_name: str
    source_path: Path
    width: int
    height: int
    annotations: list[dict[str, Any]]
    source: str
    batch: str
    original_name: str
    image_sha256: str


def _is_coco_json(path: Path) -> bool:
    if path.name in {"manifest.json", "run_config.json", "dataset_info.json"}:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list)


def _collect_coco_batches(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    root = Path(root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"input directory does not exist or is not a directory: {root}")
    batches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.json")):
        if _is_coco_json(path):
            batches.append((path, json.loads(path.read_text(encoding="utf-8"))))
    if not batches:
        raise ValueError(f"no valid COCO json found under: {root}")
    return batches


def _locate_image(coco_dir: Path, file_name: str) -> Path | None:
    raw = Path(file_name)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            coco_dir / file_name,
            coco_dir / raw.name,
            coco_dir / "images" / raw.name,
            coco_dir.parent / "images" / raw.name,
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
            return resolved
    return None


def _image_size(path: Path, image_record: dict[str, Any]) -> tuple[int, int]:
    width = int(image_record.get("width") or 0)
    height = int(image_record.get("height") or 0)
    if width > 0 and height > 0:
        return width, height
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image dimensions: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox": annotation.get("bbox", [0, 0, 0, 0]),
        "segmentation": annotation.get("segmentation", []),
        "area": annotation.get("area", 0.0),
        "iscrowd": annotation.get("iscrowd", 0),
    }


def _gather_records(root: Path, source: str, counter_start: int, config: DatasetBuildConfig) -> tuple[list[DatasetRecord], int]:
    records: list[DatasetRecord] = []
    counter = counter_start
    missing_images: list[str] = []
    for coco_path, coco in _collect_coco_batches(root):
        coco_dir = coco_path.parent
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            anns_by_image[int(annotation["image_id"])].append(annotation)
        for image_record in coco.get("images", []):
            original_file_name = str(image_record.get("file_name", ""))
            source_path = _locate_image(coco_dir, original_file_name)
            if source_path is None:
                missing_images.append(f"{coco_path}: {original_file_name}")
                continue
            counter += 1
            width, height = _image_size(source_path, image_record)
            new_name = f"{config.filename_prefix}{counter:0{config.filename_pad}d}{source_path.suffix.lower()}"
            image_id = int(image_record["id"])
            records.append(
                DatasetRecord(
                    new_name=new_name,
                    source_path=source_path,
                    width=width,
                    height=height,
                    annotations=[_copy_annotation(a) for a in anns_by_image.get(image_id, [])],
                    source=source,
                    batch=coco_dir.name,
                    original_name=Path(original_file_name).name,
                    image_sha256=_sha256_file(source_path),
                )
            )
    if missing_images:
        preview = "; ".join(missing_images[:10])
        raise FileNotFoundError(f"{len(missing_images)} image(s) referenced by COCO were not found: {preview}")
    return records, counter


def _duplicate_sha_groups(records: list[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    groups: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        groups[record.image_sha256].append(record)
    return {sha: items for sha, items in groups.items() if len(items) > 1}


def _describe_records(records: list[DatasetRecord], limit: int = 6) -> str:
    return "; ".join(
        f"{r.source}:{r.batch}/{r.original_name} ({r.source_path})"
        for r in records[:limit]
    )


def _validate_no_duplicate_pool_images(pool_records: list[DatasetRecord]) -> None:
    duplicate_groups = _duplicate_sha_groups(pool_records)
    if duplicate_groups:
        sha, records = next(iter(duplicate_groups.items()))
        raise ValueError(
            "annotation pool contains duplicate image content; refusing to split because "
            "duplicates can leak between train and val. "
            f"sha256={sha}, examples={_describe_records(records)}"
        )


def _validate_no_pool_test_overlap(pool_records: list[DatasetRecord], test_records: list[DatasetRecord]) -> None:
    pool_by_sha: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in pool_records:
        pool_by_sha[record.image_sha256].append(record)
    for test_record in test_records:
        overlaps = pool_by_sha.get(test_record.image_sha256)
        if overlaps:
            raise ValueError(
                "annotation pool and test directory contain the same image content; refusing to build "
                "because test would leak into train/val. "
                f"sha256={test_record.image_sha256}, pool={_describe_records(overlaps)}, "
                f"test={_describe_records([test_record])}"
            )


def _split_pool(records: list[DatasetRecord], val_ratio: float, seed: int) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    if not records:
        raise ValueError("annotation pool produced no usable images")
    shuffled = list(records)
    Random(seed).shuffle(shuffled)
    if val_ratio <= 0:
        val_count = 0
    else:
        val_count = round(len(shuffled) * val_ratio)
        if len(shuffled) >= 2:
            val_count = max(1, val_count)
        val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _write_split(output_dir: Path, split: str, records: list[DatasetRecord], category_name: str) -> dict[str, int]:
    split_dir = output_dir / split
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        shutil.copy2(record.source_path, image_dir / record.new_name)
        images.append(
            {
                "id": image_id,
                "file_name": record.new_name,
                "width": int(record.width),
                "height": int(record.height),
            }
        )
        for annotation in record.annotations:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": annotation["bbox"],
                    "segmentation": annotation["segmentation"],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"],
                }
            )
            annotation_id += 1
    coco = {
        "info": {"description": f"{category_name} SAM3 {split} split"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name, "supercategory": ""}],
    }
    (split_dir / "annotations.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations)}


def _write_manifest(output_dir: Path, split_records: dict[str, list[DatasetRecord]], config: DatasetBuildConfig) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["new_name", "source", "batch", "original_name", "split", "source_path"])
        for split, records in split_records.items():
            for record in records:
                writer.writerow([record.new_name, record.source, record.batch, record.original_name, split, str(record.source_path)])
    metadata = {
        "created_at": datetime.now().isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "policy": {
            "annotation_pool_splits": ["train", "val"],
            "test_dir_split": "test",
            "val_ratio_target": config.val_ratio,
            "test_never_mixed_into_train_or_val": True,
        },
        "splits": {
            split: {
                "image_count": len(records),
                "annotation_count": sum(len(r.annotations) for r in records),
                "source_counts": dict(_count_sources(records)),
            }
            for split, records in split_records.items()
        },
    }
    (output_dir / "dataset_build_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _count_sources(records: list[DatasetRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts


def _install_built_dataset(tmp_output: Path, output_dir: Path, overwrite: bool) -> Path | None:
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")
        backup = output_dir.with_name(f"{output_dir.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        os.replace(output_dir, backup)
        os.replace(tmp_output, output_dir)
        return backup
    if output_dir.exists():
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_output, output_dir)
    return None


def _validate_output_not_inside_inputs(output_dir: Path, *input_dirs: Path) -> None:
    resolved_output = Path(output_dir).expanduser().resolve(strict=False)
    for input_dir in input_dirs:
        resolved_input = Path(input_dir).expanduser().resolve(strict=False)
        if resolved_output == resolved_input or resolved_output.is_relative_to(resolved_input):
            raise ValueError(
                "output directory must not be inside an input directory; otherwise previous outputs "
                f"can be re-ingested as source COCO on rebuild: output={resolved_output}, input={resolved_input}"
            )


def build_training_dataset(config: DatasetBuildConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir).expanduser().resolve(strict=False)
    if not config.category_name.strip():
        raise ValueError("category_name must not be empty")
    if not (0 <= float(config.val_ratio) < 1):
        raise ValueError(f"val_ratio must be in [0, 1), got {config.val_ratio}")
    if output_dir == Path("/"):
        raise ValueError("refusing to write dataset to filesystem root")
    _validate_output_not_inside_inputs(output_dir, config.annotation_pool_dir, config.test_dir)

    test_records, counter = _gather_records(config.test_dir, "test", 0, config)
    pool_records, _counter = _gather_records(config.annotation_pool_dir, "pool", counter, config)
    if not test_records:
        raise ValueError("test directory produced no usable images")
    _validate_no_duplicate_pool_images(pool_records)
    _validate_no_pool_test_overlap(pool_records, test_records)
    train_records, val_records = _split_pool(pool_records, float(config.val_ratio), int(config.seed))
    split_records = {"train": train_records, "val": val_records, "test": test_records}

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.build_", dir=parent))
    try:
        split_counts = {
            split: _write_split(tmp_path, split, records, config.category_name)
            for split, records in split_records.items()
        }
        _write_manifest(tmp_path, split_records, config)
        backup = _install_built_dataset(tmp_path, output_dir, config.overwrite)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    return {
        "ok": True,
        "output_dir": str(output_dir),
        "backup_dir": str(backup) if backup else None,
        "category_name": config.category_name,
        "val_ratio_target": config.val_ratio,
        "seed": config.seed,
        "splits": split_counts,
        "paths": {
            "train_images": str(output_dir / "train" / "images"),
            "train_annotations": str(output_dir / "train" / "annotations.json"),
            "val_images": str(output_dir / "val" / "images"),
            "val_annotations": str(output_dir / "val" / "annotations.json"),
            "test_images": str(output_dir / "test" / "images"),
            "test_annotations": str(output_dir / "test" / "annotations.json"),
            "manifest_csv": str(output_dir / "manifest.csv"),
            "summary_json": str(output_dir / "dataset_build_summary.json"),
        },
    }
