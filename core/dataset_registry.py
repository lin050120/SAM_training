from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import BOOK_ROOT
from core.dataset_identity import DEFAULT_REGISTRY_PATH, load_registry

REQUIRED_SPLITS = ("train", "val")
OPTIONAL_SPLITS = ("test",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(BOOK_ROOT.resolve(strict=False)))
    except ValueError:
        return str(resolved)


def _resolve_project_path(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (BOOK_ROOT / raw).resolve(strict=False)


def _load_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotations file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("images"), list):
        raise ValueError(f"COCO file has no images list: {path}")
    if not isinstance(data.get("annotations"), list):
        raise ValueError(f"COCO file has no annotations list: {path}")
    if not isinstance(data.get("categories"), list) or not data.get("categories"):
        raise ValueError(f"COCO file has no categories: {path}")
    return data


def _image_path(images_dir: Path, file_name: str) -> Path:
    raw = Path(file_name)
    if raw.is_absolute():
        return raw.resolve(strict=False)
    # Keep any subdirectory components: preflight resolves images as
    # <image_dir>/<file_name>, and registration must agree with it.
    return (images_dir / raw).resolve(strict=False)


def summarize_split(dataset_root: Path, split: str) -> dict[str, Any]:
    split_dir = dataset_root / split
    images_dir = split_dir / "images"
    annotations_path = split_dir / "annotations.json"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{split} images directory does not exist: {images_dir}")
    coco = _load_coco(annotations_path)
    image_hashes: list[str] = []
    for image in coco["images"]:
        file_name = str(image.get("file_name", ""))
        if not file_name:
            raise ValueError(f"{split} COCO contains an image with empty file_name")
        path = _image_path(images_dir, file_name)
        if not path.is_file():
            raise FileNotFoundError(f"{split} image referenced by COCO is missing: {path}")
        image_hashes.append(sha256_file(path))
    return {
        "images_dir": _rel(images_dir),
        "annotations_path": _rel(annotations_path),
        "annotations_sha256": sha256_file(annotations_path),
        "image_count": len(coco["images"]),
        "unique_image_count": len(set(image_hashes)),
        "annotation_count": len(coco["annotations"]),
        "categories": coco["categories"],
        # consumed (and removed) by build_dataset_identity_artifacts for
        # cross-split duplicate accounting; never persisted.
        "image_hashes": image_hashes,
    }


def build_dataset_identity_artifacts(
    dataset_root: Path | str,
    dataset_id: str,
    annotation_source: str,
    annotation_source_evidence: str,
    human_reviewed: bool,
    independently_corrected_gt: bool,
    allowed_for_formal_training: bool,
    allowed_for_model_evaluation: bool = False,
    intended_use: str = "formal_training_validation_and_diagnostic_test",
    manifests_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_id = (dataset_id or "").strip()
    if not dataset_id:
        raise ValueError("dataset_id must not be empty")
    if not annotation_source.strip():
        raise ValueError("annotation_source must not be empty")
    if not annotation_source_evidence.strip():
        raise ValueError("annotation_source_evidence must not be empty")
    if allowed_for_formal_training and not (human_reviewed and independently_corrected_gt):
        raise ValueError(
            "allowed_for_formal_training=true requires human_reviewed=true and independently_corrected_gt=true"
        )

    root = _resolve_project_path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist or is not a directory: {root}")

    splits: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        splits[split] = summarize_split(root, split)
    for split in OPTIONAL_SPLITS:
        if (root / split / "annotations.json").exists() or (root / split / "images").exists():
            splits[split] = summarize_split(root, split)

    all_image_hashes: list[str] = []
    for item in splits.values():
        all_image_hashes.extend(item.pop("image_hashes"))
    image_file_count = sum(int(item["image_count"]) for item in splits.values())
    # Unique/duplicate accounting is across the WHOLE dataset, not per split:
    # the same photo appearing in train and val is exactly the leakage this
    # registry exists to make visible.
    unique_image_count = len(set(all_image_hashes))
    hash_counts = Counter(all_image_hashes)
    exact_duplicate_group_count = sum(1 for count in hash_counts.values() if count > 1)
    annotation_count = sum(int(item["annotation_count"]) for item in splits.values())
    created_at = datetime.now(timezone.utc).isoformat()
    manifest_root = Path(manifests_dir) if manifests_dir is not None else BOOK_ROOT / "data_manifests"
    dataset_manifest_path = manifest_root / f"{dataset_id}_dataset_manifest.json"
    split_manifest_path = manifest_root / f"{dataset_id}_split_manifest.json"

    dataset_manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "created_at": created_at,
        "book_root": str(BOOK_ROOT),
        "dataset_root": _rel(root),
        "annotation_source": annotation_source,
        "annotation_source_evidence": annotation_source_evidence,
        "human_reviewed": bool(human_reviewed),
        "independently_corrected_gt": bool(independently_corrected_gt),
        "intended_use": intended_use,
        "allowed_for_formal_training": bool(allowed_for_formal_training),
        "allowed_for_model_evaluation": bool(allowed_for_model_evaluation),
        "splits": splits,
        "total_image_file_count": image_file_count,
        "total_unique_image_count": unique_image_count,
        "total_annotation_count": annotation_count,
    }
    split_manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "created_at": created_at,
        "dataset_root": _rel(root),
        "split_policy": "registered from an existing train/val/test dataset directory via UI",
        "splits": splits,
        "total_image_file_count": image_file_count,
        "total_unique_image_count": unique_image_count,
        "total_annotation_count": annotation_count,
    }
    registry_entry = {
        "dataset_id": dataset_id,
        "annotation_source": annotation_source,
        "annotation_source_evidence": annotation_source_evidence,
        "human_reviewed": bool(human_reviewed),
        "independently_corrected_gt": bool(independently_corrected_gt),
        "intended_use": intended_use,
        "allowed_for_formal_training": bool(allowed_for_formal_training),
        "allowed_for_model_evaluation": bool(allowed_for_model_evaluation),
        "max_epochs_without_human_review": 1,
        "dataset_manifest_path": _rel(dataset_manifest_path),
        "dataset_manifest_sha256": None,
        "split_manifest_path": _rel(split_manifest_path),
        "split_manifest_sha256": None,
        "source_coco_file_count": len(splits),
        "image_file_count": image_file_count,
        "unique_image_count": unique_image_count,
        "annotation_count": annotation_count,
        "exact_duplicate_group_count": exact_duplicate_group_count,
        "splits": splits,
        "basis_and_review_reports": [],
    }
    return registry_entry, dataset_manifest, split_manifest


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(encoded)
        handle.flush()
    tmp.replace(path)


def register_dataset_identity(
    dataset_root: Path | str,
    dataset_id: str,
    annotation_source: str,
    annotation_source_evidence: str,
    human_reviewed: bool,
    independently_corrected_gt: bool,
    allowed_for_formal_training: bool,
    allowed_for_model_evaluation: bool = False,
    intended_use: str = "formal_training_validation_and_diagnostic_test",
    overwrite_existing: bool = False,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    manifests_dir: Path | None = None,
) -> dict[str, Any]:
    entry, dataset_manifest, split_manifest = build_dataset_identity_artifacts(
        dataset_root=dataset_root,
        dataset_id=dataset_id,
        annotation_source=annotation_source,
        annotation_source_evidence=annotation_source_evidence,
        human_reviewed=human_reviewed,
        independently_corrected_gt=independently_corrected_gt,
        allowed_for_formal_training=allowed_for_formal_training,
        allowed_for_model_evaluation=allowed_for_model_evaluation,
        intended_use=intended_use,
        manifests_dir=manifests_dir,
    )
    manifest_path = _resolve_project_path(entry["dataset_manifest_path"])
    split_path = _resolve_project_path(entry["split_manifest_path"])
    if not overwrite_existing and (manifest_path.exists() or split_path.exists()):
        raise FileExistsError(f"manifest file already exists for dataset_id={dataset_id!r}")

    registry = load_registry(registry_path)
    datasets = list(registry.get("datasets", []))
    existing_indexes = [idx for idx, item in enumerate(datasets) if item.get("dataset_id") == dataset_id]
    if existing_indexes and not overwrite_existing:
        raise ValueError(f"dataset_id already exists in registry: {dataset_id}")

    _atomic_write_json(manifest_path, dataset_manifest)
    _atomic_write_json(split_path, split_manifest)
    # The manifests must NOT contain their own hash: the registry records the
    # sha256 of the file as written, so `sha256sum <manifest>` must reproduce
    # it (same convention as the existing book_spine_human_corrected_v1 entry).
    entry["dataset_manifest_sha256"] = sha256_file(manifest_path)
    entry["split_manifest_sha256"] = sha256_file(split_path)

    if existing_indexes:
        datasets[existing_indexes[0]] = entry
    else:
        datasets.append(entry)
    registry["datasets"] = datasets
    _atomic_write_json(Path(registry_path), registry)
    return {
        "ok": True,
        "registry_path": str(registry_path),
        "dataset_manifest_path": str(manifest_path),
        "split_manifest_path": str(split_path),
        "entry": entry,
    }
