"""Validation dataset loading + provenance guard for checkpoint evaluation.

Guard policy (fail-closed):
- The validation annotations path must match a registry entry BY RESOLVED PATH
  (core.dataset_identity.resolve_validation_identity — never filename guessing).
- The matched entry must be human_reviewed=true. A machine pre-annotation val set
  would make checkpoint selection self-referential and is refused.
- The val path must differ from the run's train annotations path — never silently
  degrade to training data.
- allowed_for_model_evaluation=false is NOT a blocker for checkpoint SELECTION on a
  val split (that is what a val split is for), but it is surfaced as a warning: the
  ranking must not be quoted as final model quality until a held-out reviewed test
  set exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.dataset_identity import resolve_validation_identity


@dataclass(frozen=True)
class ValidationGuardResult:
    ok: bool
    reason: str | None
    warnings: list[str]
    identity: dict[str, Any] | None


def check_validation_guard(
    val_annotations: Path,
    train_annotations: Path | None,
    registry_path: Path | None = None,
) -> ValidationGuardResult:
    warnings: list[str] = []
    val_resolved = Path(val_annotations).expanduser().resolve(strict=False)
    if not val_resolved.is_file():
        return ValidationGuardResult(
            ok=False,
            reason=f"validation annotations file does not exist: {val_resolved}",
            warnings=[],
            identity=None,
        )
    if train_annotations is not None:
        train_resolved = Path(train_annotations).expanduser().resolve(strict=False)
        if train_resolved == val_resolved:
            return ValidationGuardResult(
                ok=False,
                reason=(
                    "validation annotations path equals the training annotations path — "
                    "selecting checkpoints on training data is forbidden"
                ),
                warnings=[],
                identity=None,
            )
    identity = resolve_validation_identity(val_resolved, registry_path=registry_path)
    if not identity.matched:
        return ValidationGuardResult(
            ok=False,
            reason=(
                f"validation dataset is not registered: {identity.warning} "
                f"(path: {val_resolved})"
            ),
            warnings=[],
            identity=identity.to_dict(),
        )
    if not identity.human_reviewed:
        return ValidationGuardResult(
            ok=False,
            reason=(
                f"validation dataset {identity.dataset_id!r} is not human-reviewed "
                f"({identity.annotation_source}); refusing to select a best checkpoint "
                "against machine pre-annotation. Register a human-corrected validation "
                "split first (data_manifests/dataset_identity_registry.json)."
            ),
            warnings=[],
            identity=identity.to_dict(),
        )
    if not identity.allowed_for_model_evaluation:
        warnings.append(
            f"dataset {identity.dataset_id!r} has allowed_for_model_evaluation=false "
            "(not approved as a final blind model-evaluation dataset): this ranking is "
            "valid for checkpoint SELECTION on the validation split, but must not be "
            "quoted as final model quality."
        )
    return ValidationGuardResult(ok=True, reason=None, warnings=warnings, identity=identity.to_dict())


@dataclass
class ImageGroundTruth:
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int
    masks: list[np.ndarray] = field(default_factory=list)
    annotation_ids: list[int] = field(default_factory=list)


def _polygons_to_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Decode one COCO annotation's segmentation (polygon list or RLE) to a bool mask."""
    from pycocotools import mask as mask_utils

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        raise ValueError(f"unsupported segmentation type: {type(segmentation)}")
    return mask_utils.decode(rle).astype(bool)


def load_validation_ground_truth(
    val_annotations: Path,
    val_images_dir: Path,
    max_images: int | None = None,
) -> list[ImageGroundTruth]:
    """Load the fixed validation set: image paths + decoded GT instance masks.

    Raises on: missing annotation file, missing image files, image/annotation
    mismatch (annotation referencing an unknown image id). Images with zero GT
    annotations are kept (they still contribute false positives).
    """
    data = json.loads(Path(val_annotations).read_text(encoding="utf-8"))
    images = sorted(data.get("images", []), key=lambda im: int(im["id"]))
    if max_images is not None:
        images = images[:max_images]
    by_id: dict[int, ImageGroundTruth] = {}
    for im in images:
        path = Path(val_images_dir) / im["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"validation image missing on disk: {path}")
        by_id[int(im["id"])] = ImageGroundTruth(
            image_id=int(im["id"]),
            file_name=im["file_name"],
            path=path,
            width=int(im["width"]),
            height=int(im["height"]),
        )
    kept_ids = set(by_id)
    all_ids = {int(im["id"]) for im in data.get("images", [])}
    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in kept_ids:
            if image_id in all_ids:
                continue  # image excluded by max_images truncation, not an error
            raise ValueError(
                f"annotation {ann.get('id')} references unknown image_id {image_id} "
                f"in {val_annotations}"
            )
        gt = by_id[image_id]
        gt.masks.append(_polygons_to_mask(ann["segmentation"], gt.height, gt.width))
        gt.annotation_ids.append(int(ann.get("id", -1)))
    return [by_id[i] for i in sorted(by_id)]
