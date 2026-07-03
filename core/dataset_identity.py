"""Dataset identity lookup: is a given train/val annotation file human-reviewed GT?

All 7185 annotations under data/formal_book_spine_sam3_dataset (and the six source
CVAT exports it derives from) are SAM3's own machine pre-annotation output — every
source COCO's info.description reads "book_spine SAM3 pre-annotation (polygon ~8pts,
NMS)" and 100% of polygons have <=8 vertices, which is inconsistent with manual
correction. This module lets preflight/UI/provenance ask "is this dataset allowed for
formal training?" without ever guessing from a file name — only registry entries
whose recorded annotation path matches (by resolved path) the dataset actually being
used are trusted; everything else is treated as NOT reviewed (fail-safe default).

See docs/E3_DATASET_IDENTITY_ERRATUM.md and data_manifests/dataset_identity_registry.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import BOOK_ROOT

DEFAULT_REGISTRY_PATH = BOOK_ROOT / "data_manifests" / "dataset_identity_registry.json"

UNKNOWN_DATASET_WARNING = (
    "no dataset identity record matched these annotation paths; treating as "
    "NOT human-reviewed (fail-safe default) — smoke mode only, max_epochs=1"
)


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str | None
    matched: bool
    annotation_source: str
    human_reviewed: bool
    independently_corrected_gt: bool
    allowed_for_formal_training: bool
    allowed_for_model_evaluation: bool
    max_epochs_without_human_review: int
    warning: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "matched": self.matched,
            "annotation_source": self.annotation_source,
            "human_reviewed": self.human_reviewed,
            "independently_corrected_gt": self.independently_corrected_gt,
            "allowed_for_formal_training": self.allowed_for_formal_training,
            "allowed_for_model_evaluation": self.allowed_for_model_evaluation,
            "max_epochs_without_human_review": self.max_epochs_without_human_review,
            "warning": self.warning,
        }


def _unknown_identity(warning: str = UNKNOWN_DATASET_WARNING) -> DatasetIdentity:
    return DatasetIdentity(
        dataset_id=None,
        matched=False,
        annotation_source="unknown",
        human_reviewed=False,
        independently_corrected_gt=False,
        allowed_for_formal_training=False,
        allowed_for_model_evaluation=False,
        max_epochs_without_human_review=1,
        warning=warning,
    )


def load_registry(registry_path: Path | None = None) -> dict[str, Any]:
    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    if not path.is_file():
        return {"schema_version": None, "datasets": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    return (BOOK_ROOT / path_str if not Path(path_str).is_absolute() else Path(path_str)).expanduser().resolve(strict=False)


def resolve_dataset_identity(
    train_annotations: str | Path | None,
    val_annotations: str | Path | None = None,
    registry_path: Path | None = None,
) -> DatasetIdentity:
    """Match by resolved annotation path against the registry — never by filename guessing."""
    if not train_annotations:
        return _unknown_identity("no train_annotations path was provided")
    train_resolved = Path(train_annotations).expanduser().resolve(strict=False)
    val_resolved = Path(val_annotations).expanduser().resolve(strict=False) if val_annotations else None

    registry = load_registry(registry_path)
    for entry in registry.get("datasets", []):
        splits = entry.get("splits", {})
        train_entry = splits.get("train", {})
        registry_train = _resolved(train_entry.get("annotations_path"))
        if registry_train is None or registry_train != train_resolved:
            continue
        if val_resolved is not None:
            val_entry = splits.get("val", {})
            registry_val = _resolved(val_entry.get("annotations_path"))
            if registry_val is not None and registry_val != val_resolved:
                # train path matched a registry entry but val does not — do not
                # silently trust a mixed-provenance pairing.
                continue
        return DatasetIdentity(
            dataset_id=entry.get("dataset_id"),
            matched=True,
            annotation_source=entry.get("annotation_source", "unknown"),
            human_reviewed=bool(entry.get("human_reviewed", False)),
            independently_corrected_gt=bool(entry.get("independently_corrected_gt", False)),
            allowed_for_formal_training=bool(entry.get("allowed_for_formal_training", False)),
            allowed_for_model_evaluation=bool(entry.get("allowed_for_model_evaluation", False)),
            max_epochs_without_human_review=int(entry.get("max_epochs_without_human_review", 1)),
            warning=(
                None
                if entry.get("human_reviewed")
                else (
                    "dataset is registered as SAM3 machine pre-annotation "
                    f"({entry.get('dataset_id')}): human_reviewed=false, "
                    "allowed_for_formal_training=false. This run is a pipeline smoke "
                    "test only and its metrics are NOT evidence of fine-tuning quality. "
                    "See docs/E3_DATASET_IDENTITY_ERRATUM.md."
                )
            ),
        )
    return _unknown_identity()


TRAINING_MODE_SMOKE = "smoke"
TRAINING_MODE_FORMAL = "formal"
VALID_TRAINING_MODES = (TRAINING_MODE_SMOKE, TRAINING_MODE_FORMAL)


def validate_training_mode_against_identity(
    training_mode: str,
    max_epochs: int | None,
    identity: DatasetIdentity,
) -> list[str]:
    """Fail-closed rules tying training_mode/max_epochs to dataset identity.

    Returns a list of blocking reasons (empty == allowed to proceed).
    """
    reasons: list[str] = []
    if training_mode not in VALID_TRAINING_MODES:
        reasons.append(f"training_mode must be one of {VALID_TRAINING_MODES}, got {training_mode!r}")
        return reasons

    if training_mode == TRAINING_MODE_FORMAL and not identity.allowed_for_formal_training:
        reasons.append(
            "formal training mode requested, but this dataset is not marked "
            "allowed_for_formal_training=true "
            f"(annotation_source={identity.annotation_source!r}, human_reviewed={identity.human_reviewed}). "
            "Replace with a human-reviewed / independently corrected GT dataset registered in "
            "data_manifests/dataset_identity_registry.json with allowed_for_formal_training=true, "
            "or run in --training-mode smoke with max_epochs<=1 for pipeline testing only."
        )

    if not identity.allowed_for_formal_training:
        limit = identity.max_epochs_without_human_review
        if max_epochs is not None and max_epochs > limit:
            reasons.append(
                f"max_epochs={max_epochs} exceeds the smoke-mode limit ({limit}) for a dataset that is "
                f"not human-reviewed (annotation_source={identity.annotation_source!r}). Multi-epoch formal "
                "training requires a dataset registered with allowed_for_formal_training=true. See "
                "docs/E3_DATASET_IDENTITY_ERRATUM.md for how to promote a dataset to formal status."
            )
    return reasons
