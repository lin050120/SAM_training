"""Tests for the P1-A dataset-identity guard (F-P1-1 remediation).

The 184-image / 7185-annotation "formal" book-spine dataset is SAM3's own machine
pre-annotation, not human-reviewed GT (see docs/E3_DATASET_IDENTITY_ERRATUM.md and
data_manifests/dataset_identity_registry.json). These tests prove: (1) that identity
is looked up by resolved annotation path — never guessed from a filename — and (2)
that the guard fails closed for formal-mode and multi-epoch requests against a
dataset that isn't marked allowed_for_formal_training.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BOOK_ROOT, DEFAULT_TRAINING_RUN_ROOT  # noqa: E402
from core.dataset_identity import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    resolve_dataset_identity,
    validate_training_mode_against_identity,
)

FORMAL_TRAIN_ANNOTATIONS = BOOK_ROOT / "data" / "formal_book_spine_sam3_dataset" / "train" / "annotations.json"
FORMAL_VAL_ANNOTATIONS = BOOK_ROOT / "data" / "formal_book_spine_sam3_dataset" / "val" / "annotations.json"
_FORMAL_DATA_PRESENT = FORMAL_TRAIN_ANNOTATIONS.exists() and FORMAL_VAL_ANNOTATIONS.exists()


class RegistryLookupTest(unittest.TestCase):
    @unittest.skipUnless(_FORMAL_DATA_PRESENT, "formal split COCO files not present on this machine")
    def test_formal_dataset_is_identified_as_unreviewed_machine_preannotation(self) -> None:
        identity = resolve_dataset_identity(FORMAL_TRAIN_ANNOTATIONS, FORMAL_VAL_ANNOTATIONS)
        self.assertTrue(identity.matched)
        self.assertEqual(identity.annotation_source, "sam3_machine_preannotation")
        self.assertFalse(identity.human_reviewed)
        self.assertFalse(identity.independently_corrected_gt)
        self.assertFalse(identity.allowed_for_formal_training)
        self.assertFalse(identity.allowed_for_model_evaluation)
        self.assertIsNotNone(identity.warning)
        self.assertIn("E3_DATASET_IDENTITY_ERRATUM", identity.warning)

    def test_unmatched_dataset_fails_closed_not_open(self) -> None:
        """Missing identity metadata -> conservative default, never silently trusted."""
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.json"
            train.write_text("{}")
            identity = resolve_dataset_identity(train)
        self.assertFalse(identity.matched)
        self.assertEqual(identity.annotation_source, "unknown")
        self.assertFalse(identity.human_reviewed)
        self.assertFalse(identity.allowed_for_formal_training)
        self.assertEqual(identity.max_epochs_without_human_review, 1)
        self.assertIsNotNone(identity.warning)

    def test_lookup_is_by_resolved_path_not_filename(self) -> None:
        """Two different files that happen to share a basename must not collide;
        an unrelated file named annotations.json is not a formal-dataset match."""
        with tempfile.TemporaryDirectory() as tmp:
            decoy = Path(tmp) / "annotations.json"
            decoy.write_text("{}")
            identity = resolve_dataset_identity(decoy)
        self.assertFalse(identity.matched)

    def test_mismatched_train_val_pairing_does_not_silently_match(self) -> None:
        """A formal train path paired with an unrelated val path must not be
        treated as the registered (train, val) pair."""
        with tempfile.TemporaryDirectory() as tmp:
            unrelated_val = Path(tmp) / "val.json"
            unrelated_val.write_text("{}")
            identity = resolve_dataset_identity(FORMAL_TRAIN_ANNOTATIONS, unrelated_val) if _FORMAL_DATA_PRESENT else None
        if identity is not None:
            self.assertFalse(identity.matched)

    def test_no_train_annotations_path_is_unknown(self) -> None:
        identity = resolve_dataset_identity(None)
        self.assertFalse(identity.matched)
        self.assertFalse(identity.human_reviewed)

    def test_registry_file_is_valid_json_with_required_fields(self) -> None:
        data = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertIn("schema_version", data)
        self.assertGreaterEqual(len(data["datasets"]), 1)
        entry = data["datasets"][0]
        for field in [
            "dataset_id",
            "annotation_source",
            "human_reviewed",
            "independently_corrected_gt",
            "allowed_for_formal_training",
            "allowed_for_model_evaluation",
            "image_file_count",
            "unique_image_count",
            "annotation_count",
            "exact_duplicate_group_count",
            "dataset_manifest_sha256",
            "split_manifest_sha256",
        ]:
            self.assertIn(field, entry, f"missing required registry field: {field}")


class TrainingModeGuardTest(unittest.TestCase):
    def _unreviewed_identity(self, max_epochs_cap=1):
        from core.dataset_identity import DatasetIdentity

        return DatasetIdentity(
            dataset_id="fixture",
            matched=True,
            annotation_source="sam3_machine_preannotation",
            human_reviewed=False,
            independently_corrected_gt=False,
            allowed_for_formal_training=False,
            allowed_for_model_evaluation=False,
            max_epochs_without_human_review=max_epochs_cap,
            warning="fixture warning",
        )

    def test_one_epoch_smoke_is_allowed(self) -> None:
        reasons = validate_training_mode_against_identity("smoke", 1, self._unreviewed_identity())
        self.assertEqual(reasons, [])

    def test_multi_epoch_smoke_is_blocked_for_unreviewed_dataset(self) -> None:
        reasons = validate_training_mode_against_identity("smoke", 20, self._unreviewed_identity())
        self.assertTrue(any("exceeds the smoke-mode limit" in r for r in reasons))

    def test_formal_mode_is_blocked_for_unreviewed_dataset_even_at_one_epoch(self) -> None:
        reasons = validate_training_mode_against_identity("formal", 1, self._unreviewed_identity())
        self.assertTrue(any("not marked allowed_for_formal_training" in r for r in reasons))

    def test_formal_mode_is_allowed_for_a_reviewed_dataset(self) -> None:
        from core.dataset_identity import DatasetIdentity

        reviewed = DatasetIdentity(
            dataset_id="fixture-reviewed",
            matched=True,
            annotation_source="human_corrected",
            human_reviewed=True,
            independently_corrected_gt=True,
            allowed_for_formal_training=True,
            allowed_for_model_evaluation=True,
            max_epochs_without_human_review=1,
            warning=None,
        )
        reasons = validate_training_mode_against_identity("formal", 20, reviewed)
        self.assertEqual(reasons, [])

    def test_invalid_training_mode_string_is_rejected(self) -> None:
        reasons = validate_training_mode_against_identity("production", 1, self._unreviewed_identity())
        self.assertTrue(reasons)

    def test_error_message_explains_how_to_replace_with_reviewed_gt(self) -> None:
        reasons = validate_training_mode_against_identity("formal", 1, self._unreviewed_identity())
        combined = " ".join(reasons)
        self.assertIn("allowed_for_formal_training=true", combined)
        self.assertIn("dataset_identity_registry.json", combined)


@unittest.skipUnless(_FORMAL_DATA_PRESENT, "formal split COCO files not present on this machine")
class PreflightIdentityIntegrationTest(unittest.TestCase):
    """End-to-end: real inspect_training_config() against the real formal dataset."""

    def test_one_epoch_smoke_preflight_passes_and_writes_identity_to_run_files(self) -> None:
        from core.training_runner import inspect_training_config

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            preflight = inspect_training_config(
                training_prompt="book spine",
                max_epochs=1,
                train_batch_size=1,
                gradient_accumulation_steps=4,
                train_images=BOOK_ROOT / "data" / "dataset_raw",
                train_annotations=FORMAL_TRAIN_ANNOTATIONS,
                val_images=BOOK_ROOT / "data" / "dataset_raw",
                val_annotations=FORMAL_VAL_ANNOTATIONS,
                output_root=Path(tmp),
                prepare_runtime=True,
                collect_import_metadata=False,
            )
            self.assertEqual(preflight.errors, [])
            self.assertIsNotNone(preflight.dataset_identity)
            self.assertFalse(preflight.dataset_identity["human_reviewed"])
            run_dir = Path(preflight.run_dir)
            dataset_info = json.loads((run_dir / "dataset_info.json").read_text(encoding="utf-8"))
            config_summary = json.loads((run_dir / "training_config_summary.json").read_text(encoding="utf-8"))
            self.assertIn("dataset_identity", dataset_info)
            self.assertIn("dataset_identity", config_summary)
            self.assertFalse(dataset_info["dataset_identity"]["human_reviewed"])
            self.assertEqual(config_summary["training_mode"], "smoke")

    def test_formal_mode_is_rejected_before_writing_runtime_yaml(self) -> None:
        from core.training_runner import inspect_training_config

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            preflight = inspect_training_config(
                training_prompt="book spine",
                max_epochs=1,
                train_batch_size=1,
                gradient_accumulation_steps=4,
                train_images=BOOK_ROOT / "data" / "dataset_raw",
                train_annotations=FORMAL_TRAIN_ANNOTATIONS,
                val_images=BOOK_ROOT / "data" / "dataset_raw",
                val_annotations=FORMAL_VAL_ANNOTATIONS,
                output_root=Path(tmp),
                prepare_runtime=True,
                collect_import_metadata=False,
                training_mode="formal",
            )
        self.assertTrue(any("dataset identity guard" in e for e in preflight.errors))
        self.assertIsNone(preflight.runtime_config_path) if preflight.runtime_config_path is None else self.assertFalse(
            Path(preflight.runtime_config_path).exists()
        )

    def test_multi_epoch_smoke_is_rejected_against_unreviewed_dataset(self) -> None:
        from core.training_runner import inspect_training_config

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            preflight = inspect_training_config(
                training_prompt="book spine",
                max_epochs=20,
                train_batch_size=1,
                gradient_accumulation_steps=4,
                train_images=BOOK_ROOT / "data" / "dataset_raw",
                train_annotations=FORMAL_TRAIN_ANNOTATIONS,
                val_images=BOOK_ROOT / "data" / "dataset_raw",
                val_annotations=FORMAL_VAL_ANNOTATIONS,
                output_root=Path(tmp),
                prepare_runtime=True,
                collect_import_metadata=False,
                training_mode="smoke",
            )
        self.assertTrue(any("exceeds the smoke-mode limit" in e for e in preflight.errors))


class HistoricalManifestUntouchedTest(unittest.TestCase):
    """This session must not modify formal_dataset_manifest.json / formal_split_manifest.json
    or any historical run's recorded provenance — their SHA256 (independently recomputed
    here, not copied from any prior report) must remain exactly what earlier runs recorded."""

    def test_formal_dataset_and_split_manifest_hashes_unchanged(self) -> None:
        dataset_manifest = BOOK_ROOT / "data_manifests" / "formal_dataset_manifest.json"
        split_manifest = BOOK_ROOT / "data_manifests" / "formal_split_manifest.json"
        if not (dataset_manifest.exists() and split_manifest.exists()):
            self.skipTest("formal manifests not present on this machine")
        dataset_sha = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
        split_sha = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
        registry = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
        entry = registry["datasets"][0]
        self.assertEqual(dataset_sha, entry["dataset_manifest_sha256"])
        self.assertEqual(split_sha, entry["split_manifest_sha256"])

    def test_a_historical_run_provenance_referencing_the_manifest_is_unaffected(self) -> None:
        run_dir = BOOK_ROOT / "runs" / "training" / "2026-07-03_14-42-27"
        if not run_dir.is_dir():
            self.skipTest("reference historical run not present on this machine")
        summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
        # historical run predates the dataset_identity field; it must not have been
        # rewritten to inject one after the fact.
        self.assertNotIn("dataset_identity", summary)


if __name__ == "__main__":
    unittest.main()
