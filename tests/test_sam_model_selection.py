from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from core import sam_model_registry as smr


def _write_trainer(path: Path, epoch: int) -> None:
    torch.save(
        {
            "model": {"backbone.x": torch.ones(1) * epoch},
            "optimizer": {"state": {}, "param_groups": []},
            "epoch": epoch,
        },
        path,
    )


def _write_inference(path: Path, source: Path, epoch: int) -> None:
    torch.save(
        {
            "format": "sam3_inference",
            "format_version": 1,
            "model": {"backbone.x": torch.ones(1)},
            "metadata": {"source_checkpoint": str(source), "epoch": epoch},
        },
        path,
    )


class CheckpointScanTest(unittest.TestCase):
    def test_checkpoint_list_sorts_epochs_numerically_and_marks_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "training" / "2026-07-04_14-28-27"
            ckpt_dir = run / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            _write_trainer(ckpt_dir / "checkpoint_10.pt", 10)
            _write_trainer(ckpt_dir / "checkpoint_5.pt", 5)
            shutil.copy2(ckpt_dir / "checkpoint_10.pt", ckpt_dir / "checkpoint.pt")
            _write_inference(ckpt_dir / "inference_checkpoint_5.pt", ckpt_dir / "checkpoint_5.pt", 5)

            items = smr.scan_trainer_checkpoints(run)

        self.assertEqual([item.name for item in items], ["checkpoint_5.pt", "checkpoint_10.pt", "checkpoint.pt"])
        alias = items[-1]
        self.assertTrue(alias.is_alias)
        self.assertEqual(alias.alias_of, "checkpoint_10.pt")
        self.assertEqual(items[0].existing_inference_model and Path(items[0].existing_inference_model).name, "inference_checkpoint_5.pt")

    def test_default_inference_name_for_numbered_checkpoint(self) -> None:
        self.assertEqual(smr.default_inference_name("checkpoint_35.pt"), "inference_checkpoint_35.pt")
        self.assertEqual(smr.default_inference_name("checkpoint_40.pt"), "inference_checkpoint_40.pt")


class ModelPathResolutionTest(unittest.TestCase):
    def test_manual_directory_and_filename_resolves_absolute_path(self) -> None:
        path = smr.resolve_model_path(
            "manual_parts",
            model_dir="/tmp/checkpoints",
            filename="inference_checkpoint_35.pt",
        )
        self.assertEqual(str(path), "/tmp/checkpoints/inference_checkpoint_35.pt")

    def test_manual_filename_rejects_embedded_path(self) -> None:
        with self.assertRaises(ValueError):
            smr.resolve_model_path("manual_parts", model_dir="/tmp/checkpoints", filename="../checkpoint.pt")

    def test_relative_manual_absolute_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            smr.resolve_model_path("manual_absolute", absolute_path="relative/model.pt")


class ModelValidationTest(unittest.TestCase):
    def test_trainer_checkpoint_is_rejected_for_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_35.pt"
            _write_trainer(path, 35)
            info = smr.model_provenance(path)
        self.assertEqual(info["sam_model_type"], "trainer")
        self.assertEqual(info["validation_status"], "error")
        self.assertIn("trainer checkpoint", info["validation_error"])

    def test_inference_sidecar_metadata_records_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "inference_checkpoint_35.pt"
            source = Path(tmp) / "checkpoint_35.pt"
            metadata = {
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": "abc",
                "epoch": 35,
                "key_mapping": {"missing_keys": [], "unexpected_keys": [], "coverage_ratio": 1.0},
                "exported_at": "2026-07-04T00:00:00+00:00",
            }
            out.write_bytes(b"fake")
            sidecar = smr.export_sidecar_metadata(out, metadata, "def", {"strict_load": "ok"})
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(data["source_trainer_checkpoint"], str(source))
        self.assertEqual(data["source_epoch"], 35)
        self.assertEqual(data["output_sha256"], "def")


if __name__ == "__main__":
    unittest.main()
