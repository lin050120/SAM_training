from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from core.checkpoint_export import sha256_of_file
from ui import checkpoint_evaluation_page as page


def _write_trainer(path: Path, epoch: int) -> None:
    torch.save(
        {
            "model": {"backbone.x": torch.ones(1) * epoch},
            "optimizer": {"state": {}, "param_groups": []},
            "epoch": epoch,
        },
        path,
    )


class CheckpointExportSelectionCacheTest(unittest.TestCase):
    def _run(self, root: Path) -> tuple[Path, dict]:
        run = root / "run"
        ckpt_dir = run / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        _write_trainer(ckpt_dir / "checkpoint_5.pt", 5)
        _write_trainer(ckpt_dir / "checkpoint_6.pt", 6)
        result = page.refresh_checkpoint_list(str(run))
        return run, result[-1]

    def test_selection_uses_refresh_snapshot_without_rescanning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, snapshot = self._run(Path(tmp))
            with mock.patch.object(
                page,
                "scan_trainer_checkpoints",
                side_effect=AssertionError("selection must not rescan"),
            ):
                detail, output_name = page.checkpoint_selection_detail(
                    str(run), "checkpoint_6.pt", snapshot
                )

        self.assertIn('"epoch": 6', detail)
        self.assertEqual(output_name, "inference_checkpoint_6.pt")

    def test_export_rechecks_only_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, snapshot = self._run(Path(tmp))
            selected = run / "checkpoints" / "checkpoint_6.pt"
            output = run / "checkpoints" / "inference_checkpoint_6.pt"
            fake_result = SimpleNamespace(
                output_path=str(output),
                output_sha256="output-sha",
                mapping_result=SimpleNamespace(
                    matched_tensors=1,
                    coverage_ratio=1.0,
                    missing=[],
                    unexpected=[],
                ),
            )

            def fake_export(*, trainer_checkpoint_path, output_path, overwrite):
                self.assertEqual(Path(trainer_checkpoint_path), selected)
                self.assertEqual(Path(output_path), output)
                return fake_result

            with (
                mock.patch.object(
                    page,
                    "scan_trainer_checkpoints",
                    side_effect=AssertionError("export must not rescan"),
                ),
                mock.patch.object(page, "sha256_of_file", wraps=sha256_of_file) as sha256,
                mock.patch(
                    "core.checkpoint_export.export_inference_checkpoint",
                    side_effect=fake_export,
                ) as export,
                mock.patch.object(page, "model_provenance", return_value={"validation_status": "ok"}),
            ):
                status, _info = page.export_selected_checkpoint(
                    str(run),
                    "checkpoint_6.pt",
                    str(run / "checkpoints"),
                    output.name,
                    False,
                    snapshot,
                )

        self.assertIn("导出完成", status)
        self.assertEqual(export.call_count, 1)
        self.assertEqual(sha256.call_count, 1)
        self.assertEqual(Path(sha256.call_args.args[0]), selected)

    def test_export_blocks_if_selected_checkpoint_changed_after_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, snapshot = self._run(Path(tmp))
            selected = run / "checkpoints" / "checkpoint_6.pt"
            backup = run / "checkpoints" / "replacement.pt"
            shutil.copy2(run / "checkpoints" / "checkpoint_5.pt", backup)
            selected.write_bytes(backup.read_bytes())

            with mock.patch(
                "core.checkpoint_export.export_inference_checkpoint"
            ) as export:
                status, _info = page.export_selected_checkpoint(
                    str(run),
                    "checkpoint_6.pt",
                    str(run / "checkpoints"),
                    "inference_checkpoint_6.pt",
                    False,
                    snapshot,
                )

        self.assertIn("BLOCKED", status)
        self.assertIn("SHA-256", status)
        export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
