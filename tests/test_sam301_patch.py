"""Tests for the SAM301 patch integrity guard (manifest, manage script, launch gates).

All apply/revert tests run against throwaway temp trees — the real
/home/book/sam301 tree is only ever hashed read-only here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DEFAULT_TRAINING_RUN_ROOT  # noqa: E402
from core.sam301_patch import collect_training_provenance, patch_status, sha256_of_file, verify_patched_for_training  # noqa: E402

ORIG_CONTENT = "line one\nline two\nline three\n"
PATCHED_CONTENT = "line one\nline two (patched)\nline three\n"


def _load_manage_module():
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "manage_sam301_patch.py"
    spec = importlib.util.spec_from_file_location("manage_sam301_patch_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_patchset(tmp: Path) -> tuple[Path, Path]:
    """Build a temp target+patch+manifest; returns (manifest_path, target_path)."""
    target = tmp / "trainer.py"
    target.write_text(ORIG_CONTENT, encoding="utf-8")
    patched_copy = tmp / "trainer_patched.py"
    patched_copy.write_text(PATCHED_CONTENT, encoding="utf-8")
    patch_file = tmp / "fix.patch"
    diff = subprocess.run(
        ["diff", "-u", str(target), str(patched_copy)],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert diff.returncode == 1, "expected differences"
    patch_file.write_text(diff.stdout, encoding="utf-8")

    import hashlib

    manifest = {
        "patch_id": "test-patch",
        "version": 1,
        "purpose": "test",
        "target_file": str(target),
        "patch_file": str(patch_file),
        "patch_file_sha256": sha256_of_file(patch_file),
        "original_sha256": hashlib.sha256(ORIG_CONTENT.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(PATCHED_CONTENT.encode()).hexdigest(),
        "expected_sam3_root": str(tmp),
        "expected_patch_root": str(tmp),
    }
    manifest_path = tmp / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, target


class PatchStatusStatesTest(unittest.TestCase):
    def test_original_hash_is_unpatched_and_verify_fails(self) -> None:
        manage = _load_manage_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _target = _make_patchset(Path(tmp))
            self.assertEqual(patch_status(manifest).state, "UNPATCHED")
            self.assertIsNotNone(verify_patched_for_training(manifest))
            self.assertNotEqual(manage.main(["--manifest", str(manifest), "verify"]), 0)
            self.assertEqual(manage.main(["--manifest", str(manifest), "status"]), 0)

    def test_patched_hash_is_patched_and_verify_succeeds(self) -> None:
        manage = _load_manage_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, target = _make_patchset(Path(tmp))
            target.write_text(PATCHED_CONTENT, encoding="utf-8")
            self.assertEqual(patch_status(manifest).state, "PATCHED")
            self.assertIsNone(verify_patched_for_training(manifest))
            self.assertEqual(manage.main(["--manifest", str(manifest), "--json", "verify"]), 0)

    def test_unknown_hash_fails_verify_apply_and_revert_safely(self) -> None:
        manage = _load_manage_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, target = _make_patchset(Path(tmp))
            target.write_text("tampered content\n", encoding="utf-8")
            self.assertEqual(patch_status(manifest).state, "UNKNOWN")
            self.assertIsNotNone(verify_patched_for_training(manifest))
            self.assertNotEqual(manage.main(["--manifest", str(manifest), "verify"]), 0)
            self.assertNotEqual(manage.main(["--manifest", str(manifest), "apply"]), 0)
            self.assertNotEqual(manage.main(["--manifest", str(manifest), "revert"]), 0)
            # the unknown file must not have been touched
            self.assertEqual(target.read_text(encoding="utf-8"), "tampered content\n")

    def test_missing_target_is_missing_and_verify_fails(self) -> None:
        manage = _load_manage_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, target = _make_patchset(Path(tmp))
            target.unlink()
            self.assertEqual(patch_status(manifest).state, "MISSING")
            self.assertIsNotNone(verify_patched_for_training(manifest))
            self.assertNotEqual(manage.main(["--manifest", str(manifest), "verify"]), 0)

    def test_manifest_targeting_old_sam3_tree_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "evil.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "patch_id": "evil",
                        "target_file": "/home/book/sam3/sam3/train/trainer.py",
                        "patch_file": "x.patch",
                        "original_sha256": "0" * 64,
                        "patched_sha256": "1" * 64,
                    }
                ),
                encoding="utf-8",
            )
            error = verify_patched_for_training(manifest_path)
            self.assertIsNotNone(error)
            self.assertIn("refusing to train", error)

    def test_manifest_target_escaping_expected_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            manifest_path = Path(tmp) / "manifest.json"
            patch_file = Path(tmp) / "x.patch"
            patch_file.write_text("x", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "patch_id": "escape",
                        "target_file": str(Path(outside) / "trainer.py"),
                        "patch_file": str(patch_file),
                        "patch_file_sha256": sha256_of_file(patch_file),
                        "original_sha256": "0" * 64,
                        "patched_sha256": "1" * 64,
                        "expected_sam3_root": str(Path(tmp) / "sam301"),
                        "expected_patch_root": str(tmp),
                    }
                ),
                encoding="utf-8",
            )
            error = verify_patched_for_training(manifest_path)
            self.assertIsNotNone(error)
            self.assertIn("manifest target", error)

    def test_manifest_target_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp) / "sam301"
            root.mkdir()
            link = root / "trainer.py"
            real_target = Path(outside) / "trainer.py"
            real_target.write_text(ORIG_CONTENT, encoding="utf-8")
            link.symlink_to(real_target)
            patch_file = Path(tmp) / "x.patch"
            patch_file.write_text("x", encoding="utf-8")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "patch_id": "symlink-escape",
                        "target_file": str(link),
                        "patch_file": str(patch_file),
                        "patch_file_sha256": sha256_of_file(patch_file),
                        "original_sha256": "0" * 64,
                        "patched_sha256": "1" * 64,
                        "expected_sam3_root": str(root),
                        "expected_patch_root": str(tmp),
                    }
                ),
                encoding="utf-8",
            )
            error = verify_patched_for_training(manifest_path)
            self.assertIsNotNone(error)
            self.assertIn("manifest target", error)

    def test_patch_file_outside_expected_patch_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            target = Path(tmp) / "trainer.py"
            target.write_text(ORIG_CONTENT, encoding="utf-8")
            patch_file = Path(outside) / "x.patch"
            patch_file.write_text("x", encoding="utf-8")
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "patch_id": "patch-outside",
                        "target_file": str(target),
                        "patch_file": str(patch_file),
                        "patch_file_sha256": sha256_of_file(patch_file),
                        "original_sha256": "0" * 64,
                        "patched_sha256": "1" * 64,
                        "expected_sam3_root": str(tmp),
                        "expected_patch_root": str(tmp),
                    }
                ),
                encoding="utf-8",
            )
            error = verify_patched_for_training(manifest_path)
            self.assertIsNotNone(error)
            self.assertIn("manifest patch file", error)

    def test_patch_file_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            target = Path(tmp) / "trainer.py"
            target.write_text(ORIG_CONTENT, encoding="utf-8")
            real_patch = Path(outside) / "x.patch"
            real_patch.write_text("x", encoding="utf-8")
            patch_link = Path(tmp) / "x.patch"
            patch_link.symlink_to(real_patch)
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "patch_id": "patch-symlink",
                        "target_file": str(target),
                        "patch_file": str(patch_link),
                        "patch_file_sha256": sha256_of_file(real_patch),
                        "original_sha256": "0" * 64,
                        "patched_sha256": "1" * 64,
                        "expected_sam3_root": str(tmp),
                        "expected_patch_root": str(tmp),
                    }
                ),
                encoding="utf-8",
            )
            error = verify_patched_for_training(manifest_path)
            self.assertIsNotNone(error)
            self.assertIn("manifest patch file", error)


class ApplyRevertRoundtripTest(unittest.TestCase):
    def test_apply_then_reapply_then_revert(self) -> None:
        manage = _load_manage_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, target = _make_patchset(Path(tmp))
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))

            # apply from UNPATCHED: succeeds, exact patched hash
            self.assertEqual(manage.main(["--manifest", str(manifest), "apply"]), 0)
            self.assertEqual(sha256_of_file(target), manifest_data["patched_sha256"])
            self.assertEqual(patch_status(manifest).state, "PATCHED")

            # second apply: refused
            self.assertNotEqual(manage.main(["--manifest", str(manifest), "apply"]), 0)
            self.assertEqual(sha256_of_file(target), manifest_data["patched_sha256"])

            # revert: back to exact original hash
            self.assertEqual(manage.main(["--manifest", str(manifest), "revert"]), 0)
            self.assertEqual(sha256_of_file(target), manifest_data["original_sha256"])
            self.assertEqual(patch_status(manifest).state, "UNPATCHED")

            # revert again: refused (already original)
            self.assertNotEqual(manage.main(["--manifest", str(manifest), "revert"]), 0)

    def test_apply_refuses_tampered_patch_file(self) -> None:
        manage = _load_manage_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, target = _make_patchset(Path(tmp))
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            Path(manifest_data["patch_file"]).write_text("corrupted", encoding="utf-8")
            self.assertNotEqual(manage.main(["--manifest", str(manifest), "apply"]), 0)
            self.assertEqual(target.read_text(encoding="utf-8"), ORIG_CONTENT)

    def test_patch_command_failure_leaves_original_file_unchanged(self) -> None:
        manage = _load_manage_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, target = _make_patchset(Path(tmp))
            before = sha256_of_file(target)

            def fail_patch(path: Path, _patch_file: Path, reverse: bool, dry_run: bool):
                if not dry_run:
                    path.write_text("partial mutation in temp\n", encoding="utf-8")
                return SimpleNamespace(returncode=1, stdout="simulated patch failure")

            with mock.patch.object(manage, "_run_patch_tool", side_effect=fail_patch):
                self.assertNotEqual(manage.main(["--manifest", str(manifest), "apply"]), 0)
            self.assertEqual(sha256_of_file(target), before)
            self.assertEqual(target.read_text(encoding="utf-8"), ORIG_CONTENT)

    def test_final_hash_mismatch_leaves_original_file_unchanged(self) -> None:
        manage = _load_manage_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest, target = _make_patchset(Path(tmp))
            before = sha256_of_file(target)

            def wrong_patch(path: Path, _patch_file: Path, reverse: bool, dry_run: bool):
                if not dry_run:
                    path.write_text("wrong final content\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="")

            with mock.patch.object(manage, "_run_patch_tool", side_effect=wrong_patch):
                self.assertNotEqual(manage.main(["--manifest", str(manifest), "apply"]), 0)
            self.assertEqual(sha256_of_file(target), before)
            self.assertEqual(target.read_text(encoding="utf-8"), ORIG_CONTENT)

    def test_collect_training_provenance_contains_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, target = _make_patchset(Path(tmp))
            target.write_text(PATCHED_CONTENT, encoding="utf-8")
            runtime = Path(tmp) / "runtime_config.yaml"
            runtime.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")
            provenance = collect_training_provenance(
                runtime_config_path=runtime,
                sam3_import_path="/home/book/sam301/sam3/__init__.py",
                python_executable="/home/book/anaconda3/envs/sam301/bin/python",
                manifest_path=manifest,
            )
            for key in [
                "book01_git_commit",
                "book01_git_dirty",
                "manifest_sha256",
                "patch_sha256",
                "trainer_sha256",
                "sam301_root",
                "sam3_import_path",
                "python_executable",
                "runtime_yaml_sha256",
            ]:
                self.assertIn(key, provenance)
            self.assertTrue(provenance["patch_guard_ok"], provenance)


_TRAINING_FIXTURES_AVAILABLE = (
    Path("/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml").exists()
    and Path("/home/book/sam301/sam3.pt").exists()
)


class LaunchGateIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real fixtures not present")
    def test_preflight_rejects_unpatched_trainer_without_writing_yaml_or_token(self) -> None:
        from core import training_runner

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            with mock.patch.object(
                training_runner,
                "verify_patched_for_training",
                return_value="trainer patch 'x' is UNPATCHED (test)",
            ):
                preflight = training_runner.inspect_training_config(
                    training_prompt="book spine",
                    max_epochs=1,
                    train_batch_size=1,
                    gradient_accumulation_steps=4,
                    output_root=Path(tmp),
                    prepare_runtime=True,
                    collect_import_metadata=False,
                )
            self.assertTrue(any("sam301 patch guard" in e for e in preflight.errors), preflight.errors)
            self.assertFalse(Path(preflight.runtime_config_path).exists())

    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real fixtures not present")
    def test_preflight_writes_machine_readable_provenance(self) -> None:
        from core import training_runner

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            preflight = training_runner.inspect_training_config(
                training_prompt="book spine",
                max_epochs=1,
                train_batch_size=1,
                gradient_accumulation_steps=4,
                output_root=Path(tmp),
                prepare_runtime=True,
                collect_import_metadata=False,
            )
            self.assertFalse(preflight.errors, preflight.errors)
            run_dir = Path(preflight.run_dir)
            provenance_path = run_dir / "provenance.json"
            self.assertTrue(provenance_path.exists())
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(provenance["trainer_patch_state"], "PATCHED")
            self.assertEqual(provenance["trainer_sha256"], provenance["expected_trainer_patched_sha256"])
            self.assertEqual(provenance["runtime_yaml_path"], str(Path(preflight.runtime_config_path)))
            self.assertIsNotNone(provenance["runtime_yaml_sha256"])
            summary = json.loads((run_dir / "training_config_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["training_provenance"]["trainer_sha256"], provenance["trainer_sha256"])

    def test_launcher_gate_blocks_without_consuming_token_or_spawning(self) -> None:
        import ui.training_preflight_page as tpp
        import ui.training_process_manager as tpm
        import uuid

        from ui.process_manager import ProcessManager

        manager = ProcessManager()
        originals = (
            tpp.training_process_manager,
            tpm.training_process_manager,
            tpp.detect_cuda,
            tpm.validate_training_run_path,
            tpp.verify_patched_for_training,
            tpp.verify_sam3_import_for_training,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_dir = base / "run"
            (run_dir / "checkpoints").mkdir(parents=True)
            runtime_yaml = run_dir / "config" / "runtime_config.yaml"
            runtime_yaml.parent.mkdir()
            runtime_yaml.write_text("x: 1")
            for name in ["ckpt.pt", "t.json", "v.json"]:
                (base / name).write_text("x")
            (base / "ti").mkdir()
            (base / "vi").mkdir()
            state = {
                "ok": True,
                "run_dir": str(run_dir),
                "runtime_yaml": str(runtime_yaml),
                "checkpoint": str(base / "ckpt.pt"),
                "command": [sys.executable, "-c", "print('never runs')"],
                "train_images": str(base / "ti"),
                "train_annotations": str(base / "t.json"),
                "val_images": str(base / "vi"),
                "val_annotations": str(base / "v.json"),
                "num_gpus": 1,
                "launch_token": uuid.uuid4().hex,
                "consumed": False,
            }
            try:
                tpp.training_process_manager = manager
                tpm.training_process_manager = manager
                tpp.detect_cuda = lambda: SimpleNamespace(available=True, device_count=1)
                tpm.validate_training_run_path = lambda _p: None
                tpp.verify_patched_for_training = lambda: "trainer patch 'x' is UNKNOWN (test)"
                tpp._consumed_preflight_tokens.clear()
                with mock.patch.object(manager, "start", wraps=manager.start) as start_mock:
                    outputs = list(tpp.start_training(state, True))
                self.assertIn("BLOCKED", outputs[0][0])
                self.assertIn("sam301 patch guard", outputs[0][0])
                # token must NOT be consumed and no subprocess must be created
                self.assertFalse(state["consumed"])
                self.assertNotIn(state["launch_token"], tpp._consumed_preflight_tokens)
                start_mock.assert_not_called()
            finally:
                (
                    tpp.training_process_manager,
                    tpm.training_process_manager,
                    tpp.detect_cuda,
                    tpm.validate_training_run_path,
                    tpp.verify_patched_for_training,
                    tpp.verify_sam3_import_for_training,
                ) = originals
                tpp._consumed_preflight_tokens.clear()

    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real fixtures not present")
    def test_real_sam301_tree_is_currently_patched(self) -> None:
        """Read-only check of the real manifest against the real trainer file."""
        status = patch_status()
        self.assertEqual(status.state, "PATCHED", status)
        self.assertIsNone(verify_patched_for_training())


if __name__ == "__main__":
    unittest.main()
