from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.machine_config import (  # noqa: E402
    LEGACY_BOOK_ROOT,
    LEGACY_SAM301_ROOT,
    load_machine_paths,
)


def load_migration_module():
    path = PROJECT_ROOT / "scripts" / "migrate_environment.py"
    module_name = "migrate_environment_for_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_fake_roots(base: Path) -> tuple[Path, Path]:
    book = base / "book 01_移行"
    sam = base / "sam 301_移行"
    for relative in (
        "app.py",
        "core/config.py",
        "scripts/manage_sam301_patch.py",
    ):
        path = book / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    trainer = sam / "sam3/train/trainer.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("patched trainer\n", encoding="utf-8")
    for relative in (
        "pyproject.toml",
        "sam3/__init__.py",
        "sam3/train/configs/book_spine/book_spine_finetune.yaml",
        "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        "sam3.pt",
    ):
        path = sam / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    digest = hashlib.sha256(trainer.read_bytes()).hexdigest()
    manifest = {
        "patch_id": "fixture",
        "version": 2,
        "target_file": "sam3/train/trainer.py",
        "patch_file": "patches/fix.patch",
        "original_sha256": "0" * 64,
        "patched_sha256": digest,
    }
    manifest_path = book / "config/sam301_patch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return book, sam


class MachineConfigTest(unittest.TestCase):
    def test_missing_local_config_retains_legacy_defaults_on_legacy_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = load_machine_paths(
                config_path=Path(tmp) / "missing.json",
                project_root=LEGACY_BOOK_ROOT,
            )
        self.assertEqual(paths.book_root, LEGACY_BOOK_ROOT)
        self.assertEqual(paths.sam301_root, LEGACY_SAM301_ROOT)
        self.assertEqual(paths.source, "legacy_defaults")

    def test_missing_local_config_on_moved_checkout_requires_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "migrate_environment"):
                load_machine_paths(config_path=root / "missing.json", project_root=root)

    def test_valid_local_config_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            sam = root.parent / "sam 301"
            config = root / "local_paths.json"
            config.write_text(
                json.dumps({"schema_version": 1, "book_root": str(root), "sam301_root": str(sam)}),
                encoding="utf-8",
            )
            paths = load_machine_paths(config_path=config, project_root=root)
        self.assertEqual(paths.book_root, root)
        self.assertEqual(paths.sam301_root, sam.resolve(strict=False))

    def test_stale_book_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = root / "local_paths.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "book_root": str(root / "old-location"),
                        "sam301_root": str(root / "sam301"),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "rerun scripts/migrate_environment.py"):
                load_machine_paths(config_path=config, project_root=root)

    def test_relative_path_and_invalid_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = root / "local_paths.json"
            config.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid local machine-path config"):
                load_machine_paths(config_path=config, project_root=root)
            config.write_text(
                json.dumps({"schema_version": 1, "book_root": str(root), "sam301_root": "relative"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "absolute path"):
                load_machine_paths(config_path=config, project_root=root)


class MigrationScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_migration_module()

    def test_validates_paths_with_spaces_and_non_ascii(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book, sam = build_fake_roots(Path(tmp))
            result = self.module.validate_roots(book, sam, expected_project_root=book)
        self.assertTrue(result.ok, result.errors)

    def test_missing_required_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book, sam = build_fake_roots(Path(tmp))
            (sam / "sam3.pt").unlink()
            result = self.module.validate_roots(book, sam, expected_project_root=book)
        self.assertFalse(result.ok)
        self.assertTrue(any("sam3.pt" in error for error in result.errors))

    def test_atomic_write_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config/local_paths.json"
            self.module.atomic_write_json(path, {"value": 1})
            backup = self.module.atomic_write_json(path, {"value": 2})
            self.assertIsNotNone(backup)
            self.assertEqual(json.loads(backup.read_text(encoding="utf-8")), {"value": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})

    def test_dry_run_does_not_write_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book, sam = build_fake_roots(Path(tmp))
            with mock.patch.object(self.module, "PROJECT_ROOT", book):
                code = self.module.main(
                    ["--book-root", str(book), "--sam301-root", str(sam), "--dry-run"]
                )
            self.assertEqual(code, 0)
            self.assertFalse((book / "config/local_paths.json").exists())
            self.assertFalse((book / "config/migration_report.json").exists())

    def test_patch_inspection_uses_selected_sam301_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book, sam = build_fake_roots(Path(tmp))
            result = self.module.inspect_patch_state(book, sam)
        self.assertEqual(result["state"], "PATCHED")
        self.assertTrue(result["target_file"].endswith("sam3/train/trainer.py"))

    def test_import_check_requires_exact_selected_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sam = Path(tmp) / "sam301"
            expected = (sam / "sam3/__init__.py").resolve(strict=False)

            def successful_run(*_args, **_kwargs):
                return SimpleNamespace(returncode=0, stdout=f"{expected}\n", stderr="")

            def wrong_run(*_args, **_kwargs):
                return SimpleNamespace(returncode=0, stdout="/old/sam3/__init__.py\n", stderr="")

            self.assertTrue(self.module.check_sam3_import(sam, run=successful_run)["ok"])
            self.assertFalse(self.module.check_sam3_import(sam, run=wrong_run)["ok"])

    def test_import_check_is_not_fooled_by_package_in_selected_root(self) -> None:
        # A sam3/ directory inside the selected root must not satisfy the check by
        # itself: the real subprocess must resolve sam3 via the installed
        # environment, not via its working directory.
        with tempfile.TemporaryDirectory() as tmp:
            sam = Path(tmp) / "sam301"
            (sam / "sam3").mkdir(parents=True)
            (sam / "sam3" / "__init__.py").write_text("", encoding="utf-8")
            result = self.module.check_sam3_import(sam)
        self.assertFalse(result["ok"])

    def test_noninteractive_success_writes_config_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book, sam = build_fake_roots(Path(tmp))
            import_ok = {"ok": True, "expected": "x", "actual": "x", "returncode": 0, "stderr": ""}
            patch_ok = {"state": "PATCHED", "target_file": str(sam / "sam3/train/trainer.py")}
            with (
                mock.patch.object(self.module, "PROJECT_ROOT", book),
                mock.patch.object(self.module, "check_sam3_import", return_value=import_ok),
                mock.patch.object(self.module, "inspect_patch_state", return_value=patch_ok),
                mock.patch.object(self.module, "check_cuda", return_value={"ok": True, "available": True}),
            ):
                code = self.module.main(
                    [
                        "--book-root",
                        str(book),
                        "--sam301-root",
                        str(sam),
                        "--non-interactive",
                    ]
                )
            self.assertEqual(code, 0)
            config = json.loads((book / "config/local_paths.json").read_text(encoding="utf-8"))
            report = json.loads((book / "config/migration_report.json").read_text(encoding="utf-8"))
            self.assertEqual(config["sam301_root"], str(sam.resolve()))
            self.assertEqual(report["status"], "complete")

    def test_noninteractive_mode_does_not_repair_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book, sam = build_fake_roots(Path(tmp))
            import_bad = {
                "ok": False,
                "expected": str(sam / "sam3/__init__.py"),
                "actual": "/old/sam3/__init__.py",
                "returncode": 0,
                "stderr": "",
            }
            patch_ok = {"state": "PATCHED", "target_file": str(sam / "sam3/train/trainer.py")}
            with (
                mock.patch.object(self.module, "PROJECT_ROOT", book),
                mock.patch.object(self.module, "check_sam3_import", return_value=import_bad),
                mock.patch.object(self.module, "inspect_patch_state", return_value=patch_ok),
                mock.patch.object(self.module, "check_cuda", return_value={"ok": True, "available": True}),
                mock.patch.object(self.module, "repair_editable_install") as repair,
            ):
                code = self.module.main(
                    [
                        "--book-root",
                        str(book),
                        "--sam301-root",
                        str(sam),
                        "--non-interactive",
                    ]
                )
            self.assertEqual(code, 3)
            repair.assert_not_called()

    def test_explicit_install_repair_is_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            book, sam = build_fake_roots(Path(tmp))
            import_bad = {"ok": False, "expected": "new", "actual": "old", "returncode": 0, "stderr": ""}
            import_ok = {"ok": True, "expected": "new", "actual": "new", "returncode": 0, "stderr": ""}
            patch_ok = {"state": "PATCHED", "target_file": str(sam / "sam3/train/trainer.py")}
            with (
                mock.patch.object(self.module, "PROJECT_ROOT", book),
                mock.patch.object(self.module, "check_sam3_import", side_effect=[import_bad, import_ok]),
                mock.patch.object(self.module, "inspect_patch_state", return_value=patch_ok),
                mock.patch.object(self.module, "check_cuda", return_value={"ok": True, "available": True}),
                mock.patch.object(self.module, "repair_editable_install", return_value={"ok": True}) as repair,
                mock.patch.dict(self.module.os.environ, {"CONDA_DEFAULT_ENV": "sam301"}),
            ):
                code = self.module.main(
                    [
                        "--book-root",
                        str(book),
                        "--sam301-root",
                        str(sam),
                        "--non-interactive",
                        "--repair-install",
                    ]
                )
            self.assertEqual(code, 0)
            repair.assert_called_once_with(sam.resolve())
