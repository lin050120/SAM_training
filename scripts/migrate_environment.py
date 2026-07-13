#!/usr/bin/env python3
"""Configure a book01 checkout after moving it and the SAM301 source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_RELATIVE = Path("config/local_paths.json")
REPORT_RELATIVE = Path("config/migration_report.json")

BOOK_REQUIRED = (
    Path("app.py"),
    Path("core/config.py"),
    Path("config/sam301_patch_manifest.json"),
    Path("scripts/manage_sam301_patch.py"),
)
SAM301_REQUIRED = (
    Path("pyproject.toml"),
    Path("sam3/__init__.py"),
    Path("sam3/train/trainer.py"),
    Path("sam3/train/configs/book_spine/book_spine_finetune.yaml"),
    Path("sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
    Path("sam3.pt"),
)


@dataclass(frozen=True)
class RootValidation:
    book_root: str
    sam301_root: str
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_user_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def validate_roots(
    book_root: str | Path,
    sam301_root: str | Path,
    expected_project_root: Path | None = None,
) -> RootValidation:
    book = resolve_user_path(book_root)
    sam = resolve_user_path(sam301_root)
    expected = resolve_user_path(expected_project_root or PROJECT_ROOT)
    errors: list[str] = []
    warnings: list[str] = []

    if book != expected:
        errors.append(f"selected book01 root {book} is not this checkout {expected}")
    if not book.is_dir():
        errors.append(f"book01 root is not a directory: {book}")
    else:
        for relative in BOOK_REQUIRED:
            if not (book / relative).is_file():
                errors.append(f"book01 required file is missing: {book / relative}")

    if not sam.is_dir():
        errors.append(f"SAM301 root is not a directory: {sam}")
    else:
        for relative in SAM301_REQUIRED:
            if not (sam / relative).is_file():
                errors.append(f"SAM301 required file is missing: {sam / relative}")

    if book == sam:
        errors.append("book01 root and SAM301 root must be different directories")
    if (book / "data").is_dir() is False:
        warnings.append(f"data directory does not exist yet: {book / 'data'}")
    if (book / "runs").is_dir() is False:
        warnings.append(f"runs directory does not exist yet: {book / 'runs'}")
    return RootValidation(str(book), str(sam), errors, warnings)


def machine_config_payload(book_root: Path, sam301_root: Path) -> dict:
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "generated_by": "scripts/migrate_environment.py",
        "book_root": str(resolve_user_path(book_root)),
        "sam301_root": str(resolve_user_path(sam301_root)),
    }


def atomic_write_json(path: Path, payload: dict, create_backup: bool = True) -> Path | None:
    path = resolve_user_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if create_backup and path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
        suffix = 1
        while backup.exists():
            backup = path.with_name(f"{path.stem}.{stamp}_{suffix}.bak{path.suffix}")
            suffix += 1
        shutil.copy2(path, backup)

    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def _clean_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def check_sam3_import(
    sam301_root: Path,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())",
    ]
    # Python prepends the cwd to sys.path, so running inside sam301_root would
    # always import sam3/ straight from disk and mask a broken editable install;
    # a neutral empty cwd makes the check reflect the installed state.
    with tempfile.TemporaryDirectory(prefix="sam3_import_check_") as neutral_cwd:
        completed = run(
            command,
            cwd=neutral_cwd,
            env=_clean_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    actual = completed.stdout.strip().splitlines()[-1] if completed.returncode == 0 and completed.stdout.strip() else None
    expected = str((sam301_root / "sam3" / "__init__.py").resolve(strict=False))
    return {
        "ok": completed.returncode == 0 and actual == expected,
        "expected": expected,
        "actual": actual,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def repair_editable_install(
    sam301_root: Path,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    command = [sys.executable, "-m", "pip", "install", "-e", str(sam301_root), "--no-deps"]
    completed = run(
        command,
        cwd=str(sam301_root),
        env=_clean_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_patch_state(book_root: Path, sam301_root: Path) -> dict:
    manifest_path = book_root / "config" / "sam301_patch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_target = Path(str(manifest["target_file"]))
    target = raw_target if raw_target.is_absolute() else sam301_root / raw_target
    target = target.resolve(strict=False)
    try:
        target.relative_to(sam301_root.resolve(strict=False))
    except ValueError:
        return {"state": "INVALID", "target_file": str(target), "error": "patch target escapes SAM301 root"}
    if not target.is_file():
        state = "MISSING"
        actual = None
    else:
        actual = sha256_file(target)
        if actual == manifest["patched_sha256"]:
            state = "PATCHED"
        elif actual == manifest["original_sha256"]:
            state = "UNPATCHED"
        else:
            state = "UNKNOWN"
    return {"state": state, "target_file": str(target), "actual_sha256": actual}


def run_patch_manager(book_root: Path, command: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(book_root / "scripts" / "manage_sam301_patch.py"), "--json", command],
        cwd=str(book_root),
        env=_clean_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"output": completed.stdout.strip()}
    return {"ok": completed.returncode == 0, "returncode": completed.returncode, **payload}


def check_cuda() -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    lines = completed.stdout.strip().splitlines()
    return {
        "ok": completed.returncode == 0 and bool(lines) and lines[0] == "True",
        "available": bool(lines) and lines[0] == "True",
        "device": lines[1] if len(lines) > 1 else None,
        "stderr": completed.stderr.strip(),
    }


def choose_directories_gui(default_book_root: Path) -> tuple[Path, Path] | None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showinfo("SAM3 migration", "Select the copied book01 project directory.", parent=root)
        book = filedialog.askdirectory(initialdir=str(default_book_root.parent), mustexist=True, parent=root)
        if not book:
            return None
        messagebox.showinfo("SAM3 migration", "Select the copied sam301 source directory.", parent=root)
        sam = filedialog.askdirectory(initialdir=str(Path(book).parent), mustexist=True, parent=root)
        if not sam:
            return None
        return resolve_user_path(book), resolve_user_path(sam)
    finally:
        root.destroy()


def confirm_gui(message: str) -> bool:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        return bool(messagebox.askyesno("SAM3 migration", message, parent=root))
    finally:
        root.destroy()


def prompt_path(label: str, default: Path) -> Path:
    value = input(f"{label} [{default}]: ").strip()
    return resolve_user_path(value or default)


def confirm_terminal(message: str) -> bool:
    return input(f"{message} [y/N]: ").strip().lower() in {"y", "yes"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-root", type=Path)
    parser.add_argument("--sam301-root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate and show the plan without writing")
    parser.add_argument("--non-interactive", action="store_true", help="never prompt for repairs")
    parser.add_argument("--repair-install", action="store_true", help="allow SAM3 editable-install repair")
    parser.add_argument("--apply-patch", action="store_true", help="allow applying the known trainer patch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    use_gui = False
    if args.book_root is not None or args.sam301_root is not None:
        if args.book_root is None or args.sam301_root is None:
            print("ERROR: --book-root and --sam301-root must be provided together", file=sys.stderr)
            return 2
        book_root = resolve_user_path(args.book_root)
        sam301_root = resolve_user_path(args.sam301_root)
    elif args.non_interactive:
        print("ERROR: non-interactive mode requires both root arguments", file=sys.stderr)
        return 2
    else:
        try:
            selected = choose_directories_gui(PROJECT_ROOT)
            if selected is None:
                print("Migration cancelled.")
                return 2
            book_root, sam301_root = selected
            use_gui = True
        except Exception as exc:
            print(f"Graphical picker unavailable ({exc}); using terminal prompts.")
            book_root = prompt_path("book01 root", PROJECT_ROOT)
            sam301_root = prompt_path("sam301 root", PROJECT_ROOT.parent / "sam301")

    validation = validate_roots(book_root, sam301_root)
    print(json.dumps(asdict(validation), ensure_ascii=False, indent=2))
    if not validation.ok:
        return 2

    payload = machine_config_payload(book_root, sam301_root)
    patch_before = inspect_patch_state(book_root, sam301_root)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "local_config": payload, "patch": patch_before}, indent=2))
        return 0

    config_path = book_root / LOCAL_CONFIG_RELATIVE
    backup = atomic_write_json(config_path, payload)
    report: dict = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "incomplete",
        "validation": asdict(validation),
        "local_config_path": str(config_path),
        "local_config_backup": str(backup) if backup else None,
        "actions": [],
    }

    import_result = check_sam3_import(sam301_root)
    report["sam3_import"] = import_result
    if not import_result["ok"]:
        allow = args.repair_install
        if not allow and not args.non_interactive:
            message = "SAM3 is imported from the wrong location. Repair the editable install now?"
            allow = confirm_gui(message) if use_gui else confirm_terminal(message)
        if allow:
            if os.environ.get("CONDA_DEFAULT_ENV") != "sam301":
                report["actions"].append({"editable_install": "blocked", "reason": "not running in sam301 env"})
            else:
                install = repair_editable_install(sam301_root)
                report["actions"].append({"editable_install": install})
                import_result = check_sam3_import(sam301_root)
                report["sam3_import_after_repair"] = import_result

    patch_result = inspect_patch_state(book_root, sam301_root)
    report["trainer_patch"] = patch_result
    if patch_result["state"] == "UNPATCHED":
        allow = args.apply_patch
        if not allow and not args.non_interactive:
            message = "The known trainer patch is not applied. Apply and verify it now?"
            allow = confirm_gui(message) if use_gui else confirm_terminal(message)
        if allow:
            applied = run_patch_manager(book_root, "apply")
            report["actions"].append({"trainer_patch_apply": applied})
            report["trainer_patch_after_apply"] = inspect_patch_state(book_root, sam301_root)

    report["cuda"] = check_cuda()
    final_import = report.get("sam3_import_after_repair", report["sam3_import"])
    final_patch = report.get("trainer_patch_after_apply", report["trainer_patch"])
    success = (
        bool(final_import.get("ok"))
        and final_patch.get("state") == "PATCHED"
        and bool(report["cuda"].get("ok"))
    )
    report["status"] = "complete" if success else "incomplete"
    report["launch_command"] = f"conda run -n sam301 python {book_root / 'app.py'}"
    atomic_write_json(book_root / REPORT_RELATIVE, report, create_backup=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 3


if __name__ == "__main__":
    raise SystemExit(main())
