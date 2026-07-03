"""Manage the SAM301 trainer grad-accum loss-scaling patch: status / verify / apply / revert.

/home/book/sam301 is not a git tree; this script plus the git-tracked manifest
(config/sam301_patch_manifest.json) make the patch reproducible and auditable on
any book01 checkout.

State model (full SHA256 comparison, no truncation):
  PATCHED   target hash == patched_sha256
  UNPATCHED target hash == original_sha256
  UNKNOWN   target exists but matches neither hash  -> never overwritten by this tool
  MISSING   target file does not exist

  status  print the state (exit 0 for any determinable state)
  verify  exit 0 only for PATCHED; everything else fails closed (formal-training gate)
  apply   UNPATCHED only: patch-utility dry-run, apply, then post-hash must equal
          patched_sha256 (restored from backup otherwise)
  revert  PATCHED only: reverse-apply, post-hash must equal original_sha256

Never touches /home/book/sam3 (manifest loader rejects such targets outright).
"""

from __future__ import annotations

import argparse
import os
import fcntl
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sam301_patch import (  # noqa: E402
    STATE_PATCHED,
    STATE_UNPATCHED,
    load_manifest,
    patch_status,
    resolve_patch_file,
    sha256_of_file,
    verify_patched_for_training,
)


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def _check_patch_file(manifest: dict, manifest_path: Path | None) -> tuple[Path, str | None]:
    patch_file = resolve_patch_file(manifest, manifest_path)
    if not patch_file.is_file():
        return patch_file, f"patch file missing: {patch_file}"
    expected = manifest.get("patch_file_sha256")
    if expected and sha256_of_file(patch_file) != expected:
        return patch_file, f"patch file sha256 does not match manifest: {patch_file}"
    return patch_file, None


def _run_patch_tool(target: Path, patch_file: Path, reverse: bool, dry_run: bool) -> subprocess.CompletedProcess:
    command = ["patch", "--silent", "--forward" if not reverse else "--reverse"]
    if dry_run:
        command.append("--dry-run")
    command += [str(target), str(patch_file)]
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)


def _fsync_parent(path: Path) -> None:
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_mode(src: Path, dst: Path) -> None:
    stat = src.stat()
    os.chmod(dst, stat.st_mode & 0o7777)


def _apply_patch_atomically(target: Path, patch_file: Path, reverse: bool, expected_after: str) -> tuple[bool, str | None]:
    """Patch a same-directory temp file, validate it, then atomically replace target."""
    lock_path = target.with_name(f".{target.name}.patch.lock")
    tmp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            shutil.copy2(target, tmp_path)
            _copy_mode(target, tmp_path)
            real = _run_patch_tool(tmp_path, patch_file, reverse=reverse, dry_run=False)
            if real.returncode != 0:
                return False, f"patch command failed: {real.stdout.strip()}"
            actual_after = sha256_of_file(tmp_path) if tmp_path.is_file() else None
            if actual_after != expected_after:
                return False, f"post-patch hash {actual_after} != expected {expected_after}"
            with tmp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
            _fsync_parent(target)
            return True, None
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def cmd_status(args) -> int:
    status = patch_status(args.manifest)
    _emit(status.to_dict(), args.json)
    return 0


def cmd_verify(args) -> int:
    error = verify_patched_for_training(args.manifest)
    status_dict = None
    try:
        status_dict = patch_status(args.manifest).to_dict()
    except Exception:
        pass
    payload = {"ok": error is None, **(status_dict or {})}
    if error:
        payload["error"] = error
    _emit(payload, args.json)
    return 0 if error is None else 1


def _apply_or_revert(args, revert: bool) -> int:
    action = "revert" if revert else "apply"
    manifest = load_manifest(args.manifest)
    status = patch_status(args.manifest)
    patch_file, patch_error = _check_patch_file(manifest, args.manifest)
    if patch_error:
        _emit({"ok": False, "action": action, "error": patch_error}, args.json)
        return 1

    required_state = STATE_PATCHED if revert else STATE_UNPATCHED
    if status.state != required_state:
        hint = ""
        if status.state == "UNKNOWN":
            hint = " — refusing to touch a file with an unknown hash; inspect it manually"
        elif status.state == STATE_PATCHED and not revert:
            hint = " — already PATCHED, refusing to re-apply"
        elif status.state == STATE_UNPATCHED and revert:
            hint = " — already at the original hash, nothing to revert"
        _emit(
            {
                "ok": False,
                "action": action,
                "error": f"{action} requires state {required_state}, current state is {status.state}{hint}",
                **status.to_dict(),
            },
            args.json,
        )
        return 1

    target = Path(status.target_file)
    expected_after = status.original_sha256 if revert else status.patched_sha256

    dry = _run_patch_tool(target, patch_file, reverse=revert, dry_run=True)
    if dry.returncode != 0:
        _emit({"ok": False, "action": action, "error": f"dry-run failed: {dry.stdout.strip()}"}, args.json)
        return 1

    ok, error = _apply_patch_atomically(target, patch_file, reverse=revert, expected_after=expected_after)
    if not ok:
        _emit({"ok": False, "action": action, "error": error}, args.json)
        return 1

    _emit(
        {"ok": True, "action": action, "state": patch_status(args.manifest).state, "sha256": expected_after},
        args.json,
    )
    return 0


def cmd_apply(args) -> int:
    return _apply_or_revert(args, revert=False)


def cmd_revert(args) -> int:
    return _apply_or_revert(args, revert=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="manifest path override (tests)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    sub.add_parser("apply").set_defaults(func=cmd_apply)
    sub.add_parser("revert").set_defaults(func=cmd_revert)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, getattr(args, "json", False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
