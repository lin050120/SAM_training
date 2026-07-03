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
import json
import shutil
import subprocess
import sys
import tempfile
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

    with tempfile.TemporaryDirectory() as tmp:
        backup = Path(tmp) / target.name
        shutil.copy2(target, backup)
        real = _run_patch_tool(target, patch_file, reverse=revert, dry_run=False)
        actual_after = sha256_of_file(target) if target.is_file() else None
        if real.returncode != 0 or actual_after != expected_after:
            shutil.copy2(backup, target)
            _emit(
                {
                    "ok": False,
                    "action": action,
                    "error": (
                        f"post-{action} hash {actual_after} != expected {expected_after}; "
                        "target restored from backup"
                    ),
                },
                args.json,
            )
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
