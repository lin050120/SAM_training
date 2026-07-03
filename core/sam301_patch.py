"""SAM301 trainer patch integrity: manifest loading, hashing, status classification.

/home/book/sam301 is not a git tree, so the grad-accum loss-scaling patch applied to
sam3/train/trainer.py is pinned here by full SHA256 (original and patched) recorded
in a git-tracked manifest. Everything that launches real training must require
state == PATCHED and fail closed otherwise.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.config import BOOK_ROOT

DEFAULT_MANIFEST_PATH = BOOK_ROOT / "config" / "sam301_patch_manifest.json"

STATE_PATCHED = "PATCHED"
STATE_UNPATCHED = "UNPATCHED"
STATE_UNKNOWN = "UNKNOWN"
STATE_MISSING = "MISSING"

# Never allowed as a patch target, no matter what a (possibly edited) manifest says.
FORBIDDEN_TARGET_ROOT = Path("/home/book/sam3")
DEFAULT_PATCH_ROOT = BOOK_ROOT / "patches"


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_inside(path: Path, root: Path) -> bool:
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def _resolve_manifest_path(manifest_path: Path | None = None) -> Path:
    return (Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH).expanduser().resolve(strict=False)


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    path = _resolve_manifest_path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "patch_id",
        "target_file",
        "patch_file",
        "original_sha256",
        "patched_sha256",
        "expected_sam3_root",
    ):
        if not data.get(key):
            raise ValueError(f"sam301 patch manifest is missing required field: {key}")
    target = Path(data["target_file"]).expanduser().resolve(strict=False)
    expected_root = Path(data["expected_sam3_root"]).expanduser().resolve(strict=False)
    if _resolved_inside(target, FORBIDDEN_TARGET_ROOT):
        raise ValueError(f"manifest target must never point into {FORBIDDEN_TARGET_ROOT}: {target}")
    if not _resolved_inside(target, expected_root):
        raise ValueError(f"manifest target must remain under {expected_root}: {target}")
    return data


def resolve_patch_file(manifest: dict[str, Any], manifest_path: Path | None = None) -> Path:
    patch_file = Path(manifest["patch_file"])
    if not patch_file.is_absolute():
        base = _resolve_manifest_path(manifest_path).parent.parent if manifest_path else BOOK_ROOT
        patch_file = base / patch_file
    resolved = patch_file.expanduser().resolve(strict=False)
    patch_root = Path(manifest.get("expected_patch_root") or DEFAULT_PATCH_ROOT).expanduser().resolve(strict=False)
    if not _resolved_inside(resolved, patch_root):
        raise ValueError(f"manifest patch file must remain under {patch_root}: {resolved}")
    return resolved


@dataclass(frozen=True)
class PatchStatus:
    state: str
    patch_id: str
    target_file: str
    actual_sha256: str | None
    original_sha256: str
    patched_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def patch_status(manifest_path: Path | None = None) -> PatchStatus:
    manifest = load_manifest(manifest_path)
    target = Path(manifest["target_file"]).expanduser().resolve(strict=False)
    if not target.is_file():
        actual = None
        state = STATE_MISSING
    else:
        actual = sha256_of_file(target)
        if actual == manifest["patched_sha256"]:
            state = STATE_PATCHED
        elif actual == manifest["original_sha256"]:
            state = STATE_UNPATCHED
        else:
            state = STATE_UNKNOWN
    return PatchStatus(
        state=state,
        patch_id=str(manifest["patch_id"]),
        target_file=str(target),
        actual_sha256=actual,
        original_sha256=str(manifest["original_sha256"]),
        patched_sha256=str(manifest["patched_sha256"]),
    )


def _git_output(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(BOOK_ROOT), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.rstrip("\n")


def collect_training_provenance(
    runtime_config_path: Path | str | None = None,
    sam3_import_path: str | None = None,
    python_executable: str | None = None,
    manifest_path: Path | None = None,
    distributed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Machine-readable provenance for runs that depend on the SAM301 patch."""
    manifest_file = _resolve_manifest_path(manifest_path)
    manifest = load_manifest(manifest_file)
    patch_file = resolve_patch_file(manifest, manifest_file)
    target = Path(manifest["target_file"]).expanduser().resolve(strict=False)
    status = patch_status(manifest_file)
    status_error = verify_patched_for_training(manifest_file)
    git_status = _git_output(["status", "--short"])
    runtime_path = Path(runtime_config_path).expanduser().resolve(strict=False) if runtime_config_path else None
    return {
        "book01_git_commit": _git_output(["rev-parse", "HEAD"]),
        "book01_git_dirty": bool(git_status),
        "book01_git_status_short": git_status or "",
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_of_file(manifest_file) if manifest_file.is_file() else None,
        "patch_path": str(patch_file),
        "patch_sha256": sha256_of_file(patch_file) if patch_file.is_file() else None,
        "expected_patch_sha256": manifest.get("patch_file_sha256"),
        "trainer_path": str(target),
        "trainer_sha256": status.actual_sha256,
        "expected_trainer_patched_sha256": status.patched_sha256,
        "expected_trainer_original_sha256": status.original_sha256,
        "trainer_patch_state": status.state,
        "patch_guard_ok": status_error is None,
        "patch_guard_error": status_error,
        "sam301_root": str(Path(manifest["expected_sam3_root"]).expanduser().resolve(strict=False)),
        "sam3_import_path": sam3_import_path,
        "python_executable": python_executable,
        "runtime_yaml_path": str(runtime_path) if runtime_path else None,
        "runtime_yaml_sha256": sha256_of_file(runtime_path) if runtime_path and runtime_path.is_file() else None,
        "distributed": distributed,
    }


def verify_patched_for_training(manifest_path: Path | None = None) -> str | None:
    """Formal-training gate: None only when the target is exactly the patched hash.

    MISSING / UNPATCHED / UNKNOWN / unreadable manifest all fail closed with a
    human-actionable message.
    """
    try:
        manifest_file = _resolve_manifest_path(manifest_path)
        manifest = load_manifest(manifest_file)
        patch_file = resolve_patch_file(manifest, manifest_file)
        expected_patch_sha = manifest.get("patch_file_sha256")
        if not patch_file.is_file():
            return f"sam301 patch file is missing: {patch_file}; refusing to train"
        if expected_patch_sha and sha256_of_file(patch_file) != expected_patch_sha:
            return f"sam301 patch file sha256 mismatch: {patch_file}; refusing to train"
        status = patch_status(manifest_file)
    except Exception as exc:
        return (
            f"sam301 patch manifest could not be evaluated ({exc!r}); refusing to train. "
            "Check /home/book/book01/config/sam301_patch_manifest.json"
        )
    if status.state == STATE_PATCHED:
        return None
    fix_hint = {
        STATE_UNPATCHED: "apply it with: conda run -n sam301 python scripts/manage_sam301_patch.py apply",
        STATE_MISSING: "the target file does not exist; restore /home/book/sam301 first",
        STATE_UNKNOWN: (
            "the file matches neither the original nor the patched hash; do NOT force-overwrite — "
            "inspect it manually (see docs/SAM301_PATCH_MANAGEMENT.md)"
        ),
    }[status.state]
    return (
        f"trainer patch '{status.patch_id}' is {status.state} "
        f"(actual sha256={status.actual_sha256}, expected patched={status.patched_sha256}); {fix_hint}"
    )
