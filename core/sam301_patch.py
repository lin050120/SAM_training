"""SAM301 trainer patch integrity: manifest loading, hashing, status classification.

/home/book/sam301 is not a git tree, so the grad-accum loss-scaling patch applied to
sam3/train/trainer.py is pinned here by full SHA256 (original and patched) recorded
in a git-tracked manifest. Everything that launches real training must require
state == PATCHED and fail closed otherwise.
"""

from __future__ import annotations

import hashlib
import json
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


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("patch_id", "target_file", "patch_file", "original_sha256", "patched_sha256"):
        if not data.get(key):
            raise ValueError(f"sam301 patch manifest is missing required field: {key}")
    target = Path(data["target_file"]).expanduser().resolve(strict=False)
    try:
        target.relative_to(FORBIDDEN_TARGET_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError(f"manifest target must never point into {FORBIDDEN_TARGET_ROOT}: {target}")
    return data


def resolve_patch_file(manifest: dict[str, Any], manifest_path: Path | None = None) -> Path:
    patch_file = Path(manifest["patch_file"])
    if not patch_file.is_absolute():
        base = Path(manifest_path).parent.parent if manifest_path else BOOK_ROOT
        patch_file = base / patch_file
    return patch_file.expanduser().resolve(strict=False)


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


def verify_patched_for_training(manifest_path: Path | None = None) -> str | None:
    """Formal-training gate: None only when the target is exactly the patched hash.

    MISSING / UNPATCHED / UNKNOWN / unreadable manifest all fail closed with a
    human-actionable message.
    """
    try:
        status = patch_status(manifest_path)
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
