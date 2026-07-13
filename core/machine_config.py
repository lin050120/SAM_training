from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


LEGACY_BOOK_ROOT = Path("/home/book/book01")
LEGACY_SAM301_ROOT = Path("/home/book/sam301")
LOCAL_PATHS_FILENAME = "local_paths.json"


@dataclass(frozen=True)
class MachinePaths:
    book_root: Path
    sam301_root: Path
    source: str


def inferred_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def local_paths_config_path(project_root: Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else inferred_project_root()
    return root.resolve(strict=False) / "config" / LOCAL_PATHS_FILENAME


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path: {value}")
    return path.resolve(strict=False)


def load_machine_paths(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> MachinePaths:
    """Load per-machine roots.

    Without a local config, the original hard-coded paths are kept only when this
    checkout IS the legacy location; any other checkout must run the migration
    wizard first, so a moved installation fails loudly instead of silently reading
    and writing under /home/book.
    """
    actual_project_root = (
        Path(project_root).expanduser().resolve(strict=False)
        if project_root is not None
        else inferred_project_root().resolve(strict=False)
    )
    path = (
        Path(config_path).expanduser().resolve(strict=False)
        if config_path is not None
        else local_paths_config_path(actual_project_root)
    )
    if not path.exists():
        if actual_project_root == LEGACY_BOOK_ROOT.resolve(strict=False):
            return MachinePaths(LEGACY_BOOK_ROOT, LEGACY_SAM301_ROOT, "legacy_defaults")
        raise RuntimeError(
            f"no config/{LOCAL_PATHS_FILENAME} found for checkout {actual_project_root}; "
            "run: conda run -n sam301 python scripts/migrate_environment.py"
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid local machine-path config {path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise RuntimeError(f"unsupported local machine-path config version in {path}")

    book_root = _absolute_path(data.get("book_root"), "book_root")
    sam301_root = _absolute_path(data.get("sam301_root"), "sam301_root")
    if book_root != actual_project_root:
        raise RuntimeError(
            f"local machine-path config points to {book_root}, but this checkout is {actual_project_root}; "
            "rerun scripts/migrate_environment.py"
        )
    return MachinePaths(book_root, sam301_root, str(path))
