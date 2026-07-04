"""Checkpoint discovery and evaluation-time model loading.

Discovery rules:
- Scan <run_dir>/checkpoints/ for checkpoint*.pt.
- checkpoint_<N>.pt: epoch parsed from the name (verified against the file's own
  epoch field at load time), sorted NUMERICALLY.
- checkpoint.pt is the trainer's latest/resume alias: if its SHA256 equals any
  epoch-named sibling it is recorded as an alias and NOT evaluated twice; if it is
  unique it is evaluated as the "latest" candidate with epoch read from the file.
- SHA256 and size are recorded for every file (SHA also feeds the result cache key).
- Unreadable/corrupt checkpoints become status="failed" candidates; the rest of the
  run is still evaluated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.checkpoint_export import (
    CHECKPOINT_TYPE_BASE,
    CHECKPOINT_TYPE_INFERENCE,
    CHECKPOINT_TYPE_TRAINER,
    identify_checkpoint,
    load_inference_checkpoint,
    load_trainer_checkpoint_model,
    sha256_of_file,
)
from core.config import SAM301_ROOT

_EPOCH_NAME_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


@dataclass
class CheckpointCandidate:
    name: str
    path: Path
    epoch: int | None
    sha256: str
    size_bytes: int
    is_baseline: bool = False
    alias_of: str | None = None
    discovery_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def discover_checkpoints(run_dir: Path) -> list[CheckpointCandidate]:
    checkpoints_dir = Path(run_dir) / "checkpoints"
    if not checkpoints_dir.is_dir():
        return []
    named: list[CheckpointCandidate] = []
    latest: CheckpointCandidate | None = None
    for path in sorted(checkpoints_dir.glob("checkpoint*.pt")):
        try:
            sha = sha256_of_file(path)
            size = path.stat().st_size
        except OSError as exc:
            named.append(
                CheckpointCandidate(
                    name=path.name, path=path, epoch=None, sha256="", size_bytes=0,
                    discovery_error=f"unreadable: {exc}",
                )
            )
            continue
        match = _EPOCH_NAME_RE.match(path.name)
        if match:
            named.append(
                CheckpointCandidate(
                    name=path.name, path=path, epoch=int(match.group(1)), sha256=sha, size_bytes=size
                )
            )
        elif path.name == "checkpoint.pt":
            latest = CheckpointCandidate(
                name=path.name, path=path, epoch=None, sha256=sha, size_bytes=size
            )
        # any other checkpoint*.pt naming (e.g. inference_model exports named
        # differently) is intentionally ignored here; evaluation targets trainer
        # epoch snapshots.
    named.sort(key=lambda c: (c.epoch if c.epoch is not None else 10**9))
    if latest is not None:
        twin = next((c for c in named if c.sha256 and c.sha256 == latest.sha256), None)
        if twin is not None:
            latest.alias_of = twin.name
            latest.epoch = twin.epoch
        named.append(latest)
    return named


def load_model_for_candidate(candidate: CheckpointCandidate, device: str = "cuda"):
    """Load a candidate into a fresh model. Returns (model, info_dict).

    trainer checkpoints -> strict in-memory load (no file export needed);
    base checkpoint -> the unmodified official build path;
    inference checkpoints -> the fail-loud inference loader.
    """
    identity = identify_checkpoint(candidate.path)
    if identity.type == CHECKPOINT_TYPE_TRAINER:
        model, info = load_trainer_checkpoint_model(candidate.path, device=device)
        file_epoch = info.get("epoch")
        if candidate.epoch is not None and file_epoch is not None and int(file_epoch) != int(candidate.epoch):
            raise ValueError(
                f"{candidate.name}: filename epoch {candidate.epoch} != checkpoint's own "
                f"epoch field {file_epoch}; refusing to evaluate an inconsistently named file"
            )
        if candidate.epoch is None and file_epoch is not None:
            candidate.epoch = int(file_epoch)
        return model, {"checkpoint_type": identity.type, **info}
    if identity.type == CHECKPOINT_TYPE_BASE:
        import sys

        if str(SAM301_ROOT) not in sys.path:
            sys.path.insert(0, str(SAM301_ROOT))
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(checkpoint_path=str(candidate.path), device=device)
        return model, {"checkpoint_type": identity.type}
    if identity.type == CHECKPOINT_TYPE_INFERENCE:
        model, metadata = load_inference_checkpoint(candidate.path, device=device)
        return model, {"checkpoint_type": identity.type, "metadata": metadata}
    raise ValueError(
        f"{candidate.path}: cannot evaluate checkpoint of type {identity.type!r} "
        f"(error={identity.error!r})"
    )
