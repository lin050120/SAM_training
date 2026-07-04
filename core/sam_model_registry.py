from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.checkpoint_export import (
    CHECKPOINT_TYPE_BASE,
    CHECKPOINT_TYPE_INFERENCE,
    CHECKPOINT_TYPE_TRAINER,
    CHECKPOINT_TYPE_UNKNOWN,
    identify_checkpoint,
    sha256_of_file,
)
from core.config import BOOK_ROOT, DEFAULT_SAM3_CHECKPOINT, DEFAULT_TRAINING_RUN_ROOT

CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")
VALID_MODEL_SUFFIXES = {".pt", ".pth"}


@dataclass(frozen=True)
class CheckpointListItem:
    name: str
    path: str
    epoch: int | None
    size_bytes: int
    sha256: str | None
    is_alias: bool
    alias_of: str | None
    checkpoint_type: str
    can_export: bool
    load_status: str
    existing_inference_model: str | None
    suggested_output_name: str


def _epoch_from_name(path: Path, identity_epoch: Any = None) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    if match:
        return int(match.group(1))
    if isinstance(identity_epoch, int):
        return identity_epoch
    return None


def default_inference_name(checkpoint_name: str) -> str:
    match = CHECKPOINT_RE.match(checkpoint_name)
    if match:
        return f"inference_checkpoint_{int(match.group(1))}.pt"
    if checkpoint_name == "checkpoint.pt":
        return "inference_checkpoint_latest.pt"
    stem = Path(checkpoint_name).stem
    return f"inference_{stem}.pt"


def scan_trainer_checkpoints(run_dir: str | Path) -> list[CheckpointListItem]:
    run_path = Path(run_dir).expanduser().resolve(strict=False)
    ckpt_dir = run_path / "checkpoints"
    if not run_path.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_path}")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {ckpt_dir}")

    candidates = sorted(ckpt_dir.glob("checkpoint*.pt"))
    hashes: dict[str, list[Path]] = {}
    rows: list[CheckpointListItem] = []
    for path in candidates:
        try:
            digest = sha256_of_file(path)
            hashes.setdefault(digest, []).append(path)
        except OSError:
            digest = None

    for path in candidates:
        digest = sha256_of_file(path) if path.is_file() else None
        identity = identify_checkpoint(path)
        epoch = _epoch_from_name(path, identity.detail.get("epoch"))
        alias_of = None
        if path.name == "checkpoint.pt" and digest and len(hashes.get(digest, [])) > 1:
            numbered = [
                p for p in hashes[digest]
                if p.name != "checkpoint.pt" and CHECKPOINT_RE.match(p.name)
            ]
            numbered.sort(key=lambda p: int(CHECKPOINT_RE.match(p.name).group(1)))  # type: ignore[union-attr]
            alias_of = numbered[-1].name if numbered else None
        suggested_name = default_inference_name(path.name)
        existing = ckpt_dir / suggested_name
        can_export = identity.type == CHECKPOINT_TYPE_TRAINER
        rows.append(
            CheckpointListItem(
                name=path.name,
                path=str(path),
                epoch=epoch,
                size_bytes=path.stat().st_size,
                sha256=digest,
                is_alias=alias_of is not None or path.name == "checkpoint.pt",
                alias_of=alias_of,
                checkpoint_type=identity.type,
                can_export=can_export,
                load_status="ok" if identity.error is None and can_export else (identity.error or f"type={identity.type}"),
                existing_inference_model=str(existing) if existing.exists() else None,
                suggested_output_name=suggested_name,
            )
        )

    def sort_key(item: CheckpointListItem) -> tuple[int, int, int, str]:
        if item.epoch is None:
            return (1, 10**9, 1 if item.is_alias else 0, item.name)
        return (0, item.epoch, 1 if item.is_alias else 0, item.name)

    return sorted(rows, key=sort_key)


def resolve_model_path(source: str, discovered_path: str | None = None, model_dir: str | None = None, filename: str | None = None, absolute_path: str | None = None) -> Path:
    if source == "default":
        return DEFAULT_SAM3_CHECKPOINT.resolve(strict=False)
    if source == "discovered":
        if not discovered_path:
            raise ValueError("no discovered model selected")
        return Path(discovered_path).expanduser().resolve(strict=False)
    if source == "manual_absolute":
        if not absolute_path:
            raise ValueError("absolute model path is empty")
        path = Path(absolute_path).expanduser()
        if not path.is_absolute():
            raise ValueError("model path must be absolute")
        return path.resolve(strict=False)
    if source == "manual_parts":
        if not model_dir or not filename:
            raise ValueError("model directory and filename are required")
        directory = Path(model_dir).expanduser()
        if not directory.is_absolute():
            raise ValueError("model directory must be absolute")
        if Path(filename).is_absolute() or Path(filename).name != filename:
            raise ValueError("model filename must be a plain filename, not a path")
        return (directory / filename).resolve(strict=False)
    raise ValueError(f"unsupported model source: {source}")


def _metadata_sidecar(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def read_inference_metadata(path: Path) -> dict[str, Any] | None:
    sidecar = _metadata_sidecar(path)
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    identity = identify_checkpoint(path)
    if identity.type == CHECKPOINT_TYPE_INFERENCE:
        try:
            import torch

            obj = torch.load(str(path), map_location="cpu", weights_only=False)
            return obj.get("metadata") if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def model_provenance(path: str | Path, *, validate_load: bool = False, device: str = "cpu", compute_sha: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=False)
    info: dict[str, Any] = {
        "sam_model_path": str(resolved),
        "sam_model_filename": resolved.name,
        "exists": resolved.exists(),
        "is_file": resolved.is_file(),
        "readable": os.access(resolved, os.R_OK) if resolved.exists() else False,
        "metadata_path": str(_metadata_sidecar(resolved)),
        "metadata_exists": _metadata_sidecar(resolved).is_file(),
        "validation_status": "unknown",
        "validation_error": None,
        "load_timestamp": None,
    }
    if not resolved.exists():
        info.update({"validation_status": "error", "validation_error": "model path does not exist"})
        return info
    if not resolved.is_file():
        info.update({"validation_status": "error", "validation_error": "model path is not a regular file"})
        return info
    if resolved.suffix.lower() not in VALID_MODEL_SUFFIXES:
        info.update({"validation_status": "error", "validation_error": f"unsupported extension: {resolved.suffix}"})
        return info
    if not os.access(resolved, os.R_OK):
        info.update({"validation_status": "error", "validation_error": "model file is not readable"})
        return info
    try:
        info["size_bytes"] = resolved.stat().st_size
        info["sam_model_sha256"] = sha256_of_file(resolved) if compute_sha else None
    except OSError as exc:
        info.update({"validation_status": "error", "validation_error": f"cannot stat/hash model: {exc}"})
        return info

    identity = identify_checkpoint(resolved)
    metadata = read_inference_metadata(resolved) or {}
    source_checkpoint = metadata.get("source_trainer_checkpoint") or metadata.get("source_checkpoint")
    source_epoch = metadata.get("source_epoch", metadata.get("epoch"))
    source_run_id = metadata.get("source_run_id")
    if not source_run_id and source_checkpoint:
        parts = Path(source_checkpoint).parts
        if "training" in parts:
            idx = parts.index("training")
            if idx + 1 < len(parts):
                source_run_id = parts[idx + 1]
    info.update(
        {
            "sam_model_type": identity.type,
            "identity_detail": identity.detail,
            "identity_error": identity.error,
            "is_baseline": identity.type == CHECKPOINT_TYPE_BASE,
            "is_inference_model": identity.type in {CHECKPOINT_TYPE_BASE, CHECKPOINT_TYPE_INFERENCE},
            "source_trainer_checkpoint": source_checkpoint,
            "source_epoch": source_epoch,
            "source_run_id": source_run_id,
            "inference_metadata": metadata,
        }
    )
    if identity.type == CHECKPOINT_TYPE_TRAINER:
        info.update(
            {
                "validation_status": "error",
                "validation_error": "当前文件是 trainer checkpoint，不能直接用于普通推理。请先在 Checkpoint 导出页面将其导出为 inference model。",
            }
        )
        return info
    if identity.type not in {CHECKPOINT_TYPE_BASE, CHECKPOINT_TYPE_INFERENCE}:
        info.update(
            {
                "validation_status": "error",
                "validation_error": f"unrecognized or incompatible model type: {identity.type} ({identity.error})",
            }
        )
        return info
    if validate_load:
        try:
            from core.sam3_adapter import Sam3Adapter

            adapter = Sam3Adapter(checkpoint=resolved, device=device)
            info["adapter_cache_hit"] = getattr(adapter, "cache_hit", False)
            info["load_timestamp"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            info.update({"validation_status": "error", "validation_error": f"Sam3Adapter load failed: {type(exc).__name__}: {exc}"})
            return info
    info["validation_status"] = "ok"
    return info


def discover_inference_models() -> list[dict[str, Any]]:
    paths: list[Path] = [DEFAULT_SAM3_CHECKPOINT]
    if DEFAULT_TRAINING_RUN_ROOT.is_dir():
        for pattern in ("*/checkpoints/inference_*.pt", "*/checkpoints/inference_best.pt"):
            paths.extend(DEFAULT_TRAINING_RUN_ROOT.glob(pattern))
    # Keep discovery dynamic; no registry is a single source of truth.
    unique = sorted({p.expanduser().resolve(strict=False) for p in paths}, key=lambda p: str(p))
    models: list[dict[str, Any]] = []
    for path in unique:
        metadata = None
        sidecar = _metadata_sidecar(path)
        if sidecar.is_file():
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = None
        source_checkpoint = metadata.get("source_trainer_checkpoint") or metadata.get("source_checkpoint") if metadata else None
        source_epoch = metadata.get("source_epoch", metadata.get("epoch")) if metadata else None
        source_run_id = metadata.get("source_run_id") if metadata else None
        if not source_run_id and source_checkpoint:
            parts = Path(source_checkpoint).parts
            if "training" in parts:
                idx = parts.index("training")
                if idx + 1 < len(parts):
                    source_run_id = parts[idx + 1]
        display = path.name
        if source_run_id or source_epoch is not None:
            display = f"{path.name} | run={source_run_id or '-'} epoch={source_epoch or '-'}"
        elif path == DEFAULT_SAM3_CHECKPOINT:
            display = f"{path.name} | default baseline"
        models.append(
            {
                "display_name": display,
                "model_path": str(path),
                "sam_model_path": str(path),
                "sam_model_filename": path.name,
                "available": path.is_file(),
                "metadata_exists": _metadata_sidecar(path).is_file(),
                "source_trainer_checkpoint": source_checkpoint,
                "source_epoch": source_epoch,
                "source_run_id": source_run_id,
                "validation_status": "not_checked",
            }
        )
    return models


def export_sidecar_metadata(output_path: Path, metadata: dict[str, Any], output_sha256: str, smoke_test: dict[str, Any] | None = None) -> Path:
    sidecar = _metadata_sidecar(output_path)
    payload = dict(metadata)
    source_checkpoint = payload.get("source_checkpoint")
    source_run_id = None
    if source_checkpoint:
        parts = Path(source_checkpoint).parts
        if "training" in parts:
            idx = parts.index("training")
            if idx + 1 < len(parts):
                source_run_id = parts[idx + 1]
    payload.update(
        {
            "source_trainer_checkpoint": source_checkpoint,
            "source_sha256": payload.get("source_checkpoint_sha256"),
            "source_epoch": payload.get("epoch"),
            "source_run_id": source_run_id,
            "source_checkpoint_type": CHECKPOINT_TYPE_TRAINER,
            "output_path": str(output_path),
            "output_sha256": output_sha256,
            "export_timestamp": payload.get("exported_at"),
            "exporter_version": payload.get("key_mapping_version"),
            "sam3_source_hash": payload.get("sam3_code_version"),
            "base_model_path": str(DEFAULT_SAM3_CHECKPOINT),
            "base_model_sha256": payload.get("base_checkpoint_sha256"),
            "missing_keys": payload.get("key_mapping", {}).get("missing_keys"),
            "unexpected_keys": payload.get("key_mapping", {}).get("unexpected_keys"),
            "coverage_ratio": payload.get("key_mapping", {}).get("coverage_ratio"),
            "smoke_test_result": smoke_test or {"load": "ok"},
        }
    )
    tmp = sidecar.with_name(f".{sidecar.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, sidecar)
    return sidecar
