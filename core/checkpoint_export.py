"""Export a SAM3 trainer checkpoint into a self-contained inference checkpoint.

Root cause this module fixes (F-P1-2, docs/FABLE5_FULL_INDEPENDENT_PROJECT_AUDIT.md):
`sam3/model_builder.py::_load_checkpoint()` (in /home/book/sam301, NOT modified by
this module) filters loaded state dict keys with `if "detector" in k`. That filter
is correct for the BASE checkpoint (/home/book/sam301/sam3.pt), whose top-level dict
IS a flat state dict already prefixed with "detector." (e.g.
"detector.backbone.vision_backbone.trunk.pos_embed"). But a TRAINER checkpoint saved
by sam3/train/trainer.py::save_checkpoint() stores the model under a "model" key,
and that inner state dict's keys have NO "detector." substring anywhere (they were
saved via `unwrap_ddp_if_wrapped(self.model).state_dict()`, where trainer.model is a
`Sam3Image` instance built by the SAME `build_sam3_image_model` target used for
inference — trainer.model IS NOT wrapped in a "detector" attribute). So the
"detector" filter matches zero keys for a trainer checkpoint, load_state_dict(strict=False)
loads nothing, and the model silently keeps its random/default init.

Empirically verified (2026-07-03) against the real run
runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt:
- fresh `build_sam3_image_model(checkpoint_path=None).state_dict()`: 1134 tensors.
- trainer checkpoint's `ckpt["model"]`: 1134 tensors, IDENTICAL keyset, 0 shape
  mismatches. `model.load_state_dict(ckpt["model"], strict=True)` succeeds directly.
- base sam3.pt, after stripping its "detector." prefix, is a SUPERSET (1156 keys,
  22 extra tracker/video-only buffers not part of a plain image-detector model).

So the correct key mapping for trainer -> inference is the IDENTITY function (no
prefix add/strip) — this is derived from the real structures above, not guessed.
build_key_mapping() below still runs a small closed set of candidate prefixes and
picks the one with the best, unambiguous match, so the mapping stays correct and
traceable if the trainer's own model wrapping ever changes.

This module never imports or modifies anything under /home/book/sam301 except by
reading files the caller names explicitly, and it never calls sam301's own
_load_checkpoint / model_builder loading path — it builds a fresh model with
checkpoint_path=None and loads state dicts itself with strict=True.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from core.config import BOOK_ROOT, DEFAULT_SAM3_CHECKPOINT, SAM301_ROOT

CHECKPOINT_TYPE_BASE = "base"
CHECKPOINT_TYPE_TRAINER = "trainer"
CHECKPOINT_TYPE_INFERENCE = "inference"
CHECKPOINT_TYPE_MISSING = "missing"
CHECKPOINT_TYPE_UNKNOWN = "unknown"

INFERENCE_FORMAT = "sam3_inference"
INFERENCE_FORMAT_VERSION = 1
KEY_MAPPING_VERSION = 1

# No prefix manipulation is currently needed (identity mapping is exact — see module
# docstring), but the candidate list is kept so a future architecture change that
# reintroduces a wrapper prefix is handled explicitly rather than silently failing.
CANDIDATE_SOURCE_PREFIXES: tuple[str, ...] = ("", "module.", "detector.")

MIN_COVERAGE_RATIO = 0.98

# Buffers that are legitimately allowed to be absent from a strict load (currently
# empty: the real architecture round-trips with zero exceptions, verified above).
# Keep this mechanism explicit rather than ever falling back to strict=False.
ALLOWED_MISSING_KEYS: frozenset[str] = frozenset()
ALLOWED_UNEXPECTED_KEYS: frozenset[str] = frozenset()


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(BOOK_ROOT), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _raw_load(path: str | Path) -> Any:
    """Load a checkpoint's raw pickled object.

    weights_only=False is required: trainer checkpoints legitimately contain
    non-tensor Python objects (optimizer state dicts, ints, floats, dicts). This is
    only ever called on a checkpoint path the caller names explicitly (a local CLI
    --input argument or a book01-managed run directory) — never on untrusted or
    remote input. Document this trust boundary at every call site that is reachable
    from user-facing code.
    """
    return torch.load(str(path), map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class CheckpointIdentity:
    type: str
    path: str
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def identify_checkpoint(path: str | Path) -> CheckpointIdentity:
    """Classify a checkpoint by its real structure — never by filename or extension."""
    path = Path(path)
    if not path.is_file():
        return CheckpointIdentity(type=CHECKPOINT_TYPE_MISSING, path=str(path))
    try:
        obj = _raw_load(path)
    except Exception as exc:
        return CheckpointIdentity(
            type=CHECKPOINT_TYPE_UNKNOWN, path=str(path), error=f"{type(exc).__name__}: {exc}"
        )
    if not isinstance(obj, dict):
        return CheckpointIdentity(
            type=CHECKPOINT_TYPE_UNKNOWN, path=str(path), error=f"top-level object is {type(obj).__name__}, not dict"
        )
    if obj.get("format") == INFERENCE_FORMAT:
        return CheckpointIdentity(
            type=CHECKPOINT_TYPE_INFERENCE,
            path=str(path),
            detail={"format_version": obj.get("format_version"), "n_tensors": len(obj.get("model", {}))},
        )
    if "model" in obj and isinstance(obj["model"], dict) and "optimizer" in obj:
        return CheckpointIdentity(
            type=CHECKPOINT_TYPE_TRAINER,
            path=str(path),
            detail={"n_tensors": len(obj["model"]), "epoch": obj.get("epoch")},
        )
    tensor_items = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    if tensor_items and len(tensor_items) == len(obj) and any(
        k.startswith("detector.") or k.startswith("tracker.") for k in tensor_items
    ):
        return CheckpointIdentity(type=CHECKPOINT_TYPE_BASE, path=str(path), detail={"n_tensors": len(tensor_items)})
    return CheckpointIdentity(type=CHECKPOINT_TYPE_UNKNOWN, path=str(path))


@dataclass(frozen=True)
class KeyMappingResult:
    mapping: dict[str, str]
    chosen_prefix: str
    matched_tensors: int
    matched_parameters: int
    missing: list[str]
    unexpected: list[str]
    shape_mismatch: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]]
    dtype_mismatch: list[tuple[str, str, str, str]]
    critical_modules: dict[str, dict[str, int]]
    coverage_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_prefix": self.chosen_prefix,
            "matched_tensors": self.matched_tensors,
            "matched_parameters": self.matched_parameters,
            "missing_keys": self.missing,
            "unexpected_keys": self.unexpected,
            "shape_mismatch": [
                {"source": s, "target": t, "source_shape": list(ss), "target_shape": list(ts)}
                for s, t, ss, ts in self.shape_mismatch
            ],
            "dtype_mismatch": [
                {"source": s, "target": t, "source_dtype": sd, "target_dtype": td}
                for s, t, sd, td in self.dtype_mismatch
            ],
            "critical_modules": self.critical_modules,
            "coverage_ratio": self.coverage_ratio,
        }


def build_key_mapping(source_sd: dict[str, Any], target_sd: dict[str, Any]) -> KeyMappingResult:
    """Derive an explicit, traceable key mapping from the REAL source/target keys.

    Tries each candidate prefix, keeps whichever yields the most matches; refuses to
    guess if two different prefixes tie with different resulting mappings. Never
    allows two source keys to map to the same target key.
    """
    candidates: list[tuple[str, dict[str, str]]] = []
    for prefix in CANDIDATE_SOURCE_PREFIXES:
        candidate_map: dict[str, str] = {}
        for k in source_sd:
            if prefix and not k.startswith(prefix):
                continue
            target_key = k[len(prefix):] if prefix else k
            if target_key in target_sd:
                candidate_map[k] = target_key
        candidates.append((prefix, candidate_map))
    candidates.sort(key=lambda item: -len(item[1]))

    best_prefix, best_map = candidates[0]
    if len(best_map) == 0:
        raise ValueError(
            "matched 0 tensors for every candidate prefix "
            f"{CANDIDATE_SOURCE_PREFIXES}; refusing to export an inference checkpoint "
            "with zero loadable weights"
        )
    if len(candidates) > 1:
        second_prefix, second_map = candidates[1]
        if len(second_map) == len(best_map) and second_map != best_map:
            raise ValueError(
                f"ambiguous key mapping: prefixes {best_prefix!r} and {second_prefix!r} "
                f"both match {len(best_map)} keys with DIFFERENT resulting mappings; "
                "refusing to guess which one is correct"
            )

    seen_targets: dict[str, str] = {}
    for src, tgt in best_map.items():
        if tgt in seen_targets:
            raise ValueError(
                f"key mapping conflict: both {seen_targets[tgt]!r} and {src!r} "
                f"map to the same target key {tgt!r}"
            )
        seen_targets[tgt] = src

    missing = sorted(set(target_sd) - set(best_map.values()))
    unexpected = sorted(set(source_sd) - set(best_map.keys()))

    shape_mismatch: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []
    dtype_mismatch: list[tuple[str, str, str, str]] = []
    matched_parameters = 0
    final_mapping: dict[str, str] = {}
    for src, tgt in best_map.items():
        s_t, t_t = source_sd[src], target_sd[tgt]
        if not (torch.is_tensor(s_t) and torch.is_tensor(t_t)):
            continue
        if tuple(s_t.shape) != tuple(t_t.shape):
            shape_mismatch.append((src, tgt, tuple(s_t.shape), tuple(t_t.shape)))
            continue
        if s_t.dtype != t_t.dtype:
            dtype_mismatch.append((src, tgt, str(s_t.dtype), str(t_t.dtype)))
        final_mapping[src] = tgt
        matched_parameters += s_t.numel()

    module_totals: dict[str, list[int]] = {}
    for tgt in target_sd:
        top = tgt.split(".")[0]
        module_totals.setdefault(top, [0, 0])
        module_totals[top][1] += 1
    for tgt in final_mapping.values():
        top = tgt.split(".")[0]
        module_totals[top][0] += 1
    critical_modules = {k: {"matched": v[0], "total": v[1]} for k, v in module_totals.items()}

    coverage_ratio = len(final_mapping) / len(target_sd) if target_sd else 0.0

    return KeyMappingResult(
        mapping=final_mapping,
        chosen_prefix=best_prefix,
        matched_tensors=len(final_mapping),
        matched_parameters=matched_parameters,
        missing=missing,
        unexpected=unexpected,
        shape_mismatch=shape_mismatch,
        dtype_mismatch=dtype_mismatch,
        critical_modules=critical_modules,
        coverage_ratio=coverage_ratio,
    )


def _fresh_model(device: str = "cpu", sam301_root: Path = SAM301_ROOT):
    if str(sam301_root) not in sys.path:
        sys.path.insert(0, str(sam301_root))
    from sam3.model_builder import build_sam3_image_model

    return build_sam3_image_model(checkpoint_path=None, load_from_HF=False, device=device, eval_mode=True)


def _base_checkpoint_target_state_dict(base_checkpoint_path: str | Path) -> dict[str, Any]:
    base_raw = _raw_load(base_checkpoint_path)
    return {k[len("detector."):]: v for k, v in base_raw.items() if k.startswith("detector.")}


@dataclass(frozen=True)
class ExportResult:
    output_path: str
    output_sha256: str
    metadata: dict[str, Any]
    mapping_result: KeyMappingResult


def export_inference_checkpoint(
    trainer_checkpoint_path: str | Path,
    output_path: str | Path,
    base_checkpoint_path: str | Path | None = DEFAULT_SAM3_CHECKPOINT,
    overwrite: bool = False,
    dataset_identity: dict[str, Any] | None = None,
    sam301_root: Path = SAM301_ROOT,
) -> ExportResult:
    trainer_path = Path(trainer_checkpoint_path).expanduser().resolve(strict=False)
    identity = identify_checkpoint(trainer_path)
    if identity.type != CHECKPOINT_TYPE_TRAINER:
        raise ValueError(
            f"expected a TRAINER checkpoint at {trainer_path}, but identified type={identity.type!r} "
            f"(error={identity.error!r}); this exporter only accepts trainer checkpoints "
            "(containing 'model' and 'optimizer' keys)"
        )

    output_path = Path(output_path).expanduser().resolve(strict=False)
    if output_path == trainer_path:
        raise ValueError("refusing to export: output path is the same as the source trainer checkpoint")
    if base_checkpoint_path is not None:
        base_resolved = Path(base_checkpoint_path).expanduser().resolve(strict=False)
        if output_path == base_resolved:
            raise ValueError(f"refusing to overwrite the base checkpoint: {output_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing file (pass overwrite=True to replace it): {output_path}"
        )

    trainer_ckpt = _raw_load(trainer_path)
    source_sd = trainer_ckpt["model"]
    target_sd = _fresh_model(device="cpu", sam301_root=sam301_root).state_dict()

    mapping_result = build_key_mapping(source_sd, target_sd)

    if mapping_result.matched_tensors == 0:
        raise ValueError("matched 0 tensors — refusing to export an inference checkpoint with zero loaded weights")
    effective_missing = [k for k in mapping_result.missing if k not in ALLOWED_MISSING_KEYS]
    effective_unexpected = [k for k in mapping_result.unexpected if k not in ALLOWED_UNEXPECTED_KEYS]
    if mapping_result.coverage_ratio < MIN_COVERAGE_RATIO:
        raise ValueError(
            f"coverage ratio {mapping_result.coverage_ratio:.4f} is below the minimum "
            f"{MIN_COVERAGE_RATIO}; missing={effective_missing[:10]}"
        )
    if mapping_result.shape_mismatch:
        raise ValueError(f"{len(mapping_result.shape_mismatch)} tensor shape mismatches: {mapping_result.shape_mismatch[:3]}")
    for module, stats in mapping_result.critical_modules.items():
        if stats["matched"] == 0:
            raise ValueError(f"critical module {module!r} has zero matched tensors: {stats}")

    diff_summary: dict[str, Any] = {"changed_tensors": None, "identical_tensors": None, "changed_examples": []}
    if base_checkpoint_path is not None:
        base_identity = identify_checkpoint(base_checkpoint_path)
        if base_identity.type == CHECKPOINT_TYPE_BASE:
            base_target_sd = _base_checkpoint_target_state_dict(base_checkpoint_path)
            common = [tgt for tgt in mapping_result.mapping.values() if tgt in base_target_sd]
            inverse = {tgt: src for src, tgt in mapping_result.mapping.items()}
            changed: list[str] = []
            identical: list[str] = []
            for tgt in common:
                a = source_sd[inverse[tgt]]
                b = base_target_sd[tgt]
                if a.shape == b.shape and torch.equal(a, b):
                    identical.append(tgt)
                else:
                    changed.append(tgt)
            diff_summary["changed_tensors"] = len(changed)
            diff_summary["identical_tensors"] = len(identical)
            diff_summary["changed_examples"] = sorted(changed)[:10]
            if len(changed) == 0:
                raise ValueError(
                    "exported weights are BIT-IDENTICAL to the base checkpoint across all "
                    f"{len(common)} common tensors — refusing to export a checkpoint that does "
                    "not represent a fine-tuned model"
                )

    mapped_sd = {mapping_result.mapping[src]: source_sd[src] for src in mapping_result.mapping}

    metadata = {
        "source_checkpoint": str(trainer_path),
        "source_checkpoint_sha256": sha256_of_file(trainer_path),
        "base_checkpoint_sha256": sha256_of_file(base_checkpoint_path) if base_checkpoint_path else None,
        "book01_commit": _git_commit(),
        "dataset_identity": dataset_identity,
        "epoch": trainer_ckpt.get("epoch"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "key_mapping_version": KEY_MAPPING_VERSION,
        "key_mapping": mapping_result.to_dict(),
        "base_diff": diff_summary,
    }

    output_obj = {
        "format": INFERENCE_FORMAT,
        "format_version": INFERENCE_FORMAT_VERSION,
        "model": mapped_sd,
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(output_obj, tmp_path)
    os.replace(tmp_path, output_path)

    # Self-verification: the file we just wrote must strict-load into a fresh model.
    load_inference_checkpoint(output_path, device="cpu", sam301_root=sam301_root)
    output_sha256 = sha256_of_file(output_path)
    try:
        from core.sam_model_registry import export_sidecar_metadata

        export_sidecar_metadata(
            output_path,
            metadata,
            output_sha256,
            smoke_test={"strict_load": "ok", "adapter_load": "not_run_in_core_export"},
        )
    except Exception:
        try:
            output_path.unlink(missing_ok=True)
        finally:
            raise

    return ExportResult(
        output_path=str(output_path),
        output_sha256=output_sha256,
        metadata=metadata,
        mapping_result=mapping_result,
    )


EXPORT_HINT = "Run: conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py --input {path} --output <inference_model.pt>"


def load_inference_checkpoint(
    path: str | Path,
    device: str = "cpu",
    sam301_root: Path = SAM301_ROOT,
):
    """Fail-loud loader: builds a FRESH model and does a real strict(-ish) load.

    Rejects trainer/base/unknown checkpoints with a clear, actionable message instead
    of silently loading zero weights. Never reuses a model object from a training
    process — this always constructs its own model instance.
    """
    identity = identify_checkpoint(path)
    if identity.type == CHECKPOINT_TYPE_TRAINER:
        raise ValueError(
            f"{path} is a TRAINER checkpoint (has 'model' + 'optimizer' keys), not an "
            f"inference checkpoint. {EXPORT_HINT.format(path=path)}"
        )
    if identity.type == CHECKPOINT_TYPE_BASE:
        raise ValueError(
            f"{path} is a BASE checkpoint (flat 'detector.*' state dict); load it via the "
            "existing build_sam3_image_model(checkpoint_path=...) path, not this inference-"
            "checkpoint loader."
        )
    if identity.type == CHECKPOINT_TYPE_MISSING:
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    if identity.type != CHECKPOINT_TYPE_INFERENCE:
        raise ValueError(f"{path}: unrecognized checkpoint type ({identity.type!r}, error={identity.error!r})")

    obj = _raw_load(path)
    if obj.get("format_version") != INFERENCE_FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported inference checkpoint format_version={obj.get('format_version')!r}, "
            f"expected {INFERENCE_FORMAT_VERSION}"
        )
    model = _fresh_model(device=device, sam301_root=sam301_root)
    state_dict = obj["model"]
    # strict=True raises RuntimeError on ANY missing/unexpected key (verified: the
    # real architecture round-trips with zero exceptions, so no allowlist is needed
    # today). ALLOWED_MISSING_KEYS/ALLOWED_UNEXPECTED_KEYS exist only so a future,
    # legitimately-non-persistent buffer can be documented explicitly rather than the
    # loader silently switching to strict=False.
    if ALLOWED_MISSING_KEYS or ALLOWED_UNEXPECTED_KEYS:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        effective_missing = [k for k in missing if k not in ALLOWED_MISSING_KEYS]
        effective_unexpected = [k for k in unexpected if k not in ALLOWED_UNEXPECTED_KEYS]
        if effective_missing or effective_unexpected:
            raise RuntimeError(
                f"load of inference checkpoint {path} left non-allowlisted "
                f"missing={effective_missing} unexpected={effective_unexpected}"
            )
    else:
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"strict load of inference checkpoint {path} into a fresh model failed: {exc}"
            ) from exc
    if missing or unexpected:
        # PyTorch's strict=True normally raises before returning; this branch only
        # matters if a future torch version returns instead of raising.
        effective_missing = [k for k in missing if k not in ALLOWED_MISSING_KEYS]
        effective_unexpected = [k for k in unexpected if k not in ALLOWED_UNEXPECTED_KEYS]
        if effective_missing or effective_unexpected:
            raise RuntimeError(
                f"strict load reported missing={effective_missing} unexpected={effective_unexpected}"
            )
    loaded_params = sum(t.numel() for t in state_dict.values() if torch.is_tensor(t))
    if loaded_params == 0:
        raise RuntimeError("strict load succeeded but zero parameters were present — refusing silent zero-load")
    return model, obj["metadata"]


def load_trainer_checkpoint_model(
    path: str | Path,
    device: str = "cpu",
    sam301_root: Path = SAM301_ROOT,
):
    """Load a TRAINER checkpoint's model weights into a FRESH inference model, strictly.

    For checkpoint evaluation only: unlike the inference entry point (which requires
    an exported inference checkpoint so downstream consumers never touch resume
    state), evaluation legitimately iterates over raw trainer checkpoints in a run
    directory. The identity key mapping is verified empirically (see module
    docstring): the trainer's ckpt["model"] keyset equals a fresh
    build_sam3_image_model state_dict exactly, so strict=True either fully succeeds
    or raises — no silent partial load is possible.

    Returns (model, info) where info records epoch and tensor/parameter counts.
    """
    identity = identify_checkpoint(path)
    if identity.type != CHECKPOINT_TYPE_TRAINER:
        raise ValueError(
            f"load_trainer_checkpoint_model expects a TRAINER checkpoint, got type={identity.type!r} "
            f"for {path} (error={identity.error!r})"
        )
    obj = _raw_load(path)
    state_dict = obj["model"]
    model = _fresh_model(device=device, sam301_root=sam301_root)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"strict load of trainer checkpoint {path} into a fresh model failed "
            f"(architecture drift between trainer and inference model?): {exc}"
        ) from exc
    loaded_params = sum(t.numel() for t in state_dict.values() if torch.is_tensor(t))
    if loaded_params == 0:
        raise RuntimeError("trainer checkpoint contained zero parameters — refusing silent zero-load")
    info = {
        "epoch": obj.get("epoch"),
        "n_tensors": len(state_dict),
        "n_parameters": loaded_params,
    }
    return model, info
