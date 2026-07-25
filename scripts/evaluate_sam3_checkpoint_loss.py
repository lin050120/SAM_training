#!/usr/bin/env python3
"""Compute SAM3 validation-style loss for trainer checkpoints on a COCO split.

Unlike the checkpoint ranking evaluator, this command reruns the model forward
path and retains the unthresholded outputs required by Sam3LossWrapper.  The
result is therefore directly comparable with ``Losses/val_book_spine_loss``.

Example:
    conda run -n sam301 python scripts/evaluate_sam3_checkpoint_loss.py \
        --run-dir runs/training/2026-07-23_13-25-53 \
        --split test
"""

from __future__ import annotations

import argparse
import atexit
import csv
import gc
import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import SAM301_ROOT  # noqa: E402

if str(SAM301_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM301_ROOT))

_NUMBERED_CHECKPOINT = re.compile(r"^checkpoint_(\d+)\.pt$")
_DIST_TEMP_DIR: tempfile.TemporaryDirectory[str] | None = None


def _init_single_process_group(device_name: str) -> None:
    """Match SAM3's DDP-dependent validation path at world size one."""
    import torch
    import torch.distributed as dist

    global _DIST_TEMP_DIR
    if dist.is_initialized():
        return
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(0)
    _DIST_TEMP_DIR = tempfile.TemporaryDirectory(prefix="sam3-loss-dist-")
    rendezvous = Path(_DIST_TEMP_DIR.name) / "rendezvous"
    dist.init_process_group(
        # SAM3's dataset queries distributed rank even for a one-process
        # evaluator. Gloo supplies that metadata without depending on NVML/NCCL.
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )

    def _cleanup() -> None:
        global _DIST_TEMP_DIR
        if dist.is_initialized():
            dist.destroy_process_group()
        if _DIST_TEMP_DIR is not None:
            _DIST_TEMP_DIR.cleanup()
            _DIST_TEMP_DIR = None

    atexit.register(_cleanup)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_checkpoints(run_dir: Path, requested: list[str] | None) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if requested:
        paths = [checkpoint_dir / name for name in requested]
    else:
        paths = [
            path
            for path in checkpoint_dir.glob("checkpoint_*.pt")
            if _NUMBERED_CHECKPOINT.match(path.name)
        ]
        paths.sort(key=lambda path: int(_NUMBERED_CHECKPOINT.match(path.name).group(1)))  # type: ignore[union-attr]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint files do not exist: {missing}")
    if not paths:
        raise FileNotFoundError(f"no numbered trainer checkpoints found in {checkpoint_dir}")
    return paths


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolved_split_paths(
    runtime_config: Any,
    split: str,
    annotations_override: Path | None,
    images_override: Path | None,
) -> tuple[Path, Path]:
    if annotations_override is not None or images_override is not None:
        if annotations_override is None or images_override is None:
            raise ValueError("--annotations and --images must be supplied together")
        return annotations_override.resolve(), images_override.resolve()
    if split == "validation":
        dataset = runtime_config.trainer.data.val.dataset
        return Path(dataset.ann_file).resolve(), Path(dataset.img_folder).resolve()
    dataset_root = Path(runtime_config.paths.dataset_root)
    return (dataset_root / "test" / "annotations.json").resolve(), (
        dataset_root / "test" / "images"
    ).resolve()


def _build_loader(runtime_config: Any, annotations: Path, images: Path, epoch: int):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    loader_config = OmegaConf.create(
        OmegaConf.to_container(runtime_config.trainer.data.val, resolve=True)
    )
    loader_config.dataset.ann_file = str(annotations)
    loader_config.dataset.img_folder = str(images)
    dataset = instantiate(loader_config)
    return dataset.get_loader(epoch=epoch)


def _evaluate_one(
    checkpoint_path: Path,
    runtime_config: Any,
    annotations: Path,
    images: Path,
    device_name: str,
) -> dict[str, Any]:
    import torch
    from hydra.utils import instantiate
    from sam3.model.utils.misc import copy_data_to_device

    device = torch.device(device_name)
    match = _NUMBERED_CHECKPOINT.match(checkpoint_path.name)
    filename_epoch = int(match.group(1)) if match else None
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"{checkpoint_path.name} is not a trainer checkpoint")
    epoch_value = checkpoint.get("epoch", filename_epoch)
    if epoch_value is None:
        raise ValueError(f"{checkpoint_path.name} has no epoch metadata")
    epoch = int(epoch_value)
    if filename_epoch is not None and epoch != filename_epoch:
        raise ValueError(
            f"{checkpoint_path.name}: filename epoch {filename_epoch} "
            f"!= checkpoint epoch {epoch}"
        )

    # Instantiate the exact training model target, including its stable internal
    # matcher. The generic checkpoint ranking loader intentionally builds the
    # inference model and is therefore not exact enough for loss reproduction.
    model = instantiate(runtime_config.trainer.model, _convert_="all")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    losses = torch.nn.ModuleDict(
        {
            key: value
            for key, value in instantiate(
                runtime_config.trainer.loss, _convert_="all"
            ).items()
        }
    )
    loss_state = checkpoint.get("loss")
    if isinstance(loss_state, dict):
        losses.load_state_dict(loss_state, strict=True)
    criterion = losses["book_spine"]
    # At world size one this is numerically equivalent to the trainer's global
    # normalization. It also avoids asking Gloo to all-reduce a CUDA tensor.
    criterion.normalization = "local"
    losses.to(device)

    training_epoch = epoch - 1
    loader = _build_loader(runtime_config, annotations, images, training_epoch)
    amp_config = runtime_config.trainer.optim.amp
    amp_enabled = bool(amp_config.enabled) and device.type == "cuda"
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(str(amp_config.amp_dtype))
    if amp_enabled and amp_dtype is None:
        raise ValueError(f"unsupported AMP dtype: {amp_config.amp_dtype}")

    if hasattr(model, "on_validation_epoch_start"):
        model.on_validation_epoch_start()

    weighted_sums: dict[str, float] = {}
    sample_count = 0
    with torch.no_grad():
        for batch_dict in loader:
            if len(batch_dict) != 1:
                raise ValueError(f"expected one dataset key per batch, got {list(batch_dict)}")
            _, datapoint = batch_dict.popitem()
            datapoint = copy_data_to_device(datapoint, device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                outputs = model(datapoint)
                targets = [model.back_convert(item) for item in datapoint.find_targets]
                loss_dict = criterion(outputs, targets)

            batch_size = len(datapoint.img_batch)
            sample_count += batch_size
            for key, value in loss_dict.items():
                scalar = float(value.detach().float().item())
                weighted_sums[key] = weighted_sums.get(key, 0.0) + scalar * batch_size

    if hasattr(model, "on_validation_epoch_end"):
        model.on_validation_epoch_end()
    if sample_count == 0:
        raise RuntimeError("split loader yielded zero samples")

    averages = {key: value / sample_count for key, value in weighted_sums.items()}
    result: dict[str, Any] = {
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "epoch": epoch,
        # The trainer saves checkpoint N after completing zero-based epoch N-1,
        # immediately before running that epoch's validation pass.
        "training_epoch": training_epoch,
        "status": "completed",
        "sample_count": sample_count,
        "loss": averages["core_loss"],
    }
    result.update({f"component/{key}": value for key, value in sorted(averages.items())})

    del loader, criterion, losses, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    from omegaconf import OmegaConf
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    try:
        register_omegaconf_resolvers()
    except ValueError:
        pass

    run_dir = args.run_dir.resolve()
    runtime_path = run_dir / "config" / "runtime_config.yaml"
    if not runtime_path.is_file():
        raise FileNotFoundError(f"runtime config does not exist: {runtime_path}")
    runtime_config = OmegaConf.load(runtime_path)
    _init_single_process_group(args.device)
    annotations, images = _resolved_split_paths(
        runtime_config,
        args.split,
        args.annotations,
        args.images,
    )
    if not annotations.is_file():
        raise FileNotFoundError(f"annotations do not exist: {annotations}")
    if not images.is_dir():
        raise FileNotFoundError(f"images directory does not exist: {images}")

    checkpoint_paths = _discover_checkpoints(run_dir, args.checkpoints)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "evaluation" / f"{args.split}_loss"
    )
    json_path = output_dir / "checkpoint_loss.json"
    csv_path = output_dir / "checkpoint_loss.csv"
    rows: list[dict[str, Any]] = []

    print(
        f"split={args.split} images={images} annotations={annotations} "
        f"checkpoints={len(checkpoint_paths)}",
        flush=True,
    )
    for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        print(
            f"[{index}/{len(checkpoint_paths)}] {checkpoint_path.name}: evaluating",
            flush=True,
        )
        try:
            row = _evaluate_one(
                checkpoint_path,
                runtime_config,
                annotations,
                images,
                args.device,
            )
            print(
                f"[{index}/{len(checkpoint_paths)}] {checkpoint_path.name}: "
                f"loss={row['loss']:.9f}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path),
                "epoch": (
                    int(_NUMBERED_CHECKPOINT.match(checkpoint_path.name).group(1))
                    if _NUMBERED_CHECKPOINT.match(checkpoint_path.name)
                    else None
                ),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[{index}/{len(checkpoint_paths)}] {checkpoint_path.name}: "
                f"FAILED: {row['error']}",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        rows.append(row)
        payload = {
            "format_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "runtime_config": str(runtime_path),
            "split": args.split,
            "annotations": str(annotations),
            "annotations_sha256": _sha256(annotations),
            "images": str(images),
            "device": args.device,
            "amp_enabled": bool(runtime_config.trainer.optim.amp.enabled),
            "amp_dtype": str(runtime_config.trainer.optim.amp.amp_dtype),
            "normalization": (
                "local (numerically equivalent to training global at world-size 1)"
            ),
            "loss_config": OmegaConf.to_container(
                runtime_config.trainer.loss.book_spine, resolve=True
            ),
            "checkpoints": rows,
        }
        _atomic_write_json(json_path, payload)
        _write_csv(csv_path, rows)

    failed = [row for row in rows if row.get("status") != "completed"]
    print(f"json={json_path}", flush=True)
    print(f"csv={csv_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
