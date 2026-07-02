from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceRunPaths:
    run_id: str
    root: Path
    input_images: Path
    npz_raw: Path
    npz_nms: Path
    visual_raw: Path
    visual_nms: Path
    coco_dir: Path
    cvat_images: Path
    cvat_annotations: Path
    logs: Path
    run_config: Path
    manifest: Path
    validation_report: Path
    errors: Path


def unique_run_id(root: Path, now: datetime | None = None) -> str:
    base = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    candidate = base
    suffix = 2
    while (root / candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def create_inference_run(output_root: Path, run_id: str | None = None) -> InferenceRunPaths:
    root_base = output_root / "inference"
    root_base.mkdir(parents=True, exist_ok=True)
    rid = run_id or unique_run_id(root_base)
    root = root_base / rid
    if root.exists():
        raise FileExistsError(f"Run directory already exists: {root}")

    paths = InferenceRunPaths(
        run_id=rid,
        root=root,
        input_images=root / "input_images",
        npz_raw=root / "npz_raw",
        npz_nms=root / "npz_nms",
        visual_raw=root / "visualizations" / "raw",
        visual_nms=root / "visualizations" / "nms",
        coco_dir=root / "coco",
        cvat_images=root / "cvat_export" / "images",
        cvat_annotations=root / "cvat_export" / "annotations",
        logs=root / "logs",
        run_config=root / "run_config.json",
        manifest=root / "manifest.json",
        validation_report=root / "validation_report.json",
        errors=root / "errors.json",
    )
    for directory in [
        paths.input_images,
        paths.npz_raw,
        paths.npz_nms,
        paths.visual_raw,
        paths.visual_nms,
        paths.coco_dir,
        paths.cvat_images,
        paths.cvat_annotations,
        paths.logs,
    ]:
        directory.mkdir(parents=True, exist_ok=False)
    return paths


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_file_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"book_spine_run.{log_path.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def runtime_info(book_root: Path, sam3_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "book_root": str(book_root),
        "sam3_root": str(sam3_root),
        "business_code_version": git_commit(book_root),
        "sam3_code_version": git_commit(sam3_root),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def paths_as_dict(paths: InferenceRunPaths) -> dict[str, str]:
    data = asdict(paths)
    return {key: str(value) for key, value in data.items()}
