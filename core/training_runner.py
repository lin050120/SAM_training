from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from core.config import (
    BOOK_ROOT,
    DEFAULT_BOOK_SPINE_DATASET_ROOT,
    DEFAULT_BOOK_SPINE_FINETUNE_CONFIG,
    DEFAULT_CONDA_ENV,
    DEFAULT_SAM3_BPE_PATH,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_SAM3_TRAIN_SCRIPT,
    DEFAULT_TRAINING_RUN_ROOT,
    SAM301_ROOT,
)


@dataclass(frozen=True)
class PathCheck:
    path: str
    resolved_path: str
    exists: bool
    kind: str
    readable: bool
    writable: bool
    size_bytes: int | None
    inside_workspace: bool
    role: str
    warning: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class CocoSummary:
    path: str
    exists: bool
    images: int | None
    annotations: int | None
    has_book_spine: bool | None
    category_names: list[str]
    missing_files: list[str]
    error: str | None = None


@dataclass(frozen=True)
class TrainingPreflight:
    base_config_path: Path
    runtime_config_path: Path | None
    config_exists: bool
    is_default_authoritative_config: bool
    warning: str | None
    train_batch_size: int | None
    num_gpus: int
    gradient_accumulation_steps: int | None
    trainer_gradient_accumulation_steps: int | None
    effective_batch_size: int | None
    initial_checkpoint: str | None
    bpe_path: str | None
    train_images: str | None
    train_annotations: str | None
    val_images: str | None
    val_annotations: str | None
    output_dir: str | None
    coco_category_id: int | None
    coco_category_name: str | None
    requested_training_prompt: str | None
    resolved_training_prompt: str | None
    prompt_source: str | None
    output_root: str
    run_dir: str | None
    command: list[str]
    path_checks: list[PathCheck]
    train_coco: CocoSummary | None
    val_coco: CocoSummary | None
    warnings: list[str]
    errors: list[str]


def _same_file_or_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve(strict=True) == b.resolve(strict=True)
    except FileNotFoundError:
        return a.absolute() == b.absolute()


def _select_int(cfg: Any, key: str) -> int | None:
    value = OmegaConf.select(cfg, key)
    return int(value) if value is not None else None


def _resolve_existing_or_absolute(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _inside_workspace(path: Path) -> bool:
    return _inside(path, BOOK_ROOT) or _inside(path, SAM301_ROOT)


def _path_kind(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_symlink():
        return "symlink"
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    return "other"


def check_path(path: Path, role: str, output: bool = False, allow_external_output: bool = False) -> PathCheck:
    resolved = _resolve_existing_or_absolute(path)
    exists = resolved.exists()
    kind = _path_kind(resolved)
    readable = os.access(resolved, os.R_OK) if exists else False
    writable = os.access(resolved, os.W_OK) if exists else False
    size = resolved.stat().st_size if exists and resolved.is_file() else None
    inside_workspace = _inside_workspace(resolved)
    warning = None
    error = None
    if not inside_workspace:
        if output and not allow_external_output:
            error = "output path is outside workspace"
        else:
            warning = "external read-only input path" if not output else "external output path"
    if output:
        parent = resolved.parent if resolved.suffix else resolved
        if parent.exists() and not os.access(parent, os.W_OK):
            error = "output parent is not writable"
    return PathCheck(
        path=str(path),
        resolved_path=str(resolved),
        exists=exists,
        kind=kind,
        readable=readable,
        writable=writable,
        size_bytes=size,
        inside_workspace=inside_workspace,
        role=role,
        warning=warning,
        error=error,
    )


def unique_training_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = root / base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base}_{suffix}"
        suffix += 1
    return candidate


def summarize_coco(annotation_path: Path, image_dir: Path) -> CocoSummary:
    resolved_ann = _resolve_existing_or_absolute(annotation_path)
    if not resolved_ann.exists():
        return CocoSummary(str(resolved_ann), False, None, None, None, [], [], "COCO file missing")
    try:
        data = json.loads(resolved_ann.read_text(encoding="utf-8"))
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        category_names = [str(cat.get("name", "")) for cat in categories]
        has_book_spine = any(name == "book_spine" or name.replace(" ", "_") == "book_spine" for name in category_names)
        missing = []
        resolved_images = _resolve_existing_or_absolute(image_dir)
        for image in images:
            file_name = image.get("file_name")
            if file_name and not (resolved_images / file_name).exists():
                missing.append(file_name)
                if len(missing) >= 20:
                    break
        return CocoSummary(
            path=str(resolved_ann),
            exists=True,
            images=len(images),
            annotations=len(annotations),
            has_book_spine=has_book_spine,
            category_names=category_names,
            missing_files=missing,
        )
    except Exception as exc:
        return CocoSummary(str(resolved_ann), True, None, None, None, [], [], repr(exc))


def _default_dataset_paths() -> dict[str, Path]:
    return {
        "train_images": DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "images",
        "train_annotations": DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "annotations.json",
        "val_images": DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "images",
        "val_annotations": DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "annotations.json",
    }


def _resolve_training_inputs(
    cfg: Any,
    initial_checkpoint: Path | None,
    train_images: Path | None,
    train_annotations: Path | None,
    val_images: Path | None,
    val_annotations: Path | None,
    output_root: Path,
) -> dict[str, Path]:
    defaults = _default_dataset_paths()
    checkpoint = initial_checkpoint or (DEFAULT_SAM3_CHECKPOINT if DEFAULT_SAM3_CHECKPOINT.exists() else Path(OmegaConf.select(cfg, "trainer.model.checkpoint_path")))
    return {
        "initial_checkpoint": _resolve_existing_or_absolute(checkpoint),
        "bpe_path": _resolve_existing_or_absolute(DEFAULT_SAM3_BPE_PATH),
        "train_images": _resolve_existing_or_absolute(train_images or defaults["train_images"]),
        "train_annotations": _resolve_existing_or_absolute(train_annotations or defaults["train_annotations"]),
        "val_images": _resolve_existing_or_absolute(val_images or defaults["val_images"]),
        "val_annotations": _resolve_existing_or_absolute(val_annotations or defaults["val_annotations"]),
        "output_root": _resolve_existing_or_absolute(output_root),
    }


def _prompt_config(category_id: int, prompt: str) -> dict[str, Any]:
    return {
        "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON",
        "include_negatives": True,
        "category_chunk_size": 2,
        "prompts": repr([{"id": int(category_id), "name": prompt}]),
        "_partial_": True,
    }


def write_runtime_yaml(
    base_config: Path,
    runtime_config: Path,
    paths: dict[str, Path],
    run_dir: Path,
    category_id: int | None = None,
    training_prompt: str | None = None,
) -> None:
    cfg = OmegaConf.load(base_config)
    dataset_root = DEFAULT_BOOK_SPINE_DATASET_ROOT.resolve(strict=False)
    OmegaConf.update(cfg, "paths.dataset_root", str(dataset_root), merge=False)
    OmegaConf.update(cfg, "paths.experiment_log_dir", str(run_dir), merge=False)
    OmegaConf.update(cfg, "paths.bpe_path", str(paths["bpe_path"]), merge=False)
    OmegaConf.update(cfg, "trainer.model.checkpoint_path", str(paths["initial_checkpoint"]), merge=False)
    OmegaConf.update(cfg, "trainer.data.train.dataset.img_folder", str(paths["train_images"]), merge=False)
    OmegaConf.update(cfg, "trainer.data.train.dataset.ann_file", str(paths["train_annotations"]), merge=False)
    OmegaConf.update(cfg, "trainer.data.val.dataset.img_folder", str(paths["val_images"]), merge=False)
    OmegaConf.update(cfg, "trainer.data.val.dataset.ann_file", str(paths["val_annotations"]), merge=False)
    if category_id is not None and training_prompt:
        prompt_loader = _prompt_config(category_id, training_prompt)
        OmegaConf.update(cfg, "trainer.data.train.dataset.coco_json_loader", prompt_loader, merge=False)
        OmegaConf.update(cfg, "trainer.data.val.dataset.coco_json_loader", prompt_loader, merge=False)
    OmegaConf.update(cfg, "trainer.meters.val.book_spine.detection.dump_dir", str(run_dir / "dumps" / "book_spine"), merge=False)
    OmegaConf.update(cfg, "trainer.meters.val.book_spine.detection.pred_file_evaluators.0.gt_path", str(paths["val_annotations"]), merge=False)
    OmegaConf.update(cfg, "trainer.checkpoint.save_dir", str(run_dir / "checkpoints"), merge=False)
    OmegaConf.update(cfg, "trainer.logging.tensorboard_writer.log_dir", str(run_dir / "tensorboard"), merge=False)
    OmegaConf.update(cfg, "trainer.logging.log_dir", str(run_dir / "logs" / "book_spine"), merge=False)
    runtime_config.parent.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, runtime_config)


def inspect_training_config(
    config_path: Path = DEFAULT_BOOK_SPINE_FINETUNE_CONFIG,
    num_gpus: int = 1,
    train_script: Path = DEFAULT_SAM3_TRAIN_SCRIPT,
    conda_env: str = DEFAULT_CONDA_ENV,
    initial_checkpoint: Path | None = None,
    train_images: Path | None = None,
    train_annotations: Path | None = None,
    val_images: Path | None = None,
    val_annotations: Path | None = None,
    training_prompt: str | None = None,
    output_root: Path = DEFAULT_TRAINING_RUN_ROOT,
    allow_external_output: bool = False,
    prepare_runtime: bool = True,
) -> TrainingPreflight:
    base_config = _resolve_existing_or_absolute(config_path)
    train_script = _resolve_existing_or_absolute(train_script)
    output_root = _resolve_existing_or_absolute(output_root)
    exists = base_config.exists()
    is_default = _same_file_or_path(base_config, DEFAULT_BOOK_SPINE_FINETUNE_CONFIG)
    warnings: list[str] = []
    errors: list[str] = []
    warning = None if is_default else (
        "WARNING: selected config is not the default authoritative book-spine config: "
        f"{DEFAULT_BOOK_SPINE_FINETUNE_CONFIG}"
    )
    if warning:
        warnings.append(warning)

    train_batch_size = None
    accumulation = None
    trainer_accumulation = None
    effective = None
    runtime_config_path = None
    run_dir = None
    command_config = base_config
    path_checks: list[PathCheck] = []
    train_coco = None
    val_coco = None
    resolved_paths: dict[str, Path] = {}
    coco_category_id = None
    coco_category_name = None
    resolved_training_prompt = None
    prompt_source = None

    if not exists:
        errors.append(f"base config does not exist: {base_config}")
    else:
        cfg = OmegaConf.load(base_config)
        train_batch_size = _select_int(cfg, "scratch.train_batch_size")
        accumulation = _select_int(cfg, "scratch.gradient_accumulation_steps")
        trainer_accumulation = _select_int(cfg, "trainer.gradient_accumulation_steps")
        if train_batch_size is not None and accumulation is not None:
            effective = train_batch_size * num_gpus * accumulation
        resolved_paths = _resolve_training_inputs(
            cfg,
            initial_checkpoint,
            train_images,
            train_annotations,
            val_images,
            val_annotations,
            output_root,
        )
        run_dir = unique_training_run_dir(resolved_paths["output_root"]) if prepare_runtime else resolved_paths["output_root"]
        runtime_config_path = run_dir / "config" / "runtime_config.yaml"
        command_config = runtime_config_path if prepare_runtime else base_config

        path_checks = [
            check_path(base_config, "base_config"),
            check_path(train_script, "train_script"),
            check_path(resolved_paths["initial_checkpoint"], "initial_checkpoint"),
            check_path(resolved_paths["bpe_path"], "bpe_path"),
            check_path(resolved_paths["train_images"], "train_images"),
            check_path(resolved_paths["train_annotations"], "train_annotations"),
            check_path(resolved_paths["val_images"], "val_images"),
            check_path(resolved_paths["val_annotations"], "val_annotations"),
            check_path(run_dir, "output_dir", output=True, allow_external_output=allow_external_output),
        ]
        for item in path_checks:
            if item.warning:
                warnings.append(f"{item.role}: {item.warning}: {item.resolved_path}")
            if item.error:
                errors.append(f"{item.role}: {item.error}: {item.resolved_path}")
            if item.role in {"initial_checkpoint", "bpe_path", "train_images", "train_annotations", "val_images", "val_annotations"} and not item.exists:
                errors.append(f"{item.role} missing: {item.resolved_path}")
            if item.role in {"initial_checkpoint", "bpe_path", "train_images", "train_annotations", "val_images", "val_annotations"} and item.exists and not item.readable:
                errors.append(f"{item.role} not readable: {item.resolved_path}")

        if resolved_paths["train_images"] == resolved_paths["val_images"]:
            warnings.append("train_images and val_images point to the same directory")
        if resolved_paths["train_annotations"] == resolved_paths["val_annotations"]:
            warnings.append("train_annotations and val_annotations point to the same COCO file")

        train_coco = summarize_coco(resolved_paths["train_annotations"], resolved_paths["train_images"])
        val_coco = summarize_coco(resolved_paths["val_annotations"], resolved_paths["val_images"])
        if train_coco and train_coco.category_names:
            train_data = json.loads(resolved_paths["train_annotations"].read_text(encoding="utf-8"))
            first_category = sorted(train_data.get("categories", []), key=lambda item: int(item.get("id", 0)))[0]
            coco_category_id = int(first_category["id"])
            coco_category_name = str(first_category["name"])
            if training_prompt:
                resolved_training_prompt = training_prompt
                prompt_source = "manual_override"
                if resolved_training_prompt != coco_category_name:
                    warnings.append(
                        f"training prompt differs from COCO category name; annotations still use category_id={coco_category_id}"
                    )
            else:
                resolved_training_prompt = coco_category_name
                prompt_source = "category_name_fallback"
        for label, summary in [("train", train_coco), ("val", val_coco)]:
            if summary.error:
                errors.append(f"{label} COCO error: {summary.error}")
            if summary.has_book_spine is False:
                errors.append(f"{label} COCO categories do not include book_spine")
            elif summary.category_names and "book_spine" not in summary.category_names:
                warnings.append(f"{label} COCO uses compatible category alias instead of exact book_spine: {summary.category_names}")
            if summary.missing_files:
                errors.append(f"{label} COCO has missing image files, first examples: {summary.missing_files[:5]}")

        if prepare_runtime and run_dir.exists() and any(run_dir.iterdir()):
            errors.append(f"output directory already exists and is non-empty: {run_dir}")
        if prepare_runtime and not errors:
            write_runtime_yaml(
                base_config,
                runtime_config_path,
                resolved_paths,
                run_dir,
                category_id=coco_category_id,
                training_prompt=resolved_training_prompt if training_prompt else None,
            )
            command_config = runtime_config_path

    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(train_script),
        "-c",
        str(command_config),
        "--use-cluster",
        "0",
        "--num-gpus",
        str(num_gpus),
    ]

    if prepare_runtime and run_dir is not None and not errors:
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        dataset_info = {
            "initial_checkpoint": str(resolved_paths["initial_checkpoint"]),
            "bpe_path": str(resolved_paths["bpe_path"]),
            "train_images": str(resolved_paths["train_images"]),
            "train_annotations": str(resolved_paths["train_annotations"]),
            "val_images": str(resolved_paths["val_images"]),
            "val_annotations": str(resolved_paths["val_annotations"]),
            "train_coco": asdict(train_coco) if train_coco else None,
            "val_coco": asdict(val_coco) if val_coco else None,
            "coco_category_id": coco_category_id,
            "coco_category_name": coco_category_name,
            "requested_training_prompt": training_prompt,
            "resolved_training_prompt": resolved_training_prompt,
            "prompt_source": prompt_source,
        }
        (run_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    return TrainingPreflight(
        base_config_path=base_config,
        runtime_config_path=runtime_config_path,
        config_exists=exists,
        is_default_authoritative_config=is_default,
        warning=warning,
        train_batch_size=train_batch_size,
        num_gpus=num_gpus,
        gradient_accumulation_steps=accumulation,
        trainer_gradient_accumulation_steps=trainer_accumulation,
        effective_batch_size=effective,
        initial_checkpoint=str(resolved_paths.get("initial_checkpoint")) if resolved_paths else None,
        bpe_path=str(resolved_paths.get("bpe_path")) if resolved_paths else None,
        train_images=str(resolved_paths.get("train_images")) if resolved_paths else None,
        train_annotations=str(resolved_paths.get("train_annotations")) if resolved_paths else None,
        val_images=str(resolved_paths.get("val_images")) if resolved_paths else None,
        val_annotations=str(resolved_paths.get("val_annotations")) if resolved_paths else None,
        output_dir=str(run_dir) if run_dir else None,
        coco_category_id=coco_category_id,
        coco_category_name=coco_category_name,
        requested_training_prompt=training_prompt,
        resolved_training_prompt=resolved_training_prompt,
        prompt_source=prompt_source,
        output_root=str(output_root),
        run_dir=str(run_dir) if run_dir else None,
        command=command,
        path_checks=path_checks,
        train_coco=train_coco,
        val_coco=val_coco,
        warnings=warnings,
        errors=errors,
    )


def preflight_as_dict(preflight: TrainingPreflight) -> dict[str, Any]:
    data = asdict(preflight)
    data["base_config_path"] = str(preflight.base_config_path)
    data["runtime_config_path"] = str(preflight.runtime_config_path) if preflight.runtime_config_path else None
    return data


def format_preflight(preflight: TrainingPreflight) -> str:
    return json.dumps(preflight_as_dict(preflight), ensure_ascii=False, indent=2)
