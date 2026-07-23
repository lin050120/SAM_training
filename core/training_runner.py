from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
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
    DEFAULT_TASK_SLUG,
    EXPECTED_SAM3_INIT,
    EXPECTED_SAM3_PACKAGE_DIR,
    EXPECTED_SAM3_ROOT,
    DEFAULT_SAM3_BPE_PATH,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_SAM3_TRAIN_SCRIPT,
    DEFAULT_TRAINING_LAUNCHER,
    DEFAULT_TRAINING_RUN_ROOT,
    SAM301_ROOT,
)

from core.dataset_identity import (
    TRAINING_MODE_SMOKE,
    resolve_dataset_identity,
    validate_training_mode_against_identity,
)
from core.online_augmentation import (
    OnlineAugmentationConfig,
    build_online_augmentation_transforms,
    resolve_online_augmentation_config,
)
from core.sam301_patch import collect_training_provenance, verify_patched_for_training

TRAINING_OUTPUT_ROOT_ERROR = "Training output must remain under"
DEFAULT_DISTRIBUTED_MASTER_ADDR = "localhost"


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
    has_expected_category: bool | None
    category_names: list[str]
    missing_files: list[str]
    expected_category_name: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TrainingPreflight:
    base_config_path: Path
    runtime_config_path: Path | None
    config_exists: bool
    is_default_authoritative_config: bool
    warning: str | None
    # train_batch_size / gradient_accumulation_steps / effective_batch_size below are the
    # *resolved* values (override if valid and given, else the authoritative base YAML
    # value) — this is what will actually run, matching requirement "预检必须显示最终
    # resolved 值". The requested_* counterparts capture what the user asked for, so a
    # mismatch (e.g. an invalid override that fell back to base) stays visible.
    train_batch_size: int | None
    num_gpus: int
    gradient_accumulation_steps: int | None
    trainer_gradient_accumulation_steps: int | None
    effective_batch_size: int | None
    max_epochs: int | None
    learning_rate: float | None
    num_workers: int | None
    requested_max_epochs: int | None
    requested_train_batch_size: int | None
    requested_gradient_accumulation_steps: int | None
    requested_learning_rate: float | None
    requested_num_workers: int | None
    unsupported_overrides: list[str]
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
    conda_environment: str
    expected_sam3_root: str
    expected_sam3_package_dir: str
    resolved_sam3_import_path: str | None
    sam3_import_guard_ok: bool | None
    sam3_import_guard_error: str | None
    effective_pythonpath: str | None
    command: list[str]
    path_checks: list[PathCheck]
    train_coco: CocoSummary | None
    val_coco: CocoSummary | None
    warnings: list[str]
    errors: list[str]
    training_provenance: dict[str, Any] | None = None
    dataset_identity: dict[str, Any] | None = None
    online_augmentation: dict[str, Any] | None = None


def _same_file_or_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve(strict=True) == b.resolve(strict=True)
    except FileNotFoundError:
        return a.absolute() == b.absolute()


def _select_int(cfg: Any, key: str) -> int | None:
    value = OmegaConf.select(cfg, key)
    return int(value) if value is not None else None


def _select_float(cfg: Any, key: str) -> float | None:
    # throw_on_resolution_failure=False: scratch.lr_transformer is
    # ${times:8e-4,${scratch.lr_scale}} in the base YAML, using a custom "times"
    # OmegaConf resolver registered by sam3.train.utils.register_omegaconf_resolvers()
    # at real training-launch time. Preflight does not import that (heavy torch/hydra
    # module, and this codebase should not guess its multiply_all() semantics), so an
    # unresolvable interpolation is reported as None rather than raising or guessing.
    value = OmegaConf.select(cfg, key, throw_on_resolution_failure=False)
    return float(value) if value is not None else None


def _validate_positive_override(
    value: Any, name: str, kind: str = "int", allow_zero: bool = False
) -> tuple[Any, str | None]:
    """Validate a caller-supplied training param override.

    None means "no override requested" and passes through untouched (the caller
    falls back to the base YAML value). This exists independently of
    ui/ui_utils.py's parser: core.training_runner is also called directly by
    scripts/training_preflight.py, so it must not trust an already-clean value from
    the UI layer — a CLI caller could pass -5 or NaN directly.

    allow_zero=True is for num_workers: 0 is PyTorch DataLoader's documented value
    for "load in the main process" and is exactly what the base YAML already uses
    for scratch.num_val_workers, so it must not be treated as invalid.
    """
    if value is None:
        return None, None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{name} is not a valid number: {value!r}"
    if not math.isfinite(number):
        return None, f"{name} must be a finite number, got {value!r}"
    if kind == "int":
        if number != int(number):
            return None, f"{name} must be a whole number, got {number}"
        number = int(number)
    minimum_ok = number >= 0 if allow_zero else number > 0
    if not minimum_ok:
        bound = "non-negative" if allow_zero else "positive"
        return None, f"{name} must be a {bound} number, got {number}"
    return number, None


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


def _inside_training_output_root(path: Path, root: Path | None = None) -> bool:
    root = root or DEFAULT_TRAINING_RUN_ROOT
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def task_slug_from_name(name: str | None) -> str:
    text = (name or "").strip().lower()
    if not text:
        return DEFAULT_TASK_SLUG
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    # A non-empty name that slugs to nothing (e.g. fully non-ASCII like "ケーブル")
    # must not silently reuse the book_spine directories of an unrelated task.
    return slug or "task"


def validate_training_output_root(path: Path, allow_external_output: bool = False) -> str | None:
    if allow_external_output:
        return None
    resolved = path.expanduser().resolve(strict=False)
    if not _inside_training_output_root(resolved):
        return f"{TRAINING_OUTPUT_ROOT_ERROR} {DEFAULT_TRAINING_RUN_ROOT}"
    return None


def _bind_host_for_addr(master_addr: str) -> str:
    return "127.0.0.1" if master_addr in {"localhost", "127.0.0.1"} else master_addr


def is_tcp_port_available(port: int, master_addr: str = DEFAULT_DISTRIBUTED_MASTER_ADDR) -> bool:
    if not isinstance(port, int) or port < 1 or port > 65535:
        return False
    host = _bind_host_for_addr(master_addr)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def allocate_distributed_port(master_addr: str = DEFAULT_DISTRIBUTED_MASTER_ADDR) -> int:
    """Return a currently bindable TCP port for SAM3's local TCPStore.

    The socket is closed before Popen because SAM3 itself must bind it. The launcher
    calls this under the training-start lock immediately before spawning to keep the
    unavoidable check/use window as small as possible.
    """
    host = _bind_host_for_addr(master_addr)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def configure_runtime_distributed_port(
    runtime_config: Path,
    master_port: int,
    master_addr: str = DEFAULT_DISTRIBUTED_MASTER_ADDR,
) -> dict[str, Any]:
    if not is_tcp_port_available(master_port, master_addr=master_addr):
        raise RuntimeError(f"distributed port is not bindable before launch: {master_addr}:{master_port}")
    cfg = OmegaConf.load(runtime_config)
    OmegaConf.update(cfg, "submitit.port_range", [int(master_port), int(master_port)], merge=False)
    OmegaConf.save(cfg, runtime_config)
    return {
        "master_addr": master_addr,
        "master_port": int(master_port),
        "port_range": [int(master_port), int(master_port)],
        "runtime_config": str(runtime_config),
    }


def validate_training_run_path(path: Path, allow_external_output: bool = False) -> str | None:
    return validate_training_output_root(path, allow_external_output=allow_external_output)


def training_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for SAM3 trainer subprocesses.

    The sam301 conda environment is the primary source of truth. PYTHONPATH is still
    prefixed defensively for child processes so an accidental future editable-package
    regression cannot silently import /home/book/sam3. The caller's value is preserved
    after the canonical source roots, and os.environ is never mutated globally.
    """
    env = dict(base_env) if base_env is not None else os.environ.copy()
    existing = env.get("PYTHONPATH")
    prefixes = [str(EXPECTED_SAM3_ROOT), str(BOOK_ROOT)]
    if existing:
        prefixes.extend(part for part in existing.split(os.pathsep) if part)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(prefixes))
    return env


def validate_sam3_import_path(actual_path: str | Path | None) -> str | None:
    if not actual_path:
        return f"sam3 import guard produced no path; expected {EXPECTED_SAM3_INIT}"
    try:
        resolved_actual = Path(actual_path).expanduser().resolve(strict=False)
    except TypeError:
        return f"sam3 import path is not parseable: {actual_path!r}"
    expected_init = EXPECTED_SAM3_INIT.expanduser().resolve(strict=False)
    expected_package = EXPECTED_SAM3_PACKAGE_DIR.expanduser().resolve(strict=False)
    if resolved_actual != expected_init:
        try:
            resolved_actual.relative_to(expected_package)
        except ValueError:
            pass
        return f"sam3 import resolved to {resolved_actual}, expected {expected_init}"
    return None


def build_sam3_import_guard_command(conda_env: str = DEFAULT_CONDA_ENV) -> list[str]:
    script = (
        "from pathlib import Path\n"
        "import json, sys\n"
        "import sam3\n"
        "print(json.dumps({'python': sys.executable, 'sam3': str(Path(sam3.__file__).resolve())}))\n"
    )
    return ["conda", "run", "-n", conda_env, "python", "-c", script]


def run_sam3_import_guard(
    conda_env: str = DEFAULT_CONDA_ENV,
    cwd: Path = BOOK_ROOT,
    env: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    command = build_sam3_import_guard_command(conda_env)
    effective_env = env if env is not None else training_subprocess_env()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=effective_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "command": command,
            "python": None,
            "sam3": None,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error": f"sam3 import guard command failed: {exc!r}",
        }

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    payload = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if completed.returncode != 0:
        error = f"sam3 import guard exited with code {completed.returncode}"
    elif payload is None:
        error = "sam3 import guard did not emit parseable JSON"
    else:
        error = validate_sam3_import_path(payload.get("sam3"))
    return {
        "ok": error is None,
        "command": command,
        "python": payload.get("python") if isinstance(payload, dict) else None,
        "sam3": payload.get("sam3") if isinstance(payload, dict) else None,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": completed.returncode,
        "error": error,
    }


def build_hydra_validation_command(config_path: Path, conda_env: str = DEFAULT_CONDA_ENV) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(DEFAULT_TRAINING_LAUNCHER),
        "-c",
        str(config_path),
        "--validate-only",
    ]


def run_hydra_config_validation(
    config_path: Path,
    conda_env: str = DEFAULT_CONDA_ENV,
    cwd: Path = BOOK_ROOT,
    env: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Prove the runtime YAML is loadable by the real launch path without training.

    Runs the launcher wrapper's --validate-only mode in the same conda env and
    subprocess env that the trainer itself will use.
    """
    command = build_hydra_validation_command(config_path, conda_env=conda_env)
    effective_env = env if env is not None else training_subprocess_env()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=effective_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "command": command,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error": f"hydra validation command failed: {exc!r}",
        }
    error = None
    if completed.returncode != 0:
        stderr_tail = "\n".join((completed.stderr or "").splitlines()[-5:])
        error = f"hydra validation exited with code {completed.returncode}: {stderr_tail}"
    return {
        "ok": error is None,
        "command": command,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "returncode": completed.returncode,
        "error": error,
    }


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
    base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = root / base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base}_{suffix}"
        suffix += 1
    return candidate


def summarize_coco(annotation_path: Path, image_dir: Path, expected_category_name: str | None = None) -> CocoSummary:
    resolved_ann = _resolve_existing_or_absolute(annotation_path)
    if not resolved_ann.exists():
        return CocoSummary(
            str(resolved_ann),
            False,
            None,
            None,
            None,
            [],
            [],
            expected_category_name=expected_category_name,
            error="COCO file missing",
        )
    try:
        data = json.loads(resolved_ann.read_text(encoding="utf-8"))
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        category_names = [str(cat.get("name", "")) for cat in categories]
        has_expected_category = None
        if expected_category_name:
            expected = expected_category_name.strip()
            expected_slug = task_slug_from_name(expected)
            has_expected_category = any(
                name == expected or task_slug_from_name(name) == expected_slug for name in category_names
            )
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
            has_expected_category=has_expected_category,
            category_names=category_names,
            missing_files=missing,
            expected_category_name=expected_category_name,
        )
    except Exception as exc:
        return CocoSummary(
            str(resolved_ann),
            True,
            None,
            None,
            None,
            [],
            [],
            expected_category_name=expected_category_name,
            error=repr(exc),
        )


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


def _wire_validation_loss(cfg: Any) -> None:
    """Use a validation-compatible copy of the train criterion.

    SAM3 selects a loss by the single key returned by each collator. The base
    config uses ``all`` for train and ``book_spine`` for validation, while only
    ``all`` is wired to the real criterion. Without this explicit alias,
    validation silently falls back to DummyLoss and records zero for every
    validation step. SAM3Image also omits matcher indices in eval mode, so the
    validation copy fills those indices before invoking the unchanged criterion.
    """
    train_key = OmegaConf.select(cfg, "scratch.collate_fn.dict_key")
    val_key = OmegaConf.select(cfg, "scratch.collate_fn_val.dict_key")
    if not isinstance(train_key, str) or not train_key.strip():
        raise ValueError("base config must define scratch.collate_fn.dict_key")
    if not isinstance(val_key, str) or not val_key.strip():
        raise ValueError("base config must define scratch.collate_fn_val.dict_key")

    loss_cfg = OmegaConf.select(cfg, "trainer.loss")
    if loss_cfg is None or train_key not in loss_cfg:
        raise ValueError(
            "base config trainer.loss must define the train collator key "
            f"{train_key!r}"
        )
    if val_key != train_key:
        train_loss_cfg = OmegaConf.to_container(
            loss_cfg[train_key],
            resolve=False,
        )
        if not isinstance(train_loss_cfg, dict):
            raise ValueError(
                f"trainer.loss.{train_key} must resolve to a loss configuration"
            )
        train_loss_cfg["_target_"] = (
            "core.training_loss_trace.ValidationMatchingSam3LossWrapper"
        )
        loss_cfg[val_key] = OmegaConf.create(train_loss_cfg)


def write_runtime_yaml(
    base_config: Path,
    runtime_config: Path,
    paths: dict[str, Path],
    run_dir: Path,
    category_id: int | None = None,
    training_prompt: str | None = None,
    task_slug: str | None = None,
    max_epochs: int | None = None,
    train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    num_workers: int | None = None,
    online_augmentation: OnlineAugmentationConfig | dict[str, Any] | None = None,
) -> None:
    """Render the base authoritative YAML into a per-run runtime YAML.

    The base YAML is only ever read here, never written. Only fields explicitly
    passed as non-None are touched — e.g. scratch.lr_transformer is an interpolated
    expression (${times:8e-4,${scratch.lr_scale}}) in the base config, and leaving
    it untouched when learning_rate is None preserves that interpolation instead of
    collapsing it to a literal for no reason.

    Field mapping confirmed against book_spine_finetune.yaml (not guessed):
      max_epochs                  -> trainer.max_epochs
      train_batch_size            -> scratch.train_batch_size (feeds
                                      trainer.data.train.batch_size via interpolation)
      gradient_accumulation_steps -> scratch.gradient_accumulation_steps (feeds
                                      trainer.gradient_accumulation_steps)
      learning_rate                -> scratch.lr_transformer (the trainable
                                      transformer/decoder LR fed to the optimizer's
                                      scheduler as base_lr)
      num_workers                  -> scratch.num_train_workers only. The val
                                      dataloader (scratch.num_val_workers) is left at
                                      its base value (0) — the val split is tiny (2
                                      images per docs/training_path_audit.md) and val
                                      workers were not part of what the user asked to
                                      control.

    Deliberately NOT exposed: scratch.lr_vision_backbone / scratch.lr_language_backbone.
    The base YAML freezes both at 0.0 on purpose (see its "冻结策略" comment) to keep
    the vision backbone and text prompt frozen; a generic "learning_rate" override must
    not silently unfreeze them.
    """
    cfg = OmegaConf.load(base_config)
    augmentation, augmentation_errors = resolve_online_augmentation_config(
        online_augmentation
    )
    if augmentation_errors:
        raise ValueError("; ".join(augmentation_errors))
    slug = task_slug_from_name(task_slug or training_prompt)
    dataset_root = DEFAULT_BOOK_SPINE_DATASET_ROOT.resolve(strict=False)
    train_ann = paths.get("train_annotations")
    if train_ann is not None and train_ann.name == "annotations.json" and train_ann.parent.name in {"train", "training"}:
        dataset_root = train_ann.parent.parent.resolve(strict=False)
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
    OmegaConf.update(cfg, "trainer.meters.val.book_spine.detection.dump_dir", str(run_dir / "dumps" / slug), merge=False)
    OmegaConf.update(cfg, "trainer.meters.val.book_spine.detection.pred_file_evaluators.0.gt_path", str(paths["val_annotations"]), merge=False)
    OmegaConf.update(cfg, "trainer.checkpoint.save_dir", str(run_dir / "checkpoints"), merge=False)
    OmegaConf.update(cfg, "trainer.logging.tensorboard_writer.log_dir", str(run_dir / "tensorboard"), merge=False)
    OmegaConf.update(cfg, "trainer.logging.log_dir", str(run_dir / "logs" / slug), merge=False)
    if max_epochs is not None:
        OmegaConf.update(cfg, "trainer.max_epochs", int(max_epochs), merge=False)
    if train_batch_size is not None:
        OmegaConf.update(cfg, "scratch.train_batch_size", int(train_batch_size), merge=False)
    if gradient_accumulation_steps is not None:
        OmegaConf.update(cfg, "scratch.gradient_accumulation_steps", int(gradient_accumulation_steps), merge=False)
    if num_workers is not None:
        OmegaConf.update(cfg, "scratch.num_train_workers", int(num_workers), merge=False)
    if learning_rate is not None:
        OmegaConf.update(cfg, "scratch.lr_transformer", float(learning_rate), merge=False)
    OmegaConf.update(
        cfg,
        "trainer._target_",
        "core.training_loss_trace.LossTracingTrainer",
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "trainer.train_loss_window_optimizer_steps",
        20,
        merge=False,
    )
    OmegaConf.update(cfg, "trainer.val_epoch_freq", 1, merge=False)
    OmegaConf.update(cfg, "trainer.skip_first_val", False, merge=False)

    # Online augmentation is inserted only into the train ComposeAPI, immediately
    # after DecodeRle and before the base resize/pad transforms. Validation remains
    # byte-for-byte equivalent to the authoritative YAML transform tree.
    OmegaConf.update(cfg, "online_augmentation", augmentation.to_dict(), merge=False)
    if augmentation.enabled:
        train_transforms = OmegaConf.to_container(
            OmegaConf.select(cfg, "book_spine.train_transforms.0.transforms"),
            resolve=False,
        )
        if not isinstance(train_transforms, list):
            raise ValueError("base config train ComposeAPI transforms must be a list")
        decode_index = next(
            (
                index
                for index, item in enumerate(train_transforms)
                if isinstance(item, dict)
                and item.get("_target_") == "sam3.train.transforms.segmentation.DecodeRle"
            ),
            None,
        )
        if decode_index is None:
            raise ValueError("base config train transforms must contain DecodeRle")
        train_transforms[decode_index + 1 : decode_index + 1] = build_online_augmentation_transforms(
            augmentation
        )
        OmegaConf.update(
            cfg,
            "book_spine.train_transforms.0.transforms",
            train_transforms,
            merge=False,
        )
    if augmentation.repeat_factor > 1:
        OmegaConf.update(
            cfg,
            "trainer.data.train.dataset._target_",
            "core.training_augmentation.RepeatedSam3ImageDataset",
            merge=False,
        )
        OmegaConf.update(
            cfg,
            "trainer.data.train.dataset.repeat_factor",
            augmentation.repeat_factor,
            merge=False,
        )

    # Gradient accumulation wiring. trainer._run_step requires the dataloader to
    # yield a LIST of exactly gradient_accumulation_steps micro-batches when
    # accumulation is enabled (sam3/train/trainer.py:920-925). The only official
    # producer of that list is sam3.train.data.collator.collate_fn_api_with_chunking,
    # which splits one DataLoader fetch into num_chunks collated micro-batches — so
    # the DataLoader batch_size must be micro_batch × accum_steps. The base YAML
    # (copied from a grad_accum=1 template) uses plain collate_fn_api and
    # batch_size=${scratch.train_batch_size}; without this wiring any accum>1 run
    # fails with "Expected a list of batches, got <class 'dict'>".
    raw_accum = OmegaConf.select(cfg, "scratch.gradient_accumulation_steps")
    raw_micro_batch = OmegaConf.select(cfg, "scratch.train_batch_size")
    if raw_accum is None or raw_micro_batch is None:
        raise ValueError(
            "base config must define scratch.train_batch_size and "
            "scratch.gradient_accumulation_steps (or overrides must be provided); "
            f"got train_batch_size={raw_micro_batch!r}, "
            f"gradient_accumulation_steps={raw_accum!r}"
        )
    effective_accum = int(raw_accum)
    effective_micro_batch = int(raw_micro_batch)
    if effective_accum > 1:
        OmegaConf.update(
            cfg,
            "scratch.collate_fn._target_",
            "sam3.train.data.collator.collate_fn_api_with_chunking",
            merge=False,
        )
        OmegaConf.update(
            cfg,
            "scratch.collate_fn.num_chunks",
            "${scratch.gradient_accumulation_steps}",
            merge=False,
        )
        OmegaConf.update(
            cfg,
            "trainer.data.train.batch_size",
            effective_micro_batch * effective_accum,
            merge=False,
        )

    _wire_validation_loss(cfg)
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
    max_epochs: int | None = None,
    train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    num_workers: int | None = None,
    output_root: Path = DEFAULT_TRAINING_RUN_ROOT,
    allow_external_output: bool = False,
    prepare_runtime: bool = True,
    collect_import_metadata: bool = False,
    training_mode: str = TRAINING_MODE_SMOKE,
    online_augmentation: OnlineAugmentationConfig | dict[str, Any] | None = None,
) -> TrainingPreflight:
    warnings: list[str] = []
    errors: list[str] = []
    resolved_augmentation, augmentation_errors = resolve_online_augmentation_config(
        online_augmentation
    )
    errors.extend(augmentation_errors)
    base_config = _resolve_existing_or_absolute(config_path)
    train_script = _resolve_existing_or_absolute(train_script)
    try:
        output_root = _resolve_existing_or_absolute(output_root if isinstance(output_root, Path) else Path(output_root))
    except TypeError:
        errors.append(f"output_root is not a valid path: {output_root!r}")
        output_root = DEFAULT_TRAINING_RUN_ROOT.expanduser().resolve(strict=False)
    exists = base_config.exists()
    is_default = _same_file_or_path(base_config, DEFAULT_BOOK_SPINE_FINETUNE_CONFIG)
    warning = None if is_default else (
        "WARNING: selected config is not the default authoritative SAM3 fine-tuning config: "
        f"{DEFAULT_BOOK_SPINE_FINETUNE_CONFIG}"
    )
    if warning:
        warnings.append(warning)

    # Validate optional overrides independently of whether the base config exists:
    # core.training_runner is also invoked directly by scripts/training_preflight.py,
    # so a caller could pass an invalid raw value even after the UI layer's own parsing.
    # None means "no override requested" and always falls back to the base YAML value.
    validated_max_epochs, err = _validate_positive_override(max_epochs, "max_epochs", "int")
    if err:
        errors.append(err)
    validated_train_batch_size, err = _validate_positive_override(train_batch_size, "train_batch_size", "int")
    if err:
        errors.append(err)
    validated_gradient_accumulation_steps, err = _validate_positive_override(
        gradient_accumulation_steps, "gradient_accumulation_steps", "int"
    )
    if err:
        errors.append(err)
    validated_learning_rate, err = _validate_positive_override(learning_rate, "learning_rate", "float")
    if err:
        errors.append(err)
    validated_num_workers, err = _validate_positive_override(num_workers, "num_workers", "int", allow_zero=True)
    if err:
        errors.append(err)
    unsupported_overrides: list[str] = []

    resolved_max_epochs = None
    resolved_train_batch_size = None
    resolved_gradient_accumulation_steps = None
    resolved_learning_rate = None
    resolved_num_workers = None
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
    import_guard_result: dict[str, Any] | None = None
    hydra_validation_result: dict[str, Any] | None = None
    dataset_identity: dict[str, Any] | None = None
    training_provenance: dict[str, Any] | None = None
    distributed_metadata: dict[str, Any] | None = None
    effective_env = training_subprocess_env()

    if not exists:
        errors.append(f"base config does not exist: {base_config}")
    else:
        cfg = OmegaConf.load(base_config)
        base_max_epochs = _select_int(cfg, "trainer.max_epochs")
        base_train_batch_size = _select_int(cfg, "scratch.train_batch_size")
        base_gradient_accumulation_steps = _select_int(cfg, "scratch.gradient_accumulation_steps")
        base_learning_rate = _select_float(cfg, "scratch.lr_transformer")
        base_num_workers = _select_int(cfg, "scratch.num_train_workers")
        trainer_accumulation = _select_int(cfg, "trainer.gradient_accumulation_steps")

        # Resolved = validated override if the user gave one, else the authoritative
        # base YAML value. Never 0/NaN/empty-string: an invalid override was already
        # turned into an error above and validated_* is None in that case, so it falls
        # back to the base value here rather than propagating a bad number.
        resolved_max_epochs = validated_max_epochs if validated_max_epochs is not None else base_max_epochs
        resolved_train_batch_size = (
            validated_train_batch_size if validated_train_batch_size is not None else base_train_batch_size
        )
        resolved_gradient_accumulation_steps = (
            validated_gradient_accumulation_steps
            if validated_gradient_accumulation_steps is not None
            else base_gradient_accumulation_steps
        )
        resolved_learning_rate = validated_learning_rate if validated_learning_rate is not None else base_learning_rate
        resolved_num_workers = validated_num_workers if validated_num_workers is not None else base_num_workers
        if resolved_learning_rate is None:
            warnings.append(
                "base learning rate (scratch.lr_transformer) uses a custom OmegaConf "
                "resolver not registered during preflight, so its value cannot be "
                "displayed unless you set an explicit learning_rate override; the "
                "authoritative base YAML expression is left untouched either way."
            )

        # R-4: these two scratch keys are load-bearing for the accumulation wiring
        # in write_runtime_yaml (int(None) would otherwise raise TypeError there for
        # non-default base configs). Reject missing/invalid values with a clean error.
        for scratch_key, resolved_value in [
            ("scratch.train_batch_size", resolved_train_batch_size),
            ("scratch.gradient_accumulation_steps", resolved_gradient_accumulation_steps),
        ]:
            if resolved_value is None:
                errors.append(
                    f"{scratch_key} is missing: the base config does not define it and no "
                    "override was given; it must be an integer >= 1"
                )
            elif resolved_value < 1:
                errors.append(f"{scratch_key} must be an integer >= 1, got {resolved_value}")

        if resolved_train_batch_size is not None and resolved_gradient_accumulation_steps is not None:
            effective = resolved_train_batch_size * num_gpus * resolved_gradient_accumulation_steps
        resolved_paths = _resolve_training_inputs(
            cfg,
            initial_checkpoint,
            train_images,
            train_annotations,
            val_images,
            val_annotations,
            output_root,
        )
        output_root_error = validate_training_output_root(
            resolved_paths["output_root"], allow_external_output=allow_external_output
        )
        if output_root_error:
            errors.append(f"output_dir: {output_root_error}: {resolved_paths['output_root']}")
        run_dir = (
            unique_training_run_dir(resolved_paths["output_root"])
            if prepare_runtime and not output_root_error
            else resolved_paths["output_root"]
        )
        runtime_config_path = run_dir / "config" / "runtime_config.yaml"
        command_config = runtime_config_path if prepare_runtime else base_config

        path_checks = [
            check_path(base_config, "base_config"),
            check_path(train_script, "train_script"),
            check_path(DEFAULT_TRAINING_LAUNCHER, "training_launcher"),
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
            if item.role in {"train_script", "training_launcher", "initial_checkpoint", "bpe_path", "train_images", "train_annotations", "val_images", "val_annotations"} and not item.exists:
                errors.append(f"{item.role} missing: {item.resolved_path}")
            if item.role in {"train_script", "training_launcher", "initial_checkpoint", "bpe_path", "train_images", "train_annotations", "val_images", "val_annotations"} and item.exists and not item.readable:
                errors.append(f"{item.role} not readable: {item.resolved_path}")

        if resolved_paths["train_images"] == resolved_paths["val_images"]:
            warnings.append("train_images and val_images point to the same directory")
        if resolved_paths["train_annotations"] == resolved_paths["val_annotations"]:
            warnings.append("train_annotations and val_annotations point to the same COCO file")

        train_parse_error = None
        try:
            train_data = json.loads(resolved_paths["train_annotations"].read_text(encoding="utf-8"))
            train_categories = sorted(train_data.get("categories", []), key=lambda item: int(item.get("id", 0)))
        except Exception as exc:
            train_parse_error = exc
            train_categories = []
        if train_categories:
            first_category = train_categories[0]
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
        elif train_parse_error is None:
            errors.append("train COCO has no categories; cannot determine the training category")
        train_coco = summarize_coco(
            resolved_paths["train_annotations"],
            resolved_paths["train_images"],
            expected_category_name=coco_category_name,
        )
        val_coco = summarize_coco(
            resolved_paths["val_annotations"],
            resolved_paths["val_images"],
            expected_category_name=coco_category_name,
        )
        for label, summary in [("train", train_coco), ("val", val_coco)]:
            if summary.error:
                errors.append(f"{label} COCO error: {summary.error}")
            if summary.has_expected_category is False:
                errors.append(
                    f"{label} COCO categories do not include expected category "
                    f"{summary.expected_category_name!r}: {summary.category_names}"
                )
            if summary.missing_files:
                errors.append(f"{label} COCO has missing image files, first examples: {summary.missing_files[:5]}")

        # R-5: the train loader uses drop_last=True, so a train set smaller than the
        # effective batch (micro_batch x accum x num_gpus) yields len(loader) == 0 and
        # the trainer would "complete" an epoch with ZERO optimizer steps yet still
        # save a checkpoint. Reject that outright; warn about partial-batch drops.
        if effective is not None and train_coco is not None and train_coco.images:
            train_samples_per_epoch = train_coco.images * resolved_augmentation.repeat_factor
            if train_samples_per_epoch < effective:
                errors.append(
                    f"train samples per epoch ({train_samples_per_epoch} = {train_coco.images} source images "
                    f"x repeat_factor {resolved_augmentation.repeat_factor}) is smaller than the effective "
                    f"batch size ({effective} = train_batch_size x gradient_accumulation_steps "
                    f"x num_gpus); with drop_last=True this trains for 0 optimizer steps and "
                    "would produce a fake 'completed' run — refusing to prepare this run"
                )
            elif train_samples_per_epoch % effective != 0:
                warnings.append(
                    f"train samples per epoch ({train_samples_per_epoch}) is not divisible by the effective "
                    f"batch size ({effective}); drop_last=True will drop "
                    f"{train_samples_per_epoch % effective} sample(s) every epoch"
                )

        # Dataset identity guard: the current formal_book_spine_sam3_dataset split is
        # SAM3's own machine pre-annotation (see data_manifests/dataset_identity_registry.json
        # and docs/E3_DATASET_IDENTITY_ERRATUM.md) — not human-reviewed GT. Look it up by
        # resolved annotation path (never by filename), and refuse formal-mode or
        # multi-epoch runs against a non-reviewed dataset. Unmatched datasets are treated
        # as NOT reviewed (fail-safe default), not silently trusted.
        dataset_identity_obj = resolve_dataset_identity(
            resolved_paths.get("train_annotations"), resolved_paths.get("val_annotations")
        )
        dataset_identity = dataset_identity_obj.to_dict()
        if dataset_identity_obj.warning:
            warnings.append(f"dataset identity: {dataset_identity_obj.warning}")
        errors.extend(
            f"dataset identity guard: {reason}"
            for reason in validate_training_mode_against_identity(
                training_mode, resolved_max_epochs, dataset_identity_obj
            )
        )

        # SAM301 patch guard (preflight side): a launchable run (runtime YAML +
        # token) must never be prepared while the trainer is not exactly the
        # expected patched hash — fail closed before anything is written.
        if prepare_runtime:
            patch_guard_error = verify_patched_for_training()
            if patch_guard_error:
                errors.append(f"sam301 patch guard: {patch_guard_error}")

        if prepare_runtime and run_dir.exists() and any(run_dir.iterdir()):
            errors.append(f"output directory already exists and is non-empty: {run_dir}")
        if prepare_runtime and not errors and collect_import_metadata:
            import_guard_result = run_sam3_import_guard(
                conda_env=conda_env,
                cwd=BOOK_ROOT,
                env=effective_env,
            )
            if not import_guard_result.get("ok"):
                errors.append(f"sam3 import guard failed: {import_guard_result.get('error')}")
        if prepare_runtime and not errors:
            write_runtime_yaml(
                base_config,
                runtime_config_path,
                resolved_paths,
                run_dir,
                category_id=coco_category_id,
                training_prompt=resolved_training_prompt,
                task_slug=task_slug_from_name(resolved_training_prompt or coco_category_name),
                max_epochs=validated_max_epochs,
                train_batch_size=validated_train_batch_size,
                gradient_accumulation_steps=validated_gradient_accumulation_steps,
                learning_rate=validated_learning_rate,
                num_workers=validated_num_workers,
                online_augmentation=resolved_augmentation,
            )
            distributed_port = allocate_distributed_port(DEFAULT_DISTRIBUTED_MASTER_ADDR)
            distributed_metadata = configure_runtime_distributed_port(
                runtime_config_path,
                distributed_port,
                master_addr=DEFAULT_DISTRIBUTED_MASTER_ADDR,
            )
            command_config = runtime_config_path
            if collect_import_metadata:
                hydra_validation_result = run_hydra_config_validation(
                    runtime_config_path,
                    conda_env=conda_env,
                    cwd=BOOK_ROOT,
                    env=effective_env,
                )
                if not hydra_validation_result.get("ok"):
                    errors.append(f"hydra config validation failed: {hydra_validation_result.get('error')}")
            if not errors:
                training_provenance = collect_training_provenance(
                    runtime_config_path=runtime_config_path,
                    sam3_import_path=import_guard_result.get("sam3") if import_guard_result else None,
                    python_executable=import_guard_result.get("python") if import_guard_result else None,
                    distributed=distributed_metadata,
                )

    # train.py resolves -c as a Hydra config name inside pkg://sam3.train, so the
    # per-run runtime YAML (which must stay in the run directory) is launched via
    # the book01 wrapper, which initialize_config_dir's the YAML's own directory
    # and then calls the official sam3.train.train.main().
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(DEFAULT_TRAINING_LAUNCHER),
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
            "conda_environment": conda_env,
            "expected_sam3_root": str(EXPECTED_SAM3_ROOT),
            "expected_sam3_package_dir": str(EXPECTED_SAM3_PACKAGE_DIR),
            "resolved_sam3_import_path": import_guard_result.get("sam3") if import_guard_result else None,
            "sam3_import_guard_ok": import_guard_result.get("ok") if import_guard_result else None,
            "hydra_validation_ok": hydra_validation_result.get("ok") if hydra_validation_result else None,
            "effective_pythonpath": effective_env.get("PYTHONPATH"),
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
            "requested_max_epochs": max_epochs,
            "resolved_max_epochs": resolved_max_epochs,
            "requested_train_batch_size": train_batch_size,
            "resolved_train_batch_size": resolved_train_batch_size,
            "requested_gradient_accumulation_steps": gradient_accumulation_steps,
            "resolved_gradient_accumulation_steps": resolved_gradient_accumulation_steps,
            "requested_learning_rate": learning_rate,
            "resolved_learning_rate": resolved_learning_rate,
            "requested_num_workers": num_workers,
            "resolved_num_workers": resolved_num_workers,
            "training_provenance": training_provenance,
            "distributed": distributed_metadata,
            "dataset_identity": dataset_identity,
            "training_mode": training_mode,
            "online_augmentation": resolved_augmentation.to_dict(),
        }
        (run_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
        training_config_summary = {
            "run_id": run_dir.name,
            "conda_environment": conda_env,
            "expected_sam3_root": str(EXPECTED_SAM3_ROOT),
            "expected_sam3_package_dir": str(EXPECTED_SAM3_PACKAGE_DIR),
            "resolved_sam3_import_path": import_guard_result.get("sam3") if import_guard_result else None,
            "sam3_import_guard_ok": import_guard_result.get("ok") if import_guard_result else None,
            "sam3_import_guard_error": import_guard_result.get("error") if import_guard_result else None,
            "hydra_validation_ok": hydra_validation_result.get("ok") if hydra_validation_result else None,
            "hydra_validation_error": hydra_validation_result.get("error") if hydra_validation_result else None,
            "effective_pythonpath": effective_env.get("PYTHONPATH"),
            "base_config_path": str(base_config),
            "runtime_config_path": str(runtime_config_path),
            "command": command,
            "max_epochs": resolved_max_epochs,
            "train_batch_size": resolved_train_batch_size,
            "gradient_accumulation_steps": resolved_gradient_accumulation_steps,
            "effective_batch_size": effective,
            "num_gpus": num_gpus,
            "learning_rate": resolved_learning_rate,
            "num_workers": resolved_num_workers,
            "initial_checkpoint": str(resolved_paths["initial_checkpoint"]),
            "output_directory": str(run_dir),
            "requested_training_prompt": training_prompt,
            "resolved_training_prompt": resolved_training_prompt,
            "training_provenance": training_provenance,
            "distributed": distributed_metadata,
            "dataset_identity": dataset_identity,
            "training_mode": training_mode,
            "online_augmentation": resolved_augmentation.to_dict(),
        }
        (run_dir / "training_config_summary.json").write_text(
            json.dumps(training_config_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "provenance.json").write_text(
            json.dumps(training_provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    return TrainingPreflight(
        base_config_path=base_config,
        runtime_config_path=runtime_config_path,
        config_exists=exists,
        is_default_authoritative_config=is_default,
        warning=warning,
        train_batch_size=resolved_train_batch_size,
        num_gpus=num_gpus,
        gradient_accumulation_steps=resolved_gradient_accumulation_steps,
        trainer_gradient_accumulation_steps=trainer_accumulation,
        effective_batch_size=effective,
        max_epochs=resolved_max_epochs,
        learning_rate=resolved_learning_rate,
        num_workers=resolved_num_workers,
        requested_max_epochs=max_epochs,
        requested_train_batch_size=train_batch_size,
        requested_gradient_accumulation_steps=gradient_accumulation_steps,
        requested_learning_rate=learning_rate,
        requested_num_workers=num_workers,
        unsupported_overrides=unsupported_overrides,
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
        conda_environment=conda_env,
        expected_sam3_root=str(EXPECTED_SAM3_ROOT),
        expected_sam3_package_dir=str(EXPECTED_SAM3_PACKAGE_DIR),
        resolved_sam3_import_path=import_guard_result.get("sam3") if import_guard_result else None,
        sam3_import_guard_ok=import_guard_result.get("ok") if import_guard_result else None,
        sam3_import_guard_error=import_guard_result.get("error") if import_guard_result else None,
        effective_pythonpath=effective_env.get("PYTHONPATH"),
        command=command,
        path_checks=path_checks,
        train_coco=train_coco,
        val_coco=val_coco,
        warnings=warnings,
        errors=errors,
        training_provenance=training_provenance,
        dataset_identity=dataset_identity,
        online_augmentation=resolved_augmentation.to_dict(),
    )


def preflight_as_dict(preflight: TrainingPreflight) -> dict[str, Any]:
    data = asdict(preflight)
    data["base_config_path"] = str(preflight.base_config_path)
    data["runtime_config_path"] = str(preflight.runtime_config_path) if preflight.runtime_config_path else None
    return data


def format_preflight(preflight: TrainingPreflight) -> str:
    return json.dumps(preflight_as_dict(preflight), ensure_ascii=False, indent=2)
