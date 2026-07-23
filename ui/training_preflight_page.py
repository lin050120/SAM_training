from __future__ import annotations

import os
import json
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
import pandas as pd

from core.config import (
    BOOK_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_BOOK_SPINE_FINETUNE_CONFIG,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_TRAINING_PROMPT,
    DEFAULT_TRAINING_RUN_ROOT,
)
from core.dataset_split_builder import DatasetBuildConfig, build_training_dataset
from core.online_augmentation import (
    LIGHT_CONFIG,
    OFF_CONFIG,
    ONLINE_AUGMENTATION_CUSTOM,
    ONLINE_AUGMENTATION_LIGHT,
    OnlineAugmentationConfig,
)
from core.sam301_patch import verify_patched_for_training
from core.sam301_patch import collect_training_provenance
from core.training_runner import (
    DEFAULT_DISTRIBUTED_MASTER_ADDR,
    allocate_distributed_port,
    configure_runtime_distributed_port,
    training_subprocess_env,
)
from ui.training_process_manager import (
    finalize_training_summary,
    inspect_resumable_training_run,
    list_resumable_training_runs,
    make_training_summary_callback,
    read_training_loss_curve,
    training_process_manager,
    training_snapshot,
    validate_can_start_training,
    validate_can_resume_training,
    verify_sam3_import_for_training,
)
from ui.ui_utils import (
    detect_cuda,
    format_json,
    logger,
    parse_optional_positive_float,
    parse_optional_positive_int,
)

BANNER = "## 必须先完成训练预检，预检通过后才能启动训练。"

STAGE_A_NOTE = (
    "留空的数值字段回退到权威基础 YAML 的值，不会写入 0、NaN 或空字符串。"
    "`learning_rate` 对应 `scratch.lr_transformer`（可训练的 transformer/decoder 学习率）；"
    "`scratch.lr_vision_backbone`/`scratch.lr_language_backbone` 按基础 YAML 设计固定冻结为 0.0，本页面不提供覆盖入口。"
    "`num_workers` 只覆盖训练集 dataloader（`scratch.num_train_workers`），验证集固定使用基础 YAML 的值（当前为 0，验证集很小）。"
)

STAGE_B_NOTE = "只有满足预检通过、runtime YAML/checkpoint/数据均存在、当前无其他活动训练任务、CUDA 可用且用户已勾选确认框时，才会真正启动训练子进程。"

_preflight_launch_lock = threading.Lock()
_consumed_preflight_tokens: set[str] = set()
LOSS_CURVE_COLUMNS = ["epoch", "loss", "split"]


def _parse_optional_nonnegative_float(value: Any) -> tuple[float | None, str | None]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None, None
    try:
        parsed = float(text)
    except ValueError:
        return None, f"must be a number, got {value!r}"
    if parsed < 0:
        return None, f"must be >= 0, got {parsed}"
    return parsed, None


def _parse_optional_nonnegative_int(value: Any) -> tuple[int | None, str | None]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None, None
    try:
        parsed_float = float(text)
    except ValueError:
        return None, f"must be an integer, got {value!r}"
    parsed = int(parsed_float)
    if parsed != parsed_float:
        return None, f"must be an integer, got {value!r}"
    if parsed < 0:
        return None, f"must be >= 0, got {parsed}"
    return parsed, None


def _preserve_dataset_split_outputs(status: str) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    return (status, *(gr.update() for _ in range(7)))


def build_dataset_split(
    annotation_pool_dir: str,
    test_dir: str,
    output_dir: str,
    category_name: str,
    val_ratio: str,
    seed: str,
    overwrite: bool,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    errors: list[str] = []
    if not annotation_pool_dir:
        errors.append("标注数据文件夹不能为空")
    if not test_dir:
        errors.append("test 数据文件夹不能为空")
    if not output_dir:
        errors.append("输出数据集目录不能为空")
    ratio_value, ratio_error = _parse_optional_nonnegative_float(val_ratio)
    if ratio_error:
        errors.append(f"val ratio: {ratio_error}")
    if ratio_value is None:
        ratio_value = 0.10
    if ratio_value < 0 or ratio_value >= 1:
        errors.append("val ratio 必须 >=0 且 <1")
    seed_value, seed_error = _parse_optional_nonnegative_int(seed)
    if seed_error:
        errors.append(f"seed: {seed_error}")
    if seed_value is None:
        seed_value = 42
    if errors:
        return _preserve_dataset_split_outputs("VALIDATION FAILED:\n" + "\n".join(errors))

    try:
        result = build_training_dataset(
            DatasetBuildConfig(
                annotation_pool_dir=Path(annotation_pool_dir).expanduser(),
                test_dir=Path(test_dir).expanduser(),
                output_dir=Path(output_dir).expanduser(),
                category_name=category_name or DEFAULT_TRAINING_PROMPT,
                val_ratio=float(ratio_value),
                seed=int(seed_value),
                overwrite=bool(overwrite),
            )
        )
    except Exception as exc:
        logger.exception("dataset_split_build_failed")
        return _preserve_dataset_split_outputs(f"ERROR: dataset split build failed: {exc!r}")

    paths = result["paths"]
    status = (
        "DATASET BUILD OK\n"
        + format_json(result)
        + "\n\n已将 train/val 路径填入下方训练预检输入框。test split 已完整来自 test 数据文件夹。"
    )
    return (
        status,
        paths["train_images"],
        paths["train_annotations"],
        paths["val_images"],
        paths["val_annotations"],
        paths["test_images"],
        paths["test_annotations"],
        result["category_name"],
    )


def _empty_preflight_state() -> dict[str, Any]:
    return {"ok": False, "run_dir": None, "runtime_yaml": None, "checkpoint": None, "command": None,
            "train_images": None, "train_annotations": None, "val_images": None, "val_annotations": None,
            "num_gpus": None, "launch_token": None, "consumed": False}


def invalidate_preflight(_changed_value: Any = None) -> tuple[dict[str, Any], str]:
    """Bound to every stage-A input's .change() event: any parameter edit invalidates
    the last preflight result server-side, so a stale runtime YAML can never be used
    to start training even if the browser still shows the old preflight output."""
    return _empty_preflight_state(), "参数已修改，之前的预检结果已失效，请重新运行训练预检。"


def augmentation_controls_for_preset(preset: str) -> tuple[Any, ...]:
    """Set fixed preset values or unlock the bounded custom controls."""
    if preset == ONLINE_AUGMENTATION_LIGHT:
        config = LIGHT_CONFIG
        interactive = False
    elif preset == ONLINE_AUGMENTATION_CUSTOM:
        config = LIGHT_CONFIG
        interactive = True
    else:
        config = OFF_CONFIG
        interactive = False
    values = [
        config.repeat_factor,
        config.affine_probability,
        config.rotation_min_degrees,
        config.rotation_max_degrees,
        config.scale_min,
        config.scale_max,
        config.translate_fraction,
        config.horizontal_flip_probability,
        config.color_jitter_probability,
        config.color_jitter_strength,
        config.motion_blur_probability,
    ]
    return tuple(gr.update(value=value, interactive=interactive) for value in values)


def run_training_preflight(
    config_path: str,
    train_images: str,
    train_annotations: str,
    val_images: str,
    val_annotations: str,
    checkpoint: str,
    training_prompt: str,
    output_root: str,
    max_epochs: str,
    train_batch_size: str,
    gradient_accumulation_steps: str,
    learning_rate: str,
    num_workers: str,
    num_gpus: str,
    training_mode: str = "smoke",
    augmentation_preset: str = "off",
    augmentation_repeat_factor: Any = 1,
    augmentation_affine_probability: Any = 0.0,
    augmentation_rotation_min_degrees: Any = 0.0,
    augmentation_rotation_max_degrees: Any = 0.0,
    augmentation_scale_min: Any = 1.0,
    augmentation_scale_max: Any = 1.0,
    augmentation_translate_fraction: Any = 0.0,
    augmentation_horizontal_flip_probability: Any = 0.0,
    augmentation_color_jitter_probability: Any = 0.0,
    augmentation_color_jitter_strength: Any = 0.0,
    augmentation_motion_blur_probability: Any = 0.0,
) -> tuple[str, dict[str, Any], str]:
    """Stage A click handler. Returns (result_json, preflight_state, status_text)."""
    parse_errors: list[str] = []
    parsed: dict[str, Any] = {}
    for field_name, raw, parser, kwargs in [
        ("max_epochs", max_epochs, parse_optional_positive_int, {}),
        ("train_batch_size", train_batch_size, parse_optional_positive_int, {}),
        ("gradient_accumulation_steps", gradient_accumulation_steps, parse_optional_positive_int, {}),
        # allow_zero: num_workers=0 (load in the main process) is a valid PyTorch
        # DataLoader setting, matching the base YAML's own scratch.num_val_workers=0.
        ("num_workers", num_workers, parse_optional_positive_int, {"allow_zero": True}),
        ("learning_rate", learning_rate, parse_optional_positive_float, {}),
    ]:
        value, error = parser(raw, **kwargs)
        if error:
            parse_errors.append(f"{field_name}: {error}")
        parsed[field_name] = value

    num_gpus_value, num_gpus_error = parse_optional_positive_int(num_gpus)
    if num_gpus_error:
        parse_errors.append(f"num_gpus: {num_gpus_error}")
    resolved_num_gpus = num_gpus_value if num_gpus_value is not None else 1

    if parse_errors:
        return (
            "VALIDATION FAILED (numeric field parsing):\n" + "\n".join(parse_errors),
            _empty_preflight_state(),
            "参数解析失败，未运行预检。",
        )

    try:
        from core.training_runner import format_preflight, inspect_training_config

        preflight = inspect_training_config(
            config_path=Path(config_path).expanduser(),
            num_gpus=resolved_num_gpus,
            initial_checkpoint=Path(checkpoint).expanduser() if checkpoint else None,
            train_images=Path(train_images).expanduser() if train_images else None,
            train_annotations=Path(train_annotations).expanduser() if train_annotations else None,
            val_images=Path(val_images).expanduser() if val_images else None,
            val_annotations=Path(val_annotations).expanduser() if val_annotations else None,
            training_prompt=training_prompt or None,
            max_epochs=parsed["max_epochs"],
            train_batch_size=parsed["train_batch_size"],
            gradient_accumulation_steps=parsed["gradient_accumulation_steps"],
            learning_rate=parsed["learning_rate"],
            num_workers=parsed["num_workers"],
            output_root=Path(output_root).expanduser() if output_root else DEFAULT_TRAINING_RUN_ROOT,
            prepare_runtime=True,
            collect_import_metadata=True,
            training_mode=training_mode,
            online_augmentation=OnlineAugmentationConfig(
                preset=augmentation_preset,
                repeat_factor=augmentation_repeat_factor,
                affine_probability=augmentation_affine_probability,
                rotation_min_degrees=augmentation_rotation_min_degrees,
                rotation_max_degrees=augmentation_rotation_max_degrees,
                scale_min=augmentation_scale_min,
                scale_max=augmentation_scale_max,
                translate_fraction=augmentation_translate_fraction,
                horizontal_flip_probability=augmentation_horizontal_flip_probability,
                color_jitter_probability=augmentation_color_jitter_probability,
                color_jitter_strength=augmentation_color_jitter_strength,
                motion_blur_probability=augmentation_motion_blur_probability,
            ),
        )
    except Exception as exc:
        logger.exception("training_preflight_failed")
        return (
            f"ERROR: training preflight failed: {exc!r}\n(full traceback is in the UI server log)",
            _empty_preflight_state(),
            "预检执行异常，未生成可用的训练配置。",
        )

    result_text = format_preflight(preflight)
    if preflight.errors:
        return result_text, _empty_preflight_state(), f"预检未通过，发现 {len(preflight.errors)} 个 error，不能启动训练。"

    identity = preflight.dataset_identity or {}
    if not identity.get("human_reviewed"):
        result_text = (
            "‼️ 数据身份警告 / DATASET IDENTITY WARNING ‼️\n"
            f"annotation_source={identity.get('annotation_source')!r}, "
            f"human_reviewed={identity.get('human_reviewed')}, "
            f"allowed_for_formal_training={identity.get('allowed_for_formal_training')}\n"
            "本次训练使用的是 SAM3 机器预标注数据，未经人工审核，仅可用于训练流程 smoke test，"
            "不构成正式模型效果证据。详见 docs/E3_DATASET_IDENTITY_ERRATUM.md。\n\n"
        ) + result_text

    state = {
        "ok": True,
        "run_dir": preflight.run_dir,
        "runtime_yaml": str(preflight.runtime_config_path) if preflight.runtime_config_path else None,
        "checkpoint": preflight.initial_checkpoint,
        "command": preflight.command,
        "train_images": preflight.train_images,
        "train_annotations": preflight.train_annotations,
        "val_images": preflight.val_images,
        "val_annotations": preflight.val_annotations,
        "num_gpus": preflight.num_gpus,
        "conda_environment": preflight.conda_environment,
        "expected_sam3_root": preflight.expected_sam3_root,
        "expected_sam3_package_dir": preflight.expected_sam3_package_dir,
        "resolved_sam3_import_path": preflight.resolved_sam3_import_path,
        "sam3_import_guard_ok": preflight.sam3_import_guard_ok,
        "effective_pythonpath": preflight.effective_pythonpath,
        "distributed": (preflight.training_provenance or {}).get("distributed"),
        "dataset_identity": identity,
        "online_augmentation": preflight.online_augmentation,
        "launch_token": uuid.uuid4().hex,
        "consumed": False,
    }
    status_text = "预检通过，可以启动训练。"
    if not identity.get("human_reviewed"):
        status_text += " [SMOKE — 数据未经人工审核，非正式训练结果]"
    return result_text, state, status_text


def _consume_preflight_for_launch(
    state: dict[str, Any],
    confirmed: bool,
) -> list[str]:
    """Atomically validate and consume one preflight launch token.

    The token is consumed before subprocess creation. If Popen later fails, the
    preflight still cannot be retried because the run directory may be in an
    uncertain partial-start state.
    """
    cuda = detect_cuda()
    token = state.get("launch_token")
    preflight_consumed = bool(state.get("consumed")) or not token or token in _consumed_preflight_tokens
    reasons = validate_can_start_training(
        preflight_ok=bool(state.get("ok")),
        run_dir=state.get("run_dir"),
        runtime_yaml=state.get("runtime_yaml"),
        checkpoint=state.get("checkpoint"),
        train_images=state.get("train_images"),
        train_annotations=state.get("train_annotations"),
        val_images=state.get("val_images"),
        val_annotations=state.get("val_annotations"),
        confirmed=confirmed,
        already_running=training_process_manager.is_running(),
        cuda_available=cuda.available,
        requested_num_gpus=int(state.get("num_gpus") or 1),
        cuda_device_count=cuda.device_count,
        preflight_consumed=preflight_consumed,
    )
    if not token:
        reasons.append("预检启动凭证缺失，请重新运行训练预检")
    # SAM301 patch guard (launcher side): re-check the trainer hash right before the
    # token would be consumed, so a file replaced after preflight is caught here and
    # the token is NOT burned.
    patch_guard_error = verify_patched_for_training()
    if patch_guard_error:
        reasons.append(f"sam301 patch guard: {patch_guard_error}")
    if reasons:
        return reasons
    try:
        master_port = allocate_distributed_port(DEFAULT_DISTRIBUTED_MASTER_ADDR)
        state["distributed"] = configure_runtime_distributed_port(
            Path(str(state["runtime_yaml"])),
            master_port,
            master_addr=DEFAULT_DISTRIBUTED_MASTER_ADDR,
        )
    except Exception as exc:
        return [f"distributed port allocation failed before token consumption: {exc!r}"]
    _consumed_preflight_tokens.add(str(token))
    state["consumed"] = True
    state["ok"] = False
    return []


def _write_launcher_provenance(run_dir: Path, training_provenance: dict[str, Any]) -> None:
    (run_dir / "provenance.json").write_text(
        format_json(training_provenance) + "\n",
        encoding="utf-8",
    )
    summary_path = run_dir / "training_config_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = {}
    summary["training_provenance"] = training_provenance
    summary["distributed"] = training_provenance.get("distributed")
    summary_path.write_text(format_json(summary) + "\n", encoding="utf-8")


def start_training(
    preflight_state: dict[str, Any] | None,
    confirmed: bool,
) -> Iterator[tuple[str, str, str, str]]:
    """Stage B click handler. Yields (status_text, log_text, monitor_json, summary_json)."""
    state = preflight_state or _empty_preflight_state()

    with _preflight_launch_lock:
        reasons = _consume_preflight_for_launch(state, confirmed)
        if reasons:
            yield "BLOCKED:\n" + "\n".join(reasons), "", "{}", "{}"
            return
        run_dir = Path(state["run_dir"])
        command = state["command"]
        env = training_subprocess_env(os.environ)
        distributed = state.get("distributed") or {}
        if distributed.get("master_addr") and distributed.get("master_port"):
            env["MASTER_ADDR"] = str(distributed["master_addr"])
            env["MASTER_PORT"] = str(distributed["master_port"])
        guard = verify_sam3_import_for_training(env=env)
        if not guard["ok"]:
            yield (
                "ERROR: SAM3 import guard failed before trainer launch:\n"
                f"expected={guard['expected']}\nactual={guard.get('sam3')}\nerror={guard.get('error')}"
            ), "", format_json(guard), "{}"
            return
        try:
            training_provenance = collect_training_provenance(
                runtime_config_path=state.get("runtime_yaml"),
                sam3_import_path=guard.get("sam3"),
                python_executable=guard.get("python"),
                distributed=distributed,
            )
            _write_launcher_provenance(run_dir, training_provenance)
        except Exception as exc:
            logger.exception("training_provenance_failed")
            yield f"ERROR: failed to collect training provenance: {exc!r}", "", "{}", "{}"
            return
        try:
            attempt_id = f"new-{uuid.uuid4().hex}"
            training_process_manager.start(
                command,
                cwd=BOOK_ROOT,
                env=env,
                on_finish=make_training_summary_callback(
                    run_dir,
                    command,
                    state.get("runtime_yaml"),
                    state.get("checkpoint"),
                    import_metadata=guard,
                    effective_pythonpath=env.get("PYTHONPATH"),
                    training_provenance=training_provenance,
                    distributed=distributed,
                    attempt_id=attempt_id,
                    launch_kind="new",
                ),
                metadata={
                    "run_dir": str(run_dir),
                    "attempt_id": attempt_id,
                    "launch_kind": "new",
                },
            )
        except Exception as exc:
            logger.exception("training_start_failed")
            yield f"ERROR: failed to start training process: {exc!r}", "", "{}", "{}"
            return

    while training_process_manager.is_running():
        snap = training_snapshot(run_dir)
        yield f"status={snap['status']} pid={snap['pid']}", snap["log"], format_json(snap), "{}"
        time.sleep(1.0)

    snap = training_snapshot(run_dir)
    summary = finalize_training_summary(
        run_dir,
        command,
        state.get("runtime_yaml"),
        state.get("checkpoint"),
        import_metadata=guard,
        effective_pythonpath=env.get("PYTHONPATH"),
        training_provenance=training_provenance,
        distributed=distributed,
        attempt_id=attempt_id,
        launch_kind="new",
    )
    yield f"status={snap['status']} pid={snap['pid']}", snap["log"], format_json(snap), format_json(summary)


def stop_training() -> str:
    if not training_process_manager.is_running():
        return "no active training task to stop"
    training_process_manager.stop(reason="cancelled")
    return "stop requested"


def pause_training() -> str:
    """Release the GPU and preserve the latest completed epoch for later resume."""
    if not training_process_manager.is_running():
        return "PAUSE BLOCKED: no active training task"
    _log, process_state = training_process_manager.snapshot()
    run_dir_value = process_state.metadata.get("run_dir")
    if not run_dir_value:
        return "PAUSE BLOCKED: active task has no trusted run directory metadata"
    run_dir = Path(run_dir_value).expanduser().resolve(strict=False)
    checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
    checkpoint_tmp = checkpoint.with_name(f"{checkpoint.name}.tmp")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        return (
            "PAUSE BLOCKED: 还没有完整的 checkpoints/checkpoint.pt。"
            "请等待至少一个 epoch 完成并保存 checkpoint 后再暂停。"
        )
    if checkpoint_tmp.exists():
        return "PAUSE BLOCKED: checkpoint.pt.tmp 存在，训练器正在保存 checkpoint，请稍后重试。"
    training_process_manager.stop(reason="paused")
    return (
        "paused: GPU 已释放；恢复时将从最近完整 checkpoint 继续。"
        "当前 epoch 中 checkpoint 之后的进度不会保留。"
    )


def current_training_loss_curve() -> pd.DataFrame:
    """Return the active or most recently finished run's epoch-level losses."""
    _log, state = training_process_manager.snapshot()
    run_dir_value = state.metadata.get("run_dir")
    points = read_training_loss_curve(
        Path(run_dir_value) if run_dir_value else None
    )
    return pd.DataFrame(points, columns=LOSS_CURVE_COLUMNS)


def resumable_run_details(run_dir: str | None) -> str:
    return format_json(inspect_resumable_training_run(run_dir))


def refresh_resumable_runs(current_run: str | None = None) -> tuple[Any, str]:
    choices = list_resumable_training_runs()
    selected = current_run if current_run in choices else (choices[0] if choices else None)
    detail = resumable_run_details(selected) if selected else format_json(
        {
            "resumable": False,
            "message": "没有发现含 checkpoints/checkpoint.pt 的 run。",
        }
    )
    return gr.update(choices=choices, value=selected), detail


def resume_training(
    run_dir_value: str | None,
    confirmed: bool,
) -> Iterator[tuple[str, str, str, str]]:
    """Resume an existing run; SAM3 auto-loads save_dir/checkpoint.pt."""
    with _preflight_launch_lock:
        candidate = inspect_resumable_training_run(run_dir_value)
        cuda = detect_cuda()
        reasons = validate_can_resume_training(
            candidate,
            confirmed=confirmed,
            already_running=training_process_manager.is_running(),
            cuda_available=cuda.available,
            cuda_device_count=cuda.device_count,
        )
        patch_guard_error = verify_patched_for_training()
        if patch_guard_error:
            reasons.append(f"sam301 patch guard: {patch_guard_error}")
        if reasons:
            yield "RESUME BLOCKED:\n" + "\n".join(reasons), "", format_json(candidate), "{}"
            return

        run_dir = Path(str(candidate["run_dir"]))
        runtime_yaml = Path(str(candidate["runtime_yaml"]))
        command = list(candidate["command"])
        resume_checkpoint = str(candidate["resume_checkpoint"])
        try:
            master_port = allocate_distributed_port(DEFAULT_DISTRIBUTED_MASTER_ADDR)
            distributed = configure_runtime_distributed_port(
                runtime_yaml,
                master_port,
                master_addr=DEFAULT_DISTRIBUTED_MASTER_ADDR,
            )
        except Exception as exc:
            yield f"RESUME BLOCKED: distributed port allocation failed: {exc!r}", "", format_json(candidate), "{}"
            return

        env = training_subprocess_env(os.environ)
        env["MASTER_ADDR"] = str(distributed["master_addr"])
        env["MASTER_PORT"] = str(distributed["master_port"])
        guard = verify_sam3_import_for_training(env=env)
        if not guard["ok"]:
            yield (
                "ERROR: SAM3 import guard failed before resume launch:\n"
                f"expected={guard['expected']}\nactual={guard.get('sam3')}\nerror={guard.get('error')}"
            ), "", format_json(guard), "{}"
            return
        try:
            training_provenance = collect_training_provenance(
                runtime_config_path=runtime_yaml,
                sam3_import_path=guard.get("sam3"),
                python_executable=guard.get("python"),
                distributed=distributed,
            )
            _write_launcher_provenance(run_dir, training_provenance)
        except Exception as exc:
            logger.exception("training_resume_provenance_failed")
            yield f"ERROR: failed to collect resume provenance: {exc!r}", "", "{}", "{}"
            return

        attempt_id = f"resume-{uuid.uuid4().hex}"
        try:
            training_process_manager.start(
                command,
                cwd=BOOK_ROOT,
                env=env,
                on_finish=make_training_summary_callback(
                    run_dir,
                    command,
                    str(runtime_yaml),
                    candidate.get("initial_checkpoint"),
                    import_metadata=guard,
                    effective_pythonpath=env.get("PYTHONPATH"),
                    training_provenance=training_provenance,
                    distributed=distributed,
                    attempt_id=attempt_id,
                    launch_kind="resume",
                    resume_from_checkpoint=resume_checkpoint,
                ),
                metadata={
                    "run_dir": str(run_dir),
                    "attempt_id": attempt_id,
                    "launch_kind": "resume",
                    "resume_from_checkpoint": resume_checkpoint,
                },
            )
        except Exception as exc:
            logger.exception("training_resume_start_failed")
            yield f"ERROR: failed to start resumed training process: {exc!r}", "", "{}", "{}"
            return

    while training_process_manager.is_running():
        snap = training_snapshot(run_dir)
        yield (
            f"status={snap['status']} pid={snap['pid']} resume_from={resume_checkpoint}",
            snap["log"],
            format_json(snap),
            "{}",
        )
        time.sleep(1.0)

    snap = training_snapshot(run_dir)
    summary = finalize_training_summary(
        run_dir,
        command,
        str(runtime_yaml),
        candidate.get("initial_checkpoint"),
        import_metadata=guard,
        effective_pythonpath=env.get("PYTHONPATH"),
        training_provenance=training_provenance,
        distributed=distributed,
        attempt_id=attempt_id,
        launch_kind="resume",
        resume_from_checkpoint=resume_checkpoint,
    )
    yield f"status={snap['status']} pid={snap['pid']}", snap["log"], format_json(snap), format_json(summary)


def build_training_tab() -> None:
    gr.Markdown(BANNER)
    gr.Markdown(STAGE_A_NOTE)
    gr.Markdown(
        "**数据身份**：预检会按当前 train/val COCO 路径查询 dataset registry。未注册或未经人工审核的"
        "数据集只能用于 smoke test，不构成正式微调效果证据。详见 `docs/E3_DATASET_IDENTITY_ERRATUM.md`。"
    )

    preflight_state = gr.State(_empty_preflight_state())

    gr.Markdown("### 数据集自动划分: 标注数据 → train/val，test 数据 → test")
    gr.Markdown(
        "输入两个已标注 COCO 数据文件夹：标注数据文件夹会随机切分为 train/val，"
        "test 数据文件夹会整体写入 test。默认 val ratio=0.10，约为标注数据总量的 1/10。"
    )
    with gr.Row():
        split_pool_dir = gr.Textbox(label="标注数据文件夹 (自动切 train/val)", value="")
        split_test_dir = gr.Textbox(label="test 数据文件夹 (全部进入 test)", value="")
    with gr.Row():
        split_output_dir = gr.Textbox(label="输出数据集目录", value=str(DEFAULT_DATASET_ROOT))
        split_category_name = gr.Textbox(label="category / training prompt", value=DEFAULT_TRAINING_PROMPT)
    with gr.Row():
        split_val_ratio = gr.Textbox(label="val ratio", value="0.10")
        split_seed = gr.Textbox(label="random seed", value="42")
        split_overwrite = gr.Checkbox(label="允许覆盖输出目录 (旧目录会先改名为 backup)", value=False)
    split_btn = gr.Button("自动生成 train/val/test 数据集")
    split_status = gr.Code(label="数据集划分结果", language="json")
    with gr.Row():
        generated_test_images = gr.Textbox(label="生成的 test images", interactive=False)
        generated_test_annotations = gr.Textbox(label="生成的 test COCO", interactive=False)

    gr.Markdown("### 阶段 A: 生成并验证训练配置")
    with gr.Row():
        config_path = gr.Textbox(label="authoritative config", value=str(DEFAULT_BOOK_SPINE_FINETUNE_CONFIG))
        checkpoint = gr.Textbox(label="initial checkpoint", value=str(DEFAULT_SAM3_CHECKPOINT))
    with gr.Row():
        train_images = gr.Textbox(label="train images", value=str(DEFAULT_DATASET_ROOT / "train" / "images"))
        train_annotations = gr.Textbox(label="train COCO", value=str(DEFAULT_DATASET_ROOT / "train" / "annotations.json"))
    with gr.Row():
        val_images = gr.Textbox(label="val images", value=str(DEFAULT_DATASET_ROOT / "val" / "images"))
        val_annotations = gr.Textbox(label="val COCO", value=str(DEFAULT_DATASET_ROOT / "val" / "annotations.json"))
    with gr.Row():
        training_prompt = gr.Textbox(label="training prompt (optional manual override)", value=DEFAULT_TRAINING_PROMPT)
        output_root = gr.Textbox(label="output root", value=str(DEFAULT_TRAINING_RUN_ROOT))
    with gr.Row():
        max_epochs = gr.Textbox(label="max_epochs (留空=基础YAML)", value="")
        train_batch_size = gr.Textbox(label="train batch size (留空=基础YAML)", value="")
        gradient_accumulation_steps = gr.Textbox(label="gradient accumulation steps (留空=基础YAML)", value="")
    with gr.Row():
        learning_rate = gr.Textbox(label="learning rate (留空=基础YAML)", value="")
        num_workers = gr.Textbox(label="num_workers (留空=基础YAML, 只作用于训练集)", value="")
        num_gpus = gr.Textbox(label="num_gpus", value="1")
    training_mode = gr.Radio(
        label="training mode",
        choices=[("smoke (max_epochs<=1, 未审核数据默认)", "smoke"), ("formal (需要人工审核 GT)", "formal")],
        value="smoke",
    )

    gr.Markdown("### 在线训练数据增强")
    augmentation_preset = gr.Radio(
        label="增强预设",
        choices=[("关闭", "off"), ("轻量", "light"), ("自定义", "custom")],
        value="off",
    )
    with gr.Accordion("增强参数", open=True):
        with gr.Row():
            augmentation_repeat_factor = gr.Number(
                label="每张源图每 epoch 的样本数",
                minimum=1,
                step=1,
                precision=0,
                value=1,
                interactive=False,
            )
            augmentation_affine_probability = gr.Slider(
                label="仿射变换概率",
                minimum=0,
                maximum=1,
                step=0.05,
                value=0,
                interactive=False,
            )
            augmentation_horizontal_flip_probability = gr.Slider(
                label="水平翻转概率",
                minimum=0,
                maximum=1,
                step=0.05,
                value=0,
                interactive=False,
            )
        with gr.Row():
            augmentation_rotation_min_degrees = gr.Slider(
                label="最小旋转角度",
                minimum=-180,
                maximum=180,
                step=1,
                value=0,
                interactive=False,
            )
            augmentation_rotation_max_degrees = gr.Slider(
                label="最大旋转角度",
                minimum=-180,
                maximum=180,
                step=1,
                value=0,
                interactive=False,
            )
            augmentation_translate_fraction = gr.Slider(
                label="最大平移比例",
                minimum=0,
                maximum=0.5,
                step=0.01,
                value=0,
                interactive=False,
            )
        with gr.Row():
            augmentation_scale_min = gr.Slider(
                label="最小缩放倍数",
                minimum=0.5,
                maximum=2,
                step=0.01,
                value=1,
                interactive=False,
            )
            augmentation_scale_max = gr.Slider(
                label="最大缩放倍数",
                minimum=0.5,
                maximum=2,
                step=0.01,
                value=1,
                interactive=False,
            )
        with gr.Row():
            augmentation_color_jitter_probability = gr.Slider(
                label="颜色抖动概率",
                minimum=0,
                maximum=1,
                step=0.05,
                value=0,
                interactive=False,
            )
            augmentation_color_jitter_strength = gr.Slider(
                label="颜色抖动强度",
                minimum=0,
                maximum=0.5,
                step=0.01,
                value=0,
                interactive=False,
            )
            augmentation_motion_blur_probability = gr.Slider(
                label="运动模糊概率",
                minimum=0,
                maximum=1,
                step=0.05,
                value=0,
                interactive=False,
            )

    augmentation_controls = [
        augmentation_repeat_factor,
        augmentation_affine_probability,
        augmentation_rotation_min_degrees,
        augmentation_rotation_max_degrees,
        augmentation_scale_min,
        augmentation_scale_max,
        augmentation_translate_fraction,
        augmentation_horizontal_flip_probability,
        augmentation_color_jitter_probability,
        augmentation_color_jitter_strength,
        augmentation_motion_blur_probability,
    ]
    augmentation_preset.change(
        fn=augmentation_controls_for_preset,
        inputs=[augmentation_preset],
        outputs=augmentation_controls,
    )

    preflight_btn = gr.Button("运行训练预检 (不会启动训练)", variant="primary")
    preflight_status = gr.Textbox(label="预检状态", interactive=False, value="尚未运行预检。")
    preflight_output = gr.Code(
        label="预检结果 (resolved paths / max_epochs / batch size / gradient accumulation / learning rate / "
        "num_workers / effective batch size / prompt / online augmentation / runtime YAML / 最终训练命令)",
        language="json",
    )

    preflight_inputs = [
        config_path, train_images, train_annotations, val_images, val_annotations, checkpoint,
        training_prompt, output_root, max_epochs, train_batch_size, gradient_accumulation_steps,
        learning_rate, num_workers, num_gpus, training_mode, augmentation_preset,
        *augmentation_controls,
    ]
    split_btn.click(
        fn=build_dataset_split,
        inputs=[
            split_pool_dir,
            split_test_dir,
            split_output_dir,
            split_category_name,
            split_val_ratio,
            split_seed,
            split_overwrite,
        ],
        outputs=[
            split_status,
            train_images,
            train_annotations,
            val_images,
            val_annotations,
            generated_test_images,
            generated_test_annotations,
            training_prompt,
        ],
    )
    preflight_btn.click(
        fn=run_training_preflight,
        inputs=preflight_inputs,
        outputs=[preflight_output, preflight_state, preflight_status],
    )
    # Any parameter edit invalidates the stored preflight result, so a stale runtime
    # YAML can never be used to launch training after the form has moved on.
    for component in preflight_inputs:
        component.change(fn=invalidate_preflight, inputs=[component], outputs=[preflight_state, preflight_status])

    gr.Markdown("### 阶段 B: 启动训练")
    gr.Markdown(STAGE_B_NOTE)
    confirm_checkbox = gr.Checkbox(label="我确认这将启动 GPU 训练任务。", value=False)
    with gr.Row():
        start_btn = gr.Button("启动训练", variant="stop")
        pause_btn = gr.Button("暂停训练并释放显存")
        stop_btn = gr.Button("停止训练")

    training_status_box = gr.Textbox(label="训练任务状态", interactive=False)
    training_log_box = gr.Textbox(label="stdout / stderr", lines=20, interactive=False, autoscroll=True)
    training_monitor_box = gr.Code(
        label="监控信息 (run_id/pid/pgid/started_at/elapsed/exit_code/output_dir/checkpoint_dir/"
        "discovered checkpoints/epoch/loss/lr/gpu memory — 指标解析为 best-effort, 解析不到时显示 unavailable)",
        language="json",
    )
    training_summary_box = gr.Code(label="training_summary.json (训练结束后生成)", language="json")
    training_loss_plot = gr.LinePlot(
        value=pd.DataFrame(columns=LOSS_CURVE_COLUMNS),
        x="epoch",
        y="loss",
        color="split",
        color_map={"train": "#18794e", "val": "#d1495b"},
        title="Train / Val Loss",
        x_title="Epoch",
        y_title="Loss",
        height=340,
        label="Train / Val Loss 曲线",
        show_fullscreen_button=True,
    )
    loss_curve_timer = gr.Timer(value=1.0, active=True)
    loss_curve_timer.tick(
        fn=current_training_loss_curve,
        inputs=[],
        outputs=[training_loss_plot],
        queue=False,
    )

    gr.Markdown("### 阶段 C: 从最近完整 Checkpoint 恢复")
    initial_resume_runs = list_resumable_training_runs()
    with gr.Row():
        resume_run_dropdown = gr.Dropdown(
            label="训练 run（含 checkpoint.pt）",
            choices=initial_resume_runs,
            value=initial_resume_runs[0] if initial_resume_runs else None,
            interactive=True,
        )
        refresh_resume_btn = gr.Button("刷新可恢复 run")
    resume_details_box = gr.Code(
        label="恢复预检信息",
        language="json",
        value=(
            resumable_run_details(initial_resume_runs[0])
            if initial_resume_runs
            else format_json({"resumable": False, "message": "没有发现含 checkpoint.pt 的 run。"})
        ),
    )
    resume_confirm_checkbox = gr.Checkbox(
        label="我确认将从最近完整 checkpoint 恢复；当前 epoch 未保存的进度会丢失。",
        value=False,
    )
    resume_btn = gr.Button("恢复训练", variant="primary")

    start_btn.click(
        fn=start_training,
        inputs=[preflight_state, confirm_checkbox],
        outputs=[training_status_box, training_log_box, training_monitor_box, training_summary_box],
    )
    pause_btn.click(fn=pause_training, inputs=[], outputs=[training_status_box])
    stop_btn.click(fn=stop_training, inputs=[], outputs=[training_status_box])
    resume_run_dropdown.change(
        fn=resumable_run_details,
        inputs=[resume_run_dropdown],
        outputs=[resume_details_box],
    )
    refresh_resume_btn.click(
        fn=refresh_resumable_runs,
        inputs=[resume_run_dropdown],
        outputs=[resume_run_dropdown, resume_details_box],
    )
    resume_btn.click(
        fn=resume_training,
        inputs=[resume_run_dropdown, resume_confirm_checkbox],
        outputs=[training_status_box, training_log_box, training_monitor_box, training_summary_box],
    )
