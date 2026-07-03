from __future__ import annotations

import os
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from core.config import (
    BOOK_ROOT,
    DEFAULT_BOOK_SPINE_DATASET_ROOT,
    DEFAULT_BOOK_SPINE_FINETUNE_CONFIG,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_TRAINING_RUN_ROOT,
)
from core.sam301_patch import verify_patched_for_training
from core.training_runner import training_subprocess_env
from ui.training_process_manager import (
    finalize_training_summary,
    make_training_summary_callback,
    verify_sam3_import_for_training,
    training_process_manager,
    training_snapshot,
    validate_can_start_training,
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


def _empty_preflight_state() -> dict[str, Any]:
    return {"ok": False, "run_dir": None, "runtime_yaml": None, "checkpoint": None, "command": None,
            "train_images": None, "train_annotations": None, "val_images": None, "val_annotations": None,
            "num_gpus": None, "launch_token": None, "consumed": False}


def invalidate_preflight(_changed_value: Any = None) -> tuple[dict[str, Any], str]:
    """Bound to every stage-A input's .change() event: any parameter edit invalidates
    the last preflight result server-side, so a stale runtime YAML can never be used
    to start training even if the browser still shows the old preflight output."""
    return _empty_preflight_state(), "参数已修改，之前的预检结果已失效，请重新运行训练预检。"


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
        "launch_token": uuid.uuid4().hex,
        "consumed": False,
    }
    return result_text, state, "预检通过，可以启动训练。"


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
    _consumed_preflight_tokens.add(str(token))
    state["consumed"] = True
    state["ok"] = False
    return []


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
        guard = verify_sam3_import_for_training(env=env)
        if not guard["ok"]:
            yield (
                "ERROR: SAM3 import guard failed before trainer launch:\n"
                f"expected={guard['expected']}\nactual={guard.get('sam3')}\nerror={guard.get('error')}"
            ), "", format_json(guard), "{}"
            return
        try:
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
                ),
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
    )
    yield f"status={snap['status']} pid={snap['pid']}", snap["log"], format_json(snap), format_json(summary)


def stop_training() -> str:
    if not training_process_manager.is_running():
        return "no active training task to stop"
    training_process_manager.stop()
    return "stop requested"


def build_training_tab() -> None:
    gr.Markdown(BANNER)
    gr.Markdown(STAGE_A_NOTE)

    preflight_state = gr.State(_empty_preflight_state())

    gr.Markdown("### 阶段 A: 生成并验证训练配置")
    with gr.Row():
        config_path = gr.Textbox(label="authoritative config", value=str(DEFAULT_BOOK_SPINE_FINETUNE_CONFIG))
        checkpoint = gr.Textbox(label="initial checkpoint", value=str(DEFAULT_SAM3_CHECKPOINT))
    with gr.Row():
        train_images = gr.Textbox(label="train images", value=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "images"))
        train_annotations = gr.Textbox(label="train COCO", value=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "annotations.json"))
    with gr.Row():
        val_images = gr.Textbox(label="val images", value=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "images"))
        val_annotations = gr.Textbox(label="val COCO", value=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "annotations.json"))
    with gr.Row():
        training_prompt = gr.Textbox(label="training prompt (optional manual override)", value="book spine")
        output_root = gr.Textbox(label="output root", value=str(DEFAULT_TRAINING_RUN_ROOT))
    with gr.Row():
        max_epochs = gr.Textbox(label="max_epochs (留空=基础YAML)", value="")
        train_batch_size = gr.Textbox(label="train batch size (留空=基础YAML)", value="")
        gradient_accumulation_steps = gr.Textbox(label="gradient accumulation steps (留空=基础YAML)", value="")
    with gr.Row():
        learning_rate = gr.Textbox(label="learning rate (留空=基础YAML)", value="")
        num_workers = gr.Textbox(label="num_workers (留空=基础YAML, 只作用于训练集)", value="")
        num_gpus = gr.Textbox(label="num_gpus", value="1")

    preflight_btn = gr.Button("运行训练预检 (不会启动训练)", variant="primary")
    preflight_status = gr.Textbox(label="预检状态", interactive=False, value="尚未运行预检。")
    preflight_output = gr.Code(
        label="预检结果 (resolved paths / max_epochs / batch size / gradient accumulation / learning rate / "
        "num_workers / effective batch size / prompt / runtime YAML / 最终训练命令)",
        language="json",
    )

    preflight_inputs = [
        config_path, train_images, train_annotations, val_images, val_annotations, checkpoint,
        training_prompt, output_root, max_epochs, train_batch_size, gradient_accumulation_steps,
        learning_rate, num_workers, num_gpus,
    ]
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
        stop_btn = gr.Button("停止训练")

    training_status_box = gr.Textbox(label="训练任务状态", interactive=False)
    training_log_box = gr.Textbox(label="stdout / stderr", lines=20, interactive=False, autoscroll=True)
    training_monitor_box = gr.Code(
        label="监控信息 (run_id/pid/pgid/started_at/elapsed/exit_code/output_dir/checkpoint_dir/"
        "discovered checkpoints/epoch/loss/lr/gpu memory — 指标解析为 best-effort, 解析不到时显示 unavailable)",
        language="json",
    )
    training_summary_box = gr.Code(label="training_summary.json (训练结束后生成)", language="json")

    start_btn.click(
        fn=start_training,
        inputs=[preflight_state, confirm_checkbox],
        outputs=[training_status_box, training_log_box, training_monitor_box, training_summary_box],
    )
    stop_btn.click(fn=stop_training, inputs=[], outputs=[training_status_box])
