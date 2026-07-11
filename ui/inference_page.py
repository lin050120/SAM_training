from __future__ import annotations

import os
import shlex
import time
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from core.config import BOOK_ROOT, DEFAULT_CONDA_ENV, DEFAULT_SAM3_CHECKPOINT, SAM301_ROOT
from core.sam_model_registry import discover_inference_models, model_provenance, resolve_model_path
from ui.process_manager import inference_process_manager
from ui.ui_utils import (
    check_input_path,
    detect_cuda,
    format_json,
    is_finite_number,
    logger,
    parse_optional_positive_int,
)

RUN_UNIFIED_INFERENCE_SCRIPT = BOOK_ROOT / "scripts" / "run_unified_inference.py"


def discovered_model_choices() -> list[tuple[str, str]]:
    return [(item["display_name"], item["model_path"]) for item in discover_inference_models()]


def resolve_ui_model_path(source: str, discovered_path: str, model_dir: str, filename: str, absolute_path: str) -> str:
    try:
        return str(
            resolve_model_path(
                source,
                discovered_path=discovered_path,
                model_dir=model_dir,
                filename=filename,
                absolute_path=absolute_path,
            )
        )
    except Exception as exc:
        return f"ERROR: {exc}"


def sync_ui_model_fields(
    source: str,
    discovered_path: str,
    model_dir: str,
    filename: str,
    absolute_path: str,
) -> tuple[str, str, str, str]:
    """Return directory/name/full-path display values plus the resolved checkpoint.

    The resolved checkpoint textbox was already updated on source changes, but the
    auxiliary display fields stayed at their previous values. Keeping all four
    values derived from the same resolver avoids UI drift without changing the
    backend inference command.
    """
    resolved = resolve_ui_model_path(source, discovered_path, model_dir, filename, absolute_path)
    if resolved.startswith("ERROR:"):
        return model_dir, filename, absolute_path, resolved

    path = Path(resolved)
    return str(path.parent), path.name, str(path), resolved


def check_ui_model(source: str, discovered_path: str, model_dir: str, filename: str, absolute_path: str) -> tuple[str, str]:
    resolved = resolve_ui_model_path(source, discovered_path, model_dir, filename, absolute_path)
    if resolved.startswith("ERROR:"):
        return resolved, "{}"
    info = model_provenance(resolved, validate_load=True, device="cpu")
    status = "可用于推理" if info.get("validation_status") == "ok" else f"不可用: {info.get('validation_error')}"
    return status, format_json(info)


def build_inference_command(
    input_dir: str,
    checkpoint: str,
    prompt: str,
    device: str,
    score_threshold: float,
    confidence_threshold: float,
    dtype_mode: str,
    nms_iou_thresh: float,
    nms_metric: str,
    nms_mode: str,
    min_area: float,
    category_name: str,
    output_root: str,
    limit: Any,
    cuda_available: bool,
) -> tuple[list[str] | None, list[str], list[str]]:
    """Build the subprocess argument list for scripts/run_unified_inference.py.

    `limit` is the raw widget payload (None, "", "10", 10.0, ...) and is parsed here;
    empty means "process all images" and adds no --limit flag.

    Returns (command_or_none, errors, warnings). command is None when errors is non-empty:
    the caller must not silently fall back or start anything in that case.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if err := check_input_path(input_dir, must_exist=True, must_be_dir=True):
        errors.append(f"input_dir: {err}")
    if err := check_input_path(checkpoint, must_exist=True, must_be_dir=False):
        errors.append(f"checkpoint: {err}")
    if not prompt or not prompt.strip():
        errors.append("prompt must not be empty")
    if device not in {"cuda", "cpu"}:
        errors.append(f"unsupported device: {device}")
    if device == "cuda" and not cuda_available:
        errors.append(
            "device=cuda was requested but CUDA is not available in the UI process. "
            "Refusing to start (no silent CPU fallback). Launch the UI from a terminal "
            "with real GPU access, or explicitly choose device=cpu."
        )
    # Guard against NaN payloads from cleared gr.Number fields before range checks:
    # NaN fails every comparison, and int(NaN) raises, so it must be rejected explicitly.
    for field_name, field_value in [
        ("score_threshold", score_threshold),
        ("confidence_threshold", confidence_threshold),
        ("nms_iou_thresh", nms_iou_thresh),
        ("min_area", min_area),
    ]:
        if not is_finite_number(field_value):
            errors.append(f"{field_name} must be a number (field is empty or invalid)")
    if is_finite_number(score_threshold) and not (0.0 <= float(score_threshold) <= 1.0):
        errors.append("score_threshold must be within [0, 1]")
    if is_finite_number(nms_iou_thresh) and not (0.0 <= float(nms_iou_thresh) <= 1.0):
        errors.append("nms_iou_thresh must be within [0, 1]")
    if is_finite_number(min_area) and float(min_area) < 0:
        errors.append("min_area must be >= 0")
    limit_value, limit_error = parse_optional_positive_int(limit)
    if limit_error:
        errors.append(f"limit: {limit_error} (leave the field empty to process all images)")

    if not output_root or not output_root.strip():
        errors.append("output_root must not be empty")
    else:
        output_root_path = Path(output_root).expanduser()
        parent = output_root_path if output_root_path.exists() else output_root_path.parent
        if not parent.exists():
            errors.append(f"output_root parent does not exist: {parent}")
        elif not os.access(parent, os.W_OK):
            errors.append(f"output_root is not writable: {parent}")

    if errors:
        return None, errors, warnings

    command = [
        "conda",
        "run",
        "-n",
        DEFAULT_CONDA_ENV,
        "python",
        str(RUN_UNIFIED_INFERENCE_SCRIPT),
        "--input-dir",
        str(Path(input_dir).expanduser()),
        "--output-root",
        str(Path(output_root).expanduser()),
        "--model-path",
        str(Path(checkpoint).expanduser()),
        "--prompt",
        prompt,
        "--score-threshold",
        str(score_threshold),
        "--confidence-threshold",
        str(confidence_threshold),
        "--dtype-mode",
        dtype_mode,
        "--device",
        device,
        "--nms-iou-thresh",
        str(nms_iou_thresh),
        "--nms-metric",
        nms_metric,
        "--nms-mode",
        nms_mode,
        "--min-area",
        str(int(min_area)),
        "--category-name",
        category_name,
    ]
    if limit_value is not None:
        command += ["--limit", str(limit_value)]
    return command, errors, warnings


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SAM301_ROOT)
    return env


def run_inference(
    input_dir: str,
    checkpoint: str,
    prompt: str,
    device: str,
    score_threshold: float,
    confidence_threshold: float,
    dtype_mode: str,
    nms_iou_thresh: float,
    nms_metric: str,
    nms_mode: str,
    min_area: float,
    category_name: str,
    output_root: str,
    limit: Any,
    segmentation_format: str,
) -> Iterator[tuple[str, str, str]]:
    """Generator driving the run button: yields (command_text, log_text, status_text)."""
    cuda_status = detect_cuda()
    command, errors, warnings = build_inference_command(
        input_dir,
        checkpoint,
        prompt,
        device,
        score_threshold,
        confidence_threshold,
        dtype_mode,
        nms_iou_thresh,
        nms_metric,
        nms_mode,
        min_area,
        category_name,
        output_root,
        limit,
        cuda_status.available,
    )
    command_text = shlex.join(command) if command else "(command not generated: validation failed)"
    if errors:
        yield command_text, "", "VALIDATION FAILED:\n" + "\n".join(errors)
        return
    if inference_process_manager.is_running():
        yield command_text, "", "ERROR: another task is already running. Stop it first."
        return

    assert command is not None
    try:
        inference_process_manager.start(command, cwd=BOOK_ROOT, env=_subprocess_env())
    except Exception as exc:
        logger.exception("inference_start_failed")
        yield command_text, "", f"ERROR: failed to start process: {exc!r}"
        return

    status = "running"
    while True:
        log_text, state = inference_process_manager.snapshot()
        if not state.running:
            break
        yield command_text, log_text, status
        time.sleep(0.5)

    log_text, state = inference_process_manager.snapshot()
    if state.stopped_by_user:
        status = "stopped by user"
    elif state.returncode == 0:
        status = "completed"
    else:
        status = f"failed (returncode={state.returncode})"
    yield command_text, log_text, status

    if state.returncode == 0 and not state.stopped_by_user and segmentation_format in {"rle", "both"}:
        run_dir = _extract_run_dir_from_log(log_text)
        if run_dir is not None:
            try:
                from core.cvat_export import export_cvat_package

                report = export_cvat_package(run_dir, segmentation_format="rle")
                log_text = log_text + f"\n[UI] additional RLE CVAT export: {format_json(report)}"
                status = status + f"; RLE export ok={report.get('ok')}"
            except Exception as exc:
                logger.exception("post_run_rle_export_failed")
                log_text = log_text + f"\n[UI] RLE export failed: {exc!r}"
                status = status + "; RLE export failed"
    yield command_text, log_text, status


def _extract_run_dir_from_log(log_text: str) -> Path | None:
    for line in reversed(log_text.splitlines()):
        candidate = Path(line.strip())
        if candidate.exists() and (candidate / "manifest.json").exists():
            return candidate
    return None


def stop_inference() -> str:
    if not inference_process_manager.is_running():
        return "no active task to stop"
    inference_process_manager.stop()
    return "stop requested"


def build_inference_tab() -> None:
    gr.Markdown(
        "调用现有 `scripts/run_unified_inference.py`，不重写推理/NMS/COCO 逻辑。"
        "device=cuda 且 CUDA 不可用时会拒绝启动，不做静默 CPU 回退。"
    )
    cuda_status = detect_cuda()
    gr.Markdown(
        f"**CUDA 检测**: available={cuda_status.available}, "
        f"torch={cuda_status.torch_version}, device={cuda_status.device_name}\n\n{cuda_status.note}"
    )

    with gr.Row():
        input_dir = gr.Textbox(label="input image directory", value=str(BOOK_ROOT / "data" / "book_spine_sam3_dataset" / "test" / "images"))
        checkpoint = gr.Textbox(label="resolved model path", value=str(DEFAULT_SAM3_CHECKPOINT), interactive=False)
    gr.Markdown("### 模型权重选择")
    with gr.Row():
        model_source = gr.Radio(
            label="模型来源",
            choices=[
                ("默认原始 SAM3", "default"),
                ("已发现 inference models", "discovered"),
                ("手动输入目录和文件名", "manual_parts"),
                ("手动输入完整绝对路径", "manual_absolute"),
            ],
            value="default",
        )
        refresh_models_btn = gr.Button("刷新模型列表")
        check_model_btn = gr.Button("检查模型")
    with gr.Row():
        discovered_model = gr.Dropdown(label="已发现模型", choices=discovered_model_choices(), value=None)
        model_dir = gr.Textbox(label="模型目录", value=str(DEFAULT_SAM3_CHECKPOINT.parent))
        model_filename = gr.Textbox(label="模型文件名", value=DEFAULT_SAM3_CHECKPOINT.name)
    absolute_model_path = gr.Textbox(label="完整绝对路径", value=str(DEFAULT_SAM3_CHECKPOINT))
    model_status = gr.Textbox(label="模型检查结果", interactive=False)
    model_info = gr.Code(label="当前选中模型信息", language="json")
    with gr.Row():
        prompt = gr.Textbox(label="prompt", value="book spine")
        device = gr.Radio(label="device", choices=["cuda", "cpu"], value="cuda")
        category_name = gr.Textbox(label="category name", value="book_spine")
    with gr.Row():
        score_threshold = gr.Number(label="inference threshold (score)", value=0.3, minimum=0.0, maximum=1.0)
        confidence_threshold = gr.Number(label="processor confidence threshold", value=0.05, minimum=0.0, maximum=1.0)
        dtype_mode = gr.Dropdown(label="dtype mode", choices=["bf16", "fp16", "none"], value="bf16")
    with gr.Row():
        nms_iou_thresh = gr.Number(label="NMS threshold", value=0.5, minimum=0.0, maximum=1.0)
        nms_metric = gr.Dropdown(label="NMS metric", choices=["iou", "iomin"], value="iou")
        nms_mode = gr.Dropdown(label="NMS mode", choices=["suppress", "merge"], value="suppress")
        min_area = gr.Number(label="min area (px)", value=200, minimum=0)
    with gr.Row():
        output_root = gr.Textbox(label="output root", value=str(BOOK_ROOT / "runs"))
        # Textbox, not gr.Number: cleared Number fields can deliver NaN (and
        # precision=0 even crashes on NaN in Gradio 5.50), so the raw text is
        # parsed by parse_optional_positive_int() instead.
        limit = gr.Textbox(label="limit (留空 = 处理全部图片)", value="")
        segmentation_format = gr.Radio(label="segmentation format", choices=["polygon", "rle", "both"], value="polygon")

    with gr.Row():
        run_btn = gr.Button("开始运行", variant="primary")
        stop_btn = gr.Button("停止")

    command_box = gr.Code(label="最终命令", language="shell")
    status_box = gr.Textbox(label="任务状态", interactive=False)
    log_box = gr.Textbox(label="stdout / stderr", lines=20, interactive=False, autoscroll=True)

    run_btn.click(
        fn=run_inference,
        inputs=[
            input_dir,
            checkpoint,
            prompt,
            device,
            score_threshold,
            confidence_threshold,
            dtype_mode,
            nms_iou_thresh,
            nms_metric,
            nms_mode,
            min_area,
            category_name,
            output_root,
            limit,
            segmentation_format,
        ],
        outputs=[command_box, log_box, status_box],
    )
    stop_btn.click(fn=stop_inference, inputs=[], outputs=[status_box])

    def _refresh_models(source, directory, filename, absolute):
        choices = discovered_model_choices()
        selected = choices[0][1] if choices else None
        next_directory, next_filename, next_absolute, next_checkpoint = sync_ui_model_fields(
            source,
            selected,
            directory,
            filename,
            absolute,
        )
        return (
            gr.update(choices=choices, value=selected),
            next_directory,
            next_filename,
            next_absolute,
            next_checkpoint,
        )

    def _sync(source, discovered, directory, filename, absolute):
        return sync_ui_model_fields(source, discovered, directory, filename, absolute)

    for component in [model_source, discovered_model, model_dir, model_filename, absolute_model_path]:
        component.change(
            fn=_sync,
            inputs=[model_source, discovered_model, model_dir, model_filename, absolute_model_path],
            outputs=[model_dir, model_filename, absolute_model_path, checkpoint],
        )
    refresh_models_btn.click(
        fn=_refresh_models,
        inputs=[model_source, model_dir, model_filename, absolute_model_path],
        outputs=[discovered_model, model_dir, model_filename, absolute_model_path, checkpoint],
    )
    check_model_btn.click(
        fn=check_ui_model,
        inputs=[model_source, discovered_model, model_dir, model_filename, absolute_model_path],
        outputs=[model_status, model_info],
    )
