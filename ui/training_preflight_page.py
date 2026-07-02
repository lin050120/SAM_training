from __future__ import annotations

from pathlib import Path

import gradio as gr

from core.config import (
    DEFAULT_BOOK_SPINE_DATASET_ROOT,
    DEFAULT_BOOK_SPINE_FINETUNE_CONFIG,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_TRAINING_RUN_ROOT,
)
from ui.ui_utils import format_json, logger

BANNER = "## 本页面只生成和验证训练配置，不会启动 SAM3 训练。"

KNOWN_LIMITATION = (
    "已知限制: `max_epochs` / `batch size` / `gradient accumulation` / `learning rate` / `num_workers` "
    "当前不能从本页面覆盖，因为 `core.training_runner.inspect_training_config()` 尚未开放这些参数的写入接口。"
    "下方只读展示权威 YAML 中解析出的 batch size / gradient accumulation / effective batch size；"
    "如需支持覆盖，应在 core 层新增参数，而不是在 UI 层另起一套训练预检逻辑。"
)


def run_training_preflight(
    config_path: str,
    train_images: str,
    train_annotations: str,
    val_images: str,
    val_annotations: str,
    checkpoint: str,
    training_prompt: str,
    output_root: str,
    num_gpus: float,
) -> str:
    try:
        from core.training_runner import format_preflight, inspect_training_config

        preflight = inspect_training_config(
            config_path=Path(config_path).expanduser(),
            num_gpus=int(num_gpus),
            initial_checkpoint=Path(checkpoint).expanduser() if checkpoint else None,
            train_images=Path(train_images).expanduser() if train_images else None,
            train_annotations=Path(train_annotations).expanduser() if train_annotations else None,
            val_images=Path(val_images).expanduser() if val_images else None,
            val_annotations=Path(val_annotations).expanduser() if val_annotations else None,
            training_prompt=training_prompt or None,
            output_root=Path(output_root).expanduser() if output_root else DEFAULT_TRAINING_RUN_ROOT,
            prepare_runtime=True,
        )
        return format_preflight(preflight)
    except Exception as exc:
        logger.exception("training_preflight_failed")
        return f"ERROR: training preflight failed: {exc!r}\n(see server logs for full traceback)"


def build_training_tab() -> None:
    gr.Markdown(BANNER)
    gr.Markdown(KNOWN_LIMITATION)

    with gr.Row():
        config_path = gr.Textbox(label="authoritative config", value=str(DEFAULT_BOOK_SPINE_FINETUNE_CONFIG))
        checkpoint = gr.Textbox(label="checkpoint", value=str(DEFAULT_SAM3_CHECKPOINT))
    with gr.Row():
        train_images = gr.Textbox(label="train images", value=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "images"))
        train_annotations = gr.Textbox(label="train COCO", value=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "annotations.json"))
    with gr.Row():
        val_images = gr.Textbox(label="val images", value=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "images"))
        val_annotations = gr.Textbox(label="val COCO", value=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "annotations.json"))
    with gr.Row():
        training_prompt = gr.Textbox(label="training prompt (optional manual override)", value="book spine")
        output_root = gr.Textbox(label="output root", value=str(DEFAULT_TRAINING_RUN_ROOT))
        num_gpus = gr.Number(label="num_gpus", value=1, minimum=1)

    preflight_btn = gr.Button("运行训练预检 (不会启动训练)", variant="primary")
    output_box = gr.Code(label="预检结果 (resolved paths / batch size / prompt / runtime YAML / 最终训练命令)", language="json")

    preflight_btn.click(
        fn=run_training_preflight,
        inputs=[
            config_path,
            train_images,
            train_annotations,
            val_images,
            val_annotations,
            checkpoint,
            training_prompt,
            output_root,
            num_gpus,
        ],
        outputs=[output_box],
    )
