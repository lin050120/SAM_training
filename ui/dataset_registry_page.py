from __future__ import annotations

from pathlib import Path

import gradio as gr

from core.config import BOOK_ROOT
from core.dataset_registry import build_dataset_identity_artifacts, register_dataset_identity
from ui.ui_utils import format_json, logger


def _fields(
    dataset_root: str,
    dataset_id: str,
    annotation_source: str,
    annotation_source_evidence: str,
    human_reviewed: bool,
    independently_corrected_gt: bool,
    allowed_for_formal_training: bool,
    allowed_for_model_evaluation: bool,
    intended_use: str,
) -> dict:
    return {
        "dataset_root": Path(dataset_root).expanduser(),
        "dataset_id": dataset_id,
        "annotation_source": annotation_source,
        "annotation_source_evidence": annotation_source_evidence,
        "human_reviewed": bool(human_reviewed),
        "independently_corrected_gt": bool(independently_corrected_gt),
        "allowed_for_formal_training": bool(allowed_for_formal_training),
        "allowed_for_model_evaluation": bool(allowed_for_model_evaluation),
        "intended_use": intended_use,
    }


def preview_dataset_registration(
    dataset_root: str,
    dataset_id: str,
    annotation_source: str,
    annotation_source_evidence: str,
    human_reviewed: bool,
    independently_corrected_gt: bool,
    allowed_for_formal_training: bool,
    allowed_for_model_evaluation: bool,
    intended_use: str,
) -> str:
    try:
        entry, dataset_manifest, split_manifest = build_dataset_identity_artifacts(
            **_fields(
                dataset_root,
                dataset_id,
                annotation_source,
                annotation_source_evidence,
                human_reviewed,
                independently_corrected_gt,
                allowed_for_formal_training,
                allowed_for_model_evaluation,
                intended_use,
            )
        )
        return format_json(
            {
                "ok": True,
                "will_update": "preview only; registry is not modified",
                "registry_entry": entry,
                "dataset_manifest": dataset_manifest,
                "split_manifest": split_manifest,
            }
        )
    except Exception as exc:
        logger.exception("dataset_registration_preview_failed")
        return f"ERROR: preview failed: {exc!r}"


def register_dataset_from_ui(
    dataset_root: str,
    dataset_id: str,
    annotation_source: str,
    annotation_source_evidence: str,
    human_reviewed: bool,
    independently_corrected_gt: bool,
    allowed_for_formal_training: bool,
    allowed_for_model_evaluation: bool,
    intended_use: str,
    overwrite_existing: bool,
) -> str:
    try:
        result = register_dataset_identity(
            **_fields(
                dataset_root,
                dataset_id,
                annotation_source,
                annotation_source_evidence,
                human_reviewed,
                independently_corrected_gt,
                allowed_for_formal_training,
                allowed_for_model_evaluation,
                intended_use,
            ),
            overwrite_existing=bool(overwrite_existing),
        )
        return format_json(result)
    except Exception as exc:
        logger.exception("dataset_registration_failed")
        return f"ERROR: registration failed: {exc!r}"


def build_dataset_registry_tab() -> None:
    gr.Markdown(
        "登记已经人工审核的数据集。登记后，训练预检会按 train/val COCO 路径匹配 registry；"
        "只有 `allowed_for_formal_training=true` 的登记数据集才能运行 formal 或多 epoch 训练。"
    )
    with gr.Row():
        dataset_root = gr.Textbox(
            label="dataset root",
            value=str(BOOK_ROOT / "data" / "cable_sam3_dataset"),
        )
        dataset_id = gr.Textbox(label="dataset_id", value="cable_human_corrected_v1")
    with gr.Row():
        annotation_source = gr.Dropdown(
            label="annotation source",
            choices=[
                "human_annotated",
                "sam3_preannotation_then_human_corrected",
                "other_human_reviewed",
            ],
            value="sam3_preannotation_then_human_corrected",
        )
        intended_use = gr.Textbox(
            label="intended use",
            value="formal_training_validation_and_diagnostic_test",
        )
    annotation_source_evidence = gr.Textbox(
        label="annotation source evidence",
        lines=3,
        value="All train/val/test annotations were manually reviewed and corrected before registration.",
    )
    with gr.Row():
        human_reviewed = gr.Checkbox(label="human reviewed", value=True)
        independently_corrected_gt = gr.Checkbox(label="independently corrected GT", value=True)
        allowed_for_formal_training = gr.Checkbox(label="allow formal training", value=True)
        allowed_for_model_evaluation = gr.Checkbox(label="allow final model evaluation", value=False)
    overwrite_existing = gr.Checkbox(
        label="覆盖同 dataset_id 的既有登记和 manifest",
        value=False,
    )
    with gr.Row():
        preview_btn = gr.Button("预览登记内容")
        register_btn = gr.Button("写入登记", variant="primary")
    output = gr.Code(label="登记预览 / 结果", language="json")

    common_inputs = [
        dataset_root,
        dataset_id,
        annotation_source,
        annotation_source_evidence,
        human_reviewed,
        independently_corrected_gt,
        allowed_for_formal_training,
        allowed_for_model_evaluation,
        intended_use,
    ]
    preview_btn.click(fn=preview_dataset_registration, inputs=common_inputs, outputs=[output])
    register_btn.click(
        fn=register_dataset_from_ui,
        inputs=[*common_inputs, overwrite_existing],
        outputs=[output],
    )
