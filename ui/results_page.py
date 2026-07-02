from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from core.config import BOOK_ROOT
from ui.run_reader import (
    find_nms_review_images,
    list_image_names,
    list_inference_runs,
    load_instance_table,
    read_manifest_row,
)
from ui.ui_utils import format_json, safe_read_json

INFERENCE_RUNS_ROOT = BOOK_ROOT / "runs" / "inference"


def list_run_ids() -> list[str]:
    return [s.run_id for s in list_inference_runs(INFERENCE_RUNS_ROOT)]


def list_images_for_run(run_id: str) -> list[str]:
    if not run_id:
        return []
    return list_image_names(INFERENCE_RUNS_ROOT / run_id)


def render_instance_rows(instance_rows: list[dict[str, Any]]) -> list[list[Any]]:
    """Instance dicts -> table rows. Older runs predate the source_instance_ids NPZ
    field; those cells show 'unavailable' instead of staying blank."""
    return [
        [
            r.get("annotation_id"),
            r["source_instance_id"] if r.get("source_instance_id") is not None else "unavailable",
            r.get("score"),
            r.get("bbox"),
            r.get("area"),
        ]
        for r in instance_rows
    ]


def load_result_view(run_id: str, file_name: str, view_mode: str) -> tuple[Any, Any, list, str, str, list, str]:
    """Returns (original_image, visual_image, instance_rows, manifest_json, run_config_json, nms_review_rows, errors_json)."""
    if not run_id or not file_name:
        empty_msg = "select a run and an image"
        return None, None, [], empty_msg, empty_msg, [], empty_msg

    run_dir = INFERENCE_RUNS_ROOT / run_id
    row = read_manifest_row(run_dir, file_name)
    original_path = run_dir / "input_images" / file_name
    visual_rel = None
    if row:
        visual_rel = row.get("nms_visualization") if view_mode == "nms" else row.get("raw_visualization")
    visual_path = (run_dir / visual_rel) if visual_rel else None

    instance_rows, instance_err = load_instance_table(run_dir, file_name)
    if instance_err:
        instance_rows = []

    run_config, _ = safe_read_json(run_dir / "run_config.json")
    nms_reviews = find_nms_review_images(run_dir, file_name)
    errors_data, _ = safe_read_json(run_dir / "errors.json")

    instance_table_rows = render_instance_rows(instance_rows)
    nms_review_rows = [[r.get("kept_instance_id"), r.get("removed_instance_id"), r.get("image_path")] for r in nms_reviews]

    return (
        str(original_path) if original_path.exists() else None,
        str(visual_path) if visual_path and visual_path.exists() else None,
        instance_table_rows,
        format_json(row) if row else "no manifest entry",
        format_json(run_config),
        nms_review_rows,
        format_json(errors_data),
    )


def load_extra_reports(run_id: str) -> tuple[str, str, str, str, str]:
    """Returns (acceptance_report, validation_report, polygon_fidelity, rle_validation, cvat_legacy_validation)."""
    if not run_id:
        empty = "select a run"
        return empty, empty, empty, empty, empty
    run_dir = INFERENCE_RUNS_ROOT / run_id
    acceptance, err1 = safe_read_json(run_dir / "acceptance_report.json")
    validation, err2 = safe_read_json(run_dir / "validation_report.json")
    polygon_fidelity, err3 = safe_read_json(run_dir / "cvat_export" / "polygon" / "polygon_fidelity_report.json")
    rle_validation, err4 = safe_read_json(run_dir / "cvat_export" / "rle" / "validation_report.json")
    legacy_validation, err5 = safe_read_json(run_dir / "cvat_export" / "validation_report.json")
    return (
        format_json(acceptance) if acceptance else (err1 or "not available"),
        format_json(validation) if validation else (err2 or "not available"),
        format_json(polygon_fidelity) if polygon_fidelity else (err3 or "not available (run CVAT export first)"),
        format_json(rle_validation) if rle_validation else (err4 or "not available (run CVAT export with rle/both first)"),
        format_json(legacy_validation) if legacy_validation else (err5 or "not available"),
    )


def build_results_tab() -> None:
    gr.Markdown(
        "选择 run 和图片查看原图、raw/NMS 可视化、实例信息、NMS 删除详情、manifest、run_config 和各类报告。"
        "本阶段不支持逐像素编辑/拆分/合并/手绘 mask。"
    )
    with gr.Row():
        run_dropdown = gr.Dropdown(label="run", choices=list_run_ids())
        refresh_runs_btn = gr.Button("刷新 run 列表")
        image_dropdown = gr.Dropdown(label="image")
        view_mode = gr.Radio(label="view", choices=["raw", "nms"], value="nms")

    with gr.Row():
        original_image = gr.Image(label="原图", type="filepath")
        visual_image = gr.Image(label="raw/NMS 可视化", type="filepath")

    instance_table = gr.Dataframe(
        headers=["annotation_id", "source_instance_id", "score", "bbox", "area"],
        label="实例信息 (来自 COCO + NMS npz)",
        interactive=False,
    )
    nms_review_table = gr.Dataframe(
        headers=["kept_instance_id", "removed_instance_id", "review_image_path"],
        label="NMS removed pairs",
        interactive=False,
    )

    with gr.Row():
        manifest_json = gr.Code(label="manifest row", language="json")
        run_config_json = gr.Code(label="run_config.json", language="json")
    errors_json = gr.Code(label="errors.json", language="json")

    gr.Markdown("### 附加报告")
    with gr.Row():
        acceptance_json = gr.Code(label="acceptance_report.json", language="json")
        validation_json = gr.Code(label="validation_report.json", language="json")
    with gr.Row():
        polygon_fidelity_json = gr.Code(label="polygon_fidelity_report.json", language="json")
        rle_validation_json = gr.Code(label="rle validation_report.json", language="json")
    legacy_validation_json = gr.Code(label="cvat_export/validation_report.json (legacy path)", language="json")

    def _refresh_runs():
        return gr.update(choices=list_run_ids())

    def _refresh_images(run_id: str):
        return gr.update(choices=list_images_for_run(run_id), value=None)

    refresh_runs_btn.click(fn=_refresh_runs, inputs=[], outputs=[run_dropdown])
    run_dropdown.change(fn=_refresh_images, inputs=[run_dropdown], outputs=[image_dropdown])
    run_dropdown.change(
        fn=load_extra_reports,
        inputs=[run_dropdown],
        outputs=[acceptance_json, validation_json, polygon_fidelity_json, rle_validation_json, legacy_validation_json],
    )

    for trigger in [image_dropdown.change, view_mode.change]:
        trigger(
            fn=load_result_view,
            inputs=[run_dropdown, image_dropdown, view_mode],
            outputs=[
                original_image,
                visual_image,
                instance_table,
                manifest_json,
                run_config_json,
                nms_review_table,
                errors_json,
            ],
        )
