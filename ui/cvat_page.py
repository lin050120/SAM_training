from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from core.config import BOOK_ROOT
from ui.results_page import list_run_ids
from ui.run_reader import find_nms_review_images, list_image_names, list_nms_removed_pairs
from ui.ui_utils import format_json, logger

INFERENCE_RUNS_ROOT = BOOK_ROOT / "runs" / "inference"

DISCLAIMER = """
**Polygon**: 可用于 CVAT 兼容流程；是 NMS mask 到外轮廓多边形的有损转换；不适合逐像素精确评价。

**RLE**: 与最终 NMS bool mask 逐像素一致；当前项目本地验证 19/19 exact（针对 `2026-07-02_12-28-40` 这个真实 run）；
CVAT 当前版本的**实际人工导入**仍待用户手动确认，本页面不会显示"RLE 已成功导入 CVAT"这类结论。

本页面只对已有 inference run 重新导出/校验 CVAT 包，不会重新运行 SAM3。
"""

_PAIR_SEPARATOR = "|"


def _pair_choices(run_id: str) -> list[tuple[str, str]]:
    """Build (label, encoded-value) dropdown choices from the run's recorded NMS pairs."""
    if not run_id:
        return []
    pairs = list_nms_removed_pairs(INFERENCE_RUNS_ROOT / run_id)
    choices: list[tuple[str, str]] = []
    for pair in pairs:
        file_name = pair.get("file_name")
        kept = pair.get("kept_source_id")
        removed = pair.get("removed_source_id")
        if file_name is None or kept is None or removed is None:
            continue
        label = f"{file_name}: kept {kept} vs removed {removed} (overlap={pair.get('overlap'):.4f})" \
            if isinstance(pair.get("overlap"), float) else f"{file_name}: kept {kept} vs removed {removed}"
        choices.append((label, _PAIR_SEPARATOR.join([str(file_name), str(kept), str(removed)])))
    return choices


def _decode_pair(encoded: str) -> tuple[str, int, int] | None:
    parts = (encoded or "").split(_PAIR_SEPARATOR)
    if len(parts) != 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return None


def load_existing_reviews(run_id: str) -> str:
    """Collect already-generated NMS review reports for every image in the run."""
    if not run_id:
        return "select a run"
    run_dir = INFERENCE_RUNS_ROOT / run_id
    reviews: list[dict[str, Any]] = []
    for file_name in list_image_names(run_dir):
        reviews.extend(find_nms_review_images(run_dir, file_name))
    if not reviews:
        return "no NMS review generated yet for this run"
    return format_json(reviews)


def run_cvat_export(
    run_id: str,
    segmentation_format: str,
    make_zip: bool,
    do_polygon_fidelity: bool,
    do_nms_review: bool,
    nms_pair: str,
) -> str:
    if not run_id:
        return "ERROR: select a run first"
    run_dir = INFERENCE_RUNS_ROOT / run_id
    if not run_dir.exists():
        return f"ERROR: run directory does not exist: {run_dir}"

    output: dict = {}
    try:
        from core.cvat_export import export_cvat_package, polygon_fidelity_report, write_nms_pair_review

        output["export"] = export_cvat_package(run_dir, make_zip=make_zip, segmentation_format=segmentation_format)
        if do_polygon_fidelity:
            output["polygon_fidelity"] = polygon_fidelity_report(run_dir)
        if do_nms_review:
            decoded = _decode_pair(nms_pair)
            if decoded is None:
                output["nms_review_error"] = (
                    "select a kept/removed pair from the dropdown (this run has no recorded NMS pairs "
                    "if the dropdown is empty)"
                )
            else:
                image_name, kept_source_id, removed_source_id = decoded
                output["nms_review"] = write_nms_pair_review(
                    run_dir,
                    image_name=image_name,
                    kept_source_id=kept_source_id,
                    removed_source_id=removed_source_id,
                )
    except Exception as exc:
        logger.exception("cvat_export_failed run_id=%s", run_id)
        return f"ERROR: CVAT export failed: {exc!r}\n(full traceback is in the UI server log)"

    return format_json(output)


def build_cvat_tab() -> None:
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        run_dropdown = gr.Dropdown(label="run", choices=list_run_ids())
        refresh_btn = gr.Button("刷新 run 列表")
    with gr.Row():
        segmentation_format = gr.Radio(label="segmentation format", choices=["polygon", "rle", "both"], value="both")
        make_zip = gr.Checkbox(label="生成 ZIP", value=False)
        do_polygon_fidelity = gr.Checkbox(label="计算 polygon fidelity", value=True)
    with gr.Row():
        do_nms_review = gr.Checkbox(label="生成 NMS pair review", value=False)
        nms_pair = gr.Dropdown(
            label="NMS kept/removed pair (自动来自 manifest，无需手填 ID)",
            choices=[],
        )

    existing_reviews_box = gr.Code(label="该 run 已有的 NMS review 信息", language="json")
    export_btn = gr.Button("导出 / 重新导出", variant="primary")
    output_box = gr.Code(label="导出结果 (images/annotations/errors/exact_match/IoU/zip_path)", language="json")

    def _on_run_selected(run_id: str):
        return gr.update(choices=_pair_choices(run_id), value=None), load_existing_reviews(run_id)

    refresh_btn.click(fn=lambda: gr.update(choices=list_run_ids()), inputs=[], outputs=[run_dropdown])
    run_dropdown.change(fn=_on_run_selected, inputs=[run_dropdown], outputs=[nms_pair, existing_reviews_box])
    export_btn.click(
        fn=run_cvat_export,
        inputs=[
            run_dropdown,
            segmentation_format,
            make_zip,
            do_polygon_fidelity,
            do_nms_review,
            nms_pair,
        ],
        outputs=[output_box],
    )
