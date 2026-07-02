from __future__ import annotations

from pathlib import Path

import gradio as gr

from core.config import BOOK_ROOT
from ui.run_reader import SUMMARY_TABLE_HEADERS, list_inference_runs, summary_table_rows

INFERENCE_RUNS_ROOT = BOOK_ROOT / "runs" / "inference"


def load_history_rows() -> list[list]:
    summaries = list_inference_runs(INFERENCE_RUNS_ROOT)
    return summary_table_rows(summaries)


def build_history_tab() -> None:
    gr.Markdown(f"读取 `{INFERENCE_RUNS_ROOT}` 下的推理 run。优先读取 run_config.json / manifest.json / errors.json / acceptance_report.json / cvat validation report；旧 run 缺字段时显示 unknown/unavailable，不会导致页面崩溃。")
    refresh_btn = gr.Button("刷新")
    table = gr.Dataframe(headers=SUMMARY_TABLE_HEADERS, value=load_history_rows(), interactive=False, wrap=True)
    refresh_btn.click(fn=load_history_rows, inputs=[], outputs=[table])
