from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from ui.cvat_page import build_cvat_tab
from ui.history_page import build_history_tab
from ui.inference_page import build_inference_tab
from ui.results_page import build_results_tab
from ui.training_preflight_page import build_training_tab


def build_app() -> gr.Blocks:
    """Assemble the stage D1 local Web UI. Page logic lives in ui/*_page.py; this
    function only wires pages into tabs and must not contain business logic."""
    with gr.Blocks(title="Book Spine SAM3 Tools") as demo:
        gr.Markdown(
            "# Book Spine SAM3 工具\n"
            "阶段 D1 本地 Web UI：把现有命令行流程可视化，不重写推理/NMS/COCO/CVAT/训练预检逻辑。"
        )
        with gr.Tabs():
            with gr.Tab("推理任务配置"):
                build_inference_tab()
            with gr.Tab("历史运行记录"):
                build_history_tab()
            with gr.Tab("结果查看"):
                build_results_tab()
            with gr.Tab("CVAT 导出"):
                build_cvat_tab()
            with gr.Tab("训练预检"):
                build_training_tab()
    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
