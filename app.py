from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from ui.cvat_page import build_cvat_tab
from ui.dataset_registry_page import build_dataset_registry_tab
from ui.history_page import build_history_tab
from ui.i18n import build_language_radio, wire_language_switch
from ui.inference_page import build_inference_tab
from ui.results_page import build_results_tab
from ui.checkpoint_evaluation_page import build_checkpoint_evaluation_tab
from ui.training_preflight_page import build_training_tab


def build_app() -> gr.Blocks:
    """Assemble the stage D1 local Web UI. Page logic lives in ui/*_page.py; this
    function only wires pages into tabs and must not contain business logic."""
    with gr.Blocks(title="SAM3 Fine-tuning Tools") as demo:
        lang_radio = build_language_radio()
        gr.Markdown(
            "# SAM3 Fine-tuning 工具\n"
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
            with gr.Tab("数据集登记"):
                build_dataset_registry_tab()
            with gr.Tab("Checkpoint 评估"):
                build_checkpoint_evaluation_tab()
        # Must run after every page is built: it snapshots all components once
        # and wires the top language radio to update their labels/texts.
        wire_language_switch(demo, lang_radio)
    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
