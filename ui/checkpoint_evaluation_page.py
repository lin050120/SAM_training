"""Checkpoint evaluation tab: rank a training run's checkpoints on the registered
human-corrected validation split and show/select the best one.

The heavy work runs as a subprocess (scripts/evaluate_sam3_checkpoints.py) through a
dedicated ProcessManager instance — same background mechanism as training — so the
web request never blocks until timeout, logs stream into the page, and only one
evaluation can run at a time (server-side guard, not just a disabled button).
"""

from __future__ import annotations

import atexit
import json
import time
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from core.config import BOOK_ROOT, DEFAULT_CONDA_ENV, DEFAULT_TRAINING_RUN_ROOT
from core.dataset_identity import resolve_validation_identity
from ui.process_manager import ProcessManager
from ui.ui_utils import format_json, logger

EVALUATE_SCRIPT = BOOK_ROOT / "scripts" / "evaluate_sam3_checkpoints.py"
EXPORT_SCRIPT = BOOK_ROOT / "scripts" / "export_sam3_inference_checkpoint.py"

evaluation_process_manager = ProcessManager()

if not globals().get("_EVALUATION_SHUTDOWN_REGISTERED", False):
    atexit.register(lambda: evaluation_process_manager.shutdown())
    _EVALUATION_SHUTDOWN_REGISTERED = True

RANKING_HEADERS = [
    "Checkpoint", "Epoch", "Mean IoU", "Boundary F1", "Recall@0.5", "Miss Rate",
    "FP/Image", "Area Ratio", "状态",
]


def list_runs_with_checkpoints() -> list[str]:
    runs = []
    root = DEFAULT_TRAINING_RUN_ROOT
    if root.is_dir():
        for run_dir in sorted(root.iterdir(), reverse=True):
            ckpt_dir = run_dir / "checkpoints"
            if ckpt_dir.is_dir() and any(ckpt_dir.glob("checkpoint*.pt")):
                runs.append(str(run_dir))
    return runs


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return "—"


def _metrics_rows(metrics_path: Path, best_name: str | None, mark_best: bool) -> list[list[Any]]:
    rows: list[list[Any]] = []
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for r in metrics:
            marker = "⭐ " if (mark_best and r.get("checkpoint_name") == best_name and not r.get("is_baseline")) else ""
            baseline_marker = "🏁 baseline " if r.get("is_baseline") else ""
            val_best_marker = "（val selected） " if ((not mark_best) and r.get("checkpoint_name") == best_name and not r.get("is_baseline")) else ""
            rows.append([
                f"{marker}{val_best_marker}{baseline_marker}{r.get('checkpoint_name')}",
                r.get("epoch") if r.get("epoch") is not None else "—",
                _fmt(r.get("mean_iou_all_gt")),
                _fmt(r.get("mean_boundary_f1_all_gt")),
                _fmt(r.get("recall_iou_50")),
                _fmt(r.get("miss_rate_iou_50")),
                _fmt(r.get("false_positive_per_image"), 2),
                _fmt(r.get("mean_area_ratio"), 3),
                r.get("evaluation_status", "?") + (f": {r['error_message']}" if r.get("error_message") else ""),
            ])
    except (OSError, json.JSONDecodeError):
        pass
    return rows


def load_evaluation_state(run_dir_str: str) -> tuple[str, list[list[Any]], list[list[Any]], str]:
    """Returns (status_markdown, validation_rows, test_rows, best_json_text)."""
    if not run_dir_str:
        return "未选择 run。", [], [], "{}"
    run_dir = Path(run_dir_str)
    evaluation_dir = run_dir / "evaluation"
    summary_path = evaluation_dir / "evaluation_summary.json"
    val_metrics_path = evaluation_dir / "validation" / "checkpoint_metrics.json"
    test_metrics_path = evaluation_dir / "test" / "checkpoint_metrics.json"
    best_path = evaluation_dir / "best_checkpoint.json"

    # dataset identity preview (works even before any evaluation ran)
    runtime_yaml = run_dir / "config" / "runtime_config.yaml"
    val_note = ""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(runtime_yaml)
        val_ann = OmegaConf.select(cfg, "trainer.data.val.dataset.ann_file")
        identity = resolve_validation_identity(val_ann) if val_ann else None
        if identity is not None:
            flag = "✅ 人工修正 GT" if identity.human_reviewed else "❌ 非人工修正（评估将被阻止）"
            val_note = f"验证集: `{val_ann}` — {identity.dataset_id or '未登记'} {flag}"
    except Exception:  # noqa: BLE001 - preview only
        val_note = "验证集: 无法从 runtime_config.yaml 读取"

    if not summary_path.is_file():
        return f"{val_note}\n\n**尚未评价** (evaluation/ 不存在或未完成)。", [], [], "{}"

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{val_note}\n\n评价结果读取失败: {exc}", [], [], "{}"

    best = summary.get("best") or {}
    validation = summary.get("validation") or {}
    test = summary.get("test") or {}
    improved = summary.get("finetuned_improved_over_baseline")
    improved_text = {True: "✅ 是", False: "❌ 否", None: "—(无 baseline 对比)"}[improved]
    smoke_tag = "（SMOKE — 非正式排名）" if summary.get("smoke") else ""
    status_md = (
        f"{val_note}\n\n"
        f"**评价状态**: {summary.get('status')} {smoke_tag}  \n"
        f"**checkpoint 数量**: {summary.get('checkpoint_count')}（完成 {summary.get('completed_count')}）  \n"
        f"**最佳 checkpoint**: `{best.get('checkpoint_name', '—')}` (epoch {best.get('epoch', '—')})  \n"
        f"**Mean IoU (all GT)**: {_fmt(best.get('mean_iou_all_gt'))}  \n"
        f"**优于原始 SAM3**: {improved_text}  \n"
        f"**结果目录**: `{summary.get('run_dir', run_dir_str)}/evaluation/`"
    )
    status_md += (
        f"\n\n### Validation\n"
        f"- 状态: `{validation.get('status', 'unknown')}`\n"
        f"- 数据: `{validation.get('dataset_path', '—')}`\n"
        f"- 原始数据: `{validation.get('files', {}).get('raw_predictions_dir', '—')}`\n"
        f"- 可视化: `{validation.get('files', {}).get('visualizations_dir', '—')}`\n\n"
        f"### Test（diagnostic only）\n"
        f"- 状态: `{test.get('status', 'unknown')}`\n"
        f"- 数据: `{test.get('dataset_path', '—')}`\n"
        f"- 原始数据: `{test.get('files', {}).get('raw_predictions_dir', '—')}`\n"
        f"- 可视化: `{test.get('files', {}).get('visualizations_dir', '—')}`\n"
        f"- ⚠️ best checkpoint 由 validation set 唯一决定。test 结果仅作独立对照和人工核实，"
        f"不参与模型选择；展示全部 checkpoint 后，该 test 不应再视为完全未查看的最终盲测集。"
    )
    if summary.get("reason"):
        status_md += f"  \n**原因**: {summary['reason']}"
    for warning in summary.get("guard_warnings", []):
        status_md += f"\n\n⚠️ {warning}"

    best_name = best.get("checkpoint_name")
    val_rows = _metrics_rows(val_metrics_path, best_name, mark_best=True)
    test_rows = _metrics_rows(test_metrics_path, best_name, mark_best=False)

    best_text = "{}"
    if best_path.is_file():
        try:
            best_text = best_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return status_md, val_rows, test_rows, best_text


def _run_evaluation(run_dir_str: str, extra_args: list[str]) -> Iterator[tuple[str, str]]:
    """Launch the evaluator subprocess and stream (status, log)."""
    if not run_dir_str:
        yield "BLOCKED: 未选择 run", ""
        return
    run_dir = Path(run_dir_str)
    if not (run_dir / "checkpoints").is_dir():
        yield f"BLOCKED: {run_dir} 下没有 checkpoints/ 目录", ""
        return
    if evaluation_process_manager.is_running():
        yield "BLOCKED: 已有一个评价任务在运行，请等待其结束", ""
        return
    command = [
        "conda", "run", "-n", DEFAULT_CONDA_ENV, "python", "-u", str(EVALUATE_SCRIPT),
        "--run-dir", str(run_dir), *extra_args,
    ]
    try:
        evaluation_process_manager.start(command, cwd=BOOK_ROOT)
    except Exception as exc:  # noqa: BLE001
        logger.exception("evaluation_start_failed")
        yield f"ERROR: 评价进程启动失败: {exc!r}", ""
        return
    while evaluation_process_manager.is_running():
        log_text, _state = evaluation_process_manager.snapshot()
        yield "评价运行中…（每秒刷新，一个 checkpoint 约需 1-2 分钟）", log_text
        time.sleep(1.0)
    log_text, state = evaluation_process_manager.snapshot()
    final = "评价完成" if state.returncode == 0 else f"评价失败 (exit={state.returncode})——详见日志与 evaluation.log"
    yield final, log_text


def start_evaluation(run_dir_str: str) -> Iterator[tuple[str, str]]:
    yield from _run_evaluation(run_dir_str, ["--split", "all", "--export-best"])


def start_re_evaluation(run_dir_str: str) -> Iterator[tuple[str, str]]:
    yield from _run_evaluation(run_dir_str, ["--split", "all", "--export-best", "--force"])


def start_validation_evaluation(run_dir_str: str) -> Iterator[tuple[str, str]]:
    yield from _run_evaluation(run_dir_str, ["--split", "validation", "--export-best"])


def start_test_evaluation(run_dir_str: str) -> Iterator[tuple[str, str]]:
    yield from _run_evaluation(run_dir_str, ["--split", "test"])


def start_validation_re_evaluation(run_dir_str: str) -> Iterator[tuple[str, str]]:
    yield from _run_evaluation(run_dir_str, ["--split", "validation", "--export-best", "--force"])


def start_test_re_evaluation(run_dir_str: str) -> Iterator[tuple[str, str]]:
    yield from _run_evaluation(run_dir_str, ["--split", "test", "--force"])


def export_best_model(run_dir_str: str) -> str:
    """Export the already-selected best trainer checkpoint to inference_best.pt."""
    if not run_dir_str:
        return "BLOCKED: 未选择 run"
    best_path = Path(run_dir_str) / "evaluation" / "best_checkpoint.json"
    if not best_path.is_file():
        return "BLOCKED: 还没有 best_checkpoint.json，请先运行评价"
    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"ERROR: best_checkpoint.json 读取失败: {exc}"
    if best.get("status") != "completed" or not best.get("best_checkpoint"):
        return f"BLOCKED: best 选择状态为 {best.get('status')}，无法导出"
    if evaluation_process_manager.is_running():
        return "BLOCKED: 已有一个评价/导出任务在运行"
    output = Path(run_dir_str) / "checkpoints" / "inference_best.pt"
    command = [
        "conda", "run", "-n", DEFAULT_CONDA_ENV, "python", "-u", str(EXPORT_SCRIPT),
        "--input", str(best["best_checkpoint"]), "--output", str(output), "--overwrite",
    ]
    try:
        evaluation_process_manager.start(command, cwd=BOOK_ROOT)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: 导出进程启动失败: {exc!r}"
    while evaluation_process_manager.is_running():
        time.sleep(0.5)
    log_text, state = evaluation_process_manager.snapshot()
    if state.returncode == 0:
        return f"导出完成: {output}\n{log_text[-500:]}"
    return f"导出失败 (exit={state.returncode}):\n{log_text[-1000:]}"


def build_checkpoint_evaluation_tab() -> None:
    gr.Markdown("## Checkpoint 评估：Validation 选最佳，Test 仅作诊断对照")
    gr.Markdown(
        "评价指标为 mask 级（Mean IoU / Boundary F1 / 漏检 / 误检 / 面积偏差），"
        "不使用 bbox AP，也不默认最后一个 epoch 最好。验证集必须已在 "
        "`data_manifests/dataset_identity_registry.json` 登记为人工修正 GT，否则评价会被服务端阻止。"
        "Test 结果不会改变 validation 选出的 best checkpoint。"
        "详见 `docs/CHECKPOINT_EVALUATION_CN.md`。"
    )
    run_dropdown = gr.Dropdown(
        label="训练 run（自动列出含 checkpoint 的 run）",
        choices=list_runs_with_checkpoints(),
        value=(list_runs_with_checkpoints() or [None])[0],
    )
    with gr.Row():
        refresh_btn = gr.Button("刷新状态/结果")
        evaluate_btn = gr.Button("依次评价 Validation 和 Test", variant="primary")
        re_evaluate_btn = gr.Button("重新评价 Validation 和 Test（忽略缓存）")
        export_btn = gr.Button("导出最佳推理模型")
    with gr.Row():
        val_btn = gr.Button("评价 Validation 全部 Checkpoint")
        val_force_btn = gr.Button("重新评价 Validation")
        test_btn = gr.Button("评价 Test 全部 Checkpoint")
        test_force_btn = gr.Button("重新评价 Test")

    status_md = gr.Markdown("选择 run 后点击“刷新状态/结果”。")
    gr.Markdown("### Validation 排名（唯一用于 best selection）")
    validation_table = gr.Dataframe(headers=RANKING_HEADERS, value=[], interactive=False, wrap=True)
    gr.Markdown("### Test 结果（diagnostic only，不参与 best selection）")
    test_table = gr.Dataframe(headers=RANKING_HEADERS, value=[], interactive=False, wrap=True)
    best_json_box = gr.Code(label="best_checkpoint.json", language="json")
    progress_box = gr.Textbox(label="评价进度", interactive=False)
    log_box = gr.Textbox(label="评价日志 (stdout/stderr)", lines=14, interactive=False, autoscroll=True)

    def _refresh(run_dir_str: str):
        status, val_rows, test_rows, best_text = load_evaluation_state(run_dir_str)
        return status, val_rows, test_rows, best_text, gr.update(choices=list_runs_with_checkpoints())

    refresh_btn.click(
        fn=_refresh, inputs=[run_dropdown], outputs=[status_md, validation_table, test_table, best_json_box, run_dropdown]
    )
    run_dropdown.change(
        fn=lambda r: load_evaluation_state(r), inputs=[run_dropdown],
        outputs=[status_md, validation_table, test_table, best_json_box],
    )
    evaluate_btn.click(fn=start_evaluation, inputs=[run_dropdown], outputs=[progress_box, log_box])
    re_evaluate_btn.click(fn=start_re_evaluation, inputs=[run_dropdown], outputs=[progress_box, log_box])
    val_btn.click(fn=start_validation_evaluation, inputs=[run_dropdown], outputs=[progress_box, log_box])
    val_force_btn.click(fn=start_validation_re_evaluation, inputs=[run_dropdown], outputs=[progress_box, log_box])
    test_btn.click(fn=start_test_evaluation, inputs=[run_dropdown], outputs=[progress_box, log_box])
    test_force_btn.click(fn=start_test_re_evaluation, inputs=[run_dropdown], outputs=[progress_box, log_box])
    export_btn.click(fn=export_best_model, inputs=[run_dropdown], outputs=[progress_box])
