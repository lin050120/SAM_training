from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import DEFAULT_CONDA_ENV, EXPECTED_SAM3_INIT, EXPECTED_SAM3_PACKAGE_DIR, EXPECTED_SAM3_ROOT
from core.training_runner import run_sam3_import_guard, validate_training_run_path
from ui.process_manager import ProcessManager, ProcessState
from ui.ui_utils import logger

# A single module-level instance, distinct from ui.process_manager.inference_process_manager:
# at most one active training task is allowed, independent of any active inference task.
training_process_manager = ProcessManager()
_summary_lock = threading.Lock()
_finalized_summary_paths: set[str] = set()


def validate_can_start_training(
    preflight_ok: bool,
    run_dir: str | None,
    runtime_yaml: str | None,
    checkpoint: str | None,
    train_images: str | None,
    train_annotations: str | None,
    val_images: str | None,
    val_annotations: str | None,
    confirmed: bool,
    already_running: bool,
    cuda_available: bool,
    requested_num_gpus: int,
    cuda_device_count: int,
    preflight_consumed: bool = False,
) -> list[str]:
    """Every condition that must hold before a training subprocess may be spawned.

    Pure function (no Gradio, no subprocess) so it is directly unit-testable. Returns
    a list of human-readable blocking reasons; empty means training may start. This
    is the real, server-side gate — a disabled button in the browser is only a UX
    hint, this function is what actually decides.
    """
    reasons: list[str] = []
    if not preflight_ok:
        reasons.append("必须先完成训练预检且通过（预检结果没有 errors），或参数已修改后未重新预检")
    if preflight_consumed:
        reasons.append("这次训练预检已经被启动消费，请重新运行训练预检生成新的 run directory")
    if not run_dir or not Path(run_dir).exists():
        reasons.append(f"training run 目录不存在，请重新预检: {run_dir}")
    else:
        run_dir_path = Path(run_dir).expanduser().resolve(strict=False)
        path_error = validate_training_run_path(run_dir_path)
        if path_error:
            reasons.append(f"{path_error}: {run_dir}")
        elif (run_dir_path / "training_summary.json").exists():
            reasons.append(f"training run 已经有 training_summary.json，拒绝复用旧 run directory: {run_dir}")
        elif (run_dir_path / "checkpoints").exists() and any((run_dir_path / "checkpoints").iterdir()):
            reasons.append(f"training run 已经有 checkpoint 产物，拒绝复用旧 run directory: {run_dir}")
    if not runtime_yaml or not Path(runtime_yaml).exists():
        reasons.append(f"runtime YAML 不存在，请重新预检: {runtime_yaml}")
    elif run_dir:
        runtime_path = Path(runtime_yaml).expanduser().resolve(strict=False)
        run_dir_path = Path(run_dir).expanduser().resolve(strict=False)
        runtime_path_error = validate_training_run_path(runtime_path)
        if runtime_path_error:
            reasons.append(f"{runtime_path_error}: runtime YAML: {runtime_yaml}")
        try:
            runtime_path.relative_to(run_dir_path)
        except ValueError:
            reasons.append(f"runtime YAML must remain inside run directory: {runtime_yaml}")
    if not checkpoint or not Path(checkpoint).exists():
        reasons.append(f"checkpoint 不存在: {checkpoint}")
    for label, value in [
        ("train_images", train_images),
        ("train_annotations", train_annotations),
        ("val_images", val_images),
        ("val_annotations", val_annotations),
    ]:
        if not value or not Path(value).exists():
            reasons.append(f"{label} 不存在: {value}")
    if already_running:
        reasons.append("已有一个训练任务在运行，请先停止")
    if not confirmed:
        reasons.append("请勾选确认框: 我确认这将启动 GPU 训练任务。")
    if not cuda_available:
        reasons.append("CUDA 不可用，拒绝启动训练（训练固定使用 accelerator=cuda，不会回退 CPU）")
    elif requested_num_gpus > cuda_device_count:
        reasons.append(f"请求 num_gpus={requested_num_gpus}，但只检测到 {cuda_device_count} 个可用 GPU")
    return reasons


def _shutdown_training_process_manager() -> None:
    training_process_manager.shutdown()


if not globals().get("_TRAINING_SHUTDOWN_REGISTERED", False):
    atexit.register(_shutdown_training_process_manager)
    _TRAINING_SHUTDOWN_REGISTERED = True


_METRIC_PATTERNS = {
    "epoch": r"[Ee]poch[:\s]+(\d+)",
    "iteration": r"[Ii]ter(?:ation)?[:\s]+(\d+)",
    "loss": r"[Ll]oss[:\s]+([\d.eE+-]+)",
    "learning_rate": r"\b[Ll][Rr][:\s]+([\d.eE+-]+)",
    "gpu_memory": r"(?:[Mm]em(?:ory)?)[:\s]+([\d.]+\s*[MG]i?B)",
}


def parse_training_metrics(log_text: str) -> dict[str, str]:
    """Best-effort scrape of epoch/iteration/loss/lr/gpu memory from raw stdout.

    This has never been validated against a real SAM3 training run's actual log
    format (no real training has been run in this project yet, see
    docs/stage_e1_training_ui.md). It must never fabricate a value or raise: a field
    that cannot be found stays "unavailable", and any parsing exception is caught so
    a log-format surprise can never be mistaken for a training failure.
    """
    result: dict[str, str] = {key: "unavailable" for key in _METRIC_PATTERNS}
    if not log_text:
        return result
    try:
        tail = "\n".join(log_text.splitlines()[-200:])
        for key, pattern in _METRIC_PATTERNS.items():
            matches = re.findall(pattern, tail)
            if matches:
                result[key] = matches[-1]
    except Exception:
        logger.exception("training_metric_parse_failed")
    return result


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def training_status_label(state: ProcessState) -> str:
    if state.running:
        return "running"
    if state.returncode is None:
        return "idle"
    if state.stopped_by_user:
        return "cancelled"
    if state.returncode == 0:
        return "completed"
    return "failed"


def verify_sam3_import_for_training(env: dict[str, str] | None = None) -> dict[str, Any]:
    result = run_sam3_import_guard(conda_env=DEFAULT_CONDA_ENV, env=env)
    result["conda_environment"] = DEFAULT_CONDA_ENV
    result["expected_sam3_root"] = str(EXPECTED_SAM3_ROOT)
    result["expected_sam3_package_dir"] = str(EXPECTED_SAM3_PACKAGE_DIR)
    result["expected"] = str(EXPECTED_SAM3_INIT)
    result["effective_pythonpath"] = env.get("PYTHONPATH") if env else None
    return result


def training_snapshot(run_dir: Path | None) -> dict[str, Any]:
    """Everything the UI needs to render the monitoring panel for one poll tick."""
    log_text, state = training_process_manager.snapshot()
    status = training_status_label(state)
    elapsed = None
    if state.started_at is not None:
        end = state.finished_at if state.finished_at is not None else time.time()
        elapsed = end - state.started_at
    metrics = parse_training_metrics(log_text)
    checkpoint_files: list[str] = []
    if run_dir is not None:
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_files = sorted(str(p) for p in checkpoints_dir.iterdir() if p.is_file())
    return {
        "status": status,
        "pid": training_process_manager.pid,
        "pgid": training_process_manager.pgid,
        "started_at": _iso(state.started_at),
        "finished_at": _iso(state.finished_at),
        "elapsed_seconds": elapsed,
        "exit_code": state.returncode,
        "output_directory": str(run_dir) if run_dir else None,
        "checkpoint_directory": str(run_dir / "checkpoints") if run_dir else None,
        "discovered_checkpoint_files": checkpoint_files,
        "metrics": metrics,
        "command": state.command,
        "log": log_text,
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _load_existing_summary(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def finalize_training_summary(
    run_dir: Path,
    command: list[str],
    runtime_config_path: str | None,
    initial_checkpoint: str | None,
    log_text: str | None = None,
    state: ProcessState | None = None,
    import_metadata: dict[str, Any] | None = None,
    effective_pythonpath: str | None = None,
) -> dict[str, Any]:
    """Write training_summary.json for a training run that has stopped (any status).

    Only lists checkpoint files that actually exist on disk — never invents a
    checkpoint path just because training reportedly completed.
    """
    summary_path = run_dir / "training_summary.json"
    resolved_summary_path = str(summary_path.expanduser().resolve(strict=False))
    with _summary_lock:
        existing = _load_existing_summary(summary_path) if summary_path.exists() else None
        if existing is not None and resolved_summary_path in _finalized_summary_paths:
            return existing
        if existing is not None and resolved_summary_path not in _finalized_summary_paths:
            _finalized_summary_paths.add(resolved_summary_path)
            return existing

    if state is None:
        captured_log, state = training_process_manager.snapshot()
    else:
        captured_log = log_text if log_text is not None else ""
    status = training_status_label(state)
    checkpoints_dir = run_dir / "checkpoints"
    discovered = sorted(str(p) for p in checkpoints_dir.iterdir() if p.is_file()) if checkpoints_dir.exists() else []
    duration = None
    if state.started_at is not None and state.finished_at is not None:
        duration = state.finished_at - state.started_at

    warnings: list[str] = []
    errors: list[str] = []
    if status == "failed":
        errors.append(f"training process exited with non-zero code: {state.returncode}")
    if status == "completed" and not discovered:
        warnings.append("training reported exit code 0 but no checkpoint files were found in checkpoints/")

    summary = {
        "run_id": run_dir.name,
        "status": status,
        "command": command,
        "runtime_config": runtime_config_path,
        "start_time": _iso(state.started_at),
        "end_time": _iso(state.finished_at),
        "duration_seconds": duration,
        "exit_code": state.returncode,
        "conda_environment": DEFAULT_CONDA_ENV,
        "expected_sam3_root": str(EXPECTED_SAM3_ROOT),
        "expected_sam3_package_dir": str(EXPECTED_SAM3_PACKAGE_DIR),
        "resolved_sam3_import_path": import_metadata.get("sam3") if import_metadata else None,
        "sam3_import_guard_ok": import_metadata.get("ok") if import_metadata else None,
        "sam3_import_guard_error": import_metadata.get("error") if import_metadata else None,
        "effective_pythonpath": effective_pythonpath,
        "initial_checkpoint": initial_checkpoint,
        "output_directory": str(run_dir),
        "discovered_checkpoint_files": discovered,
        "stdout_stderr_tail": "\n".join((captured_log or "").splitlines()[-200:]),
        "warnings": warnings,
        "errors": errors,
    }
    with _summary_lock:
        existing = _load_existing_summary(summary_path) if summary_path.exists() else None
        if existing is not None:
            _finalized_summary_paths.add(resolved_summary_path)
            return existing
        _atomic_write_json(summary_path, summary)
        _finalized_summary_paths.add(resolved_summary_path)
    return summary


def make_training_summary_callback(
    run_dir: Path,
    command: list[str],
    runtime_config_path: str | None,
    initial_checkpoint: str | None,
    import_metadata: dict[str, Any] | None = None,
    effective_pythonpath: str | None = None,
):
    def _callback(log_text: str, state: ProcessState) -> None:
        finalize_training_summary(
            run_dir,
            command,
            runtime_config_path,
            initial_checkpoint,
            log_text=log_text,
            state=state,
            import_metadata=import_metadata,
            effective_pythonpath=effective_pythonpath,
        )

    return _callback
