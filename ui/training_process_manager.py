from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import (
    DEFAULT_CONDA_ENV,
    DEFAULT_TRAINING_LAUNCHER,
    DEFAULT_TRAINING_RUN_ROOT,
    EXPECTED_SAM3_INIT,
    EXPECTED_SAM3_PACKAGE_DIR,
    EXPECTED_SAM3_ROOT,
)
from core.dataset_identity import resolve_dataset_identity, validate_training_mode_against_identity
from core.sam301_patch import collect_training_provenance
from core.training_runner import run_sam3_import_guard, validate_training_run_path
from ui.process_manager import ProcessManager, ProcessState
from ui.ui_utils import logger

# A single module-level instance, distinct from ui.process_manager.inference_process_manager:
# at most one active training task is allowed, independent of any active inference task.
training_process_manager = ProcessManager()
_summary_lock = threading.Lock()


def _read_json_dict(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} must contain a JSON object"
    return value, None


def _path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _active_pids_referencing_runtime(runtime_yaml: Path) -> list[int]:
    """Find live Linux processes whose argv contains this exact runtime config."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    target = str(runtime_yaml.resolve(strict=False))
    matches: list[int] = []
    for proc_dir in proc_root.iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            argv = (proc_dir / "cmdline").read_bytes().split(b"\0")
            decoded = [item.decode("utf-8", errors="surrogateescape") for item in argv if item]
        except (OSError, PermissionError):
            continue
        if target in decoded:
            matches.append(int(proc_dir.name))
    return sorted(matches)


def inspect_resumable_training_run(run_dir: str | Path | None) -> dict[str, Any]:
    """Inspect one existing run without loading its multi-gigabyte checkpoint."""
    result: dict[str, Any] = {
        "resumable": False,
        "run_dir": str(run_dir) if run_dir else None,
        "runtime_yaml": None,
        "resume_checkpoint": None,
        "checkpoint_size_bytes": None,
        "checkpoint_modified_at": None,
        "max_epochs": None,
        "num_gpus": None,
        "training_mode": None,
        "training_prompt": None,
        "initial_checkpoint": None,
        "bpe_path": None,
        "previous_status": None,
        "active_process_pids": [],
        "command": None,
        "dataset_identity": None,
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]
    if not run_dir:
        errors.append("未选择 training run")
        return result

    resolved_run = Path(run_dir).expanduser().resolve(strict=False)
    result["run_dir"] = str(resolved_run)
    path_error = validate_training_run_path(resolved_run)
    if path_error:
        errors.append(f"{path_error}: {resolved_run}")
        return result
    if not resolved_run.is_dir():
        errors.append(f"training run 目录不存在: {resolved_run}")
        return result

    runtime_yaml = resolved_run / "config" / "runtime_config.yaml"
    checkpoint = resolved_run / "checkpoints" / "checkpoint.pt"
    checkpoint_tmp = checkpoint.with_name(f"{checkpoint.name}.tmp")
    result["runtime_yaml"] = str(runtime_yaml)
    result["resume_checkpoint"] = str(checkpoint)
    if not _path_is_inside(runtime_yaml, resolved_run) or not runtime_yaml.is_file():
        errors.append(f"runtime YAML 不存在或不在 run directory 内: {runtime_yaml}")
    else:
        active_pids = _active_pids_referencing_runtime(runtime_yaml)
        result["active_process_pids"] = active_pids
        if active_pids:
            errors.append(f"该 run 已被活动训练进程使用，拒绝重复启动: pids={active_pids}")
    if not _path_is_inside(checkpoint, resolved_run) or not checkpoint.is_file():
        errors.append(f"缺少最新恢复 checkpoint: {checkpoint}")
    else:
        stat = checkpoint.stat()
        result["checkpoint_size_bytes"] = stat.st_size
        result["checkpoint_modified_at"] = _iso(stat.st_mtime)
        if stat.st_size <= 0:
            errors.append(f"恢复 checkpoint 为空文件: {checkpoint}")
    if checkpoint_tmp.exists():
        warnings.append(
            f"发现残留的临时 checkpoint，将忽略并使用完整 checkpoint.pt 恢复: {checkpoint_tmp}"
        )

    config_summary, summary_error = _read_json_dict(resolved_run / "training_config_summary.json")
    if summary_error:
        errors.append(summary_error)
        config_summary = {}
    dataset_info, dataset_error = _read_json_dict(resolved_run / "dataset_info.json")
    if dataset_error:
        errors.append(dataset_error)
        dataset_info = {}

    training_summary_path = resolved_run / "training_summary.json"
    training_summary: dict[str, Any] = {}
    if training_summary_path.exists():
        training_summary, training_summary_error = _read_json_dict(training_summary_path)
        if training_summary_error:
            errors.append(training_summary_error)
            training_summary = {}
    previous_status = training_summary.get("status")
    result["previous_status"] = previous_status
    if previous_status == "completed":
        errors.append("该 run 已完成训练，不需要恢复；延长 max_epochs 应创建新的训练计划")

    try:
        num_gpus = int(config_summary.get("num_gpus"))
        if num_gpus <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("training_config_summary.json 缺少有效的 num_gpus")
        num_gpus = None
    try:
        max_epochs = int(config_summary.get("max_epochs"))
        if max_epochs <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("training_config_summary.json 缺少有效的 max_epochs")
        max_epochs = None
    result["num_gpus"] = num_gpus
    result["max_epochs"] = max_epochs
    result["training_mode"] = config_summary.get("training_mode") or dataset_info.get("training_mode")
    result["training_prompt"] = (
        config_summary.get("resolved_training_prompt") or dataset_info.get("resolved_training_prompt")
    )
    result["initial_checkpoint"] = (
        config_summary.get("initial_checkpoint") or dataset_info.get("initial_checkpoint")
    )
    result["bpe_path"] = dataset_info.get("bpe_path")
    for label in ("initial_checkpoint", "bpe_path"):
        value = result[label]
        if not value or not Path(str(value)).expanduser().is_file():
            errors.append(f"恢复训练所需的 {label} 不存在: {value}")

    data_paths = {
        key: dataset_info.get(key)
        for key in ("train_images", "train_annotations", "val_images", "val_annotations")
    }
    result["data_paths"] = data_paths
    for label, value in data_paths.items():
        if not value or not Path(str(value)).expanduser().exists():
            errors.append(f"恢复训练所需的 {label} 不存在: {value}")

    if data_paths.get("train_annotations") and data_paths.get("val_annotations"):
        identity = resolve_dataset_identity(
            data_paths["train_annotations"],
            data_paths["val_annotations"],
        )
        result["dataset_identity"] = identity.to_dict()
        training_mode = result["training_mode"]
        if not training_mode:
            errors.append("run 记录中缺少 training_mode，无法重新执行数据身份守卫")
        else:
            errors.extend(validate_training_mode_against_identity(training_mode, max_epochs, identity))

    if runtime_yaml.is_file():
        try:
            from omegaconf import OmegaConf

            runtime_cfg = OmegaConf.load(runtime_yaml)
            save_dir = OmegaConf.select(runtime_cfg, "trainer.checkpoint.save_dir")
            runtime_max_epochs = OmegaConf.select(runtime_cfg, "trainer.max_epochs")
            runtime_initial_checkpoint = OmegaConf.select(runtime_cfg, "trainer.model.checkpoint_path")
            runtime_bpe_path = OmegaConf.select(runtime_cfg, "paths.bpe_path")
            expected_save_dir = resolved_run / "checkpoints"
            if not save_dir or Path(str(save_dir)).expanduser().resolve(strict=False) != expected_save_dir.resolve(strict=False):
                errors.append(
                    "runtime YAML 的 trainer.checkpoint.save_dir 不指向当前 run/checkpoints: "
                    f"{save_dir}"
                )
            if max_epochs is not None and int(runtime_max_epochs) != max_epochs:
                errors.append(
                    "runtime YAML 与 training_config_summary.json 的 max_epochs 不一致: "
                    f"{runtime_max_epochs} != {max_epochs}"
                )
            if runtime_initial_checkpoint != result["initial_checkpoint"]:
                errors.append(
                    "runtime YAML 与 run 记录的 initial checkpoint 不一致: "
                    f"{runtime_initial_checkpoint} != {result['initial_checkpoint']}"
                )
            if runtime_bpe_path != result["bpe_path"]:
                errors.append(
                    f"runtime YAML 与 run 记录的 BPE 路径不一致: {runtime_bpe_path} != {result['bpe_path']}"
                )
        except Exception as exc:
            errors.append(f"无法验证 runtime YAML 的恢复配置: {exc!r}")

    if num_gpus is not None:
        result["command"] = [
            "conda",
            "run",
            "-n",
            DEFAULT_CONDA_ENV,
            "python",
            str(DEFAULT_TRAINING_LAUNCHER),
            "-c",
            str(runtime_yaml),
            "--use-cluster",
            "0",
            "--num-gpus",
            str(num_gpus),
        ]
    if previous_status not in (None, "paused", "cancelled", "failed"):
        warnings.append(f"上一次 training_summary 状态无法识别: {previous_status!r}")
    result["resumable"] = not errors
    return result


def list_resumable_training_runs(root: Path = DEFAULT_TRAINING_RUN_ROOT) -> list[str]:
    """List runs that have a resume checkpoint; validity is shown separately."""
    if not root.is_dir():
        return []
    runs: list[str] = []
    for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
        if checkpoint.is_file():
            runs.append(str(run_dir.resolve(strict=False)))
    return runs


def validate_can_resume_training(
    candidate: dict[str, Any],
    confirmed: bool,
    already_running: bool,
    cuda_available: bool,
    cuda_device_count: int,
) -> list[str]:
    reasons = list(candidate.get("errors") or [])
    if not candidate.get("resumable") and not reasons:
        reasons.append("所选 run 不能恢复")
    if already_running:
        reasons.append("已有一个训练任务在运行，不能同时恢复另一个 run")
    if not confirmed:
        reasons.append("请勾选确认框: 我确认将从最近完整 checkpoint 恢复训练。")
    requested_num_gpus = candidate.get("num_gpus")
    if not cuda_available:
        reasons.append("CUDA 不可用，拒绝恢复训练")
    elif isinstance(requested_num_gpus, int) and requested_num_gpus > cuda_device_count:
        reasons.append(
            f"原 run 请求 num_gpus={requested_num_gpus}，但只检测到 {cuda_device_count} 个可用 GPU"
        )
    return reasons


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
        if state.stop_reason == "paused":
            return "paused"
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
        "process_metadata": state.metadata,
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
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_existing_summary(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _attempt_id_for(
    run_dir: Path,
    command: list[str],
    runtime_config_path: str | None,
    state: ProcessState,
) -> str:
    payload = json.dumps(
        {
            "run_dir": str(run_dir.resolve(strict=False)),
            "command": command,
            "runtime_config": runtime_config_path,
            "started_at": state.started_at,
        },
        ensure_ascii=True,
        sort_keys=True,
    ).encode("utf-8")
    return f"attempt-{hashlib.sha256(payload).hexdigest()[:16]}"


def _legacy_attempt(existing: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(existing, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    keys = (
        "status",
        "command",
        "runtime_config",
        "start_time",
        "end_time",
        "duration_seconds",
        "exit_code",
        "distributed",
        "initial_checkpoint",
        "discovered_checkpoint_files",
        "stdout_stderr_tail",
        "warnings",
        "errors",
    )
    attempt = {key: existing.get(key) for key in keys}
    attempt.update(
        {
            "attempt_id": f"legacy-{hashlib.sha256(payload).hexdigest()[:16]}",
            "launch_kind": existing.get("launch_kind", "legacy"),
            "resume_from_checkpoint": existing.get("resume_from_checkpoint"),
        }
    )
    return attempt


def finalize_training_summary(
    run_dir: Path,
    command: list[str],
    runtime_config_path: str | None,
    initial_checkpoint: str | None,
    log_text: str | None = None,
    state: ProcessState | None = None,
    import_metadata: dict[str, Any] | None = None,
    effective_pythonpath: str | None = None,
    training_provenance: dict[str, Any] | None = None,
    distributed: dict[str, Any] | None = None,
    attempt_id: str | None = None,
    launch_kind: str = "new",
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Write or append one completed process attempt to training_summary.json.

    Only lists checkpoint files that actually exist on disk — never invents a
    checkpoint path just because training reportedly completed. The top-level
    fields describe the latest attempt for backward compatibility; attempts keeps
    the full pause/resume history.
    """
    summary_path = run_dir / "training_summary.json"
    if state is None:
        captured_log, state = training_process_manager.snapshot()
    else:
        captured_log = log_text if log_text is not None else ""
    resolved_attempt_id = attempt_id or _attempt_id_for(run_dir, command, runtime_config_path, state)
    with _summary_lock:
        existing = _load_existing_summary(summary_path) if summary_path.exists() else None
        if existing is not None:
            attempts = existing.get("attempts")
            if isinstance(attempts, list) and any(
                isinstance(item, dict) and item.get("attempt_id") == resolved_attempt_id for item in attempts
            ):
                return existing

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
    provenance = training_provenance
    if provenance is None:
        try:
            provenance = collect_training_provenance(
                runtime_config_path=runtime_config_path,
                sam3_import_path=import_metadata.get("sam3") if import_metadata else None,
                python_executable=import_metadata.get("python") if import_metadata else None,
                distributed=distributed,
            )
        except Exception as exc:
            provenance = {"patch_guard_ok": False, "patch_guard_error": repr(exc)}

    attempt = {
        "attempt_id": resolved_attempt_id,
        "launch_kind": launch_kind,
        "resume_from_checkpoint": resume_from_checkpoint,
        "status": status,
        "command": command,
        "runtime_config": runtime_config_path,
        "start_time": _iso(state.started_at),
        "end_time": _iso(state.finished_at),
        "duration_seconds": duration,
        "exit_code": state.returncode,
        "distributed": distributed or (provenance.get("distributed") if isinstance(provenance, dict) else None),
        "initial_checkpoint": initial_checkpoint,
        "discovered_checkpoint_files": discovered,
        "stdout_stderr_tail": "\n".join((captured_log or "").splitlines()[-200:]),
        "warnings": warnings,
        "errors": errors,
    }
    summary = {
        "run_id": run_dir.name,
        "latest_attempt_id": resolved_attempt_id,
        "launch_kind": launch_kind,
        "resume_from_checkpoint": resume_from_checkpoint,
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
        "training_provenance": provenance,
        "distributed": distributed or (provenance.get("distributed") if isinstance(provenance, dict) else None),
        "initial_checkpoint": initial_checkpoint,
        "output_directory": str(run_dir),
        "discovered_checkpoint_files": discovered,
        "stdout_stderr_tail": "\n".join((captured_log or "").splitlines()[-200:]),
        "warnings": warnings,
        "errors": errors,
    }
    with _summary_lock:
        existing = _load_existing_summary(summary_path) if summary_path.exists() else None
        attempts: list[dict[str, Any]] = []
        if existing is not None:
            existing_attempts = existing.get("attempts")
            if isinstance(existing_attempts, list):
                attempts = [item for item in existing_attempts if isinstance(item, dict)]
            else:
                attempts = [_legacy_attempt(existing)]
            if any(item.get("attempt_id") == resolved_attempt_id for item in attempts):
                return existing
        attempts.append(attempt)
        merged = dict(existing or {})
        merged.update(summary)
        merged["attempts"] = attempts
        merged["attempt_count"] = len(attempts)
        _atomic_write_json(summary_path, merged)
        summary = merged
    return summary


def make_training_summary_callback(
    run_dir: Path,
    command: list[str],
    runtime_config_path: str | None,
    initial_checkpoint: str | None,
    import_metadata: dict[str, Any] | None = None,
    effective_pythonpath: str | None = None,
    training_provenance: dict[str, Any] | None = None,
    distributed: dict[str, Any] | None = None,
    attempt_id: str | None = None,
    launch_kind: str = "new",
    resume_from_checkpoint: str | None = None,
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
            training_provenance=training_provenance,
            distributed=distributed,
            attempt_id=attempt_id,
            launch_kind=launch_kind,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    return _callback
