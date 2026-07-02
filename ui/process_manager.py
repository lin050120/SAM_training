from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ui.ui_utils import logger


@dataclass
class ProcessState:
    running: bool = False
    returncode: int | None = None
    stopped_by_user: bool = False
    command: list[str] = field(default_factory=list)
    cwd: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


class ProcessManager:
    """Manages a single active long-running subprocess for the UI.

    Only one task may be active at a time, matching the single-active-inference-task
    requirement. Uses subprocess.Popen with list-form arguments only (never shell=True).

    On POSIX the child is started with start_new_session=True so it becomes the leader
    of its own session and process group (pgid == child pid). Stopping signals the whole
    group (SIGTERM, then SIGKILL after a timeout), so grandchildren spawned by the task
    — e.g. `conda run` wrapping the real python process — are terminated too instead of
    being orphaned.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._log_lines: list[str] = []
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._state = ProcessState()
        self._on_finish: Callable[[str, ProcessState], None] | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._state.running

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def pgid(self) -> int | None:
        """Process group id of the active/last child; equals pid on POSIX since
        start_new_session=True makes the child its own group leader. None once the
        process has exited and its pgid can no longer be queried."""
        if self._process is None:
            return None
        try:
            return os.getpgid(self._process.pid)
        except (ProcessLookupError, OSError):
            return None

    def start(
        self,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        on_finish: Callable[[str, ProcessState], None] | None = None,
    ) -> None:
        with self._lock:
            if self._state.running:
                raise RuntimeError("A task is already running. Stop it before starting a new one.")
            self._log_lines = []
            self._on_finish = on_finish
            self._state = ProcessState(
                running=True,
                command=list(command),
                cwd=str(cwd) if cwd else None,
                started_at=time.time(),
            )
        logger.info("process_start command=%s cwd=%s", command, cwd)
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception:
            with self._lock:
                self._state.running = False
                self._state.returncode = None
                self._state.finished_at = time.time()
                self._on_finish = None
            self._process = None
            raise
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                with self._lock:
                    self._log_lines.append(line.rstrip("\n"))
        finally:
            returncode = process.wait()
            with self._lock:
                self._state.running = False
                self._state.returncode = returncode
                self._state.finished_at = time.time()
                log_text = "\n".join(self._log_lines)
                state = ProcessState(**vars(self._state))
                on_finish = self._on_finish
                self._on_finish = None
            logger.info("process_finished returncode=%s stopped_by_user=%s", returncode, self._state.stopped_by_user)
            if on_finish is not None:
                try:
                    on_finish(log_text, state)
                except Exception:
                    logger.exception("process_finish_callback_failed")

    @staticmethod
    def _signal_group(process: subprocess.Popen, sig: signal.Signals) -> None:
        """Signal the child's whole process group; fall back to the child alone."""
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            logger.warning("killpg failed for pid=%s sig=%s, signalling child only", process.pid, sig)
            try:
                process.send_signal(sig)
            except ProcessLookupError:
                pass

    def stop(self, timeout: float = 5.0) -> None:
        process = self._process
        if process is None:
            return
        with self._lock:
            self._state.stopped_by_user = True
        logger.info("process_stop_requested pid=%s", process.pid)
        if process.poll() is None:
            self._signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("process_terminate_timeout_killing_group pid=%s", process.pid)
                self._signal_group(process, signal.SIGKILL)
                process.wait(timeout=timeout)
        reader_thread = self._reader_thread
        if reader_thread is not None and reader_thread is not threading.current_thread():
            reader_thread.join(timeout=timeout)
        # The reader thread drains remaining output and records returncode/finished_at;
        # captured stdout/stderr lines stay available via snapshot() after the stop.

    def shutdown(self) -> None:
        """Best-effort cleanup when the UI process itself exits."""
        try:
            if self.is_running():
                logger.info("ui_shutdown_stopping_active_task")
                self.stop(timeout=3.0)
        except Exception:  # pragma: no cover - never block interpreter exit
            logger.exception("ui_shutdown_cleanup_failed")

    def snapshot(self) -> tuple[str, ProcessState]:
        with self._lock:
            log_text = "\n".join(self._log_lines)
            state = ProcessState(**vars(self._state))
        return log_text, state


# A single module-level instance: this UI only supports one active inference
# task at a time, per the stage D1 handoff requirements.
inference_process_manager = ProcessManager()

# When the UI process exits normally (Ctrl+C in the terminal, demo.close(), interpreter
# shutdown), try to take the active task's process group down with it instead of
# leaving orphans. A SIGKILL of the UI itself cannot be intercepted; that case is
# documented as a known limitation.
atexit.register(inference_process_manager.shutdown)
