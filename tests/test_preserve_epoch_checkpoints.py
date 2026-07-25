"""Regression tests for the checkpoint-preserving watcher.

The watcher numbers each atomically replaced checkpoint.pt from the newest
``Trainer/epoch`` in train_stats.json. Those two files are written by the
trainer at slightly different moments, so a poll can land while the counter
still lags the checkpoint that was just saved. The watcher must retry then --
if it instead treats the stale number as "already preserved" it stops watching
that inode and the checkpoint is lost when the next epoch overwrites it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATCHER_PATH = PROJECT_ROOT / "scripts" / "preserve_epoch_checkpoints.py"


def _load_watcher():
    spec = importlib.util.spec_from_file_location("_preserve_watcher", WATCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class PreserveEpochCheckpointsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.watcher = _load_watcher()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True)
        stats_dir = self.run_dir / "logs" / "book_spine"
        stats_dir.mkdir(parents=True)
        self.stats_path = stats_dir / "train_stats.json"

    def _set_completed_epoch(self, epoch: int) -> None:
        self.stats_path.write_text(
            "".join(json.dumps({"Trainer/epoch": i}) + "\n" for i in range(epoch + 1)),
            encoding="utf-8",
        )

    def _save_checkpoint(self, marker: str) -> None:
        """Write through a temp file and rename, exactly like the trainer."""
        temporary = self.checkpoint_dir / f".{marker}.tmp"
        temporary.write_text(marker, encoding="utf-8")
        os.replace(temporary, self.checkpoint_dir / "checkpoint.pt")

    def _numbered(self, number: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_{number}.pt"

    def _start_watcher(self, max_checkpoint: int = 5) -> None:
        argv = [
            "preserve_epoch_checkpoints.py",
            "--run-dir",
            str(self.run_dir),
            "--poll-seconds",
            "0.02",
            "--min-free-gib",
            "0",
            "--max-checkpoint",
            str(max_checkpoint),
        ]
        original_argv = sys.argv
        sys.argv = argv
        self.addCleanup(lambda: setattr(sys, "argv", original_argv))
        thread = threading.Thread(target=self.watcher.main, daemon=True)
        thread.start()

    def test_preserves_each_checkpoint_as_a_hard_link(self) -> None:
        self._set_completed_epoch(0)
        self._save_checkpoint("c1")
        self._start_watcher()

        self.assertTrue(_wait_until(self._numbered(1).is_file))
        current = self.checkpoint_dir / "checkpoint.pt"
        # A hard link, not a copy: same inode, link count two.
        self.assertEqual(self._numbered(1).stat().st_ino, current.stat().st_ino)
        self.assertEqual(self._numbered(1).stat().st_nlink, 2)

    def test_stale_epoch_counter_does_not_drop_the_new_checkpoint(self) -> None:
        self._set_completed_epoch(0)
        self._save_checkpoint("c1")
        self._start_watcher()
        self.assertTrue(_wait_until(self._numbered(1).is_file))

        # The trainer saved the next checkpoint, but train_stats.json has not
        # recorded the finished epoch yet, so the derived number is one too low.
        self._save_checkpoint("c2")
        time.sleep(0.3)
        self.assertEqual(self._numbered(1).read_text(encoding="utf-8"), "c1")
        self.assertFalse(self._numbered(2).exists())

        # Once the counter catches up the checkpoint must still be preserved.
        self._set_completed_epoch(1)
        self.assertTrue(_wait_until(self._numbered(2).is_file))
        self.assertEqual(self._numbered(2).read_text(encoding="utf-8"), "c2")

    def test_unchanged_checkpoint_is_not_relinked(self) -> None:
        self._set_completed_epoch(0)
        self._save_checkpoint("c1")
        self._start_watcher()
        self.assertTrue(_wait_until(self._numbered(1).is_file))

        # Same inode across polls must stay a single preserved link.
        self._set_completed_epoch(1)
        time.sleep(0.3)
        self.assertFalse(self._numbered(2).exists())


if __name__ == "__main__":
    unittest.main()
