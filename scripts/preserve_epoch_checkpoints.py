#!/usr/bin/env python3
"""Preserve each atomically replaced SAM3 checkpoint.pt as a numbered hard link.

This is intended for an already-running trainer whose checkpoint save frequency
cannot be changed without a restart. The trainer writes checkpoint.pt through a
temporary file and atomic rename, so linking the completed inode is safe and
does not copy 10 GB while training is active.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


def _latest_completed_epoch(stats_path: Path) -> int | None:
    try:
        lines = stats_path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line).get("Trainer/epoch")
        except json.JSONDecodeError:
            continue
        if value is not None:
            return int(value)
    return None


def _log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    parser.add_argument("--max-checkpoint", type=int, default=100)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    current_path = checkpoint_dir / "checkpoint.pt"
    stats_path = run_dir / "logs" / "book_spine" / "train_stats.json"
    minimum_free = int(args.min_free_gib * 1024**3)
    last_inode: tuple[int, int] | None = None

    _log(
        f"watching {current_path}; min_free={args.min_free_gib:g} GiB; "
        f"max_checkpoint={args.max_checkpoint}"
    )

    while True:
        try:
            source_stat = current_path.stat()
        except FileNotFoundError:
            time.sleep(args.poll_seconds)
            continue

        source_identity = (source_stat.st_dev, source_stat.st_ino)
        if source_identity == last_inode:
            time.sleep(args.poll_seconds)
            continue

        completed_epoch = _latest_completed_epoch(stats_path)
        if completed_epoch is None:
            _log("checkpoint changed, but no completed epoch is available yet; retrying")
            time.sleep(args.poll_seconds)
            continue

        checkpoint_number = completed_epoch + 1
        if checkpoint_number > args.max_checkpoint:
            _log(f"reached checkpoint {checkpoint_number}; watcher complete")
            return 0

        numbered_path = checkpoint_dir / f"checkpoint_{checkpoint_number}.pt"
        try:
            numbered_stat: os.stat_result | None = numbered_path.stat()
        except FileNotFoundError:
            numbered_stat = None
        if numbered_stat is not None:
            # Only stop watching this inode when the numbered file really is it.
            # A different inode means the completed-epoch counter still lags the
            # checkpoint that was just written, so the number is one too low;
            # retrying lets train_stats.json catch up instead of dropping the
            # new checkpoint before the next epoch overwrites it.
            if (numbered_stat.st_dev, numbered_stat.st_ino) == source_identity:
                _log(f"{numbered_path.name} already exists; tracking current checkpoint")
                last_inode = source_identity
            else:
                _log(
                    f"{numbered_path.name} exists with a different inode; "
                    "completed-epoch counter lags the current checkpoint, retrying"
                )
            time.sleep(args.poll_seconds)
            continue

        free_bytes = shutil.disk_usage(checkpoint_dir).free
        if free_bytes - source_stat.st_size < minimum_free:
            _log(
                f"STOP: refusing to preserve {numbered_path.name}; "
                f"free space would fall below {args.min_free_gib:g} GiB"
            )
            return 2

        try:
            os.link(current_path, numbered_path)
        except FileExistsError:
            # Appeared between the check above and here; re-poll so the same
            # inode comparison decides whether it is genuinely preserved.
            time.sleep(args.poll_seconds)
            continue

        linked_stat = numbered_path.stat()
        current_stat = current_path.stat()
        linked_identity = (linked_stat.st_dev, linked_stat.st_ino)
        current_identity = (current_stat.st_dev, current_stat.st_ino)
        if linked_identity != current_identity:
            numbered_path.unlink()
            _log("checkpoint changed during linking; removed stale link and will retry")
            time.sleep(args.poll_seconds)
            continue

        directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

        _log(
            f"preserved {numbered_path.name} "
            f"(inode={linked_stat.st_ino}, size={linked_stat.st_size})"
        )
        last_inode = current_identity
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
