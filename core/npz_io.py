from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class InstanceSet:
    masks: np.ndarray
    scores: np.ndarray
    bboxes: np.ndarray
    instance_ids: np.ndarray
    extra: dict[str, np.ndarray]

    @property
    def count(self) -> int:
        return int(self.masks.shape[0])


def load_npz(path: Path) -> InstanceSet:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = {"masks", "scores", "bboxes"} - keys
        if missing:
            raise KeyError(f"{path} missing required NPZ keys: {sorted(missing)}")
        masks = data["masks"].astype(bool)
        scores = data["scores"].astype(np.float32)
        bboxes = data["bboxes"].astype(np.int32)
        if "instance_ids" in keys:
            instance_ids = data["instance_ids"].astype(np.int32)
        else:
            instance_ids = np.arange(1, len(scores) + 1, dtype=np.int32)
        extra = {k: data[k] for k in data.files if k not in {"masks", "scores", "bboxes", "instance_ids"}}
    return InstanceSet(masks=masks, scores=scores, bboxes=bboxes, instance_ids=instance_ids, extra=extra)


def save_npz(path: Path, instances: InstanceSet, **extra: np.ndarray) -> None:
    payload = {
        "masks": instances.masks.astype(bool),
        "scores": instances.scores.astype(np.float32),
        "bboxes": instances.bboxes.astype(np.int32),
        "instance_ids": instances.instance_ids.astype(np.int32),
    }
    payload.update(instances.extra)
    payload.update(extra)
    np.savez_compressed(path, **payload)


def empty_instances(height: int, width: int) -> InstanceSet:
    return InstanceSet(
        masks=np.zeros((0, height, width), dtype=bool),
        scores=np.zeros((0,), dtype=np.float32),
        bboxes=np.zeros((0, 4), dtype=np.int32),
        instance_ids=np.zeros((0,), dtype=np.int32),
        extra={},
    )

