from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.npz_io import InstanceSet


@dataclass(frozen=True)
class SuppressedInstance:
    source_instance_id: int
    kept_instance_id: int
    score: float
    overlap: float
    reason: str


@dataclass
class NmsResult:
    instances: InstanceSet
    kept_source_ids: list[int]
    suppressed: list[SuppressedInstance]


def mask_bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def bbox_overlaps(a: np.ndarray | list[int], b: np.ndarray | list[int]) -> bool:
    ax, ay, aw, ah = [int(v) for v in a]
    bx, by, bw, bh = [int(v) for v in b]
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def apply_mask_nms(
    instances: InstanceSet,
    iou_thresh: float = 0.5,
    metric: str = "iou",
    mode: str = "suppress",
) -> NmsResult:
    if metric not in {"iou", "iomin"}:
        raise ValueError(f"Unsupported NMS metric: {metric}")
    if mode not in {"suppress", "merge"}:
        raise ValueError(f"Unsupported NMS mode: {mode}")
    if instances.count == 0:
        return NmsResult(instances=instances, kept_source_ids=[], suppressed=[])

    masks = [np.asarray(mask, dtype=bool) for mask in instances.masks]
    scores = np.asarray(instances.scores, dtype=float)
    bboxes = [list(map(int, bbox)) for bbox in instances.bboxes]
    areas = [int(mask.sum()) for mask in masks]
    order = np.argsort(scores)[::-1].tolist()
    suppressed_flags = [False] * len(masks)
    kept_masks: list[np.ndarray] = []
    kept_scores: list[float] = []
    kept_bboxes: list[list[int]] = []
    kept_ids: list[int] = []
    suppressions: list[SuppressedInstance] = []

    for i in order:
        if suppressed_flags[i]:
            continue
        cur_mask = masks[i].copy() if mode == "merge" else masks[i]
        cur_bbox = list(bboxes[i])
        cur_area = int(cur_mask.sum())
        source_id = int(instances.instance_ids[i])

        for j in order:
            if j == i or suppressed_flags[j] or scores[j] > scores[i]:
                continue
            if not bbox_overlaps(cur_bbox, bboxes[j]):
                continue
            inter = int(np.logical_and(cur_mask, masks[j]).sum())
            if inter == 0:
                continue
            if metric == "iomin":
                denom = min(cur_area, areas[j])
            else:
                denom = cur_area + areas[j] - inter
            overlap = inter / denom if denom else 0.0
            if overlap >= iou_thresh:
                suppressed_flags[j] = True
                suppressions.append(
                    SuppressedInstance(
                        source_instance_id=int(instances.instance_ids[j]),
                        kept_instance_id=source_id,
                        score=float(scores[j]),
                        overlap=float(overlap),
                        reason=f"{metric}>={iou_thresh}",
                    )
                )
                if mode == "merge":
                    cur_mask = np.logical_or(cur_mask, masks[j])
                    cur_bbox = mask_bbox_xywh(cur_mask)
                    cur_area = int(cur_mask.sum())

        kept_masks.append(cur_mask)
        kept_scores.append(float(scores[i]))
        kept_bboxes.append(mask_bbox_xywh(cur_mask) if mode == "merge" else cur_bbox)
        kept_ids.append(source_id)

    h, w = instances.masks.shape[1], instances.masks.shape[2]
    result_instances = InstanceSet(
        masks=np.stack(kept_masks, axis=0).astype(bool) if kept_masks else np.zeros((0, h, w), dtype=bool),
        scores=np.asarray(kept_scores, dtype=np.float32),
        bboxes=np.asarray(kept_bboxes, dtype=np.int32) if kept_bboxes else np.zeros((0, 4), dtype=np.int32),
        instance_ids=np.arange(1, len(kept_masks) + 1, dtype=np.int32),
        extra={"source_instance_ids": np.asarray(kept_ids, dtype=np.int32)},
    )
    return NmsResult(instances=result_instances, kept_source_ids=kept_ids, suppressed=suppressions)

