from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.npz_io import InstanceSet


def make_overlay(image_bgr: np.ndarray, instances: InstanceSet, alpha: float = 0.45) -> np.ndarray:
    vis = image_bgr.copy()
    rng = np.random.default_rng(0)
    for idx, mask in enumerate(instances.masks):
        color = rng.integers(60, 255, 3).astype(np.uint8)
        vis[mask] = ((1 - alpha) * vis[mask] + alpha * color).astype(np.uint8)
        if idx < len(instances.bboxes):
            x, y, w, h = [int(v) for v in instances.bboxes[idx]]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color.tolist(), 1)
            score = float(instances.scores[idx]) if idx < len(instances.scores) else 0.0
            cv2.putText(
                vis,
                f"{idx + 1}:{score:.2f}",
                (x, max(12, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color.tolist(),
                1,
            )
    return vis


def write_overlay(image_path: Path, instances: InstanceSet, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), make_overlay(image, instances))

