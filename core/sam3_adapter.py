from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from core.config import DEFAULT_SAM3_CHECKPOINT, SAM301_ROOT
from core.mask_nms import mask_bbox_xywh
from core.npz_io import InstanceSet


class Sam3Adapter:
    """Small SAM3 image inference adapter used by the unified inference pipeline."""

    def __init__(
        self,
        checkpoint: Path = DEFAULT_SAM3_CHECKPOINT,
        sam3_root: Path = SAM301_ROOT,
        confidence_threshold: float = 0.05,
        dtype_mode: str = "bf16",
        device: str = "cuda",
    ) -> None:
        self.checkpoint = checkpoint.expanduser().resolve(strict=False)
        self.sam3_root = sam3_root.expanduser().resolve(strict=False)
        if str(self.sam3_root) not in sys.path:
            sys.path.insert(0, str(self.sam3_root))

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for SAM3 inference, but torch.cuda.is_available() is False")
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        self.device = device
        try:
            self.model = build_sam3_image_model(checkpoint_path=str(self.checkpoint), device=self.device)
        except TypeError:
            self.model = build_sam3_image_model(checkpoint_path=str(self.checkpoint))
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor(self.model)
        self.processor.set_confidence_threshold(confidence_threshold)
        self.dtype_mode = dtype_mode
        self.autocast_dtype = self._decide_dtype(dtype_mode)

    def _decide_dtype(self, dtype_mode: str) -> torch.dtype | None:
        if self.device != "cuda":
            return None
        if dtype_mode == "bf16":
            return torch.bfloat16
        if dtype_mode == "fp16":
            return torch.float16
        if dtype_mode == "none":
            return None
        raise ValueError(f"Unsupported dtype_mode: {dtype_mode}")

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    @staticmethod
    def _to_pil(image: np.ndarray | str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        raise TypeError(f"Unsupported image type: {type(image)}")

    def predict(self, image: np.ndarray | str | Path | Image.Image, prompt: str, score_threshold: float = 0.3, min_area: int = 200) -> InstanceSet:
        pil = self._to_pil(image)
        width, height = pil.size
        with torch.inference_mode():
            with self._autocast_context():
                state = self.processor.set_image(pil)
                output = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks_np = output["masks"].detach().to(torch.float32).cpu().numpy()
        scores_np = output["scores"].detach().to(torch.float32).cpu().numpy()
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        masks: list[np.ndarray] = []
        scores: list[float] = []
        bboxes: list[list[int]] = []
        for mask, score in zip(masks_np, scores_np):
            if float(score) < score_threshold:
                continue
            if mask.ndim == 3:
                mask = mask[0]
            if mask.shape != (height, width):
                binary = cv2.resize((mask > 0.5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                binary = mask > 0.5
            if int(binary.sum()) < min_area:
                continue
            masks.append(binary)
            scores.append(float(score))
            bboxes.append(mask_bbox_xywh(binary))

        if masks:
            masks_arr = np.stack(masks, axis=0).astype(bool)
            scores_arr = np.asarray(scores, dtype=np.float32)
            bboxes_arr = np.asarray(bboxes, dtype=np.int32)
            ids_arr = np.arange(1, len(masks) + 1, dtype=np.int32)
        else:
            masks_arr = np.zeros((0, height, width), dtype=bool)
            scores_arr = np.zeros((0,), dtype=np.float32)
            bboxes_arr = np.zeros((0, 4), dtype=np.int32)
            ids_arr = np.zeros((0,), dtype=np.int32)
        return InstanceSet(masks=masks_arr, scores=scores_arr, bboxes=bboxes_arr, instance_ids=ids_arr, extra={})
