"""Boundary-aware mask loss for SAM3 fine-tuning (loss ablation arm D).

`MasksWithBoundaryDice` mirrors the dense (non-sampled) branch of SAM3's
`Masks` loss (`sam3.train.loss.loss_fns.Masks`) and adds `loss_boundary_dice`:
a soft Dice restricted to the boundary band of the GT mask union the
(thresholded) predicted mask. Motivation and weight calibration:
`experiments/loss_ablation_202609/` — on the 2026-07-23 baseline the boundary
band error (~0.24) is ~5x the region dice error (~0.05), matching the weak
test Boundary F1 (0.75), so the band gets its own supervised term.

Band definition (BoundaryDoU-style, max-pool morphology, differentiable
w.r.t. the mask logits): for a mask m in [0,1],
``band(m) = maxpool_k(m) * maxpool_k(1-m)`` is ~1 within ~k//2 px of the 0/1
transition and ~0 elsewhere. The loss weights pixels by
``band = max(band(gt), band(pred > 0.5))`` so both "GT boundary the prediction
missed" and "spurious prediction boundary" are covered; the threshold branch
carries no gradient — the band acts as fixed pixel weights within each step.
The band lives at the loss resolution (the padded training resolution the GT
masks are stored at), not the original-image resolution used by the
Boundary-F1 metric.

``boundary_kernel`` takes one kernel or a list of them; multiple kernels are
AVERAGED so the term keeps its magnitude and a configured weight means the same
thing whichever is used. Setting ``num_sample_points`` switches focal+dice to
SAM3's PointRend importance sampling (the stock ``Masks`` behaviour) while the
boundary band stays on the dense, upsampled masks.

Hydra usage (replacing the stock ``Masks`` entry). ``loss_mask``/``loss_dice``
apply wherever the stock mask loss applies (main output and, during training,
the o2m branch); ``loss_boundary_dice`` applies to the main o2o-matched output
only — see the comment in ``get_loss``. Aux layers stay excluded via
``compute_aux=False``:

    # arm D: dense, single band
    - _target_: core.boundary_loss.MasksWithBoundaryDice
      focal_alpha: 0.25
      focal_gamma: 2.0
      boundary_kernel: 5
      weight_dict:
        loss_mask: 200.0
        loss_dice: 30.0
        loss_boundary_dice: 5.0
      compute_aux: false

    # arm E: multi-scale band, 3x the weight
    #   boundary_kernel: [3, 9]   loss_boundary_dice: 15.0
    # arm F: arm D's band plus PointRend focal/dice
    #   num_sample_points: 12544  oversample_ratio: 3.0
    #   importance_sample_ratio: 0.75
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate

from sam3.train.loss.loss_fns import Masks, dice_loss, sigmoid_focal_loss


class MasksWithBoundaryDice(Masks):
    def __init__(self, *args, boundary_kernel: int | Sequence[int] = 5, **kwargs):
        super().__init__(*args, **kwargs)
        kernels = (
            [boundary_kernel]
            if isinstance(boundary_kernel, int)
            else [int(k) for k in boundary_kernel]
        )
        if not kernels:
            raise ValueError("boundary_kernel must name at least one kernel")
        for kernel in kernels:
            if kernel % 2 != 1 or kernel < 3:
                raise ValueError(
                    f"boundary_kernel entries must be odd integers >= 3, got {kernel}"
                )
        # Multi-scale bands are AVERAGED, not summed, so `loss_boundary_dice`
        # keeps the same magnitude whether one kernel or three are configured
        # and the weight stays comparable across arms.
        self.boundary_kernels = kernels

    # Instances per chunk for the band morphology. The max-pool temporaries are
    # several N x H x W float32 tensors; unchunked, a crowded image OOMed a
    # 32 GB GPU (2026-09-03, epoch 8 of expD). 32 masks/chunk bounds the
    # transient peak at ~half a GB regardless of N.
    BAND_CHUNK = 32

    @classmethod
    def _soft_boundary_band(cls, masks: torch.Tensor, kernel: int) -> torch.Tensor:
        """masks: [N, H, W] in [0, 1] -> band weights [N, H, W] in [0, 1]."""
        pad = kernel // 2
        chunks = []
        for start in range(0, masks.shape[0], cls.BAND_CHUNK):
            stacked = masks[start : start + cls.BAND_CHUNK].unsqueeze(1)
            outer = F.max_pool2d(stacked, kernel, stride=1, padding=pad)
            inner = F.max_pool2d(1.0 - stacked, kernel, stride=1, padding=pad)
            chunks.append((outer * inner).squeeze(1))
        return torch.cat(chunks, dim=0)

    def _boundary_dice(self, probs, gt, num_boxes):
        """Mean over configured kernels of the band-restricted soft Dice."""
        total = None
        for kernel in self.boundary_kernels:
            band = torch.maximum(
                self._soft_boundary_band(gt, kernel),
                self._soft_boundary_band((probs > 0.5).float(), kernel),
            )
            intersection = (probs * gt * band).flatten(1).sum(-1)
            denominator = ((probs + gt) * band).flatten(1).sum(-1)
            per_kernel = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            total = per_kernel if total is None else total + per_kernel
        return (total / len(self.boundary_kernels)).sum() / num_boxes

    def get_loss(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]

        def _zeros() -> dict[str, torch.Tensor]:
            # Tied to the graph via sum()*0 so DDP still sees the parameters.
            zero = src_masks.sum() * 0.0
            return {
                "loss_mask": zero,
                "loss_dice": zero.clone(),
                "loss_boundary_dice": zero.clone(),
            }

        if targets["masks"] is None:
            return _zeros()

        target_masks = (
            targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
        )
        target_masks = target_masks.to(src_masks)
        keep = (
            targets["is_valid_mask"]
            if indices[2] is None
            else targets["is_valid_mask"][indices[2]]
        )
        src_masks = src_masks[(indices[0], indices[1])][keep]
        target_masks = target_masks[keep]
        if src_masks.shape[0] == 0:
            return _zeros()

        # The boundary term is supervised on the main (o2o-matched) output only:
        # the weight calibration in experiments/loss_ablation_202609 measured it
        # there, and the o2m branch (up to matcher-topk duplicates per GT) blows
        # the mask count up ~4x, which is what OOMed the band morphology. The
        # o2m call receives a synthesized dict that carries no "indices" key,
        # while every main per-layer output does — that is the discriminator.
        # Focal + dice still apply to o2m, exactly like the stock Masks loss.
        want_boundary = "indices" in outputs

        if self.num_sample_points is not None:
            # PointRend arm: focal + dice come from importance-sampled points at
            # the head's native resolution (`_sampled_loss` grid-samples both the
            # low-res logits and the full-res GT, so no upsampling happens here).
            losses = self._sampled_loss(src_masks, target_masks, num_boxes)
            if want_boundary:
                probs, gt = self._upsampled_pair(src_masks, target_masks)
                losses["loss_boundary_dice"] = self._boundary_dice(probs, gt, num_boxes)
            else:
                losses["loss_boundary_dice"] = src_masks.sum() * 0.0
            return losses

        probs_logits, gt = self._upsampled_pair(src_masks, target_masks, logits=True)
        probs = probs_logits.sigmoid()
        loss_boundary = (
            self._boundary_dice(probs, gt, num_boxes)
            if want_boundary
            else src_masks.sum() * 0.0
        )
        flat_src = probs_logits.flatten(1)
        flat_gt = gt.flatten(1)
        return {
            "loss_mask": sigmoid_focal_loss(
                flat_src,
                flat_gt,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            ),
            "loss_dice": dice_loss(flat_src, flat_gt, num_boxes),
            "loss_boundary_dice": loss_boundary,
        }

    @staticmethod
    def _upsampled_pair(src_masks, target_masks, logits: bool = False):
        """Predicted logits upsampled to the GT resolution, plus float GT.

        Returns probabilities unless `logits` is set (the dense branch needs the
        raw logits for focal/dice and the probabilities for the band).
        """
        if len(src_masks.shape) == 3:
            src_masks = src_masks[:, None]
        # Bilinear interpolation does not support bf16 (and the band math wants
        # full precision anyway).
        src_masks = src_masks.to(dtype=torch.float32)
        src_masks = interpolate(
            src_masks,
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        gt = target_masks.float()
        return (src_masks if logits else src_masks.sigmoid()), gt
