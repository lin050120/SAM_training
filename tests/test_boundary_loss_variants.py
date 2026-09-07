"""Boundary-loss behaviour that the loss-ablation arms depend on.

Arms D (dense, kernel 5), E (dense, multi-kernel, heavier weight) and F
(PointRend-sampled focal/dice + dense boundary) all run through
`MasksWithBoundaryDice`. The tests below pin the properties the cross-arm
comparison rests on:

- the dense single-kernel path still computes exactly what arm D computed
  before multi-kernel/sampling support was added (otherwise D's numbers are
  not comparable with E's and F's);
- multi-kernel bands are averaged, so `loss_boundary_dice` keeps its
  magnitude and the configured weight means the same thing across arms;
- the boundary term is skipped on the o2m branch (the OOM fix), on both the
  dense and the sampled path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra in (PROJECT_ROOT, Path("/home/book/sam301")):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

pytest.importorskip("sam3.train.loss.loss_fns")

from core.boundary_loss import MasksWithBoundaryDice  # noqa: E402

WEIGHTS = {"loss_mask": 200.0, "loss_dice": 30.0, "loss_boundary_dice": 5.0}

# sam3's focal/dice losses are Triton kernels that reject CPU tensors, so these
# run on the GPU or not at all.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sam3 focal/dice loss requires CUDA (Triton)"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _batch(seed: int = 0, n: int = 3, low: int = 8, high: int = 32):
    """(outputs, targets, indices, num_boxes) with low-res logits and full-res GT."""
    generator = torch.Generator().manual_seed(seed)
    pred = torch.randn(1, n, low, low, generator=generator).to(DEVICE).requires_grad_(True)
    gt = torch.zeros(n, high, high, device=DEVICE)
    for i in range(n):
        gt[i, 4 + i : high - 6 + i, 5 : high - 5 - i] = 1.0
    indices = (
        torch.zeros(n, dtype=torch.long, device=DEVICE),
        torch.arange(n, device=DEVICE),
        None,
    )
    targets = {
        "masks": gt,
        "is_valid_mask": torch.ones(n, dtype=torch.bool, device=DEVICE),
    }
    return pred, gt, indices, targets, float(n)


def _reference_arm_d(pred, gt, indices, num_boxes, kernel=5):
    """Arm D's original dense implementation, transcribed independently."""
    from torch.nn.functional import interpolate, max_pool2d

    src = pred[(indices[0], indices[1])][:, None].float()
    src = interpolate(src, size=gt.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    probs = src.sigmoid()

    def band(mask):
        pad = kernel // 2
        stacked = mask.unsqueeze(1)
        outer = max_pool2d(stacked, kernel, stride=1, padding=pad)
        inner = max_pool2d(1.0 - stacked, kernel, stride=1, padding=pad)
        return (outer * inner).squeeze(1)

    weights = torch.maximum(band(gt), band((probs > 0.5).float()))
    intersection = (probs * gt * weights).flatten(1).sum(-1)
    denominator = ((probs + gt) * weights).flatten(1).sum(-1)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return dice.sum() / num_boxes


def test_dense_single_kernel_matches_arm_d():
    pred, gt, indices, targets, num_boxes = _batch()
    loss_fn = MasksWithBoundaryDice(weight_dict=WEIGHTS, boundary_kernel=5)
    got = loss_fn.get_loss({"pred_masks": pred, "indices": indices}, targets, indices, num_boxes)
    expected = _reference_arm_d(pred, gt, indices, num_boxes, kernel=5)
    assert torch.allclose(got["loss_boundary_dice"], expected, atol=0, rtol=0)


def test_multi_kernel_is_the_mean_of_single_kernels():
    pred, gt, indices, targets, num_boxes = _batch(seed=1)
    outputs = {"pred_masks": pred, "indices": indices}
    singles = [
        MasksWithBoundaryDice(weight_dict=WEIGHTS, boundary_kernel=k)
        .get_loss(outputs, targets, indices, num_boxes)["loss_boundary_dice"]
        for k in (3, 9)
    ]
    multi = MasksWithBoundaryDice(weight_dict=WEIGHTS, boundary_kernel=[3, 9])
    got = multi.get_loss(outputs, targets, indices, num_boxes)["loss_boundary_dice"]
    assert torch.allclose(got, (singles[0] + singles[1]) / 2, atol=1e-6)


@pytest.mark.parametrize("sample_points", [None, 512])
def test_boundary_term_is_skipped_without_indices(sample_points):
    """The o2m branch passes a dict with no 'indices' key; boundary must be 0."""
    pred, _gt, indices, targets, num_boxes = _batch(seed=2)
    kwargs = {}
    if sample_points is not None:
        kwargs = {
            "num_sample_points": sample_points,
            "oversample_ratio": 3.0,
            "importance_sample_ratio": 0.75,
        }
    loss_fn = MasksWithBoundaryDice(weight_dict=WEIGHTS, boundary_kernel=5, **kwargs)
    o2m = loss_fn.get_loss({"pred_masks": pred}, targets, indices, num_boxes)
    assert o2m["loss_boundary_dice"].item() == 0.0
    main = loss_fn.get_loss(
        {"pred_masks": pred, "indices": indices}, targets, indices, num_boxes
    )
    assert main["loss_boundary_dice"].item() > 0.0
    assert set(main) == {"loss_mask", "loss_dice", "loss_boundary_dice"}


def test_sampled_arm_produces_finite_gradients():
    """Arm F: sampled focal/dice + dense boundary must backprop cleanly."""
    pred, _gt, indices, targets, num_boxes = _batch(seed=3)
    loss_fn = MasksWithBoundaryDice(
        weight_dict=WEIGHTS,
        boundary_kernel=5,
        num_sample_points=512,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )
    losses = loss_fn.get_loss(
        {"pred_masks": pred, "indices": indices}, targets, indices, num_boxes
    )
    total = sum(WEIGHTS[k] * v for k, v in losses.items())
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0
