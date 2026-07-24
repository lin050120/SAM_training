"""Regression test for the SAM3 Triton sigmoid focal-loss gamma=0 backward-NaN
bug (issue #575 / PRs #484,#576), fixed by the
sam301-loss-fns-focal-gamma0-fallback patch.

The Triton backward computes (1 - p_t) ** (gamma - 1); at gamma == 0 and a
saturated logit (p_t -> 1) that is (1 - p_t) ** -1 = Inf, then 0 * Inf = NaN:
a FINITE forward/loss but a NaN gradient. The patch routes gamma == 0 to the
finite PyTorch path. Requires CUDA + Triton, so it is skipped elsewhere.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="focal-loss gamma=0 bug is CUDA/Triton-only"
)


def _grad_finite(gamma: float) -> bool:
    from sam3.train.loss.loss_fns import sigmoid_focal_loss

    # +/-18 saturate the sigmoid (p_t -> 1); this is what a confident presence
    # logit late in training looks like.
    x = torch.tensor([[18.0, -18.0, 0.5, -0.5]], device="cuda", requires_grad=True)
    t = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda")
    loss = sigmoid_focal_loss(
        x, t, num_boxes=1, alpha=0.5, gamma=gamma, reduce=True, triton=True
    )
    assert torch.isfinite(loss).all(), "forward/loss should be finite even when buggy"
    (grad,) = torch.autograd.grad(loss, x)
    return bool(torch.isfinite(grad).all())


def test_gamma0_gradient_is_finite_at_saturated_logits():
    # The bug: with the unpatched Triton path this gradient is NaN.
    assert _grad_finite(0.0) is True


def test_gamma2_gradient_still_finite():
    # gamma != 0 keeps the Triton path and must be unaffected by the fix.
    assert _grad_finite(2.0) is True


def test_gamma0_forward_matches_triton_and_fallback():
    # The fix must not change the forward loss VALUE, only make the backward
    # finite: gamma=0 focal loss == alpha-weighted BCE regardless of path.
    import torch.nn.functional as F

    from sam3.train.loss.loss_fns import sigmoid_focal_loss

    x = torch.tensor([[2.0, -1.0, 0.3, -0.7]], device="cuda")
    t = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device="cuda")
    got = sigmoid_focal_loss(
        x, t, num_boxes=1, alpha=0.5, gamma=0.0, reduce=True, triton=True
    )
    ce = F.binary_cross_entropy_with_logits(x, t, reduction="none")
    alpha_t = 0.5 * t + 0.5 * (1 - t)
    expected = (alpha_t * ce).mean(1).sum() / 1
    assert torch.allclose(got, expected, atol=1e-5)
