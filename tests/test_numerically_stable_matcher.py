from __future__ import annotations

import torch

from core.numerically_stable_matcher import (
    GUARD_TAG,
    NumericallyStableBinaryHungarianMatcherV2,
)


def _stable() -> NumericallyStableBinaryHungarianMatcherV2:
    return NumericallyStableBinaryHungarianMatcherV2(
        focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2, stable=False
    )


def _base():
    from sam3.train.matcher import BinaryHungarianMatcherV2

    return BinaryHungarianMatcherV2(
        focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2, stable=False
    )


def _zero_area_case() -> tuple[dict, dict]:
    return (
        {
            "pred_logits": torch.tensor([[[0.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]),
        },
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.0, 0.0]]),
            "boxes_padded": torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]),
            "num_boxes": torch.tensor([1]),
        },
    )


def _clean_case() -> tuple[dict, dict]:
    return (
        {
            "pred_logits": torch.tensor([[[2.0], [-1.0], [0.5]]]),
            "pred_boxes": torch.tensor(
                [[[0.50, 0.50, 0.20, 0.20], [0.30, 0.30, 0.10, 0.15], [0.70, 0.60, 0.25, 0.20]]]
            ),
        },
        {
            "boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20], [0.30, 0.30, 0.10, 0.15]]),
            "boxes_padded": torch.tensor(
                [[[0.50, 0.50, 0.20, 0.20], [0.30, 0.30, 0.10, 0.15]]]
            ),
            "num_boxes": torch.tensor([2]),
        },
    )


def test_zero_area_matching_is_stabilized_and_inputs_untouched(capsys) -> None:
    outputs, targets = _zero_area_case()
    batch_idx, source_idx, target_idx = _stable()(outputs, targets)

    assert batch_idx.tolist() == [0]
    assert source_idx.tolist() == [0]
    assert target_idx is None
    # The real model outputs / loss inputs must be left exactly as they were.
    assert outputs["pred_boxes"][0, 0, 2:].tolist() == [0.0, 0.0]
    assert targets["boxes_padded"][0, 0, 2:].tolist() == [0.0, 0.0]
    # A degenerate-box diagnostic naming the tensor is emitted.
    err = capsys.readouterr().err
    assert GUARD_TAG in err
    assert "degenerate box" in err


def test_non_finite_model_outputs_are_sanitized_not_fatal(capsys) -> None:
    outputs, targets = _zero_area_case()
    outputs["pred_logits"].reshape(-1)[0] = float("nan")
    outputs["pred_boxes"].reshape(-1)[0] = float("inf")

    # Must NOT raise: training has to survive a bad batch.
    batch_idx, source_idx, target_idx = _stable()(outputs, targets)
    assert batch_idx.tolist() == [0]
    assert source_idx.tolist() == [0]

    err = capsys.readouterr().err
    assert GUARD_TAG in err
    assert "non-finite pred_logits" in err
    assert "non-finite pred_boxes" in err
    # Inputs are left untouched for the real loss.
    assert torch.isnan(outputs["pred_logits"].reshape(-1)[0])
    assert torch.isinf(outputs["pred_boxes"].reshape(-1)[0])


def test_extreme_finite_logits_are_bounded() -> None:
    outputs, targets = _zero_area_case()
    outputs["pred_logits"].fill_(torch.finfo(torch.float32).max)
    batch_idx, source_idx, _ = _stable()(outputs, targets)
    assert batch_idx.tolist() == [0]
    assert source_idx.tolist() == [0]


def test_clean_inputs_match_the_base_matcher(capsys) -> None:
    outputs, targets = _clean_case()
    base_batch, base_src, base_tgt = _base()(outputs, targets)
    stable_batch, stable_src, stable_tgt = _stable()(outputs, targets)

    assert stable_batch.tolist() == base_batch.tolist()
    assert stable_src.tolist() == base_src.tolist()
    if base_tgt is None:
        assert stable_tgt is None
    else:
        assert stable_tgt.tolist() == base_tgt.tolist()
    # No warnings on clean, non-degenerate inputs.
    assert GUARD_TAG not in capsys.readouterr().err
