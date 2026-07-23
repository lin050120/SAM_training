from __future__ import annotations

import sys
import types
from unittest import mock

from core.numerically_stable_matcher import NumericallyStableBinaryHungarianMatcherV2


class _FakeMatcher:
    cost_class = 2.0
    cost_bbox = 5.0
    cost_giou = 2.0
    focal = True
    alpha = 0.25
    gamma = 2
    stable = False
    remove_samples_with_0_gt = True


class _FakeModel:
    def __init__(self, matcher):
        self.matcher = matcher


def _install_fake_builder(returned_model, captured_kwargs):
    def _build(**kwargs):
        captured_kwargs.update(kwargs)
        return returned_model

    fake_module = types.ModuleType("sam3.model_builder")
    fake_module.build_sam3_image_model = _build
    return mock.patch.dict(sys.modules, {"sam3.model_builder": fake_module})


def test_wrapper_replaces_internal_matcher_and_forwards_kwargs() -> None:
    from core.sam3_stable_model import build_sam3_image_model_with_stable_matcher

    model = _FakeModel(_FakeMatcher())
    captured: dict = {}
    with _install_fake_builder(model, captured):
        returned = build_sam3_image_model_with_stable_matcher(
            checkpoint_path="/x/sam3.pt", eval_mode=False, device="cpus"
        )

    assert returned is model
    assert captured == {"checkpoint_path": "/x/sam3.pt", "eval_mode": False, "device": "cpus"}
    assert isinstance(model.matcher, NumericallyStableBinaryHungarianMatcherV2)
    # Cost weights / options mirror the original hardcoded matcher.
    assert model.matcher.cost_bbox == 5.0
    assert model.matcher.cost_giou == 2.0
    assert model.matcher.focal is True
    assert model.matcher.remove_samples_with_0_gt is True


def test_wrapper_is_noop_when_model_has_no_matcher() -> None:
    from core.sam3_stable_model import build_sam3_image_model_with_stable_matcher

    model = _FakeModel(matcher=None)
    with _install_fake_builder(model, {}):
        returned = build_sam3_image_model_with_stable_matcher(eval_mode=True)

    assert returned is model
    assert model.matcher is None
