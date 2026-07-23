from __future__ import annotations

from typing import Any

from core.numerically_stable_matcher import NumericallyStableBinaryHungarianMatcherV2


def build_sam3_image_model_with_stable_matcher(**kwargs: Any):
    """Build the stock SAM3 image model, then harden its internal matcher.

    ``sam3/model_builder.py`` hardcodes ``model.matcher = BinaryHungarianMatcherV2``
    for training and exposes no config hook for it, yet that internal matcher —
    called from ``SAM3Image.forward_grounding._compute_matching`` — is exactly
    where training died with ``matrix contains invalid numeric entries``. Swapping
    the loss's matcher via config does NOT touch this one. So we build the model
    normally and replace ``model.matcher`` in place with the numerically stable,
    self-reporting drop-in, mirroring the original's cost weights and options so
    matching behavior on clean inputs is unchanged. The matcher has no learnable
    parameters or buffers, so this does not affect the model state_dict or
    checkpoint resume.
    """
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(**kwargs)

    original = getattr(model, "matcher", None)
    if original is None:
        # eval_mode build: SAM3 leaves matcher=None and never matches. Nothing to do.
        return model

    model.matcher = NumericallyStableBinaryHungarianMatcherV2(
        cost_class=original.cost_class,
        cost_bbox=original.cost_bbox,
        cost_giou=original.cost_giou,
        focal=getattr(original, "focal", False),
        alpha=getattr(original, "alpha", 0.25),
        gamma=getattr(original, "gamma", 2.0),
        stable=getattr(original, "stable", False),
        remove_samples_with_0_gt=getattr(original, "remove_samples_with_0_gt", True),
    )
    return model
