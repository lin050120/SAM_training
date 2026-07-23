from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


ONLINE_AUGMENTATION_OFF = "off"
ONLINE_AUGMENTATION_LIGHT = "light"
ONLINE_AUGMENTATION_CUSTOM = "custom"
ONLINE_AUGMENTATION_PRESETS = {
    ONLINE_AUGMENTATION_OFF,
    ONLINE_AUGMENTATION_LIGHT,
    ONLINE_AUGMENTATION_CUSTOM,
}


@dataclass(frozen=True)
class OnlineAugmentationConfig:
    preset: str = ONLINE_AUGMENTATION_OFF
    repeat_factor: int = 1
    affine_probability: float = 0.0
    rotation_min_degrees: float = 0.0
    rotation_max_degrees: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    translate_fraction: float = 0.0
    horizontal_flip_probability: float = 0.0
    color_jitter_probability: float = 0.0
    color_jitter_strength: float = 0.0
    motion_blur_probability: float = 0.0
    motion_blur_kernel_size: int = 3

    @property
    def enabled(self) -> bool:
        return self.preset != ONLINE_AUGMENTATION_OFF

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["enabled"] = self.enabled
        result["samples_per_source_image"] = self.repeat_factor
        return result


OFF_CONFIG = OnlineAugmentationConfig()
LIGHT_CONFIG = OnlineAugmentationConfig(
    preset=ONLINE_AUGMENTATION_LIGHT,
    repeat_factor=1,
    affine_probability=0.5,
    rotation_min_degrees=-8.0,
    rotation_max_degrees=8.0,
    scale_min=0.9,
    scale_max=1.1,
    translate_fraction=0.05,
    horizontal_flip_probability=0.5,
    color_jitter_probability=0.3,
    color_jitter_strength=0.15,
    motion_blur_probability=0.1,
    motion_blur_kernel_size=3,
)


def _finite_float(raw: Any, field: str, default: float, errors: list[str]) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        errors.append(f"online augmentation {field} must be a number, got {raw!r}")
        return default
    if not math.isfinite(value):
        errors.append(f"online augmentation {field} must be finite, got {raw!r}")
        return default
    return value


def _whole_int(raw: Any, field: str, default: int, errors: list[str]) -> int:
    value = _finite_float(raw, field, float(default), errors)
    if value != int(value):
        errors.append(f"online augmentation {field} must be a whole number, got {value}")
        return default
    return int(value)


def resolve_online_augmentation_config(
    raw: OnlineAugmentationConfig | Mapping[str, Any] | None,
) -> tuple[OnlineAugmentationConfig, list[str]]:
    """Normalize a requested preset and validate custom values fail-closed."""
    if raw is None:
        return OFF_CONFIG, []
    if isinstance(raw, OnlineAugmentationConfig):
        values: Mapping[str, Any] = asdict(raw)
    elif isinstance(raw, Mapping):
        values = raw
    else:
        return OFF_CONFIG, [
            "online augmentation config must be an OnlineAugmentationConfig or mapping"
        ]

    preset = str(values.get("preset", ONLINE_AUGMENTATION_OFF)).strip().lower()
    if preset == ONLINE_AUGMENTATION_OFF:
        return OFF_CONFIG, []
    if preset == ONLINE_AUGMENTATION_LIGHT:
        return LIGHT_CONFIG, []
    if preset not in ONLINE_AUGMENTATION_PRESETS:
        return OFF_CONFIG, [
            f"online augmentation preset must be one of {sorted(ONLINE_AUGMENTATION_PRESETS)}, got {preset!r}"
        ]

    errors: list[str] = []
    defaults = LIGHT_CONFIG
    config = OnlineAugmentationConfig(
        preset=ONLINE_AUGMENTATION_CUSTOM,
        repeat_factor=_whole_int(
            values.get("repeat_factor", defaults.repeat_factor),
            "repeat_factor",
            defaults.repeat_factor,
            errors,
        ),
        affine_probability=_finite_float(
            values.get("affine_probability", defaults.affine_probability),
            "affine_probability",
            defaults.affine_probability,
            errors,
        ),
        rotation_min_degrees=_finite_float(
            values.get("rotation_min_degrees", defaults.rotation_min_degrees),
            "rotation_min_degrees",
            defaults.rotation_min_degrees,
            errors,
        ),
        rotation_max_degrees=_finite_float(
            values.get("rotation_max_degrees", defaults.rotation_max_degrees),
            "rotation_max_degrees",
            defaults.rotation_max_degrees,
            errors,
        ),
        scale_min=_finite_float(
            values.get("scale_min", defaults.scale_min),
            "scale_min",
            defaults.scale_min,
            errors,
        ),
        scale_max=_finite_float(
            values.get("scale_max", defaults.scale_max),
            "scale_max",
            defaults.scale_max,
            errors,
        ),
        translate_fraction=_finite_float(
            values.get("translate_fraction", defaults.translate_fraction),
            "translate_fraction",
            defaults.translate_fraction,
            errors,
        ),
        horizontal_flip_probability=_finite_float(
            values.get(
                "horizontal_flip_probability",
                defaults.horizontal_flip_probability,
            ),
            "horizontal_flip_probability",
            defaults.horizontal_flip_probability,
            errors,
        ),
        color_jitter_probability=_finite_float(
            values.get("color_jitter_probability", defaults.color_jitter_probability),
            "color_jitter_probability",
            defaults.color_jitter_probability,
            errors,
        ),
        color_jitter_strength=_finite_float(
            values.get("color_jitter_strength", defaults.color_jitter_strength),
            "color_jitter_strength",
            defaults.color_jitter_strength,
            errors,
        ),
        motion_blur_probability=_finite_float(
            values.get("motion_blur_probability", defaults.motion_blur_probability),
            "motion_blur_probability",
            defaults.motion_blur_probability,
            errors,
        ),
        motion_blur_kernel_size=3,
    )

    if config.repeat_factor < 1:
        errors.append(
            f"online augmentation repeat_factor must be at least 1, got {config.repeat_factor}"
        )

    bounds = [
        ("affine_probability", config.affine_probability, 0.0, 1.0),
        ("rotation_min_degrees", config.rotation_min_degrees, -180.0, 180.0),
        ("rotation_max_degrees", config.rotation_max_degrees, -180.0, 180.0),
        ("scale_min", config.scale_min, 0.5, 2.0),
        ("scale_max", config.scale_max, 0.5, 2.0),
        ("translate_fraction", config.translate_fraction, 0.0, 0.5),
        (
            "horizontal_flip_probability",
            config.horizontal_flip_probability,
            0.0,
            1.0,
        ),
        ("color_jitter_probability", config.color_jitter_probability, 0.0, 1.0),
        ("color_jitter_strength", config.color_jitter_strength, 0.0, 0.5),
        ("motion_blur_probability", config.motion_blur_probability, 0.0, 1.0),
    ]
    for field, value, minimum, maximum in bounds:
        if not minimum <= value <= maximum:
            errors.append(
                f"online augmentation {field} must be between {minimum} and {maximum}, got {value}"
            )
    if config.rotation_min_degrees > config.rotation_max_degrees:
        errors.append(
            "online augmentation rotation_min_degrees must be <= rotation_max_degrees"
        )
    if config.scale_min > config.scale_max:
        errors.append("online augmentation scale_min must be <= scale_max")
    return config, errors


def build_online_augmentation_transforms(
    config: OnlineAugmentationConfig,
) -> list[dict[str, Any]]:
    if not config.enabled:
        return []
    return [
        {
            "_target_": "sam3.train.transforms.basic_for_api.RandomHorizontalFlip",
            "consistent_transform": True,
            "p": config.horizontal_flip_probability,
        },
        {
            "_target_": "sam3.train.transforms.basic_for_api.RandomSelectAPI",
            "p": config.affine_probability,
            "transforms1": {
                "_target_": "sam3.train.transforms.basic_for_api.RandomAffine",
                "_convert_": "all",
                "degrees": [
                    config.rotation_min_degrees,
                    config.rotation_max_degrees,
                ],
                "scale": [config.scale_min, config.scale_max],
                "translate": [
                    config.translate_fraction,
                    config.translate_fraction,
                ],
                "consistent_transform": True,
                "image_interpolation": "bilinear",
                "num_tentatives": 4,
            },
        },
        {
            "_target_": "sam3.train.transforms.segmentation.RecomputeBoxesFromMasks",
        },
        {
            "_target_": "sam3.train.transforms.basic_for_api.RandomSelectAPI",
            "p": config.color_jitter_probability,
            "transforms1": {
                "_target_": "sam3.train.transforms.basic_for_api.ColorJitter",
                "consistent_transform": True,
                "brightness": config.color_jitter_strength,
                "contrast": config.color_jitter_strength,
                "saturation": config.color_jitter_strength,
                "hue": config.color_jitter_strength,
            },
        },
        {
            "_target_": "sam3.train.transforms.basic_for_api.MotionBlur",
            "kernel_size": config.motion_blur_kernel_size,
            "consistent_transform": True,
            "p": config.motion_blur_probability,
        },
    ]
