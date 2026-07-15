from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image as PILImage

from core.config import (
    DEFAULT_BOOK_SPINE_DATASET_ROOT,
    DEFAULT_BOOK_SPINE_FINETUNE_CONFIG,
    DEFAULT_SAM3_BPE_PATH,
    DEFAULT_SAM3_CHECKPOINT,
)
from core.online_augmentation import (
    LIGHT_CONFIG,
    OFF_CONFIG,
    OnlineAugmentationConfig,
    resolve_online_augmentation_config,
)
from core.training_runner import inspect_training_config, write_runtime_yaml


def _runtime_paths() -> dict[str, Path]:
    return {
        "initial_checkpoint": DEFAULT_SAM3_CHECKPOINT,
        "bpe_path": DEFAULT_SAM3_BPE_PATH,
        "train_images": DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "images",
        "train_annotations": DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "annotations.json",
        "val_images": DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "images",
        "val_annotations": DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "annotations.json",
    }


def _write_runtime(tmp_path: Path, augmentation: OnlineAugmentationConfig):
    runtime_path = tmp_path / "config" / "runtime.yaml"
    write_runtime_yaml(
        DEFAULT_BOOK_SPINE_FINETUNE_CONFIG,
        runtime_path,
        _runtime_paths(),
        tmp_path / "run",
        online_augmentation=augmentation,
    )
    return OmegaConf.load(runtime_path)


def _targets(cfg, split: str) -> list[str]:
    transforms = OmegaConf.select(cfg, f"book_spine.{split}_transforms.0.transforms")
    return [str(item._target_) for item in transforms]


def test_off_preserves_base_train_and_val_transforms(tmp_path: Path) -> None:
    base = OmegaConf.load(DEFAULT_BOOK_SPINE_FINETUNE_CONFIG)
    runtime = _write_runtime(tmp_path, OFF_CONFIG)
    for split in ("train", "val"):
        assert OmegaConf.to_container(
            OmegaConf.select(runtime, f"book_spine.{split}_transforms"), resolve=False
        ) == OmegaConf.to_container(
            OmegaConf.select(base, f"book_spine.{split}_transforms"), resolve=False
        )
    assert runtime.trainer.data.train.dataset._target_ == (
        "sam3.train.data.sam3_image_dataset.Sam3ImageDataset"
    )
    assert "repeat_factor" not in runtime.trainer.data.train.dataset
    assert runtime.online_augmentation.preset == "off"


def test_light_inserts_expected_train_order_and_leaves_val_unchanged(tmp_path: Path) -> None:
    base = OmegaConf.load(DEFAULT_BOOK_SPINE_FINETUNE_CONFIG)
    runtime = _write_runtime(tmp_path, LIGHT_CONFIG)
    assert _targets(runtime, "train") == [
        "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
        "sam3.train.transforms.point_sampling.RandomizeInputBbox",
        "sam3.train.transforms.segmentation.DecodeRle",
        "sam3.train.transforms.basic_for_api.RandomHorizontalFlip",
        "sam3.train.transforms.basic_for_api.RandomSelectAPI",
        "sam3.train.transforms.segmentation.RecomputeBoxesFromMasks",
        "sam3.train.transforms.basic_for_api.RandomSelectAPI",
        "sam3.train.transforms.basic_for_api.MotionBlur",
        "sam3.train.transforms.basic_for_api.RandomResizeAPI",
        "sam3.train.transforms.basic_for_api.PadToSizeAPI",
        "sam3.train.transforms.basic_for_api.ToTensorAPI",
        "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
        "sam3.train.transforms.basic_for_api.NormalizeAPI",
        "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
    ]
    assert OmegaConf.to_container(runtime.book_spine.val_transforms, resolve=False) == (
        OmegaConf.to_container(base.book_spine.val_transforms, resolve=False)
    )
    inserted = runtime.book_spine.train_transforms[0].transforms
    assert inserted[4].p == pytest.approx(0.5)
    assert list(inserted[4].transforms1.degrees) == [-8.0, 8.0]
    assert list(inserted[4].transforms1.scale) == [0.9, 1.1]
    assert list(inserted[4].transforms1.translate) == [0.05, 0.05]
    assert inserted[6].p == pytest.approx(0.3)
    assert inserted[7].p == pytest.approx(0.1)
    assert inserted[7].kernel_size == 3


def test_repeat_factor_switches_only_train_dataset_target(tmp_path: Path) -> None:
    requested = OnlineAugmentationConfig(
        preset="custom",
        repeat_factor=3,
        affine_probability=0.5,
        rotation_min_degrees=-10,
        rotation_max_degrees=10,
        scale_min=0.8,
        scale_max=1.2,
        translate_fraction=0.1,
        horizontal_flip_probability=0.5,
        color_jitter_probability=0.2,
        color_jitter_strength=0.1,
        motion_blur_probability=0.1,
    )
    runtime = _write_runtime(tmp_path, requested)
    assert runtime.trainer.data.train.dataset._target_ == (
        "core.training_augmentation.RepeatedSam3ImageDataset"
    )
    assert runtime.trainer.data.train.dataset.repeat_factor == 3
    assert runtime.trainer.data.val.dataset._target_ == (
        "sam3.train.data.sam3_image_dataset.Sam3ImageDataset"
    )
    assert "repeat_factor" not in runtime.trainer.data.val.dataset


def test_hydra_instantiates_repeated_dataset_and_online_transforms(tmp_path: Path) -> None:
    from hydra.utils import instantiate

    requested = OnlineAugmentationConfig(
        preset="custom",
        repeat_factor=2,
        affine_probability=0.5,
        rotation_min_degrees=-8,
        rotation_max_degrees=8,
        scale_min=0.9,
        scale_max=1.1,
        translate_fraction=0.05,
        horizontal_flip_probability=0.5,
        color_jitter_probability=0.3,
        color_jitter_strength=0.15,
        motion_blur_probability=0.1,
    )
    runtime = _write_runtime(tmp_path, requested)
    dataset = instantiate(runtime.trainer.data.train.dataset)
    assert type(dataset).__module__ == "core.training_augmentation"
    assert type(dataset).__name__ == "RepeatedSam3ImageDataset"
    assert len(dataset) == dataset._source_length * 2
    assert len(dataset._transforms[0].transforms) == 14


@pytest.mark.parametrize(
    "changes, expected_error",
    [
        ({"repeat_factor": 11}, "repeat_factor"),
        ({"rotation_min_degrees": 20, "rotation_max_degrees": -20}, "rotation_min_degrees"),
        ({"scale_min": 1.5, "scale_max": 1.0}, "scale_min"),
        ({"translate_fraction": 0.6}, "translate_fraction"),
        ({"color_jitter_strength": 0.6}, "color_jitter_strength"),
        ({"affine_probability": float("nan")}, "must be finite"),
    ],
)
def test_custom_boundaries_fail_closed(changes: dict, expected_error: str) -> None:
    values = {
        "preset": "custom",
        "repeat_factor": 1,
        "affine_probability": 0.5,
        "rotation_min_degrees": -8,
        "rotation_max_degrees": 8,
        "scale_min": 0.9,
        "scale_max": 1.1,
        "translate_fraction": 0.05,
        "horizontal_flip_probability": 0.5,
        "color_jitter_probability": 0.3,
        "color_jitter_strength": 0.15,
        "motion_blur_probability": 0.1,
    }
    values.update(changes)
    _resolved, errors = resolve_online_augmentation_config(values)
    assert any(expected_error in error for error in errors), errors


def test_core_preflight_rejects_invalid_custom_config() -> None:
    preflight = inspect_training_config(
        max_epochs=1,
        prepare_runtime=False,
        online_augmentation={
            "preset": "custom",
            "repeat_factor": 1,
            "rotation_min_degrees": -8,
            "rotation_max_degrees": 8,
            "scale_min": 0.9,
            "scale_max": 1.1,
            "translate_fraction": 0.8,
        },
    )
    assert any("translate_fraction" in error for error in preflight.errors)


def test_preflight_and_training_summary_record_final_augmentation(tmp_path: Path) -> None:
    from ui.process_manager import ProcessState
    from ui.training_process_manager import finalize_training_summary

    distributed = {
        "master_addr": "localhost",
        "master_port": 43111,
        "port_range": [43111, 43111],
    }
    with mock.patch(
        "core.training_runner.allocate_distributed_port", return_value=43111
    ), mock.patch(
        "core.training_runner.configure_runtime_distributed_port",
        return_value=distributed,
    ):
        preflight = inspect_training_config(
            max_epochs=1,
            output_root=tmp_path,
            allow_external_output=True,
            prepare_runtime=True,
            collect_import_metadata=False,
            online_augmentation=LIGHT_CONFIG,
        )
    assert preflight.errors == []
    run_dir = Path(preflight.run_dir)
    dataset_info = OmegaConf.load(run_dir / "dataset_info.json")
    config_summary = OmegaConf.load(run_dir / "training_config_summary.json")
    runtime = OmegaConf.load(run_dir / "config" / "runtime_config.yaml")
    for recorded in (
        preflight.online_augmentation,
        dataset_info.online_augmentation,
        config_summary.online_augmentation,
        runtime.online_augmentation,
    ):
        assert recorded["preset"] == "light"
        assert recorded["rotation_min_degrees"] == -8.0
        assert recorded["rotation_max_degrees"] == 8.0

    state = ProcessState(
        running=False,
        returncode=0,
        started_at=1.0,
        finished_at=2.0,
    )
    summary = finalize_training_summary(
        run_dir,
        ["fake"],
        str(preflight.runtime_config_path),
        preflight.initial_checkpoint,
        state=state,
        training_provenance={"patch_guard_ok": True},
    )
    assert summary["online_augmentation"]["preset"] == "light"
    assert summary["attempts"][0]["online_augmentation"]["preset"] == "light"


def test_repeated_dataset_maps_indices_and_restores_length_mode() -> None:
    from core.training_augmentation import RepeatedSam3ImageDataset
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset

    dataset = object.__new__(RepeatedSam3ImageDataset)
    dataset.repeat_factor = 4
    dataset._source_length = 3
    dataset._fetching_source_item = False
    with mock.patch.object(Sam3ImageDataset, "__getitem__", autospec=True) as parent_get:
        parent_get.side_effect = lambda _self, idx: {"source_index": idx}
        assert len(dataset) == 12
        assert dataset[7] == {"source_index": 1}
        parent_get.assert_called_once_with(dataset, 1)
    assert dataset._fetching_source_item is False
    assert len(dataset) == 12


def test_fixed_affine_keeps_image_and_mask_aligned_and_recomputes_metadata() -> None:
    from sam3.model.box_ops import masks_to_boxes
    from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object
    from sam3.train.transforms.basic_for_api import RandomAffine
    from sam3.train.transforms.segmentation import RecomputeBoxesFromMasks

    mask = torch.zeros((24, 24), dtype=torch.uint8)
    mask[5:16, 7:14] = 1
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[mask.numpy().astype(bool)] = 255
    obj = Object(
        bbox=torch.tensor([[7.0, 5.0, 13.0, 15.0]]),
        area=float(mask.sum()),
        segment=mask,
    )
    datapoint = Datapoint(
        find_queries=[],
        images=[Image(data=PILImage.fromarray(rgb), objects=[obj], size=(24, 24))],
    )
    transform = RandomAffine(
        degrees=[90.0, 90.0],
        scale=[1.0, 1.0],
        translate=[0.0, 0.0],
        consistent_transform=True,
        image_interpolation="bilinear",
    )
    transformed = RecomputeBoxesFromMasks()(transform(datapoint))
    transformed_mask = transformed.images[0].objects[0].segment
    transformed_image = np.asarray(transformed.images[0].data)[:, :, 0] > 200
    mask_pixels = transformed_mask.numpy().astype(bool)
    intersection = np.logical_and(transformed_image, mask_pixels).sum()
    union = np.logical_or(transformed_image, mask_pixels).sum()
    assert intersection / union > 0.98
    assert transformed_mask.dtype == torch.uint8
    assert set(torch.unique(transformed_mask).tolist()) <= {0, 1}
    expected_box = masks_to_boxes(transformed_mask)
    assert torch.equal(transformed.images[0].objects[0].bbox, expected_box)
    assert transformed.images[0].objects[0].area == int(transformed_mask.sum())


def test_ui_preset_updates_and_japanese_labels_are_registered() -> None:
    from ui.i18n import JA
    from ui.training_preflight_page import augmentation_controls_for_preset

    light_updates = augmentation_controls_for_preset("light")
    custom_updates = augmentation_controls_for_preset("custom")
    assert len(light_updates) == 11
    assert light_updates[0]["value"] == 1
    assert light_updates[1]["value"] == pytest.approx(0.5)
    assert all(update["interactive"] is False for update in light_updates)
    assert all(update["interactive"] is True for update in custom_updates)
    assert JA["增强预设"] == "拡張プリセット"
    assert "在线训练数据增强" in next(
        key for key in JA if "在线训练数据增强" in key
    )
