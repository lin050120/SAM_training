import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from pycocotools import mask as mask_utils


APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = APP_DIR / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import random_mixup_aug as augmentation  # noqa: E402


def make_args(input_images, input_json, output_dir, **overrides):
    values = {
        "input_images": input_images,
        "input_json": input_json,
        "output_dir": output_dir,
        "augmentations_per_image": 2,
        "num_images": None,
        "mixup_ratio": 0.25,
        "include_originals": True,
        "overwrite": False,
        "seed": 13,
        "segmentation_format": "auto",
        "workers": 1,
        "limit": None,
        "image_id": [],
        "file_name": [],
        "file_contains": [],
        "weight_color_jitter": 0.25,
        "weight_rotate": 0.20,
        "weight_scale": 0.18,
        "weight_crop": 0.18,
        "weight_translate": 0.16,
        "weight_flip": 0.15,
        "weight_noise": 0.14,
        "weight_cutout": 0.08,
        "min_angle": -45.0,
        "max_angle": 45.0,
        "min_scale": 0.80,
        "max_scale": 1.20,
        "min_crop_ratio": 0.85,
        "max_crop_ratio": 0.95,
        "min_noise_sigma": 5.0,
        "max_noise_sigma": 20.0,
        "min_cutout_box_ratio": 0.06,
        "max_cutout_box_ratio": 0.18,
        "min_cutout_boxes": 1,
        "max_cutout_boxes": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_test_dataset(root):
    image_dir = root / "source_images"
    image_dir.mkdir()
    height, width = 72, 96
    images = []
    annotations = []
    rectangles = [(12, 14, 28, 32), (46, 20, 30, 36)]

    for image_id, (x, y, box_width, box_height) in enumerate(rectangles, start=1):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[y : y + box_height, x : x + box_width] = (255, 255, 255)
        file_name = f"source_{image_id}.png"
        assert cv2.imwrite(str(image_dir / file_name), image)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [
                    [
                        x,
                        y,
                        x + box_width,
                        y,
                        x + box_width,
                        y + box_height,
                        x,
                        y + box_height,
                    ]
                ],
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
        )

    data = {
        "info": {"description": "augmentation test"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }
    json_path = root / "source_annotations.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return image_dir, json_path


def decode_annotation(ann, height, width):
    segmentation = ann["segmentation"]
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
    else:
        rle = mask_utils.frPyObjects(segmentation, height, width)
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def synthetic_sample():
    height, width = 96, 128
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[22:70, 30:50] = 1
    mask[50:72, 50:88] = 1
    image = np.repeat((mask * 255)[:, :, None], 3, axis=2)
    ann = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "segmentation": augmentation.encode_mask(mask),
        "bbox": augmentation.mask_to_bbox(mask),
        "area": int(mask.sum()),
        "iscrowd": 0,
    }
    return image, [ann], [mask]


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("rotate", SimpleNamespace(min_angle=31.0, max_angle=31.0)),
        ("scale_down", SimpleNamespace(min_scale=0.78, max_scale=0.78)),
        ("scale_up", SimpleNamespace(min_scale=1.18, max_scale=1.18)),
        ("crop", SimpleNamespace(min_crop_ratio=0.82, max_crop_ratio=0.82)),
        ("translate", SimpleNamespace()),
        ("flip", SimpleNamespace()),
    ],
)
def test_geometric_augmentation_keeps_image_and_mask_aligned(operation, args):
    image, anns, masks = synthetic_sample()
    rng = random.Random(21)

    if operation == "rotate":
        result, result_anns, result_masks = augmentation.apply_rotate(
            image, anns, masks, rng, args
        )
    elif operation.startswith("scale"):
        result, result_anns, result_masks = augmentation.apply_scale(
            image, anns, masks, rng, args
        )
    elif operation == "crop":
        result, result_anns, result_masks = augmentation.apply_crop(
            image, anns, masks, rng, args
        )
    elif operation == "translate":
        result, result_anns, result_masks = augmentation.apply_translate(
            image, anns, masks, rng
        )
    else:
        result, result_anns, result_masks = augmentation.apply_flip(
            image, anns, masks, rng
        )

    assert len(result_anns) == len(result_masks) == 1
    transformed_mask = result_masks[0].astype(bool)
    visible_object = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) >= 128
    union = np.logical_or(transformed_mask, visible_object).sum()
    intersection = np.logical_and(transformed_mask, visible_object).sum()
    assert intersection / union >= 0.95
    one_pixel_kernel = np.ones((3, 3), dtype=np.uint8)
    expanded_mask = cv2.dilate(
        transformed_mask.astype(np.uint8), one_pixel_kernel
    ).astype(bool)
    expanded_object = cv2.dilate(
        visible_object.astype(np.uint8), one_pixel_kernel
    ).astype(bool)
    assert np.all(~visible_object | expanded_mask)
    assert np.all(~transformed_mask | expanded_object)

    finalized = augmentation.finalize_annotations(result_anns, result_masks, "rle", {})
    encoded_mask = decode_annotation(finalized[0], *transformed_mask.shape)
    assert np.array_equal(encoded_mask.astype(bool), transformed_mask)
    assert result_anns[0]["bbox"] == augmentation.mask_to_bbox(result_masks[0])
    assert result_anns[0]["area"] == pytest.approx(float(result_masks[0].sum()))


def test_generates_complete_train_dataset_from_arbitrary_coco_input(tmp_path):
    input_images, input_json = write_test_dataset(tmp_path)
    output_dir = tmp_path / "augmented_train"

    augmentation.run(make_args(input_images, input_json, output_dir))

    output = json.loads((output_dir / "annotations.json").read_text(encoding="utf-8"))
    stats = json.loads(
        (output_dir / "augmentation_stats.json").read_text(encoding="utf-8")
    )
    image_ids = [image["id"] for image in output["images"]]
    annotation_ids = [ann["id"] for ann in output["annotations"]]
    assert len(output["images"]) == 6
    assert len(set(image_ids)) == len(image_ids)
    assert len(set(annotation_ids)) == len(annotation_ids)
    assert stats["source_image_count"] == 2
    assert stats["generated_count"] == 4
    assert stats["standard_count"] == 3
    assert stats["mixup_count"] == 1
    assert stats["final_image_count"] == 6

    known_images = {image["id"]: image for image in output["images"]}
    for image_info in output["images"]:
        assert (output_dir / "images" / image_info["file_name"]).is_file()
    for ann in output["annotations"]:
        image_info = known_images[ann["image_id"]]
        mask = decode_annotation(ann, image_info["height"], image_info["width"])
        assert mask.shape == (image_info["height"], image_info["width"])
        assert np.any(mask)


def load_generated_annotations(output_dir):
    output = json.loads((output_dir / "annotations.json").read_text(encoding="utf-8"))
    images_by_id = {image["id"]: image for image in output["images"]}
    generated = [
        ann
        for ann in output["annotations"]
        if "attributes" in ann and "augmentation_type" in ann["attributes"]
    ]
    assert generated
    return output, images_by_id, generated


@pytest.mark.parametrize(
    ("segmentation_format", "expected_type"),
    [("auto", list), ("polygon", list), ("rle", dict)],
)
def test_generated_segmentation_format_matches_choice(
    tmp_path, segmentation_format, expected_type
):
    input_images, input_json = write_test_dataset(tmp_path)
    output_dir = tmp_path / "augmented_train"

    augmentation.run(
        make_args(
            input_images,
            input_json,
            output_dir,
            segmentation_format=segmentation_format,
        )
    )

    _, images_by_id, generated = load_generated_annotations(output_dir)
    for ann in generated:
        segmentation = ann["segmentation"]
        assert isinstance(segmentation, expected_type)
        if isinstance(segmentation, dict):
            assert isinstance(segmentation["counts"], str)
        image_info = images_by_id[ann["image_id"]]
        mask = decode_annotation(ann, image_info["height"], image_info["width"])
        assert np.any(mask)
        assert ann["bbox"] == augmentation.mask_to_bbox(mask)
        assert ann["area"] == pytest.approx(float(mask.sum()))


def test_auto_format_preserves_uncompressed_rle_source(tmp_path):
    input_images, input_json = write_test_dataset(tmp_path)
    data = json.loads(input_json.read_text(encoding="utf-8"))
    for ann in data["annotations"]:
        image_info = next(
            image for image in data["images"] if image["id"] == ann["image_id"]
        )
        mask = decode_annotation(ann, image_info["height"], image_info["width"])
        ann["segmentation"] = augmentation.encode_uncompressed_rle(mask)
    input_json.write_text(json.dumps(data), encoding="utf-8")
    output_dir = tmp_path / "augmented_train"

    augmentation.run(make_args(input_images, input_json, output_dir))

    _, images_by_id, generated = load_generated_annotations(output_dir)
    for ann in generated:
        segmentation = ann["segmentation"]
        assert isinstance(segmentation, dict)
        assert isinstance(segmentation["counts"], list)
        image_info = images_by_id[ann["image_id"]]
        mask = decode_annotation(ann, image_info["height"], image_info["width"])
        assert np.any(mask)


def test_uncompressed_rle_roundtrip_matches_mask():
    rng = np.random.default_rng(3)
    mask = (rng.random((37, 53)) > 0.6).astype(np.uint8)
    encoded = augmentation.encode_uncompressed_rle(mask)
    decoded = decode_annotation({"segmentation": encoded}, 37, 53)
    assert np.array_equal(decoded.astype(bool), mask.astype(bool))


def test_polygon_conversion_stays_close_to_mask():
    image, anns, masks = synthetic_sample()
    converted = augmentation.finalize_annotations(anns, masks, "polygon", {})[0]
    assert isinstance(converted["segmentation"], list)
    height, width = masks[0].shape
    polygon_mask = decode_annotation(converted, height, width).astype(bool)
    original = masks[0].astype(bool)
    intersection = np.logical_and(polygon_mask, original).sum()
    union = np.logical_or(polygon_mask, original).sum()
    assert intersection / union >= 0.95
    assert converted["bbox"] == augmentation.mask_to_bbox(polygon_mask)
    assert converted["area"] == pytest.approx(float(polygon_mask.sum()))


def test_parallel_workers_produce_identical_output(tmp_path):
    input_images, input_json = write_test_dataset(tmp_path)
    single_dir = tmp_path / "augmented_single"
    parallel_dir = tmp_path / "augmented_parallel"

    augmentation.run(make_args(input_images, input_json, single_dir, workers=1))
    augmentation.run(make_args(input_images, input_json, parallel_dir, workers=2))

    single = json.loads((single_dir / "annotations.json").read_text(encoding="utf-8"))
    parallel = json.loads(
        (parallel_dir / "annotations.json").read_text(encoding="utf-8")
    )
    assert single == parallel
    for image_info in single["images"]:
        single_bytes = (single_dir / "images" / image_info["file_name"]).read_bytes()
        parallel_bytes = (
            parallel_dir / "images" / image_info["file_name"]
        ).read_bytes()
        assert single_bytes == parallel_bytes


def test_nonempty_output_requires_overwrite_and_is_backed_up(tmp_path):
    input_images, input_json = write_test_dataset(tmp_path)
    output_dir = tmp_path / "augmented_train"
    args = make_args(input_images, input_json, output_dir, mixup_ratio=0.0)
    augmentation.run(args)
    original_json = (output_dir / "annotations.json").read_bytes()

    with pytest.raises(FileExistsError, match="not empty"):
        augmentation.run(args)
    assert (output_dir / "annotations.json").read_bytes() == original_json

    args.overwrite = True
    augmentation.run(args)
    backups = list(tmp_path.glob("augmented_train.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "annotations.json").read_bytes() == original_json


def test_rejects_coco_dimensions_that_do_not_match_image(tmp_path):
    input_images, input_json = write_test_dataset(tmp_path)
    data = json.loads(input_json.read_text(encoding="utf-8"))
    data["images"][0]["width"] += 1
    input_json.write_text(json.dumps(data), encoding="utf-8")
    output_dir = tmp_path / "augmented_train"

    with pytest.raises(ValueError, match="dimensions do not match"):
        augmentation.run(make_args(input_images, input_json, output_dir))
    assert not output_dir.exists()
