import json
import os
import random
import shutil
import sys
import uuid
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from aug_common import (  # noqa: E402
    add_common_args,
    copied_file_name,
    decode_mask,
    encode_mask,
    encode_uncompressed_rle,
    mask_to_bbox,
    mask_to_polygons,
    read_image,
    write_image,
)


SAMPLE_INPUT_NAME = "peteck202401-202412_133_20260503"
DEFAULT_INPUT_JSON = APP_DIR / "inputs" / f"{SAMPLE_INPUT_NAME}_mod.json"
DEFAULT_INPUT_IMAGES = APP_DIR / "inputs" / SAMPLE_INPUT_NAME
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs" / "augmented_train"


METHOD_WEIGHTS = {
    "color_jitter": 0.25,
    "rotate": 0.20,
    "scale": 0.18,
    "crop": 0.18,
    "translate": 0.16,
    "flip": 0.15,
    "noise": 0.14,
    "cutout": 0.08,
}
METHOD_ORDER = [
    "crop",
    "scale",
    "translate",
    "rotate",
    "flip",
    "color_jitter",
    "noise",
    "cutout",
]
OBJECT_HUE_SHIFTS = tuple(range(0, 360, 30))
MAX_TRANSLATE_RATIO = 0.05
MIN_TRANSLATE_PIXELS = 20


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create random 2-3 augmentation COCO data with optional strict MixUp."
    )
    add_common_args(parser)
    parser.add_argument("--input-images", type=Path, default=DEFAULT_INPUT_IMAGES)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--augmentations-per-image", type=int, default=5)
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Legacy total generated-image count; overrides --augmentations-per-image.",
    )
    parser.add_argument("--mixup-ratio", type=float, default=0.0)
    parser.add_argument(
        "--segmentation-format",
        choices=("auto", "polygon", "rle"),
        default="auto",
        help=(
            "Output segmentation format for generated annotations: "
            "auto keeps each source annotation's format (polygon, uncompressed "
            "RLE, or compressed RLE), polygon writes COCO point lists, rle "
            "writes compressed RLE."
        ),
    )
    parser.add_argument(
        "--include-originals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include copied source images and their original annotations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Back up and replace an existing output directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel worker processes for generation. Default and 0 use all "
            "CPU cores; 1 disables multiprocessing. Results are identical for "
            "any worker count."
        ),
    )
    parser.add_argument(
        "--weight-color-jitter", type=float, default=METHOD_WEIGHTS["color_jitter"]
    )
    parser.add_argument("--weight-rotate", type=float, default=METHOD_WEIGHTS["rotate"])
    parser.add_argument("--weight-scale", type=float, default=METHOD_WEIGHTS["scale"])
    parser.add_argument("--weight-crop", type=float, default=METHOD_WEIGHTS["crop"])
    parser.add_argument(
        "--weight-translate", type=float, default=METHOD_WEIGHTS["translate"]
    )
    parser.add_argument("--weight-flip", type=float, default=METHOD_WEIGHTS["flip"])
    parser.add_argument("--weight-noise", type=float, default=METHOD_WEIGHTS["noise"])
    parser.add_argument("--weight-cutout", type=float, default=METHOD_WEIGHTS["cutout"])
    parser.add_argument("--min-angle", type=float, default=-90.0)
    parser.add_argument("--max-angle", type=float, default=90.0)
    parser.add_argument("--min-scale", type=float, default=0.85)
    parser.add_argument("--max-scale", type=float, default=1.15)
    parser.add_argument("--min-crop-ratio", type=float, default=0.85)
    parser.add_argument("--max-crop-ratio", type=float, default=0.95)
    parser.add_argument("--min-noise-sigma", type=float, default=5.0)
    parser.add_argument("--max-noise-sigma", type=float, default=20.0)
    parser.add_argument("--min-cutout-box-ratio", type=float, default=0.06)
    parser.add_argument("--max-cutout-box-ratio", type=float, default=0.18)
    parser.add_argument("--min-cutout-boxes", type=int, default=1)
    parser.add_argument("--max-cutout-boxes", type=int, default=3)
    return parser.parse_args()


def filter_data(data, args):
    selected_images = data["images"]

    if args.limit is not None:
        selected_images = selected_images[: args.limit]

    if args.image_id:
        image_ids = set(args.image_id)
        selected_images = [
            image_info
            for image_info in selected_images
            if image_info["id"] in image_ids
        ]

    if args.file_name:
        file_names = set(args.file_name)
        selected_images = [
            image_info
            for image_info in selected_images
            if image_info["file_name"] in file_names
        ]

    if args.file_contains:
        selected_images = [
            image_info
            for image_info in selected_images
            if any(text in image_info["file_name"] for text in args.file_contains)
        ]

    selected_image_ids = {image_info["id"] for image_info in selected_images}
    if not selected_image_ids:
        raise ValueError("No images matched the selected filters.")

    filtered_data = deepcopy(data)
    filtered_data["images"] = selected_images
    filtered_data["annotations"] = [
        ann for ann in data["annotations"] if ann["image_id"] in selected_image_ids
    ]
    return filtered_data


def resolve_dataset_image(image_dir, file_name):
    raw = Path(str(file_name))
    if raw.is_absolute():
        raise ValueError(f"COCO file_name must be relative: {file_name}")
    root = image_dir.resolve(strict=False)
    resolved = (root / raw).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"COCO file_name escapes the image directory: {file_name}"
        ) from exc
    return resolved


def validate_coco_dataset(data, image_dir):
    if not isinstance(data, dict):
        raise ValueError("COCO JSON root must be an object.")
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"COCO JSON must contain a {key} list.")
    if not data["images"]:
        raise ValueError("COCO images list must not be empty.")
    if not data["annotations"]:
        raise ValueError("COCO annotations list must not be empty.")
    if not data["categories"]:
        raise ValueError("COCO categories list must not be empty.")

    image_ids = [image.get("id") for image in data["images"]]
    annotation_ids = [ann.get("id") for ann in data["annotations"]]
    category_ids = [category.get("id") for category in data["categories"]]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in image_ids
    ):
        raise ValueError("COCO image ids must be integers.")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in annotation_ids
    ):
        raise ValueError("COCO annotation ids must be integers.")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in category_ids
    ):
        raise ValueError("COCO category ids must be integers.")
    if None in image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("COCO image ids must be present and unique.")
    if None in annotation_ids or len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("COCO annotation ids must be present and unique.")
    if None in category_ids or len(set(category_ids)) != len(category_ids):
        raise ValueError("COCO category ids must be present and unique.")

    known_images = set(image_ids)
    known_categories = set(category_ids)
    image_info_by_id = {image_info["id"]: image_info for image_info in data["images"]}
    for category in data["categories"]:
        if not str(category.get("name", "")).strip():
            raise ValueError(f"COCO category {category['id']} is missing a name.")
    for image_info in data["images"]:
        for key in ("file_name", "width", "height"):
            if image_info.get(key) in (None, ""):
                raise ValueError(f"COCO image {image_info.get('id')} is missing {key}.")
        image_path = resolve_dataset_image(image_dir, image_info["file_name"])
        image = read_image(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read COCO image: {image_path}")
        height, width = image.shape[:2]
        if int(image_info["width"]) != width or int(image_info["height"]) != height:
            raise ValueError(
                f"COCO dimensions do not match {image_path}: "
                f"JSON={image_info['width']}x{image_info['height']}, actual={width}x{height}"
            )
    for ann in data["annotations"]:
        for key in ("image_id", "category_id", "segmentation", "bbox"):
            if key not in ann:
                raise ValueError(f"COCO annotation {ann.get('id')} is missing {key}.")
        if ann["image_id"] not in known_images:
            raise ValueError(
                f"Annotation {ann['id']} references unknown image_id {ann['image_id']}."
            )
        if ann["category_id"] not in known_categories:
            raise ValueError(
                f"Annotation {ann['id']} references unknown category_id {ann['category_id']}."
            )
        bbox = ann["bbox"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            raise ValueError(f"COCO annotation {ann['id']} has an invalid bbox.")
        image_info = image_info_by_id[ann["image_id"]]
        try:
            mask = decode_mask(
                ann["segmentation"], image_info["height"], image_info["width"]
            )
        except Exception as exc:
            raise ValueError(
                f"Could not decode segmentation for COCO annotation {ann['id']}."
            ) from exc
        if mask.shape != (image_info["height"], image_info["width"]):
            raise ValueError(
                f"COCO annotation {ann['id']} mask dimensions do not match its image."
            )
        if not np.any(mask):
            raise ValueError(f"COCO annotation {ann['id']} has an empty segmentation.")


def weighted_sample_without_replacement(rng, weights, count):
    available = [(name, weight) for name, weight in weights.items() if weight > 0]
    selected = []
    for _ in range(count):
        total = sum(weight for _, weight in available)
        if total <= 0:
            break
        pick = rng.uniform(0.0, total)
        cursor = 0.0
        for index, (name, weight) in enumerate(available):
            cursor += weight
            if cursor >= pick:
                selected.append(name)
                available.pop(index)
                break
    return set(selected)


def method_weights_from_args(args):
    weights = {
        "color_jitter": args.weight_color_jitter,
        "rotate": args.weight_rotate,
        "scale": args.weight_scale,
        "crop": args.weight_crop,
        "translate": args.weight_translate,
        "flip": args.weight_flip,
        "noise": args.weight_noise,
        "cutout": args.weight_cutout,
    }
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Augmentation weights must be zero or positive.")
    if sum(weights.values()) <= 0:
        raise ValueError("At least one augmentation weight must be greater than zero.")
    positive_count = sum(1 for weight in weights.values() if weight > 0)
    if positive_count < 3:
        raise ValueError(
            "At least three augmentation weights must be greater than zero."
        )
    return weights


def choose_methods(rng, method_weights):
    method_count = 2 if rng.random() < 0.65 else 3
    selected = weighted_sample_without_replacement(rng, method_weights, method_count)
    return [name for name in METHOD_ORDER if name in selected]


def union_mask(masks, shape):
    object_mask = np.zeros(shape, dtype=bool)
    for mask in masks:
        object_mask |= mask.astype(bool)
    return object_mask


def apply_color_jitter(img, masks, rng):
    object_mask = union_mask(masks, img.shape[:2])
    if not np.any(object_mask):
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shifted_hsv = hsv.copy()
    hue_shift = rng.choice(OBJECT_HUE_SHIFTS) // 2
    shifted_hsv[..., 0] = (
        (shifted_hsv[..., 0].astype(np.int16) + hue_shift) % 180
    ).astype(np.uint8)
    shifted = cv2.cvtColor(shifted_hsv, cv2.COLOR_HSV2BGR)

    result = img.copy()
    result[object_mask] = shifted[object_mask]
    return result


def apply_noise(img, np_rng, args):
    sigma = float(np_rng.uniform(args.min_noise_sigma, args.max_noise_sigma))
    noise = np_rng.standard_normal(img.shape, dtype=np.float32) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def choose_crop_box(width, height, anns, rng, args):
    if not anns:
        return 0, 0, width, height

    x_min = min(float(ann["bbox"][0]) for ann in anns)
    y_min = min(float(ann["bbox"][1]) for ann in anns)
    x_max = max(float(ann["bbox"][0] + ann["bbox"][2]) for ann in anns)
    y_max = max(float(ann["bbox"][1] + ann["bbox"][3]) for ann in anns)

    target_ratio = rng.uniform(args.min_crop_ratio, args.max_crop_ratio)
    crop_w = max(int(width * target_ratio), int(np.ceil(x_max - x_min)))
    crop_h = max(int(height * target_ratio), int(np.ceil(y_max - y_min)))
    crop_w = min(width, max(1, crop_w))
    crop_h = min(height, max(1, crop_h))

    min_x0 = max(0, int(np.ceil(x_max - crop_w)))
    max_x0 = min(int(np.floor(x_min)), width - crop_w)
    min_y0 = max(0, int(np.ceil(y_max - crop_h)))
    max_y0 = min(int(np.floor(y_min)), height - crop_h)

    x0 = 0 if min_x0 > max_x0 else rng.randint(min_x0, max_x0)
    y0 = 0 if min_y0 > max_y0 else rng.randint(min_y0, max_y0)
    if min_x0 > max_x0:
        crop_w = width
    if min_y0 > max_y0:
        crop_h = height

    return x0, y0, x0 + crop_w, y0 + crop_h


def crop_and_resize(array, crop_box, output_size, interpolation):
    x0, y0, x1, y1 = crop_box
    cropped = array[y0:y1, x0:x1]
    out_width, out_height = output_size
    return cv2.resize(cropped, (out_width, out_height), interpolation=interpolation)


def transform_masks(anns, masks, transform_fn):
    # segmentation stays stale here on purpose: it is rebuilt once from the
    # final mask in finalize_annotations, so intermediate steps only need to
    # keep bbox/area in sync for the transforms that read them.
    transformed_anns = []
    transformed_masks = []
    for ann, mask in zip(anns, masks):
        new_mask = transform_fn(mask)
        new_mask = (new_mask > 0).astype(np.uint8)
        bbox = mask_to_bbox(new_mask)
        if bbox is None:
            continue

        new_ann = deepcopy(ann)
        new_ann["bbox"] = bbox
        new_ann["area"] = float(new_mask.sum())
        transformed_anns.append(new_ann)
        transformed_masks.append(new_mask)

    return transformed_anns, transformed_masks


def choose_translate_shift(width, height, anns, rng):
    max_dx = max(MIN_TRANSLATE_PIXELS, int(width * MAX_TRANSLATE_RATIO))
    max_dy = max(MIN_TRANSLATE_PIXELS, int(height * MAX_TRANSLATE_RATIO))

    min_dx = -max_dx
    max_dx_allowed = max_dx
    min_dy = -max_dy
    max_dy_allowed = max_dy

    for ann in anns:
        x, y, w, h = ann["bbox"]
        min_dx = max(min_dx, int(np.ceil(-x)))
        max_dx_allowed = min(max_dx_allowed, int(np.floor(width - (x + w))))
        min_dy = max(min_dy, int(np.ceil(-y)))
        max_dy_allowed = min(max_dy_allowed, int(np.floor(height - (y + h))))

    if min_dx > max_dx_allowed or min_dy > max_dy_allowed:
        return 0, 0
    if min_dx == max_dx_allowed == 0 and min_dy == max_dy_allowed == 0:
        return 0, 0

    while True:
        dx = rng.randint(min_dx, max_dx_allowed)
        dy = rng.randint(min_dy, max_dy_allowed)
        if dx != 0 or dy != 0:
            return dx, dy


def translate_array(array, dx, dy, interpolation, border_value):
    height, width = array.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def apply_translate(img, anns, masks, rng):
    height, width = img.shape[:2]
    dx, dy = choose_translate_shift(width, height, anns, rng)
    translated_img = translate_array(img, dx, dy, cv2.INTER_LINEAR, (0, 0, 0))

    def transform_mask(mask):
        return translate_array(mask, dx, dy, cv2.INTER_NEAREST, 0)

    new_anns, new_masks = transform_masks(anns, masks, transform_mask)
    return translated_img, new_anns, new_masks


def apply_crop(img, anns, masks, rng, args):
    height, width = img.shape[:2]
    crop_box = choose_crop_box(width, height, anns, rng, args)
    cropped_img = crop_and_resize(img, crop_box, (width, height), cv2.INTER_LINEAR)

    def transform_mask(mask):
        return crop_and_resize(mask, crop_box, (width, height), cv2.INTER_NEAREST)

    new_anns, new_masks = transform_masks(anns, masks, transform_mask)
    return cropped_img, new_anns, new_masks


def apply_rotate(img, anns, masks, rng, args):
    height, width = img.shape[:2]
    angle = rng.uniform(args.min_angle, args.max_angle)
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    def transform_mask(mask):
        return cv2.warpAffine(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    new_anns, new_masks = transform_masks(anns, masks, transform_mask)
    return rotated_img, new_anns, new_masks


def apply_scale(img, anns, masks, rng, args):
    height, width = img.shape[:2]
    scale = rng.uniform(args.min_scale, args.max_scale)
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
    scaled_img = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    def transform_mask(mask):
        return cv2.warpAffine(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    new_anns, new_masks = transform_masks(anns, masks, transform_mask)
    return scaled_img, new_anns, new_masks


def apply_flip(img, anns, masks, rng):
    flip_code = 1 if rng.random() < 0.5 else 0
    flipped_img = cv2.flip(img, flip_code)

    def transform_mask(mask):
        return cv2.flip(mask, flip_code)

    new_anns, new_masks = transform_masks(anns, masks, transform_mask)
    return flipped_img, new_anns, new_masks


def apply_cutout(img, masks, rng, args):
    result = img.copy()
    height, width = img.shape[:2]
    object_mask = np.zeros((height, width), dtype=bool)
    object_mask = union_mask(masks, (height, width))

    box_count = rng.randint(args.min_cutout_boxes, args.max_cutout_boxes)
    for _ in range(box_count):
        box_w = max(
            1,
            int(
                width
                * rng.uniform(args.min_cutout_box_ratio, args.max_cutout_box_ratio)
            ),
        )
        box_h = max(
            1,
            int(
                height
                * rng.uniform(args.min_cutout_box_ratio, args.max_cutout_box_ratio)
            ),
        )
        x0 = rng.randint(0, max(0, width - box_w))
        y0 = rng.randint(0, max(0, height - box_h))
        x1 = min(width, x0 + box_w)
        y1 = min(height, y0 + box_h)
        background_pixels = ~object_mask[y0:y1, x0:x1]
        if np.any(background_pixels):
            patch = result[y0:y1, x0:x1]
            patch[background_pixels] = 0

    return result


_SAMPLE_CACHE = OrderedDict()
_SAMPLE_CACHE_SIZE = 4


def load_sample(image_info, anns, image_dir):
    # Jobs for the same source image run back to back, so a tiny LRU avoids
    # re-reading and re-decoding the image and masks for every augmentation.
    # Cached arrays are never mutated downstream: every transform allocates
    # new arrays and every annotation edit deepcopies first.
    cache_key = (str(image_dir), int(image_info["id"]))
    cached = _SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        _SAMPLE_CACHE.move_to_end(cache_key)
        return cached

    image_path = resolve_dataset_image(image_dir, image_info["file_name"])
    img = read_image(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]
    masks = [decode_mask(ann["segmentation"], height, width) for ann in anns]
    normalized_anns = []
    for ann, mask in zip(anns, masks):
        normalized_ann = deepcopy(ann)
        normalized_ann["bbox"] = mask_to_bbox(mask)
        normalized_ann["area"] = float(mask.sum())
        normalized_anns.append(normalized_ann)

    _SAMPLE_CACHE[cache_key] = (img, normalized_anns, masks)
    if len(_SAMPLE_CACHE) > _SAMPLE_CACHE_SIZE:
        _SAMPLE_CACHE.popitem(last=False)
    return img, normalized_anns, masks


def apply_methods(img, anns, masks, methods, rng, np_rng, args):
    for method in methods:
        if method == "crop":
            img, anns, masks = apply_crop(img, anns, masks, rng, args)
        elif method == "scale":
            img, anns, masks = apply_scale(img, anns, masks, rng, args)
        elif method == "translate":
            img, anns, masks = apply_translate(img, anns, masks, rng)
        elif method == "rotate":
            img, anns, masks = apply_rotate(img, anns, masks, rng, args)
        elif method == "flip":
            img, anns, masks = apply_flip(img, anns, masks, rng)
        elif method == "color_jitter":
            img = apply_color_jitter(img, masks, rng)
        elif method == "noise":
            img = apply_noise(img, np_rng, args)
        elif method == "cutout":
            img = apply_cutout(img, masks, rng, args)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
    return img, anns, masks


def source_segmentation_formats(annotations):
    formats = {}
    for ann in annotations:
        segmentation = ann["segmentation"]
        if isinstance(segmentation, dict):
            if isinstance(segmentation.get("counts"), list):
                formats[ann["id"]] = "rle_uncompressed"
            else:
                formats[ann["id"]] = "rle"
        else:
            formats[ann["id"]] = "polygon"
    return formats


def finalize_annotations(anns, masks, segmentation_format, source_formats):
    output = []
    for ann, mask in zip(anns, masks):
        if segmentation_format == "auto":
            target_format = source_formats[ann["id"]]
        else:
            target_format = segmentation_format

        new_ann = deepcopy(ann)
        if target_format == "rle":
            new_ann["segmentation"] = encode_mask(mask)
        elif target_format == "rle_uncompressed":
            new_ann["segmentation"] = encode_uncompressed_rle(mask)
        else:
            polygons = mask_to_polygons(mask)
            if not polygons:
                continue
            height, width = mask.shape[:2]
            polygon_mask = decode_mask(polygons, height, width)
            bbox = mask_to_bbox(polygon_mask)
            if bbox is None:
                continue
            new_ann["segmentation"] = polygons
            new_ann["bbox"] = bbox
            new_ann["area"] = float(polygon_mask.sum())
        output.append(new_ann)
    return output


def add_mixup_attributes(anns, weight, source, source_image_id, methods):
    output = []
    for ann in anns:
        new_ann = deepcopy(ann)
        attributes = deepcopy(new_ann.get("attributes", {}))
        attributes["mixup_weight"] = float(weight)
        attributes["mixup_source"] = source
        attributes["mixup_source_image_id"] = int(source_image_id)
        attributes["mixup_augmentations"] = list(methods)
        attributes["augmentation_type"] = "mixup"
        new_ann["attributes"] = attributes
        output.append(new_ann)
    return output


def add_standard_attributes(anns, source_image_id, methods):
    output = []
    for ann in anns:
        new_ann = deepcopy(ann)
        attributes = deepcopy(new_ann.get("attributes", {}))
        attributes["augmentation_type"] = "standard"
        attributes["source_image_id"] = int(source_image_id)
        attributes["augmentations"] = list(methods)
        new_ann["attributes"] = attributes
        output.append(new_ann)
    return output


def make_standard_aug(
    index,
    source_image,
    annotations_by_image_id,
    image_dir,
    rng,
    np_rng,
    args,
    method_weights,
):
    anns = annotations_by_image_id[source_image["id"]]
    img, anns, masks = load_sample(source_image, anns, image_dir)

    methods = choose_methods(rng, method_weights)
    img, anns, masks = apply_methods(img, anns, masks, methods, rng, np_rng, args)

    image_info = deepcopy(source_image)
    image_info["id"] = index
    image_info["file_name"] = f"aug_{source_image['id']:06d}_{index:06d}.png"

    output_anns = add_standard_attributes(anns, source_image["id"], methods)
    return image_info, img, output_anns, masks, methods


def make_mixup(
    index,
    image_a,
    images,
    annotations_by_image_id,
    image_dir,
    rng,
    np_rng,
    args,
    method_weights,
):
    compatible_images = [
        image_info
        for image_info in images
        if image_info["id"] != image_a["id"]
        and image_info["width"] == image_a["width"]
        and image_info["height"] == image_a["height"]
    ]
    image_b = rng.choice(compatible_images) if compatible_images else image_a
    anns_a = annotations_by_image_id[image_a["id"]]
    anns_b = annotations_by_image_id[image_b["id"]]

    img_a, anns_a, masks_a = load_sample(image_a, anns_a, image_dir)
    img_b, anns_b, masks_b = load_sample(image_b, anns_b, image_dir)

    methods_a = choose_methods(rng, method_weights)
    methods_b = choose_methods(rng, method_weights)
    img_a, anns_a, masks_a = apply_methods(
        img_a, anns_a, masks_a, methods_a, rng, np_rng, args
    )
    img_b, anns_b, masks_b = apply_methods(
        img_b, anns_b, masks_b, methods_b, rng, np_rng, args
    )

    lam = float(np.clip(np_rng.beta(8.0, 8.0), 0.35, 0.65))
    mixed = np.clip(
        img_a.astype(np.float32) * lam + img_b.astype(np.float32) * (1.0 - lam),
        0,
        255,
    ).astype(np.uint8)

    mixed_image_info = deepcopy(image_a)
    mixed_image_info["id"] = index
    mixed_image_info["file_name"] = f"mixup_{image_a['id']:06d}_{index:06d}.png"

    mixed_anns = []
    mixed_anns.extend(add_mixup_attributes(anns_a, lam, "a", image_a["id"], methods_a))
    mixed_anns.extend(
        add_mixup_attributes(anns_b, 1.0 - lam, "b", image_b["id"], methods_b)
    )
    mixed_masks = list(masks_a) + list(masks_b)
    return mixed_image_info, mixed, mixed_anns, mixed_masks, methods_a, methods_b, lam


_JOB_CONTEXT = {}


def job_rngs(seed, offset):
    # Seed each job independently of execution order so any worker count and
    # scheduling produce identical output for the same --seed.
    base = int(seed) & 0xFFFFFFFF
    return (
        random.Random(base * 1_000_003 + offset),
        np.random.default_rng((base, offset)),
    )


def resolve_worker_count(requested, job_count):
    if requested is None or requested <= 0:
        requested = os.cpu_count() or 1
    return max(1, min(requested, job_count))


def _init_job_context(context):
    _JOB_CONTEXT.update(context)


def _execute_job(job):
    offset, image_id, source_image, use_mixup = job
    context = _JOB_CONTEXT
    args = context["args"]
    rng, np_rng = job_rngs(args.seed, offset)

    if use_mixup:
        image_info, augmented, anns, masks, methods_a, methods_b, lam = make_mixup(
            image_id,
            source_image,
            context["images"],
            context["annotations_by_image_id"],
            context["input_images"],
            rng,
            np_rng,
            args,
            context["method_weights"],
        )
        methods_info = ("mixup", methods_a, methods_b, lam)
    else:
        image_info, augmented, anns, masks, methods = make_standard_aug(
            image_id,
            source_image,
            context["annotations_by_image_id"],
            context["input_images"],
            rng,
            np_rng,
            args,
            context["method_weights"],
        )
        methods_info = ("standard", methods)

    save_path = context["staging_images"] / image_info["file_name"]
    if not write_image(save_path, augmented):
        raise OSError(f"Could not write image: {save_path}")
    final_anns = finalize_annotations(
        anns, masks, args.segmentation_format, context["source_formats"]
    )
    return offset, image_id, image_info, final_anns, methods_info


def initialize_stats(
    args,
    selected_images,
    standard_count,
    mixup_count,
    method_weights,
    input_json,
    input_images,
    output_dir,
):
    generated_count = standard_count + mixup_count
    return {
        "schema_version": 1,
        "seed": int(args.seed),
        "input_json": str(input_json),
        "input_images": str(input_images),
        "output_dir": str(output_dir),
        "source_image_count": len(selected_images),
        "augmentations_per_image": (
            int(args.augmentations_per_image) if args.num_images is None else None
        ),
        "legacy_total_override": int(args.num_images)
        if args.num_images is not None
        else None,
        "include_originals": bool(args.include_originals),
        "generated_count": int(generated_count),
        "standard_count": int(standard_count),
        "mixup_count": int(mixup_count),
        "mixup_ratio": float(args.mixup_ratio),
        "segmentation_format": args.segmentation_format,
        "selected_images": [
            {
                "id": int(image_info["id"]),
                "file_name": image_info["file_name"],
            }
            for image_info in selected_images
        ],
        "method_weights": deepcopy(method_weights),
        "method_usage": {name: 0 for name in METHOD_ORDER},
        "standard_method_usage": {name: 0 for name in METHOD_ORDER},
        "mixup_method_usage": {name: 0 for name in METHOD_ORDER},
        "combo_size_usage": {"2": 0, "3": 0},
        "lambda": {
            "count": 0,
            "min": None,
            "max": None,
            "sum": 0.0,
            "mean": None,
        },
    }


def record_methods(stats, methods, group):
    stats["combo_size_usage"][str(len(methods))] += 1
    for method in methods:
        stats["method_usage"][method] += 1
        stats[group][method] += 1


def record_lambda(stats, lam):
    lam_stats = stats["lambda"]
    lam_stats["count"] += 1
    lam_stats["sum"] += float(lam)
    lam_stats["min"] = (
        float(lam) if lam_stats["min"] is None else min(lam_stats["min"], float(lam))
    )
    lam_stats["max"] = (
        float(lam) if lam_stats["max"] is None else max(lam_stats["max"], float(lam))
    )


def finalize_stats(stats):
    lam_stats = stats["lambda"]
    if lam_stats["count"]:
        lam_stats["mean"] = lam_stats["sum"] / lam_stats["count"]
    return stats


def build_output_coco(
    data,
    generated_images,
    generated_annotations_by_image_id,
    include_originals,
):
    output_data = deepcopy(data)
    next_ann_id = max(ann["id"] for ann in data["annotations"]) + 1

    base_images = []
    for image_info in data["images"]:
        copied = deepcopy(image_info)
        copied["file_name"] = copied_file_name(image_info)
        base_images.append(copied)

    output_annotations = (
        [deepcopy(ann) for ann in data["annotations"]] if include_originals else []
    )
    for image_info in generated_images:
        for ann in generated_annotations_by_image_id[image_info["id"]]:
            new_ann = deepcopy(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = image_info["id"]
            output_annotations.append(new_ann)

    output_data["images"] = (
        base_images if include_originals else []
    ) + generated_images
    output_data["annotations"] = output_annotations
    return output_data


def validate_args(args):
    if not 0.0 <= args.mixup_ratio <= 1.0:
        raise ValueError("--mixup-ratio must be between 0.0 and 1.0.")
    if args.segmentation_format not in ("auto", "polygon", "rle"):
        raise ValueError("--segmentation-format must be auto, polygon, or rle.")
    if args.num_images is not None and args.num_images <= 0:
        raise ValueError("--num-images must be a positive integer.")
    if args.num_images is None and args.augmentations_per_image <= 0:
        raise ValueError("--augmentations-per-image must be a positive integer.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    ranges = (
        ("angle", args.min_angle, args.max_angle),
        ("scale", args.min_scale, args.max_scale),
        ("crop ratio", args.min_crop_ratio, args.max_crop_ratio),
        ("noise sigma", args.min_noise_sigma, args.max_noise_sigma),
        ("cutout box ratio", args.min_cutout_box_ratio, args.max_cutout_box_ratio),
        ("cutout boxes", args.min_cutout_boxes, args.max_cutout_boxes),
    )
    for name, minimum, maximum in ranges:
        if minimum > maximum:
            raise ValueError(f"Minimum {name} must not exceed maximum {name}.")
    if not 0.0 < args.min_crop_ratio <= args.max_crop_ratio <= 1.0:
        raise ValueError("Crop ratios must be greater than 0.0 and at most 1.0.")
    if args.min_scale <= 0:
        raise ValueError("Scale values must be greater than 0.0.")
    if args.min_noise_sigma < 0:
        raise ValueError("Noise sigma must be zero or positive.")
    if not 0.0 < args.min_cutout_box_ratio <= args.max_cutout_box_ratio <= 1.0:
        raise ValueError("Cutout box ratios must be greater than 0.0 and at most 1.0.")
    if args.min_cutout_boxes <= 0:
        raise ValueError("Cutout box counts must be positive integers.")


def build_generation_jobs(images, args, rng):
    if args.num_images is not None:
        source_images = [rng.choice(images) for _ in range(args.num_images)]
    else:
        source_images = [
            image_info
            for image_info in images
            for _ in range(args.augmentations_per_image)
        ]

    mixup_count = int(round(len(source_images) * args.mixup_ratio))
    mixup_indexes = set(rng.sample(range(len(source_images)), mixup_count))
    jobs = [
        (source_image, index in mixup_indexes)
        for index, source_image in enumerate(source_images)
    ]
    return jobs, len(source_images) - mixup_count, mixup_count


def paths_overlap(first, second):
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    return first == second or first in second.parents or second in first.parents


def validate_paths(input_images, input_json, output_dir):
    if not input_images.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {input_images}")
    if not input_json.is_file():
        raise FileNotFoundError(f"Input COCO JSON not found: {input_json}")
    if paths_overlap(input_images, output_dir):
        raise ValueError("Output directory must not overlap the input image directory.")
    if (
        output_dir == input_json
        or output_dir in input_json.parents
        or input_json in output_dir.parents
    ):
        raise ValueError("Output directory must not overlap the input COCO JSON path.")


def create_staging_directory(output_dir):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    staging_dir.mkdir()
    return staging_dir


def commit_output_directory(staging_dir, output_dir, overwrite):
    backup_dir = None
    if output_dir.exists():
        has_content = not output_dir.is_dir() or any(output_dir.iterdir())
        if has_content and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Choose another directory or use --overwrite."
            )
        if has_content:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = output_dir.with_name(f"{output_dir.name}.backup-{timestamp}")
            suffix = 1
            while backup_dir.exists():
                backup_dir = output_dir.with_name(
                    f"{output_dir.name}.backup-{timestamp}-{suffix}"
                )
                suffix += 1
            output_dir.rename(backup_dir)
        else:
            output_dir.rmdir()
    try:
        staging_dir.rename(output_dir)
    except Exception:
        if backup_dir is not None and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    return backup_dir


def run(args):
    validate_args(args)
    input_images = Path(args.input_images).expanduser().resolve()
    input_json = Path(args.input_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    validate_paths(input_images, input_json, output_dir)

    if (
        output_dir.exists()
        and (not output_dir.is_dir() or any(output_dir.iterdir()))
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose another directory or use --overwrite."
        )

    method_weights = method_weights_from_args(args)
    rng = random.Random(args.seed)

    with input_json.open("r", encoding="utf-8") as f:
        full_data = json.load(f)
    validate_coco_dataset(full_data, input_images)
    data = filter_data(full_data, args)

    annotations_by_image_id = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image_id[ann["image_id"]].append(ann)
    source_formats = source_segmentation_formats(data["annotations"])

    jobs, standard_count, mixup_count = build_generation_jobs(data["images"], args, rng)
    stats = initialize_stats(
        args,
        data["images"],
        standard_count,
        mixup_count,
        method_weights,
        input_json,
        input_images,
        output_dir,
    )

    staging_dir = create_staging_directory(output_dir)
    staging_images = staging_dir / "images"
    staging_images.mkdir()

    next_image_id = max(image_info["id"] for image_info in data["images"]) + 1
    generated_images = []
    generated_annotations_by_image_id = {}

    try:
        if args.include_originals:
            for image_info in data["images"]:
                source_path = resolve_dataset_image(
                    input_images, image_info["file_name"]
                )
                shutil.copy2(source_path, staging_images / copied_file_name(image_info))

        job_specs = [
            (offset, next_image_id + offset, source_image, use_mixup)
            for offset, (source_image, use_mixup) in enumerate(jobs)
        ]
        job_context = {
            "args": args,
            "images": data["images"],
            "annotations_by_image_id": annotations_by_image_id,
            "input_images": input_images,
            "staging_images": staging_images,
            "method_weights": method_weights,
            "source_formats": source_formats,
        }
        workers = resolve_worker_count(args.workers, len(job_specs))
        stats["workers"] = workers

        standard_done = 0
        mixup_done = 0

        def consume(results):
            nonlocal standard_done, mixup_done
            completed = 0
            for offset, image_id, image_info, final_anns, methods_info in results:
                if methods_info[0] == "mixup":
                    record_methods(stats, methods_info[1], "mixup_method_usage")
                    record_methods(stats, methods_info[2], "mixup_method_usage")
                    record_lambda(stats, methods_info[3])
                    mixup_done += 1
                else:
                    record_methods(stats, methods_info[1], "standard_method_usage")
                    standard_done += 1
                generated_images.append(image_info)
                generated_annotations_by_image_id[image_id] = final_anns
                completed += 1
                if completed % 50 == 0 or completed == len(job_specs):
                    print(f"generated: {completed} / {len(job_specs)}", flush=True)

        if workers > 1:
            # Contiguous chunks keep same-source jobs on the same worker, so
            # each worker's sample cache stays effective.
            chunksize = max(1, -(-len(job_specs) // (workers * 4)))
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_job_context,
                initargs=(job_context,),
            ) as pool:
                consume(pool.map(_execute_job, job_specs, chunksize=chunksize))
        else:
            _init_job_context(job_context)
            consume(map(_execute_job, job_specs))

        output_data = build_output_coco(
            data,
            generated_images,
            generated_annotations_by_image_id,
            args.include_originals,
        )
        stats["final_image_count"] = len(output_data["images"])
        stats["final_annotation_count"] = len(output_data["annotations"])
        stats["standard_count"] = standard_done
        stats["mixup_count"] = mixup_done

        output_json_path = staging_dir / "annotations.json"
        stats_json_path = staging_dir / "augmentation_stats.json"
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, allow_nan=False)
        with stats_json_path.open("w", encoding="utf-8") as f:
            json.dump(
                finalize_stats(stats), f, ensure_ascii=False, indent=2, allow_nan=False
            )

        backup_dir = commit_output_directory(staging_dir, output_dir, args.overwrite)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    print("saved dataset:", output_dir)
    print("saved images:", output_dir / "images")
    print("saved json:", output_dir / "annotations.json")
    print("saved stats:", output_dir / "augmentation_stats.json")
    if backup_dir is not None:
        print("previous output backup:", backup_dir)
    return output_dir


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
