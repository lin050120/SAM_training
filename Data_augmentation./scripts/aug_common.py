import os

import cv2
import numpy as np
from pycocotools import mask as maskUtils


def add_common_args(parser):
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images for a quick run check.",
    )
    parser.add_argument(
        "--image-id",
        action="append",
        type=int,
        default=[],
        help="Process only this image id. Can be used multiple times.",
    )
    parser.add_argument(
        "--file-name",
        action="append",
        default=[],
        help="Process only this exact image file name. Can be used multiple times.",
    )
    parser.add_argument(
        "--file-contains",
        action="append",
        default=[],
        help="Process images whose file names contain this text. Can be used multiple times.",
    )


def copied_file_name(image_info):
    _, ext = os.path.splitext(image_info["file_name"])
    return f"image_{int(image_info['id']):06d}{ext}"


def read_image(image_path):
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if image_bytes.size == 0:
        return None
    return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)


def write_image(image_path, image):
    ext = image_path.suffix or ".png"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    encoded.tofile(str(image_path))
    return True


def decode_mask(segmentation, height, width):
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(segmentation.get("counts"), list):
            rle = maskUtils.frPyObjects(segmentation, height, width)
    else:
        rle = maskUtils.frPyObjects(segmentation, height, width)

    decoded = maskUtils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def encode_mask(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "counts": counts,
        "size": [int(mask.shape[0]), int(mask.shape[1])],
    }


def encode_uncompressed_rle(mask):
    flat = (np.asarray(mask) > 0).astype(np.uint8).flatten(order="F")
    change_positions = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], change_positions, [flat.size]))
    runs = [int(run) for run in np.diff(boundaries)]
    if flat.size and flat[0] == 1:
        runs = [0] + runs
    return {
        "counts": runs,
        "size": [int(mask.shape[0]), int(mask.shape[1])],
    }


def mask_to_polygons(mask):
    # Trace contours on a 2x nearest-neighbor upscale: contour points sit on
    # pixel centers, so at 1x the rasterized polygon shrinks by half a pixel
    # per edge, which visibly erodes thin objects.
    height, width = mask.shape[:2]
    upscaled = cv2.resize(
        mask.astype(np.uint8),
        (width * 2, height * 2),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(
        upscaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2).astype(np.float64) / 2.0
        polygons.append([float(value) for value in points.flatten()])
    return polygons


def mask_to_bbox(mask):
    rows = np.flatnonzero((mask > 0).any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero((mask > 0).any(axis=0))

    x_min = int(cols[0])
    y_min = int(rows[0])
    x_max = int(cols[-1])
    y_max = int(rows[-1])
    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1),
    ]
