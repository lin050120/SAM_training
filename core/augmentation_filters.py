from __future__ import annotations

import torch
from PIL import Image as PILImage

from sam3.train.transforms.filter_query_transforms import FilterDataPointQueries

# A box whose normalized width or height falls below this fraction of the image
# side is treated as a degenerate sliver and dropped.
#
# Why this exists: the SAM3 Hungarian matcher (BinaryHungarianMatcherV2) feeds
# normalized cxcywh boxes into generalized_box_iou. A near-zero-area box makes
# that GIoU compute 0/0 -> NaN, so the cost matrix reaches
# scipy.optimize.linear_sum_assignment with non-finite entries and training dies
# with "matrix contains invalid numeric entries". Aggressive affine augmentation
# (rotation + scale + translate) can shrink a mask to a 1-2px corner sliver whose
# recomputed box triggers exactly this, whereas a real book spine — even a thin
# one — occupies a far larger fraction of the frame. 0.5% of the image side sits
# well above the numeric danger zone (< ~0.1%) and well below any genuine target.
DEFAULT_MIN_BOX_SIDE_FRACTION = 0.005


def _image_hw(image_data) -> tuple[int, int]:
    """Return (height, width) for a PIL image or CxHxW / HxW tensor."""
    if isinstance(image_data, PILImage.Image):
        width, height = image_data.size
        return height, width
    if isinstance(image_data, torch.Tensor):
        height, width = image_data.shape[-2:]
        return int(height), int(width)
    raise RuntimeError(f"Unexpected image type {type(image_data)}")


class FilterTinyBoxes(FilterDataPointQueries):
    """Drop objects whose recomputed box is a degenerate sliver.

    Meant to run right after RecomputeBoxesFromMasks in the online-augmentation
    chain, where boxes are xyxy in pixel coordinates. Filtering by a fraction of
    the current image side is invariant to the later resize, which is exactly the
    normalized scale the matcher and box/GIoU losses ultimately see. Only objects
    are marked for removal (find queries are kept, matching FilterEmptyTargets),
    so a query that loses every object simply becomes a negative query.
    """

    def __init__(self, min_box_side_fraction: float = DEFAULT_MIN_BOX_SIDE_FRACTION):
        fraction = float(min_box_side_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("min_box_side_fraction must be in the open interval (0, 1)")
        self.min_box_side_fraction = fraction

    def identify_queries_to_filter(self, datapoint) -> None:
        self.obj_ids_to_filter = set()
        for img_id, img in enumerate(datapoint.images):
            height, width = _image_hw(img.data)
            min_w = self.min_box_side_fraction * float(width)
            min_h = self.min_box_side_fraction * float(height)
            for obj_id, obj in enumerate(img.objects):
                if obj.bbox is None:
                    continue
                box = obj.bbox.reshape(-1)
                if box.numel() < 4:
                    continue
                box_w = float(box[2] - box[0])
                box_h = float(box[3] - box[1])
                if box_w < min_w or box_h < min_h:
                    self.obj_ids_to_filter.add((img_id, obj_id))
        self.find_ids_to_filter = set()
