from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from pycocotools import mask as cocomask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.coco_export import build_coco
from core.npz_io import InstanceSet


class CocoRleExportTest(unittest.TestCase):
    def test_rle_decodes_to_original_mask(self) -> None:
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:6, 3:8] = True
        instances = InstanceSet(
            masks=np.stack([mask]),
            scores=np.asarray([0.9], dtype=np.float32),
            bboxes=np.asarray([[3, 2, 5, 4]], dtype=np.int32),
            instance_ids=np.asarray([1], dtype=np.int32),
            extra={},
        )
        coco, ann_ids_by_image, errors = build_coco(
            [{"coco_image_id": 1, "file_name": "a.png", "width": 10, "height": 8}],
            {1: instances},
            segmentation_format="rle",
        )

        self.assertEqual(errors, [])
        self.assertEqual(ann_ids_by_image, {1: [1]})
        ann = coco["annotations"][0]
        rle = dict(ann["segmentation"])
        rle["counts"] = rle["counts"].encode("ascii")
        decoded = cocomask.decode(rle).astype(bool)
        self.assertTrue(np.array_equal(decoded, mask))
        self.assertEqual(int(ann["area"]), int(mask.sum()))
        self.assertEqual([int(v) for v in ann["bbox"]], [3, 2, 5, 4])


if __name__ == "__main__":
    unittest.main()
