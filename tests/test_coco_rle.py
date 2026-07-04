from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as cocomask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.coco_export import POLYGON_TARGET_POINTS, build_coco, mask_to_polygon
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


class CocoPolygonExportTest(unittest.TestCase):
    def test_rectangle_polygon_stays_four_points(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[4:16, 7:24] = True
        polygons, bbox, area = mask_to_polygon(mask, min_area=1)

        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(polygons[0]) // 2, 4)
        self.assertEqual([int(v) for v in bbox], [7, 4, 17, 12])
        self.assertEqual(int(area), int(mask.sum()))

    def test_irregular_polygon_is_simplified_to_target_points(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        yy, xx = np.ogrid[:80, :80]
        center_y, center_x = 40, 40
        # Wavy radial shape: enough boundary detail to prove we are not exporting
        # the raw contour, but still a single connected book-like blob.
        angles = np.arctan2(yy - center_y, xx - center_x)
        radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        boundary = 22 + 4 * np.sin(5 * angles)
        mask[radius <= boundary] = True

        polygons, bbox, area = mask_to_polygon(mask, min_area=1)

        self.assertEqual(len(polygons), 1)
        self.assertLessEqual(len(polygons[0]) // 2, POLYGON_TARGET_POINTS)
        self.assertGreaterEqual(len(polygons[0]) // 2, 3)
        self.assertIsNotNone(bbox)
        self.assertEqual(int(area), int(mask.sum()))

    def test_build_coco_polygon_uses_simplified_polygon(self) -> None:
        mask = np.zeros((80, 80), dtype=bool)
        cv2_mask = np.zeros((80, 80), dtype=np.uint8)

        cv2.ellipse(cv2_mask, (40, 40), (25, 12), 25, 0, 360, 1, -1)
        mask = cv2_mask.astype(bool)
        instances = InstanceSet(
            masks=np.stack([mask]),
            scores=np.asarray([0.9], dtype=np.float32),
            bboxes=np.asarray([[0, 0, 1, 1]], dtype=np.int32),
            instance_ids=np.asarray([1], dtype=np.int32),
            extra={},
        )
        coco, _, errors = build_coco(
            [{"coco_image_id": 1, "file_name": "a.png", "width": 80, "height": 80}],
            {1: instances},
            segmentation_format="polygon",
        )

        self.assertEqual(errors, [])
        polygon = coco["annotations"][0]["segmentation"][0]
        self.assertLessEqual(len(polygon) // 2, POLYGON_TARGET_POINTS)


if __name__ == "__main__":
    unittest.main()
