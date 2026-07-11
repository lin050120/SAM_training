from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_fixture_batch(root: Path, name: str, count: int, category_name: str = "book spine") -> None:
    batch = root / name
    images_dir = batch / "images"
    images_dir.mkdir(parents=True)
    images = []
    annotations = []
    ann_id = 1
    for idx in range(1, count + 1):
        file_name = f"src_{idx:03d}.png"
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, :] = (idx * 7) % 255
        cv2.imwrite(str(images_dir / file_name), image)
        images.append({"id": idx, "file_name": file_name, "width": 30, "height": 20})
        annotations.append(
            {
                "id": ann_id,
                "image_id": idx,
                "category_id": 7,
                "bbox": [1, 2, 10, 12],
                "segmentation": [[1, 2, 11, 2, 11, 14, 1, 14]],
                "area": 120,
                "iscrowd": 0,
            }
        )
        ann_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 7, "name": category_name, "supercategory": ""}],
    }
    (batch / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")


class DatasetSplitBuilderTest(unittest.TestCase):
    def test_pool_splits_train_val_and_test_stays_test_only(self) -> None:
        from core.dataset_split_builder import DatasetBuildConfig, build_training_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            test = root / "test_input"
            _write_fixture_batch(pool, "pool_batch", 20)
            _write_fixture_batch(test, "test_batch", 5)
            out = root / "dataset"

            result = build_training_dataset(
                DatasetBuildConfig(
                    annotation_pool_dir=pool,
                    test_dir=test,
                    output_dir=out,
                    category_name="book spine",
                    val_ratio=0.10,
                    seed=42,
                )
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["splits"]["train"]["images"], 18)
            self.assertEqual(result["splits"]["val"]["images"], 2)
            self.assertEqual(result["splits"]["test"]["images"], 5)
            for split in ("train", "val", "test"):
                coco = json.loads((out / split / "annotations.json").read_text(encoding="utf-8"))
                self.assertEqual(coco["categories"], [{"id": 1, "name": "book spine", "supercategory": ""}])
                self.assertTrue(all(a["category_id"] == 1 for a in coco["annotations"]))
            manifest_rows = (out / "manifest.csv").read_text(encoding="utf-8").splitlines()[1:]
            self.assertEqual(len(manifest_rows), 25)
            self.assertTrue(all(",test," in row for row in manifest_rows if row.split(",")[4] == "test"))
            test_coco = json.loads((out / "test" / "annotations.json").read_text(encoding="utf-8"))
            self.assertEqual({img["file_name"] for img in test_coco["images"]}, {f"im_{i:06d}.png" for i in range(1, 6)})

    def test_existing_output_requires_overwrite(self) -> None:
        from core.dataset_split_builder import DatasetBuildConfig, build_training_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            test = root / "test_input"
            out = root / "dataset"
            _write_fixture_batch(pool, "pool_batch", 4)
            _write_fixture_batch(test, "test_batch", 2)
            out.mkdir()
            (out / "keep.txt").write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                build_training_dataset(DatasetBuildConfig(pool, test, out))

            result = build_training_dataset(DatasetBuildConfig(pool, test, out, overwrite=True))

            self.assertTrue(result["ok"])
            self.assertIsNotNone(result["backup_dir"])
            self.assertTrue(Path(result["backup_dir"]).is_dir())
            self.assertTrue((Path(result["backup_dir"]) / "keep.txt").is_file())

    def test_training_page_handler_returns_paths_for_preflight_fields(self) -> None:
        from ui.training_preflight_page import build_dataset_split

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            test = root / "test_input"
            out = root / "dataset"
            _write_fixture_batch(pool, "pool_batch", 10)
            _write_fixture_batch(test, "test_batch", 3)

            status, train_images, train_ann, val_images, val_ann, test_images, test_ann, prompt = build_dataset_split(
                str(pool),
                str(test),
                str(out),
                "cable",
                "0.10",
                "42",
                False,
            )

        self.assertIn("DATASET BUILD OK", status)
        self.assertTrue(train_images.endswith("/train/images"))
        self.assertTrue(train_ann.endswith("/train/annotations.json"))
        self.assertTrue(val_images.endswith("/val/images"))
        self.assertTrue(val_ann.endswith("/val/annotations.json"))
        self.assertTrue(test_images.endswith("/test/images"))
        self.assertTrue(test_ann.endswith("/test/annotations.json"))
        self.assertEqual(prompt, "cable")


if __name__ == "__main__":
    unittest.main()
