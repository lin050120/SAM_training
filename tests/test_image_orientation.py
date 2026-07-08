from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.inference_run import _copy_image
from core.sam3_adapter import Sam3Adapter


class ExifOrientationNormalizationTest(unittest.TestCase):
    def test_copy_image_applies_exif_orientation_3_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            copied = root / "copied.jpg"
            cvat = root / "cvat.jpg"

            # Distinct corner colors make a 180-degree rotation unambiguous.
            arr = np.zeros((40, 50, 3), dtype=np.uint8)
            arr[:20, :25] = [255, 0, 0]
            arr[:20, 25:] = [0, 255, 0]
            arr[20:, :25] = [0, 0, 255]
            arr[20:, 25:] = [255, 255, 0]
            image = Image.fromarray(arr, mode="RGB")
            exif = image.getexif()
            exif[274] = 3
            image.save(source, exif=exif, quality=100)

            width, height, orientation = _copy_image(source, copied, cvat)

            self.assertEqual((width, height), (50, 40))
            self.assertEqual(orientation, 3)
            copied_rgb = cv2.cvtColor(cv2.imread(str(copied)), cv2.COLOR_BGR2RGB)
            cvat_rgb = cv2.cvtColor(cv2.imread(str(cvat)), cv2.COLOR_BGR2RGB)
            self.assertTrue(np.allclose(copied_rgb, cvat_rgb, atol=8))
            self.assertTrue(np.allclose(copied_rgb[5, 5], arr[-6, -6], atol=8))
            self.assertTrue(np.allclose(copied_rgb[-6, -6], arr[5, 5], atol=8))

    def test_copy_image_can_preserve_legacy_raw_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            copied = root / "copied.jpg"
            cvat = root / "cvat.jpg"
            arr = np.zeros((40, 50, 3), dtype=np.uint8)
            arr[:20, :25] = [255, 0, 0]
            arr[:20, 25:] = [0, 255, 0]
            arr[20:, :25] = [0, 0, 255]
            arr[20:, 25:] = [255, 255, 0]
            image = Image.fromarray(arr, mode="RGB")
            exif = image.getexif()
            exif[274] = 3
            image.save(source, exif=exif, quality=100)

            width, height, orientation = _copy_image(source, copied, cvat, normalize_exif_orientation=False)

            self.assertEqual((width, height), (50, 40))
            self.assertEqual(orientation, 3)
            copied_raw = np.asarray(Image.open(copied).convert("RGB"))
            self.assertTrue(np.allclose(copied_raw[5, 5], arr[5, 5], atol=8))
            self.assertTrue(np.allclose(copied_raw[-6, -6], arr[-6, -6], atol=8))

    def test_adapter_path_loader_applies_exif_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jpg"
            arr = np.zeros((40, 50, 3), dtype=np.uint8)
            arr[:20, :25] = [255, 0, 0]
            arr[:20, 25:] = [0, 255, 0]
            arr[20:, :25] = [0, 0, 255]
            arr[20:, 25:] = [255, 255, 0]
            image = Image.fromarray(arr, mode="RGB")
            exif = image.getexif()
            exif[274] = 3
            image.save(source, exif=exif, quality=100)

            pil = Sam3Adapter._to_pil(source)
            loaded = np.asarray(pil)

            self.assertEqual(pil.size, (50, 40))
            self.assertTrue(np.allclose(loaded[5, 5], arr[-6, -6], atol=8))
            self.assertTrue(np.allclose(loaded[-6, -6], arr[5, 5], atol=8))


if __name__ == "__main__":
    unittest.main()
