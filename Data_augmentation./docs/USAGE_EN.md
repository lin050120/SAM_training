# COCO Data Augmentation Usage Guide

## 1. When to use it

Complete human review and split the original dataset into train, val, and test first. Augment **train only**. Keep val and test unchanged so evaluation remains comparable.

## 2. Input requirements

Prepare an image folder and its COCO 1.0 JSON. They may be stored anywhere.

```text
train/
  images/
    image_001.jpg
    image_002.png
  annotations.json
```

The JSON must contain non-empty `images`, `annotations`, and `categories` lists. Every `images[].file_name` must be relative to the selected image folder. Image, annotation, and category IDs must be unique integers. JSON dimensions must match the actual image. COCO polygon, uncompressed RLE, and compressed RLE segmentations are supported.

The program validates the full input before generation. A validation failure does not create the final output.

## 3. Start the GUI

Run this from `/home/book/book01`:

```bash
conda run -n sam301 python "Data_augmentation./data_augmentation_gui.py"
```

On Windows, activate an environment containing OpenCV, NumPy, pycocotools, and tkinter, then double-click `start_data_augmentation.bat`.

## 4. GUI steps

1. `Input image folder`: select the train image folder, such as `train/images`.
2. `COCO annotation JSON`: select the COCO JSON for those images.
3. `Output dataset folder`: select a new or empty output directory.
4. `Augmentations per source image`: generated images per original image. The default is 5.
5. `MixUp ratio`: fraction of generated images that use MixUp. Start with `0` unless it is specifically needed.
6. `Random seed`: identical input, settings, and seed produce the same random choices.
7. `Segmentation format`: format of generated segmentations. `Auto (match source)` (default) keeps each source annotation's format (polygon stays polygon, uncompressed RLE stays uncompressed RLE, compressed RLE stays compressed RLE); `Polygon (point lists)` writes COCO 1.0 point coordinates for everything; `RLE (mask)` writes compressed RLE for everything.
8. `Parallel workers`: number of generation processes. `0` (default) uses all CPU cores; `1` runs single-process. Output is identical for any worker count with the same seed.
9. `Include original images and annotations`: enabled by default, making the result a complete train dataset.
10. `Back up and replace...`: enable only when replacing a non-empty output. The old directory is renamed to a timestamped backup.
11. Click `Generate dataset`.

For `M` source images and `N` augmentations per source, including originals produces:

```text
M * (N + 1)
```

For example, 175 source images and a value of 5 produce 875 augmented images and 1050 images in total.

## 5. Methods and masks

Each generated image randomly receives two or three methods. A larger weight makes a method more likely to be selected. Weights do not need to sum to 1.

| Method | Effect | Mask handling |
| --- | --- | --- |
| Object hue | Changes hue only inside annotated objects | Position unchanged |
| Rotate | Rotates from -90 to +90 degrees by default | Uses the same rotation matrix |
| Scale | Shrinks or enlarges from 0.85x to 1.15x by default | Uses the same scale matrix |
| Crop/zoom | Crops and resizes to create a zoom-in effect | Same crop and resize |
| Translate | Translates the image | Uses the same translation matrix |
| Flip | Horizontal or vertical flip | Applies the same flip |
| Noise | Adds Gaussian noise | Position unchanged |
| Cutout | Covers background outside masks only | Position unchanged |

Images use linear interpolation and masks use nearest-neighbor interpolation. Every geometric operation shares exactly the same geometry between image and mask. Interpolation can create a one-pixel boundary difference, but the mask cannot remain at its pre-transform position.

MixUp prefers a different image with matching dimensions and retains all transformed masks from both sources. It falls back to the current image when no other compatible image exists.

## 6. Output format

```text
<output-dataset-folder>/
  images/
    image_000001.png
    aug_000001_000176.png
    mixup_000002_000177.png
  annotations.json
  augmentation_stats.json
```

`annotations.json` is a complete COCO file. Original annotation formats are preserved; the format of generated segmentations follows `Segmentation format` (`--segmentation-format` on the command line): the default `auto` matches each source annotation's format, or force polygon point lists / compressed RLE for everything. Polygon output traces contours at 2x precision so thin objects are not distorted by the conversion; holes in a mask are filled in polygon format (COCO polygons cannot represent holes). `augmentation_stats.json` records input paths, seed, generated counts, segmentation format, method usage, and MixUp details.

The program builds everything in a sibling staging directory and publishes the final output only after all files succeed. A non-empty output is rejected by default. Replacement first preserves the old directory as a backup.

## 7. Use with SAM3 training

In the training preflight page:

1. Select `<output>/images` as the train image path.
2. Select `<output>/annotations.json` as the train COCO path.
3. Continue using the original, unaugmented val/test data.
4. Enter the real category in training prompt, such as `book spine` or `cable`.
5. Before formal training, use the UI dataset registration feature to register the combination of augmented train and original val/test. Augmentation changes the train files and JSON, so an old registration does not identify the new data.
6. Run smoke before formal training.

## 8. Command line

```bash
conda run -n sam301 python "Data_augmentation./scripts/random_mixup_aug.py" \
  --input-images /path/to/train/images \
  --input-json /path/to/train/annotations.json \
  --output-dir /path/to/augmented_train \
  --augmentations-per-image 5 \
  --mixup-ratio 0 \
  --seed 42
```

Common options:

```text
--segmentation-format auto|polygon|rle   Generated segmentation format; auto (default) matches the source
--workers 0              Parallel processes; 0 or omitted uses all CPU cores, 1 is single-process
--no-include-originals   Output only augmented images and annotations
--overwrite              Back up and replace a non-empty output
--limit 2                Quick check using the first two images
--weight-scale 0.18      Change a method's relative weight
--min-scale 0.85 --max-scale 1.15
```

The compatibility option `--num-images N` sets a total generated count and overrides the per-image value. For normal full-dataset use, prefer `--augmentations-per-image` so every source image receives the same number of augmentations.

## 9. Count recommendations

- More than 100 real train images: start with 3–5 per image.
- 20–100 images: start with 5–10 per image.
- Fewer than 20 images: try 10–20, but prioritize collecting more real images.
- Generating 1000 samples from one source image is not recommended for a formal training dataset.

Choose the final count from metrics on the original val/test data. Before formal training, also manually sample output images and masks to confirm they match the target definition.
