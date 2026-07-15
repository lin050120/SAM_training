# SAM3 Online Training Augmentation Guide

## Where to configure it

1. Start the UI with `conda run -n sam301 python app.py`.
2. Open Training Preflight and fill in the train/val data, prompt, checkpoint, and training settings.
3. Select a preset under Online Training Augmentation.
4. Run preflight again. Check `online_augmentation` in the result before starting training.

## Presets

- **Off**: adds no online transform and uses each source image once per epoch. The base YAML's existing random resize and pad still run.
- **Light**: fixed conservative settings: affine probability 0.5, rotation -8 to +8 degrees, scale 0.9 to 1.1, maximum translation 0.05, horizontal flip probability 0.5, color jitter probability 0.3 with strength 0.15, motion blur probability 0.1, and repeat factor 1.
- **Custom**: unlocks all bounded controls for samples per source image, rotation, scale, translation, and transform probabilities.

Start with Light for one complete train/validation cycle. Increase the repeat factor or transform ranges only when the dataset is small or validation results show a need.

## Parameters

- **Samples per source image per epoch**: 1 to 10. A value of 3 reads each source image three times per epoch with independent random transforms. It does not create image files on disk.
- **Affine probability**: probability of applying the combined rotation, scale, and translation transform.
- **Minimum/maximum rotation**: -180 to 180 degrees; minimum must not exceed maximum.
- **Minimum/maximum scale**: 0.5 to 2.0; minimum must not exceed maximum.
- **Maximum translation fraction**: 0 to 0.5 of image width and height.
- **Horizontal flip, color jitter, and motion blur probabilities**: 0 to 1.
- **Color jitter strength**: 0 to 0.5 and controls brightness, contrast, saturation, and hue variation.

## Masks and scope

Geometric transforms use the same random parameters for the image and every instance mask. Masks use nearest-neighbor interpolation; bbox and area are recomputed from each transformed mask. Online augmentation applies only to train. Val, test, and inference do not receive these random transforms.

This feature deliberately excludes random crop, noise, cutout, and mixup so reviewed target semantics are not changed or cropped away. Preflight rejects out-of-range values and reversed min/max ranges.

## Pause and resume

The final augmentation settings are stored in the run's `runtime_config.yaml`, `dataset_info.json`, and `training_config_summary.json`. Resume uses the same runtime YAML, so augmentation cannot change within an existing run. Create a new preflight/run for different settings.

## Records

- `config/runtime_config.yaml`: actual Hydra transforms and dataset target.
- `dataset_info.json`: data paths and `online_augmentation`.
- `training_config_summary.json`: final settings before launch.
- `training_summary.json`: augmentation settings and process-attempt history after completion or pause.
