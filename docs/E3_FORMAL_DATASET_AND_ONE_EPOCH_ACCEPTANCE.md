# E3 Formal Dataset and One-Epoch Acceptance

Date: 2026-07-03

Final result: **PASS**

## Baseline

- E2 tag: `E2_READY_FOR_FORMAL_TRAINING_PREPARATION`
- E3 branch: `e3-formal-training-prep`
- E2 baseline commit: `29b1bfacaaf09280f3ff0d3f1fc6dfcc101f003f`
- Formal dataset manifest commit: `302b4b15fefb3ff3248fba9f5dac9a1d88eaa8dd`
- SAM301 trainer SHA256: `bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2`
- Patch manifest SHA256: `cc55ef49343969cf471c58090cb65f85a8fbe9cfa7424899e65bf6a2aa22b604`
- Patch SHA256: `f28588dda179af4133ead23a7ceed2c4615822f420fa83eade282e4775caa85d`
- Python: `/home/book/anaconda3/envs/sam301/bin/python`
- PyTorch: `2.10.0+cu128`
- CUDA build: `12.8`
- NVIDIA driver: `595.71.05`
- GPU: `NVIDIA GeForce RTX 5090`
- SAM3 import path: `/home/book/sam301/sam3/__init__.py`

## Formal Dataset

Source:

- Six CVAT polygon COCO exports: `data/cvat_import_polygon_*_book_spine_iou0.5_book_spine.json`
- Raw images: `data/dataset_raw/<timestamp>_book_spine/images`
- Existing `data/book_spine_sam3_dataset` is a 22-image smoke split and was not used as the formal source.

Audit result:

- Total images: `184`
- Total annotations: `7185`
- Category: `id=1`, `name=book_spine`
- Missing images: `0`
- Empty images: `0`
- Invalid polygons / zero-area masks / bbox out-of-bounds / duplicate IDs: `0`
- Extremely small masks (`area_ratio < 0.001`): `286`
- Exact duplicate image byte groups: `44`; duplicate clusters were kept within one split.
- Exact duplicate cross-split leakage: `0`
- Average-hash near duplicate cross-split pairs: `0`

Managed artifacts:

- `data_manifests/formal_dataset_manifest.json`
- `data_manifests/formal_split_manifest.json`
- `docs/FORMAL_DATASET_AUDIT.md`

Ignored generated COCO split files:

- `data/formal_book_spine_sam3_dataset/train/annotations.json`
- `data/formal_book_spine_sam3_dataset/val/annotations.json`
- `data/formal_book_spine_sam3_dataset/test/annotations.json`

## Split

No project-level formal train/val/test ratio was found. The split is duplicate-aware:

- Seed: `20260703`
- Unit: source/acquisition group, with one exception to isolate four byte-identical duplicate images into test.
- Training count is divisible by the E3 effective batch size (`4`) so `drop_last=True` does not drop training images.

Split counts:

- train: `160` images, `6749` annotations
- val: `12` images, `253` annotations
- test: `12` images, `183` annotations

Leakage conclusion:

- No exact duplicate or average-hash near duplicate crosses split.
- Test data was not used for preflight or training.

## Training Parameters

Formal-data smoke parameters:

- `max_epochs`: `1`
- `train_batch_size`: `1`
- `gradient_accumulation_steps`: `4`
- effective batch size: `4`
- `num_gpus`: `1`
- prompt: `book spine`
- `resume_from`: `None`

Resolved runtime:

- train loader batch size: `4`
- chunking collate: `collate_fn_api_with_chunking`
- `num_chunks`: `4`
- train images root: `/home/book/book01/data/dataset_raw`
- train annotations: `/home/book/book01/data/formal_book_spine_sam3_dataset/train/annotations.json`
- val images root: `/home/book/book01/data/dataset_raw`
- val annotations: `/home/book/book01/data/formal_book_spine_sam3_dataset/val/annotations.json`
- num_workers: `10`
- learning rate: base YAML expression retained; preflight cannot resolve custom OmegaConf resolver value.
- validation frequency: base YAML `val_epoch_freq=5`, but validation ran at epoch 0 in this one-epoch run.
- checkpoint save frequency: base YAML `save_freq=5`; checkpoint was saved for this one-epoch run.

## Preflight

Run:

`/home/book/book01/runs/training/2026-07-03_14-42-27`

Preflight result:

- patch guard: passed
- SAM3 import guard: passed
- Hydra validate-only: passed
- train COCO: `160` images, `6749` annotations
- val COCO: `12` images, `253` annotations
- effective batch <= train images: yes
- train count divisible by effective batch: yes
- token initially unconsumed: yes
- logs/checkpoints/summary initially absent: yes

Warnings:

- learning rate value not displayed because base YAML uses a custom resolver;
- train and val use the same image root (`data/dataset_raw`) with disjoint annotation files;
- training prompt differs from COCO category name by design (`book spine` vs `book_spine`).

## Real One-Epoch Acceptance

Formal launcher:

`ui.training_preflight_page.start_training(...)`

Command:

```bash
conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py -c /home/book/book01/runs/training/2026-07-03_14-42-27/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
```

Launch:

- Start time: `2026-07-03T05:43:17.145403+00:00`
- End time: `2026-07-03T05:44:52.189648+00:00`
- Duration: `95.04424524307251` seconds
- PID: `1531559`
- `MASTER_ADDR`: `localhost`
- `MASTER_PORT`: `53387`
- Startup GPU memory: `804 MiB used / 31286 MiB free`
- Observed peak during run: `26667 MiB used` by `nvidia-smi`; trainer log peak reports `23.00 GB`.

Result:

- summary status: `completed`
- exit code: `0`
- train outer iterations: `40`
- micro-batches: `160`
- optimizer steps: `40`
- train images covered: `160 / 160`
- final train loss (`Losses/train_all_loss`): `575.6918604850769`
- validation executed: yes
- val iterations: `12`
- val bbox AP: `0.8380942391597286`
- val bbox AP50: `0.9771321010827602`

Summary:

`/home/book/book01/runs/training/2026-07-03_14-42-27/training_summary.json`

Checkpoint:

- Path: `/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt`
- Size: `10081250310` bytes
- mtime: `2026-07-03 14:44:48.114274448 +0900`

## Integrity

Base checkpoint:

- Path: `/home/book/sam301/sam3.pt`
- Size before/after: `3450062241` bytes
- mtime before/after: `2026-06-15 19:22:15.022366000 +0900`
- SHA256 before: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- SHA256 after: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- Result: unchanged

Residual process check:

- No residual `sam3/train/train.py`, `book_spine_finetune`, `runs/training`, `torchrun`, or `hydra` process was found.
- GPU after run: `803 MiB used / 31286 MiB free`, utilization `0%`.

Git hygiene:

- No run, checkpoint, log, summary, image, or data file was added to Git.
- Existing untracked Claude/handoff documents were preserved and not submitted.

## Conclusion

E3 formal dataset preparation and one-epoch formal-data acceptance passed.

The formal split is usable for the next stage, with one important data note: the source set contains many byte-identical duplicates, and the split deliberately keeps duplicate clusters within one split. Before full multi-epoch training, approve whether this duplicate-aware split should remain the formal split or whether duplicate records should be pruned from the formal dataset.

Do not start multi-epoch training until the next-stage training parameters are explicitly approved.
