# Human Acceptance Checklist

Date: 2026-07-03 (updated after P1 remediation, see `docs/P1_REMEDIATION_AND_INFERENCE_ACCEPTANCE.md`)

Scope: E3 formal dataset preparation and one-epoch acceptance at branch `e3-formal-training-prep`.

Do not start multi-epoch training during this checklist. Only use existing artifacts unless a step explicitly says it is a manual UI smoke action.

**Dataset identity (read before anything else)**: the 184-image / 7185-annotation dataset this checklist reviews is **SAM3's own machine pre-annotation**, not human-reviewed GT — see `docs/E3_DATASET_IDENTITY_ERRATUM.md`. It is registered in `data_manifests/dataset_identity_registry.json` with `human_reviewed=false`, `allowed_for_formal_training=false`. Everything below is a **pipeline smoke test acceptance**, not a model-quality acceptance. `bbox AP`/`AP50` numbers in the E3 report are not evidence of fine-tuning quality.

## Reference Paths

- Project: `/home/book/book01`
- UI command: `conda run -n sam301 python /home/book/book01/app.py`
- E2 tag: `E2_READY_FOR_FORMAL_TRAINING_PREPARATION`
- E3 one-epoch run: `/home/book/book01/runs/training/2026-07-03_14-42-27`
- E3 checkpoint: `/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt`
- Human review artifacts: `/home/book/book01/runs/review_artifacts/e3_human_acceptance_20260703`
- Data/duplicate report: `/home/book/book01/runs/review_artifacts/e3_human_acceptance_20260703/reports/DATA_SOURCE_AND_DUPLICATES.md`
- Checkpoint inference smoke report: `/home/book/book01/runs/review_artifacts/e3_human_acceptance_20260703/reports/CHECKPOINT_INFERENCE_SUMMARY.md`

## Data Review

### 1. Source COCO Inventory

Steps:

1. Open `runs/review_artifacts/e3_human_acceptance_20260703/reports/DATA_SOURCE_AND_DUPLICATES.md`.
2. Confirm all six `data/cvat_import_polygon_*_book_spine_iou0.5_book_spine.json` files are listed.
3. Confirm total formal source count is `184` images and `7185` annotations.

Expected result:

- Each COCO file has an image/annotation count.
- Source paths match `data/dataset_raw/<timestamp>_book_spine/images`.

Result: PASS / FAIL

Notes:

### 1b. Dataset Identity (new)

Steps:

1. Open `data_manifests/dataset_identity_registry.json`.
2. Confirm `annotation_source: "sam3_machine_preannotation"`, `human_reviewed: false`, `allowed_for_formal_training: false`.
3. In the UI training-preflight tab, confirm the "training mode" selector defaults to `smoke` and that selecting `formal` (without a human-reviewed dataset registered) produces a preflight error rather than allowing a run.

Expected result:

- Registry entry is present and matches the numbers above.
- `training_mode="formal"` is rejected by preflight for this dataset.
- `max_epochs>1` is rejected by preflight for this dataset even in `smoke` mode.

Result: PASS / FAIL

Notes:

### 2. Duplicate Groups

Steps:

1. In `DATA_SOURCE_AND_DUPLICATES.md`, inspect all 44 exact duplicate groups.
2. Open duplicate contact sheets under `runs/review_artifacts/e3_human_acceptance_20260703/gt_overlays/duplicates`.
3. Decide whether duplicates are acceptable for formal training or should be pruned before multi-epoch training.

Expected result:

- No duplicate file has been deleted.
- Duplicate groups are documented with SHA256, paths, split, likely source, and recommendation.
- Exact duplicates do not cross split in the E3 formal split.

Result: PASS / FAIL

Notes:

### 3. Random GT Overlay Samples

Steps:

1. Open these directories:
   - `gt_overlays/random_train`
   - `gt_overlays/random_val`
   - `gt_overlays/random_test`
2. Visually confirm masks align with book spines.

Expected result:

- Overlays are readable.
- Masks mostly cover book spines, not background.
- Obvious annotation defects are noted.

Result: PASS / FAIL

Notes:

### 4. Edge-Case Samples

Steps:

1. Open:
   - `gt_overlays/tiny_masks`
   - `gt_overlays/pattern_samples`
2. Inspect very small masks and dense/complex pattern samples.

Expected result:

- Tiny masks are intentional book spines or acceptable small objects.
- Dense/complex images are still annotated plausibly.

Result: PASS / FAIL

Notes:

## UI Review

### 5. UI Startup

Steps:

1. Run `conda run -n sam301 python /home/book/book01/app.py` from a normal terminal.
2. Open `http://127.0.0.1:7860`.

Expected result:

- UI starts without traceback.
- Tabs are visible: inference config, history, results, CVAT export, training preflight.

Result: PASS / FAIL

Notes:

### 6. Inference Configuration UI

Steps:

1. Open the inference config tab.
2. Set checkpoint to `/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt`.
3. Set prompt to `book spine`.
4. Use a small input directory only if manually testing inference.

Expected result:

- Fields accept the E3 checkpoint path.
- CUDA unavailable in the UI process should block CUDA inference rather than silently falling back to CPU.

Result: PASS / FAIL

Notes:

### 7. Training Preflight UI

Steps:

1. Open the training preflight tab.
2. Use formal paths:
   - train images: `/home/book/book01/data/dataset_raw`
   - train COCO: `/home/book/book01/data/formal_book_spine_sam3_dataset/train/annotations.json`
   - val images: `/home/book/book01/data/dataset_raw`
   - val COCO: `/home/book/book01/data/formal_book_spine_sam3_dataset/val/annotations.json`
3. Set `max_epochs=1`, `train batch size=1`, `gradient accumulation steps=4`, `num_gpus=1`.
4. Run preflight only.

Expected result:

- Preflight passes.
- It reports train `160` images / `6749` annotations and val `12` images / `253` annotations.
- Runtime YAML and command are generated under a new run directory.
- Editing any input invalidates the old preflight state.

Result: PASS / FAIL

Notes:

### 8. Training Start Guard

Steps:

1. Confirm the start button requires the confirmation checkbox.
2. Do not start multi-epoch training.
3. If performing only a manual one-epoch smoke, verify that the button consumes a single preflight token and cannot relaunch the same run.

Expected result:

- Start is blocked without confirmation.
- Old preflight cannot be reused after launch.
- No stale runtime YAML can be launched after changing parameters.

Result: PASS / FAIL

Notes:

### 9. Training Status, Logs, Summary

Steps:

1. Open or inspect existing E3 run `/home/book/book01/runs/training/2026-07-03_14-42-27`.
2. Review `training_summary.json`, `logs/book_spine/log.txt`, `train_stats.json`, and `val_stats.json`.

Expected result:

- Summary status is `completed`.
- Exit code is `0`.
- Train shows `40` outer iterations and `160` micro-batches.
- Val shows `12` steps.

Result: PASS / FAIL

Notes:

### 10. Stop / Resume Controls

Steps:

1. Inspect the training preflight tab.
2. Confirm a stop button exists.
3. Confirm there is no separate resume-training control in the current UI.

Expected result:

- Stop is implemented for an active task and should mark cancelled if used.
- Resume is not implemented as a separate UI feature; future resume behavior requires explicit design and approval.

Result: PASS / FAIL

Notes:

## Checkpoint And Inference Review

### 11. Checkpoint File

Steps:

1. Inspect `/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt`.
2. Confirm file size is `10081250310` bytes.

Expected result:

- Checkpoint exists and is non-zero.
- It is under the E3 run directory, not under `/home/book/sam301`.

Result: PASS / FAIL

Notes:

### 12. Checkpoint Inference Smoke (superseded by P1-B remediation — see below)

The original version of this step accepted a **zero-prediction** result as "functional-path verification only." That is no longer the acceptance bar: zero predictions caused by a silent zero-weight-load must be treated as **FAIL**, not as an accepted known limitation. Use step 12b below instead.

### 12a. Checkpoint Type Rejection (new)

Steps:

1. In the inference config UI (or via `core.sam3_adapter.Sam3Adapter`), point the checkpoint field at `runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt` (the raw trainer checkpoint) directly.

Expected result:

- Loading is refused immediately with an error naming it a TRAINER checkpoint and pointing at `scripts/export_sam3_inference_checkpoint.py`.
- No inference is attempted with an unexported trainer checkpoint.

Result: PASS / FAIL

Notes:

### 12b. Exported Inference Checkpoint Acceptance (new)

Steps:

1. Confirm `runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt` exists (produced by `scripts/export_sam3_inference_checkpoint.py`).
2. Open `docs/P1_REMEDIATION_AND_INFERENCE_ACCEPTANCE.md` and confirm: `coverage_ratio=1.0`, `missing_keys=[]`, `unexpected_keys=[]`, and a non-zero base-checkpoint weight diff (`changed_tensors=385`).
3. Open `runs/review_artifacts/checkpoint_inference_acceptance/checkpoint_inference_acceptance_results.json` and the visualizations under `runs/review_artifacts/checkpoint_inference_acceptance/visualizations/` for the 4 real smoke images (normal spine, comic/complex pattern, tilted spine, thin/dense shelf).

Expected result:

- All 4 images produce **non-empty** predictions with the exported inference checkpoint. **A zero-prediction result on any image is a FAIL for this step**, not an accepted limitation — investigate the checkpoint/prompt/threshold before signing off.
- Mask counts/scores visibly differ from the base-checkpoint run on the same 4 images (proof the exported weights are actually being used, not just present).
- Visualizations show masks plausibly aligned to book spines (not background) on all 4 categories, including the comic/complex-pattern and tilted-spine images specifically.

Result: PASS / FAIL

Notes:

## History And Output Files

### 13. History Tab

Steps:

1. Open the UI history tab.
2. Confirm E3 run `2026-07-03_14-42-27` appears in training history.

Expected result:

- Status is visible as completed.
- Key metadata and paths are readable.

Result: PASS / FAIL

Notes:

### 14. Result Viewing

Steps:

1. Use the result/history tabs to inspect the checkpoint inference smoke run if inference runs are listed.
2. Open raw and NMS visualizations for the four sampled images.

Expected result:

- UI does not crash on zero-prediction images.
- Tables and extra reports render as empty or unavailable where appropriate.

Result: PASS / FAIL

Notes:

### 15. Output Isolation

Steps:

1. Confirm generated review files are under `runs/review_artifacts/e3_human_acceptance_20260703`.
2. Confirm no original images, annotations, checkpoints, or run outputs were added to Git.

Expected result:

- Review artifacts are not tracked by Git.
- Existing source data and checkpoints are unchanged.

Result: PASS / FAIL

Notes:

## Final Human Decision

Data accepted for multi-epoch training parameter approval: PASS / FAIL

UI accepted for supervised training operation: PASS / FAIL

Checkpoint inference path accepted as functional smoke only: PASS / FAIL

Known follow-up before multi-epoch training:

- Dataset must become human-reviewed (`independently_corrected_gt`) and be registered with `allowed_for_formal_training=true` before `training_mode="formal"` or `max_epochs>1` will be accepted by preflight for this data (see `docs/E3_DATASET_IDENTITY_ERRATUM.md`).
- Decide whether to prune exact duplicate records before full training.
- ~~Decide whether the training checkpoint should be adapted/exported differently for inference~~ — **done**: `scripts/export_sam3_inference_checkpoint.py` / `core/checkpoint_export.py`, see `docs/CHECKPOINT_EXPORT_AND_INFERENCE.md` and `docs/P1_REMEDIATION_AND_INFERENCE_ACCEPTANCE.md`.
- Approve final multi-epoch hyperparameters separately.

