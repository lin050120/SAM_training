# Codex Checkpoint Evaluation Review and Val/Test Extension

Date: 2026-07-04

## Git Baseline

- Branch: `codex-stage-e4-checkpoint-evaluation`
- Reviewed commit: `076e973ccff69922235822a9cad5cb66bd37f964`
- Implementation commit: `a76c4ec`
- SAM301 source was not modified.
- No new training run was created and no trainer was started.

## Independent Review Findings

### Closed / Accepted

- Checkpoint discovery sorts `checkpoint_N.pt` numerically.
- `checkpoint.pt` is treated as an alias when its SHA256 matches an epoch checkpoint.
- Trainer checkpoints are loaded through the strict trainer checkpoint loader.
- Baseline and fine-tuned checkpoints share the same `Sam3Adapter.predict` path and inference conditions.
- Hungarian matching is one-to-one and rejects zero-IoU assignments.
- Missed GT contributes `0` to all-GT IoU and all-GT Boundary F1.
- Boundary F1 is computed on original-size masks.
- Validation best selection uses the documented hierarchy and selected `checkpoint_20.pt`.

### Blocking Gaps Fixed

- Existing implementation was validation-only.
- Existing outputs did not contain raw prediction masks, GT snapshots, per-instance metrics, IoU matrices, assignment records, or visualizations.
- Existing cache key did not include split name or image hashes.
- Test split was not registered in dataset identity registry.
- UI did not separate validation ranking from diagnostic test results.

## Dataset Audit

Run:

`/home/book/book01/runs/training/2026-07-04_14-28-27`

Audit output:

- `evaluation/dataset_split_audit.json`
- `evaluation/dataset_split_audit.csv`

Counts:

- Train: 44 COCO images, 857 annotations, category `book spine`
- Validation: 8 COCO images, 155 annotations, category `book spine`
- Test: 12 COCO images, 229 annotations, category `book spine`

Cross-split overlap:

- train vs validation: 0 by normalized filename, file size, SHA256, and pixel hash
- train vs test: 0 by normalized filename, file size, SHA256, and pixel hash
- validation vs test: 0 by normalized filename, file size, SHA256, and pixel hash

The dataset remains `allowed_for_model_evaluation=false`; test output is diagnostic only.

## Evaluation Outputs

Validation:

- `evaluation/validation/checkpoint_metrics.csv`
- `evaluation/validation/checkpoint_metrics.json`
- `evaluation/validation/per_image_metrics.csv`
- `evaluation/validation/per_instance_metrics.csv`
- `evaluation/validation/gt_snapshot.json`
- `evaluation/validation/raw_predictions/`
- `evaluation/validation/match_records/`
- `evaluation/validation/visualizations/`
- `evaluation/validation/human_review_index.csv`

Test:

- `evaluation/test/checkpoint_metrics.csv`
- `evaluation/test/checkpoint_metrics.json`
- `evaluation/test/per_image_metrics.csv`
- `evaluation/test/per_instance_metrics.csv`
- `evaluation/test/gt_snapshot.json`
- `evaluation/test/raw_predictions/`
- `evaluation/test/match_records/`
- `evaluation/test/visualizations/`
- `evaluation/test/human_review_index.csv`

Selection:

- `evaluation/best_checkpoint.json`
- `evaluation/test_checkpoint_comparison.json`
- `evaluation/evaluation_summary.json`

## Validation Result

- Status: `completed`
- GT instances: 155
- Best checkpoint: `checkpoint_20.pt`
- Best epoch: 20
- Primary metric: `mean_iou_all_gt = 0.9265702815938421`
- Selection trace: top IoU tie set, Boundary F1 tie-break, miss-rate tie-break, FP/image tie-break, selected epoch 20.

## Test Result

- Status: `completed`
- GT instances: 229
- Diagnostic highest metric checkpoint: `checkpoint_20.pt`
- `test_checkpoint_comparison.json` records `diagnostic_only=true` and `does_not_affect_model_selection=true`.
- Test did not rewrite `best_checkpoint.json` and did not change `inference_best.pt` source.

## Raw Recompute Check

Sampled raw prediction NPZ files were reopened and their match-record IoU matrix dimensions were verified against saved prediction counts for both validation and test splits. Example sampled records included:

- validation `checkpoint_5.pt` / `im_000022.png`: matrix `19 x 19`
- validation `checkpoint_5.pt` / `im_000036.png`: matrix `20 x 20`
- test `checkpoint_5.pt` / `im_000001.png`: matrix `15 x 15`
- test `checkpoint_5.pt` / `im_000002.png`: matrix `25 x 26`

## Tests

- `conda run -n sam301 python -m pytest tests/test_checkpoint_evaluation.py -v`: 39 passed
- `tests/test_checkpoint_evaluation.py` repeated three times: 39 passed each run
- `conda run -n sam301 python -m pytest tests/test_ui_smoke.py -v`: 1 passed
- `conda run -n sam301 python -m pytest tests/ -v`: 270 passed, 17 subtests passed, 11 warnings
- `conda run -n sam301 python scripts/manage_sam301_patch.py status`: PATCHED
- `conda run -n sam301 python scripts/manage_sam301_patch.py verify`: ok
- `conda run -n sam301 python scripts/launch_sam3_training.py --config ... --validate-only`: hydra validation ok

## Remaining Risks

- Because all checkpoints are now shown on the registered test split, that test split is no longer a blind final evaluation set.
- `allowed_for_model_evaluation=false` remains correct.
- Formal final quality claims still require a separate blind, human-reviewed evaluation set.
