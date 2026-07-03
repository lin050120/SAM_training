# Codex Strict Review of Data Identity and Checkpoint Export

Generated: 2026-07-03

Scope: independent, read-only review of Sonnet 5 remediation commits for F-P1-1
and F-P1-2. I did not modify project source, SAM301 source, data, historical
runs, registry, manifests, checkpoints, or Conda/CUDA state. I did not start
training or consume a launch token.

## Verdict

**APPROVE_WITH_FOLLOWUPS**

- F-P1-1: **closed** for the current code path. The 184-file / 7185-annotation
  dataset is now recorded as SAM3 machine pre-annotation, not human-reviewed GT,
  and formal/multi-epoch training is blocked fail-closed before runtime YAML and
  launch token creation.
- F-P1-2: **closed** for the current inference path. The trainer checkpoint is
  exported into a fail-loud inference checkpoint, strict-loads into a fresh model,
  and produces non-empty real predictions.
- Human acceptance: **allowed**, as pipeline/checkpoint smoke acceptance only.
- Formal multi-epoch training: **not allowed** with the current dataset. The data
  remains unreviewed machine pre-annotation and the guard correctly blocks it.

## Git Baseline

- Branch: `e3-formal-training-prep`
- HEAD: `20cbe045a47adaeb49f352a4af8fd5d1ee982955`
- Reviewed range: `36704e1..HEAD`
- Relevant commits:
  - `30c50e4 Correct smoke dataset identity`
  - `1661d0c Add SAM3 inference checkpoint export`
  - `20cbe04 Document checkpoint inference acceptance`
- Untracked files present and left untouched:
  - `docs/CLAUDE_REVIEW_OF_GRAD_ACCUM_FIX.md`
  - `docs/CLAUDE_REVIEW_OF_SAM301_ENV_MIGRATION.md`
  - `docs/HANDOFF_FOR_REVIEW_E2_FIXES.md`
  - `docs/HUMAN_ACCEPTANCE_CHECKLIST.md`

Diff scope from `36704e1` was limited to README/docs, new dataset identity and
checkpoint export modules/scripts/tests, and small training/UI guard integration.
`git ls-files` showed no tracked `runs/`, `data/`, model, checkpoint, or image
artifacts. `/home/book/sam301` was not modified by these commits.

## Data Identity Review

Reviewed:

- `core/dataset_identity.py`
- `data_manifests/dataset_identity_registry.json`
- `core/training_runner.py`
- `ui/training_preflight_page.py`
- `README.md`
- `docs/stage_e1_training_ui.md`
- `tests/test_dataset_identity.py`
- `tests/test_e1_training.py`
- `tests/test_ui.py`

Independent recomputation:

- Source COCO files: 6
- Source descriptions: all `book_spine SAM3 pre-annotation (polygon ~8pts, NMS)`
- Source annotations: 7185
- Polygon vertex counts: 4=8, 5=61, 6=223, 7=1049, 8=5844
- Split files:
  - train: 160 image files, 120 unique SHA256 images, 6749 annotations
  - val: 12 image files, 12 unique SHA256 images, 253 annotations
  - test: 12 image files, 4 unique SHA256 images, 183 annotations
- Total image files: 184
- Unique image SHA256 count: 136
- Exact duplicate groups: 44
- Cross-split exact duplicate groups: 0
- Missing images: 0
- Category: `book_spine`, id 1

Registry values match the independently recomputed data:

- `annotation_source="sam3_machine_preannotation"`
- `human_reviewed=false`
- `independently_corrected_gt=false`
- `allowed_for_formal_training=false`
- `allowed_for_model_evaluation=false`
- `max_epochs_without_human_review=1`

The lookup matches by resolved train/val annotation paths. It does not trust a
filename-only match. Mismatched train/val pairings and unmatched datasets fail
closed to non-human-reviewed smoke-only status.

Guard placement:

- `core.training_runner.inspect_training_config()` resolves identity before
  `write_runtime_yaml()`.
- `training_mode="formal"` is rejected for this dataset.
- `max_epochs>1` is rejected even in smoke mode.
- One-epoch smoke remains allowed.
- On errors, `inspect_training_config()` returns before run files, runtime YAML,
  or launchable state are created.
- UI exposes a `training_mode` selector and shows an explicit unreviewed-data
  warning.

Historical artifacts:

- `formal_dataset_manifest.json` SHA256:
  `1e56e47f64a9b08c774e8d2e5765815d8b008a0a2f6189939b85fcf7bd47899a`
- `formal_split_manifest.json` SHA256:
  `4bb39f138fd4d936ee12fe24b636b002a97793a76232fe6d6dabf72185beb7e0`
- The historical run `2026-07-03_14-42-27` was not rewritten; its
  `training_summary.json` predates `dataset_identity`, which is expected.

Conclusion: F-P1-1 is closed as a code/policy guard. The previous one-epoch result
must be interpreted only as a pipeline smoke test, not a formal model-quality
acceptance.

## Checkpoint Structure Review

Reviewed real trainer checkpoint:

`runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt`

Independent structure:

- SHA256: `a62cc3da0c007c3274ec39f994cf5177d07e2cfbfa9470eeb79a138961a45cf7`
- Top-level keys:
  - `model`
  - `optimizer`
  - `epoch`
  - `loss`
  - `steps`
  - `time_elapsed`
  - `best_meter_values`
  - `scaler`
- Epoch: 1
- Steps: `{"train": 160, "val": 0}`
- Model tensors: 1134
- Model parameters: 841,689,398
- Prefix distribution:
  - `backbone`: 737
  - `transformer`: 283
  - `geometry_encoder`: 76
  - `segmentation_head`: 28
  - `dot_prod_scoring`: 10
- Dtype distribution:
  - `torch.float32`: 1102
  - `torch.complex64`: 32
- Keys containing `detector`: 0
- Keys containing `tracker`: 0
- Example keys:
  - `backbone.vision_backbone.trunk.pos_embed`
  - `backbone.vision_backbone.trunk.patch_embed.proj.weight`
  - `transformer...`

Fresh model comparison under host GPU-visible shell:

- Fresh `build_sam3_image_model(checkpoint_path=None).state_dict()` tensors: 1134
- Fresh model parameters: 841,689,398
- Source and target key sets: exact match
- Shape mismatches: 0
- Dtype mismatches: 0
- `load_state_dict(trainer_ckpt["model"], strict=True)`: all keys matched

Base checkpoint:

- `/home/book/sam301/sam3.pt` SHA256:
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- Base raw tensor keys: 1465
- Base `detector.*` keys after stripping `detector.`: 1156
- Common trainer/base model keys: 1134
- Changed tensors: 385
- Identical tensors: 749
- Changed prefixes:
  - `transformer`: 283
  - `geometry_encoder`: 68
  - `segmentation_head`: 24
  - `dot_prod_scoring`: 10

## Original Zero-Load Root Cause

The SAM301 base loader in `/home/book/sam301/sam3/model_builder.py::_load_checkpoint`
unwraps `ckpt["model"]` and then filters keys with `if "detector" in k`.

That works for base `sam3.pt`, whose keys use `detector.*`, but it matches 0 keys
from the trainer checkpoint because trainer `ckpt["model"]` uses the same
unprefixed key namespace as the fresh image model. Since the SAM301 loader then
calls `load_state_dict(..., strict=False)`, an empty load can silently leave the
model uninitialized for finetuned inference. Sonnet's root-cause description is
accurate.

## Exporter and Loader Review

Reviewed:

- `core/checkpoint_export.py`
- `scripts/export_sam3_inference_checkpoint.py`
- `tests/test_checkpoint_export.py`
- `core/sam3_adapter.py`

Exporter behavior:

- Checkpoint type is identified by real object structure, not filename.
- Base, trainer, inference, missing, and unknown types are distinct.
- The real trainer checkpoint uses identity key mapping (`chosen_prefix=""`).
- Candidate prefixes are limited to `""`, `module.`, and `detector.`.
- Ambiguous mappings are rejected.
- Multiple source keys mapping to the same target are rejected.
- Shape mismatches are rejected.
- Dtype mismatches are reported.
- Low coverage is rejected.
- Matched-zero is rejected.
- Critical-module zero coverage is rejected.
- Export refuses to overwrite the source trainer checkpoint or base checkpoint.
- Output contains only `format`, `format_version`, `model`, and `metadata`; no
  optimizer/scheduler/scaler state is present.
- Export self-verifies by loading the output into a fresh model.

Real export result for Sonnet artifact:

- Path:
  `runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt`
- SHA256:
  `e3aea4edbadc684f7807aed0d981a481f8650e01dff3041df21d29fe09afd222`
- Format: `sam3_inference`
- Format version: 1
- Model tensors: 1134
- Matched tensors: 1134
- Matched parameters: 841,689,398
- Coverage: 1.0
- Missing keys: 0
- Unexpected keys: 0
- Shape mismatches: 0
- Dtype mismatches: 0
- Inference model tensors equal trainer checkpoint tensors bit-for-bit.

Independent temporary export:

- Output:
  `runs/review_artifacts/codex_checkpoint_inference_review/codex_temp_inference_model.pt`
- SHA256:
  `1a70fdfe545dbc70c76dc2fb687e6d76552f4c5c5156d4fbcdc19e3ce43fe022`
- Matched tensors: 1134
- Coverage: 1.0
- Changed vs base: 385
- Identical vs base: 749
- Tensor payload is bit-identical to Sonnet `inference_model.pt`.
- SHA differs only because metadata differs (`book01_commit`, `exported_at`).

CLI export smoke:

- Output:
  `runs/review_artifacts/codex_checkpoint_inference_review/cli_export_smoke.pt`
- Matched tensors: 1134
- Matched parameters: 841,689,398
- Coverage: 1.0
- Missing/unexpected: 0/0

Adapter behavior:

- `Sam3Adapter` identifies checkpoint type before loading.
- Trainer checkpoints are rejected with an exporter command hint.
- Base checkpoint uses the existing SAM301 loader path.
- Inference checkpoint uses `core.checkpoint_export.load_inference_checkpoint()`.
- Unknown or damaged checkpoint structures fail loud.
- Project inference entry point `core/inference_run.py` constructs `Sam3Adapter`,
  so normal inference uses the new type checks.

Conclusion: F-P1-2 is closed for the normal project inference path.

## Four-Image Inference Reproduction

Output written under ignored review artifacts:

`runs/review_artifacts/codex_checkpoint_inference_review/codex_4_image_inference_results.json`

Prompt/config:

- Prompt: `book spine`
- Adapter confidence threshold: 0.05
- Final score threshold: 0.3
- Minimum area: 200
- Device: CUDA, RTX 5090
- Fresh model instances used for base and inference checkpoints.

Images:

- Normal:
  `data/dataset_raw/20260630_180628_book_spine/images/img_0001.png`
- Comic/complex:
  `data/dataset_raw/20260626_161819_book_spine/images/img_0001.png`
- Tilted:
  `data/dataset_raw/20260623_233120_book_spine/images/img_0050.png`
- Thin/dense:
  `data/dataset_raw/20260622_231208_book_spine/images/img_0006.png`

Results:

| Checkpoint | Normal | Comic | Tilted | Thin/dense |
|---|---:|---:|---:|---:|
| Base `sam3.pt` final masks | 51 | 22 | 52 | 21 |
| Exported inference final masks | 51 | 22 | 54 | 25 |

Raw proposal counts:

| Checkpoint | Normal | Comic | Tilted | Thin/dense |
|---|---:|---:|---:|---:|
| Base `sam3.pt` raw proposals | 162 | 83 | 155 | 105 |
| Exported inference raw proposals | 178 | 154 | 171 | 163 |

Direct trainer checkpoint load through `Sam3Adapter` was rejected before inference.
No image produced zero predictions for either base or exported inference
checkpoint. This proves the checkpoint->export->fresh inference model->mask
prediction chain is functional. It does not prove model quality improvement.

## Tests Run

Full suite:

```text
conda run -n sam301 python -m pytest tests/ -q
228 passed, 9 warnings, 17 subtests passed in 87.27s
```

Repeated new tests:

```text
for i in 1 2 3; do conda run -n sam301 python -m pytest \
  tests/test_dataset_identity.py tests/test_checkpoint_export.py -q; done
46 passed, 4 warnings
46 passed, 4 warnings
46 passed, 4 warnings
```

Gradio smoke:

```text
conda run -n sam301 python -m pytest tests/test_ui_smoke.py -q
1 passed, 5 warnings
```

Hydra validate-only:

```text
conda run -n sam301 python scripts/launch_sam3_training.py \
  -c runs/training/2026-07-03_14-42-27/config/runtime_config.yaml \
  --use-cluster 0 --num-gpus 1 --validate-only
hydra validation ok
```

SAM301 patch status/verify:

```text
state=PATCHED
actual_sha256=bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2
verify ok=true
```

No training process remained after checks:

```text
pgrep -af '[s]am3/train/train.py|[r]untime_config.yaml|[t]orchrun'
# no output
```

## Findings

### P1

None.

### P2

None.

### P3-1: `training_summary.json` does not directly record `dataset_identity`

New preflights write `dataset_identity` into `dataset_info.json` and
`training_config_summary.json`, and the guard itself correctly blocks before launch.
However, `ui.training_process_manager.finalize_training_summary()` has no
`dataset_identity` parameter and does not copy it from the config summary.

Impact: final run summaries are less self-contained for audit and downstream
automation. This does not reopen F-P1-1 because launch blocking happens earlier.

Recommended fix: pass/copy `dataset_identity` into final summary, including
`annotation_source`, `human_reviewed`, `intended_use`, and
`allowed_for_formal_training`.

### P3-2: Export CLI does not auto-populate dataset identity metadata

`export_inference_checkpoint()` can record `dataset_identity`, but the CLI only
records it when `--dataset-identity-json` is manually provided. The real
`inference_model.pt` and my independent CLI export both had
`metadata.dataset_identity=null`.

Impact: exported inference checkpoint weights are correct, but their metadata does
not by itself prove whether the source data was formal GT or smoke pre-annotation.

Recommended fix: when the input checkpoint is inside a managed run directory, have
the exporter read `training_config_summary.json` / `dataset_info.json` and include
dataset identity by default, while still allowing explicit override for external
checkpoints.

### P3-3: Existing `inference_model.pt` commit metadata is not a clean committed state

The existing exported checkpoint records:

- `book01_commit=36704e13057d381da80e1488958f1be23abec286`

That is the pre-remediation audit commit, not the later committed exporter code
state. A fresh export from the current checkout records current HEAD correctly.
The tensor payload is identical, so this is a provenance issue, not a weight-load
issue.

Impact: the existing inference artifact is functionally correct but its metadata is
not ideal as a reproducibility record.

Recommended fix: after P3-1/P3-2, regenerate inference checkpoints from a clean,
committed tree and record `git_dirty` plus dataset identity in metadata.

### Observations

- The base checkpoint has 1465 raw tensor keys; after filtering `detector.*` and
  stripping the prefix, it has 1156 image-detector-compatible keys. This matches
  the documented explanation that the base checkpoint contains extra tracker/video
  tensors.
- `tests/test_e1_training.py` fixture updates mostly add explicit `max_epochs=1`
  where tests are not about epoch count. One fallback test mocks a reviewed dataset
  identity to preserve the `max_epochs=20` fallback assertion. I did not find test
  weakening that masks either P1 fix.
- The historical E3 run predates dataset identity fields and was not rewritten.

## Final Conclusions

### A. F-P1-1

Closed. The data is now correctly treated as unreviewed SAM3 machine
pre-annotation, and unsafe training modes fail closed before runtime YAML/token
creation.

### B. F-P1-2

Closed. The trainer checkpoint cannot be silently zero-loaded through the normal
adapter path; exporting produces a strict-loadable inference checkpoint with full
key coverage, and real 4-image inference produces non-empty predictions.

### C. Human Acceptance

Allowed. The user can proceed with human acceptance of data visibility, UI
workflow, checkpoint export, and inference smoke behavior, while treating all
training/inference outputs as pipeline smoke artifacts.

### D. Formal Multi-Epoch Training

Not allowed. The current dataset remains `human_reviewed=false` and
`allowed_for_formal_training=false`; formal or multi-epoch training must remain
blocked until a genuinely human-reviewed dataset is registered.
