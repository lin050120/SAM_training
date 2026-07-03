# P1 Remediation and Inference Acceptance

Generated: 2026-07-03

Result: **PASS**

Both P1 findings from `docs/FABLE5_FULL_INDEPENDENT_PROJECT_AUDIT.md` are remediated with fail-closed guards, tests, and a real 4-image inference acceptance. This is a code/policy fix, not a data fix: it does not and cannot make the underlying dataset human-reviewed (that remains open, tracked as a prerequisite for formal training).

## P1-A: Dataset identity (F-P1-1)

### Root cause

The 184-image / 7185-annotation dataset used by `data/formal_book_spine_sam3_dataset` is SAM3's own machine pre-annotation, not human-reviewed GT — every source COCO's `info.description` reads `"book_spine SAM3 pre-annotation (polygon ~8pts, NMS)"`, and 100% of 7185 polygons have <=8 vertices (recomputed independently this session, not copied from any prior report). Nothing in the codebase previously recorded or enforced this.

### Fix

- New `data_manifests/dataset_identity_registry.json` (git-tracked, **does not modify** `formal_dataset_manifest.json` / `formal_split_manifest.json` — those two files and their SHA256 hashes are byte-for-byte unchanged by this session; `tests/test_dataset_identity.py::HistoricalManifestUntouchedTest` verifies this and would fail if they had been touched).
- New `core/dataset_identity.py`: `resolve_dataset_identity()` matches by **resolved annotation file path** against the registry — never by filename. Unmatched paths default to `human_reviewed=false` (fail-safe, not fail-open).
- `core/training_runner.py::inspect_training_config()` gained a `training_mode` parameter (`"smoke"` default / `"formal"`) and now, before writing any runtime YAML or issuing a launch token: rejects `training_mode="formal"` against a non-`allowed_for_formal_training` dataset; rejects `max_epochs` above the dataset's `max_epochs_without_human_review` (1) in either mode. One-epoch smoke runs continue to work unchanged.
- `dataset_identity` and `training_mode` are written into every new run's `dataset_info.json` and `training_config_summary.json`.
- UI (`ui/training_preflight_page.py`) gained a training-mode selector (default `smoke`) and a prominent warning banner whenever the resolved dataset is not human-reviewed.
- `docs/E3_DATASET_IDENTITY_ERRATUM.md` (new) corrects how to interpret `docs/E3_FORMAL_DATASET_AND_ONE_EPOCH_ACCEPTANCE.md` **without modifying that report**.

### Dataset identity registry (independently recomputed, not copied from the audit report)

- Path: `data_manifests/dataset_identity_registry.json`
- `image_file_count=184`, `unique_image_count=136`, `annotation_count=7185`, `exact_duplicate_group_count=44`
- Splits: train 160 files / 120 unique / 6749 annotations; val 12/12/253; test 12/4/183
- `human_reviewed=false`, `independently_corrected_gt=false`, `allowed_for_formal_training=false`, `allowed_for_model_evaluation=false`, `max_epochs_without_human_review=1`

### Does this block formal multi-epoch training?

**Yes.** Verified directly:

```
formal-mode errors: ["dataset identity guard: formal training mode requested, but this
dataset is not marked allowed_for_formal_training=true ...", "dataset identity guard:
max_epochs=20 exceeds the smoke-mode limit (1) for a dataset that is not
human-reviewed ..."]
smoke-mode max_epochs=20 errors: ["dataset identity guard: max_epochs=20 exceeds the
smoke-mode limit (1) ..."]
```

Both the explicit `training_mode="formal"` path and an implicit multi-epoch smoke request against this dataset are rejected before any runtime YAML is written and before a launch token is issued.

### Tests

`tests/test_dataset_identity.py` (17 tests): registry lookup by resolved path (not filename), mismatched train/val pairing does not silently match, unmatched dataset fails closed, formal-mode rejection, multi-epoch rejection, one-epoch-smoke acceptance, real end-to-end preflight integration (identity written to `dataset_info.json`/`training_config_summary.json`), and the historical-manifest-untouched guard.

Fixing existing tests to satisfy the new guard: several pre-existing tests (`RuntimeYamlOverrideTest`, `Sam301EnvironmentMigrationTest`, `HydraLaunchRegressionTest`, `TrainingOutputConfinementTest`, `UnicodeAndSpacePathTest`, `TrainingPageStageATest`, `test_ui.py::TrainingPreflightCallTest`) used the default (non-registered) smoke dataset with the base YAML's `max_epochs=20` default and had no assertions about epoch count — these were given an explicit `max_epochs=1` input (or, for the one test whose actual purpose is proving the base-YAML-fallback for `max_epochs=20` specifically, a mocked "allowed for formal training" identity) so the new guard's precondition is satisfied without weakening any assertion.

## P1-B: Checkpoint export (F-P1-2)

### Root cause (see `docs/CHECKPOINT_EXPORT_AND_INFERENCE.md` for full derivation)

`sam301/sam3/model_builder.py::_load_checkpoint()` filters `if "detector" in k` after unwrapping a `"model"` key if present. For the **base** checkpoint (`/home/book/sam301/sam3.pt`) this is correct: its top-level dict is a flat state dict already prefixed `detector.*`/`tracker.*`. For a **trainer** checkpoint, `ckpt["model"]`'s keys (e.g. `backbone.vision_backbone.trunk.pos_embed`) have **no** `"detector"` substring anywhere — `trainer.model` is a `Sam3Image` built by the same `build_sam3_image_model` target used for inference, never wrapped in a `.detector` attribute. The filter therefore matches **zero** of the trainer checkpoint's 1134 keys, `load_state_dict({}, strict=False)` loads nothing, and the model silently stays at its random/default init.

### Real checkpoint structure (audited, not assumed)

`runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt`:

- Top-level keys: `model`, `optimizer`, `epoch`, `loss`, `steps`, `time_elapsed`, `best_meter_values`, `scaler`.
- `model`: 1134 tensors, 841,689,398 parameters. Top-level key prefixes: `backbone` (737), `transformer` (283), `segmentation_head` (28), `geometry_encoder` (76), `dot_prod_scoring` (10).
- Sample keys: `backbone.vision_backbone.trunk.pos_embed`, `transformer.decoder.layers.4.ca_text.out_proj.weight`, `segmentation_head.pixel_decoder.conv_layers.0.weight` — **zero** contain `"detector"` or `"tracker"`.
- A fresh `build_sam3_image_model(checkpoint_path=None).state_dict()` has an **identical** 1134-key keyset, 0 shape mismatches. `model.load_state_dict(ckpt["model"], strict=True)` succeeds directly — this is why the mapping is identity, derived empirically, not guessed.
- `/home/book/sam301/sam3.pt` (base), after stripping its `detector.` prefix: 1156 keys — a strict superset (22 extra tracker/video-only buffers).
- vs base: 385 tensors changed, 749 identical (consistent with the training config's frozen-backbone policy — only `transformer`/`segmentation_head`/`dot_prod_scoring` trained).

### Key mapping rules

`core.checkpoint_export.build_key_mapping()`: tries candidate prefixes (`""`, `"module."`, `"detector."`) against the *real* source/target keys, picks the best unambiguous match (refuses to guess on a tie between two prefixes with different resulting mappings), forbids two source keys mapping to one target key, checks every mapped pair's shape and dtype, and reports matched/missing/unexpected/shape-mismatch/per-module coverage. For this checkpoint the chosen mapping is the identity function (empty prefix) with 100% coverage.

### Fail-loud rules

Export (`core.checkpoint_export.export_inference_checkpoint`) rejects: non-trainer input; `matched_tensors==0`; `coverage_ratio<0.98`; any shape mismatch; any critical module with zero matched tensors; overwriting the source/base checkpoint or an existing output without `--overwrite`; and — separately from key coverage — **weights bit-identical to the base checkpoint** (would mean no real fine-tuning happened). After writing, export self-verifies by strict-loading the file into a **fresh** model. `core.checkpoint_export.load_inference_checkpoint()` (used by `Sam3Adapter`) rejects trainer and base checkpoints with actionable messages, and refuses a zero-parameter load even if `strict=True` didn't raise.

### Exporter and scripts

- `core/checkpoint_export.py` (core logic)
- `scripts/export_sam3_inference_checkpoint.py` (CLI)
- `core/sam3_adapter.py` modified: `Sam3Adapter.__init__` now calls `identify_checkpoint()` first and dispatches to the base-checkpoint path (unchanged), the new fail-loud inference-checkpoint loader, or raises immediately for a trainer checkpoint with the export command in the error message. **`/home/book/sam301` was not modified** — the buggy `_load_checkpoint` filter is bypassed entirely, never patched.

### Export output

```
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input runs/training/2026-07-03_14-42-27/checkpoints/checkpoint.pt \
  --output runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt
```

- Output: `/home/book/book01/runs/training/2026-07-03_14-42-27/checkpoints/inference_model.pt`
- SHA256: `e3aea4edbadc684f7807aed0d981a481f8650e01dff3041df21d29fe09afd222`
- `matched_tensors=1134`, `matched_parameters=841689398`, `coverage_ratio=1.0`, `missing_keys=[]`, `unexpected_keys=[]`
- `base_diff`: `changed_tensors=385`, `identical_tensors=749`
- Source `checkpoint.pt` (mtime unchanged at 14:44, from the original training run) and `/home/book/sam301/sam3.pt` (SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`, unchanged) were **not modified** by the export.

## Real 4-image inference acceptance (fresh model instances, no reuse of the training process's model)

Prompt `"book spine"`, score threshold `0.3`, min area `200`, device `cuda`. Images (from the real smoke dataset, chosen for category diversity):

| Category | Image |
|---|---|
| Normal spine | `data/dataset_raw/20260630_180628_book_spine/images/img_0001.png` |
| Comic/complex pattern | `data/dataset_raw/20260626_161819_book_spine/images/img_0001.png` |
| Tilted spine | `data/dataset_raw/20260623_233120_book_spine/images/img_0050.png` |
| Thin/dense shelf | `data/dataset_raw/20260622_231208_book_spine/images/img_0006.png` |

| Checkpoint | Result | normal | comic | tilted | thin/dense |
|---|---|---|---|---|---|
| Base (`sam3.pt`) | loaded, ran | 51 masks | 22 masks | 52 masks | 21 masks |
| Trainer (`checkpoint.pt`) direct | **rejected before any inference** — `ValueError: ... is a TRAINER checkpoint ... Export it first: ...` | — | — | — | — |
| Inference (`inference_model.pt`) | loaded, ran | 51 masks | 22 masks | 54 masks | 25 masks |

No image produced zero predictions with either the base or the exported inference checkpoint. Scores differ materially between base and exported (e.g. normal-spine base scores mixed 0.30-0.89; exported scores mostly 0.75-0.96) and mask counts differ on 2 of 4 images (tilted 52->54, thin/dense 21->25) — the model's output is materially different with the exported weights loaded, which is direct behavioral evidence the loaded weights are actually being exercised by the inference code, not merely present in memory unused.

Full numeric results (not tracked by git, under gitignored `runs/`): `runs/review_artifacts/checkpoint_inference_acceptance/checkpoint_inference_acceptance_results.json`; visualizations under `runs/review_artifacts/checkpoint_inference_acceptance/visualizations/`.

### Why this is not judged on "non-empty predictions" alone

Per the acceptance criteria, success required all of:

1. **High/strict key coverage**: `coverage_ratio=1.0`, `missing_keys=[]`, `unexpected_keys=[]` — verified above.
2. **Real fresh-model load**: `Sam3Adapter` always constructs a new model (`load_from_HF=False, checkpoint_path=None` then `load_state_dict`); no model object from the training process was reused. `tests/test_checkpoint_export.py::RealCheckpointIntegrationTest::test_export_and_strict_load_into_fresh_model_round_trips` performs this independently of the adapter and of the export's own self-verification.
3. **Source vs loaded tensor equality**: the same test samples a real changed tensor (`dot_prod_scoring.hs_proj.bias`) from the raw trainer checkpoint and asserts `torch.equal()` against the tensor actually present in the loaded fresh model — bit-for-bit, not "close enough."
4. **Real difference from base**: 385/1134 tensors changed (not size-based, not file-size-based — computed via `torch.equal()` on every common tensor); export explicitly refuses to proceed if this were 0.
5. **Inference code actually uses the exported weights**: proven behaviorally (different scores/mask counts vs base, item above), not just by successful loading.
6. **No silent zero-load**: `load_inference_checkpoint` raises if the loaded parameter count is 0, independent of whatever `strict=True` returns.

All six were independently checked; none was skipped or assumed.

## Tests

New: `tests/test_dataset_identity.py` (17), `tests/test_checkpoint_export.py` (29, including 4 gated on real fixtures that build a real fresh SAM3 model and do a real strict load — not mocked).

Repeated runs (per instructions, 3x on the new tests):

```
46 passed, 4 warnings in 53.64s
46 passed, 4 warnings in 56.30s
46 passed, 4 warnings in 55.03s
```

Full suite: `228 passed, 9 warnings, 17 subtests passed in 83.87s` (`conda run -n sam301 python -m pytest tests/ -q`; 182 (baseline at commit `36704e1`, before this session) -> 228, +46 new P1-A/P1-B tests, 0 pre-existing tests removed or weakened — several had their input fixtures adjusted (explicit `max_epochs=1`, or a mocked "allowed for formal training" identity for the one test whose purpose is the base-YAML `max_epochs=20` fallback) to satisfy the new dataset-identity precondition; their original assertions and behavior-under-test are unchanged).

Gradio smoke: `1 passed` (`tests/test_ui_smoke.py`). Hydra `--validate-only` on the real run's runtime YAML: `hydra validation ok`. SAM301 patch: `status`/`verify --json` both report `PATCHED`, `verify` exit code `0`.

No training was started, no launch token was consumed, no new training run directory was created during this remediation.

## Scope discipline

Not modified: `/home/book/sam3`, `/home/book/sam301` (source or patch), any historical run's `training_summary.json`/`provenance.json`/runtime YAML/checkpoint, `data_manifests/formal_dataset_manifest.json`, `data_manifests/formal_split_manifest.json`, the original `checkpoint.pt` or `sam3.pt`, batch size/learning rate/other training parameters. COCO merge tooling, duplicate pruning, a new split, mask IoU metrics, best-checkpoint selection, resume, RNG state, directory restructuring, and 20-epoch formal training are explicitly out of scope for this remediation and remain open per `docs/FABLE5_AUDIT_ACTION_PLAN.md`.

## Conclusion

**PASS.** Both P1 findings are closed at the code/policy layer: the dataset's true identity is now recorded, enforced fail-closed, and cannot silently permit formal or multi-epoch training; the trainer-checkpoint-to-inference gap is closed with a verified, fail-loud export/load path and a real 4-image behavioral acceptance. The underlying dataset is still not human-reviewed — that is a data-acquisition prerequisite, not something a code fix can close, and formal multi-epoch training remains correctly blocked until it is addressed (`docs/FABLE5_AUDIT_ACTION_PLAN.md`, items 6-8).
