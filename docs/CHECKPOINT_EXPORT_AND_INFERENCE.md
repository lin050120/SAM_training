# Checkpoint Export and Inference

> Languages: **English** | [中文](CHECKPOINT_EXPORT_AND_INFERENCE_CN.md) | [日本語](CHECKPOINT_EXPORT_AND_INFERENCE_JA.md)

Generated: 2026-07-03

## Three checkpoint types (never guess from filename)

| Type | Example path | Top-level structure | Loaded via |
|---|---|---|---|
| **Base** | `/home/book/sam301/sam3.pt` | Flat dict, no wrapper, keys prefixed `detector.*` / `tracker.*` (1156 tensors) | `sam3.model_builder.build_sam3_image_model(checkpoint_path=...)` (unmodified, sam301) |
| **Trainer/resume** | `<run_dir>/checkpoints/checkpoint.pt` | `{"model": {...}, "optimizer": {...}, "epoch": int, "loss": {...}, "steps": {...}, "scaler": {...}}` | Never load directly for inference. Contains full resume state. |
| **Inference** | `<run_dir>/checkpoints/inference_model.pt` | `{"format": "sam3_inference", "format_version": 1, "model": {...}, "metadata": {...}}` | `core.checkpoint_export.load_inference_checkpoint()` |

`core.checkpoint_export.identify_checkpoint(path)` classifies a file by loading it and inspecting its real keys — never by extension or filename. `Sam3Adapter` (`core/sam3_adapter.py`) calls this on construction and picks the right loading path, or refuses.

## Why a trainer checkpoint can't be passed directly to inference (root cause)

`sam301/sam3/model_builder.py::_load_checkpoint()` (not modified by this project) does:

```python
if "model" in ckpt and isinstance(ckpt["model"], dict):
    ckpt = ckpt["model"]
sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
```

This is correct for a **base** checkpoint, whose top-level keys already carry a `detector.` prefix. But a **trainer** checkpoint's `ckpt["model"]` sub-dict has keys like `backbone.vision_backbone.trunk.pos_embed` — no `"detector"` substring anywhere, because `trainer.model` is built by the *same* `build_sam3_image_model` target used for inference and is never wrapped in a `.detector` attribute (`Sam3Image.__init__` sets `self.backbone`, `self.transformer`, etc. directly). The `if "detector" in k` filter therefore matches **zero** of the trainer checkpoint's 1134 keys, `load_state_dict({}, strict=False)` loads nothing, and the model silently keeps its random/default initialization — this is exactly what produced the 4/4-images-zero-predictions result recorded in the earlier human-acceptance inference smoke.

Verified empirically (2026-07-03) against the real run `2026-07-03_14-42-27`:

- Fresh `build_sam3_image_model(checkpoint_path=None).state_dict()`: 1134 tensors.
- Trainer checkpoint's `ckpt["model"]`: 1134 tensors, **identical keyset**, 0 shape mismatches. `model.load_state_dict(ckpt["model"], strict=True)` succeeds directly — the correct mapping is the **identity function**, not a guess.
- Base `sam3.pt`, after stripping its `detector.` prefix: 1156 keys (a strict superset — 22 extra tracker/video-only buffers not part of a plain image detector).

## How to export

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

Options: `--base-checkpoint <path>` (default `/home/book/sam301/sam3.pt`, used for the "did the weights actually change" check below), `--no-base-diff` to skip it, `--overwrite` to replace an existing output file, `--json` for machine-readable output.

Never overwrites: the source trainer checkpoint, the base checkpoint path, or an existing output file (unless `--overwrite`).

## Key mapping rules (`core.checkpoint_export.build_key_mapping`)

- Tries a small closed set of candidate prefixes (`""`, `"module."`, `"detector."`) against the *real* source keys, picks whichever produces the most matches against the *real* target model's `state_dict()`.
- Refuses to guess: if two different prefixes tie with **different** resulting mappings, export fails outright rather than picking one arbitrarily.
- No two source keys may map to the same target key (conflict -> hard error).
- Every mapped pair is checked for tensor shape and dtype; any shape mismatch is a hard error (not silently dropped).
- Reports `matched_tensors`, `matched_parameters`, `coverage_ratio`, `missing_keys`, `unexpected_keys`, `shape_mismatch`, and per-top-level-module (`backbone`/`transformer`/`segmentation_head`/`dot_prod_scoring`/`geometry_encoder`) matched/total counts.

## Fail-loud rules (export time)

Export raises (does not write a file) when:

- Input is not identified as a **trainer** checkpoint.
- `matched_tensors == 0`.
- `coverage_ratio < 0.98` (`MIN_COVERAGE_RATIO`).
- Any shape mismatch exists.
- Any top-level module has zero matched tensors.
- Output path already exists without `--overwrite`, or would overwrite the source/base checkpoint.
- **The exported weights are bit-identical to the base checkpoint across every common tensor** — this would mean the "export" doesn't represent a fine-tuned model at all.

After writing, export immediately self-verifies by strict-loading the file it just wrote into a **fresh** model instance (not the training process's model object) — this is not optional and cannot be skipped.

## strict / coverage rules (load time)

`core.checkpoint_export.load_inference_checkpoint(path)`:

- Rejects a **trainer** checkpoint with a message pointing at the exporter command.
- Rejects a **base** checkpoint with a message pointing at the existing `build_sam3_image_model(checkpoint_path=...)` path.
- Builds a fresh model (`build_sam3_image_model(checkpoint_path=None)`) and calls `model.load_state_dict(state_dict, strict=True)` — for the real architecture this succeeds with **zero** missing/unexpected keys (verified). `ALLOWED_MISSING_KEYS` / `ALLOWED_UNEXPECTED_KEYS` exist as an explicit, currently-empty allowlist mechanism for a future architecture change that legitimately needs one — this loader never silently falls back to `strict=False`.
- Refuses a checkpoint that loads zero parameters, even if `strict=True` didn't raise (defense in depth).

## Real acceptance result (2026-07-03, run `2026-07-03_14-42-27`)

- Exported `inference_model.pt`, SHA256 `e3aea4edbadc684f7807aed0d981a481f8650e01dff3041df21d29fe09afd222`.
- `matched_tensors=1134`, `matched_parameters=841689398`, `coverage_ratio=1.0`, `missing=[]`, `unexpected=[]`.
- vs base: 385 tensors changed, 749 identical (matches the frozen-backbone training config — only `transformer`/`segmentation_head`/`dot_prod_scoring` were trained).
- 4 real smoke images (normal spine, comic/complex pattern, tilted spine, thin/dense shelf): base checkpoint and exported inference checkpoint both produced **non-empty** predictions on all 4, with materially different scores and mask counts between the two (proof the exported weights are actually exercised, not silently ignored). Direct load of the raw trainer checkpoint was correctly **rejected** before any inference was attempted.
- Full numeric results: `runs/review_artifacts/checkpoint_inference_acceptance/checkpoint_inference_acceptance_results.json` (not tracked by git — under `runs/`).

See `docs/P1_REMEDIATION_AND_INFERENCE_ACCEPTANCE.md` for the full acceptance record.

## Common errors

- `ValueError: ... is a TRAINER checkpoint ... Export it first: ...` — you pointed inference at `checkpoints/checkpoint.pt` directly; run the exporter.
- `ValueError: ... is a BASE checkpoint ...` — you pointed the inference-checkpoint loader at `sam3.pt`; that path already works via the existing `build_sam3_image_model(checkpoint_path=...)` call, don't route it through the exporter/loader.
- `coverage ratio ... below the minimum` / `N tensor shape mismatches` / `critical module ... has zero matched tensors` — the trainer checkpoint's model no longer matches the current `build_sam3_image_model` architecture (e.g. after a SAM3 code upgrade); investigate before forcing anything through.
- `exported weights are BIT-IDENTICAL to the base checkpoint` — the "trainer" checkpoint you're exporting was never actually trained (e.g. 0 optimizer steps ran); this is a real problem with the run, not a bug in the exporter.
