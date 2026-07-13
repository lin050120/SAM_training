# SAM3 Fine-tuning Workflow

> Languages: **English** | [中文](README_CN.md) | [日本語](README_JA.md)

单目标 SAM3 微调工作流（默认书脊，也支持 cable 等任意新目标）。The local web UI has a
中文/日本語 language switch at the top.

- 简明使用说明与注意事项：[`docs/QUICK_START_CN.md`](docs/QUICK_START_CN.md)（日本語: [`docs/QUICK_START_JA.md`](docs/QUICK_START_JA.md)）
- 中文完整使用说明：[`docs/USER_GUIDE_ZH_CN.md`](docs/USER_GUIDE_ZH_CN.md)（日本語: [`docs/USER_GUIDE_JA.md`](docs/USER_GUIDE_JA.md)）
- Program migration guide: [`docs/PROGRAM_MIGRATION_EN.md`](docs/PROGRAM_MIGRATION_EN.md)（中文: [`docs/PROGRAM_MIGRATION_CN.md`](docs/PROGRAM_MIGRATION_CN.md)，日本語: [`docs/PROGRAM_MIGRATION_JA.md`](docs/PROGRAM_MIGRATION_JA.md)）

## Local Web UI (Stage D1 / D1.1 / E1)

A local Gradio UI wraps the existing CLI workflow (inference, history browsing, result
viewing, CVAT export, training preflight and orchestration) without reimplementing any
of it. See `docs/stage_d_ui.md` (D1/D1.1) and `docs/stage_e1_training_ui.md` (E1
training orchestration and monitoring) for full details.

The training tab is a two-stage flow: preflight generates and validates a runtime
config without starting anything; a start button (gated server-side, not just a
disabled widget) launches the official SAM3 trainer (via `scripts/launch_sam3_training.py`,
which hands the per-run runtime YAML to `sam3.train.train.main()`) only once
preflight passed, the checkpoint/data/runtime YAML all exist, no other training task
is running, CUDA is available, and the user has explicitly confirmed. Editing any
preflight input invalidates the stored preflight result immediately, so a stale
runtime YAML can never be used to start training.

```bash
conda run -n sam301 python /home/book/book01/app.py
```

Launch this from a normal terminal, not a restricted coding-agent sandbox, so the
UI process's CUDA detection reflects the terminal's real GPU visibility. Listens on
`127.0.0.1:7860` only (`share=False`). The UI never starts real SAM3 training and,
when `device=cuda` is requested but CUDA is unavailable in the UI process, it refuses
to start inference instead of silently falling back to CPU.

## SAM301 Trainer Patch Guard

`/home/book/sam301` is not a git tree; the grad-accum loss-scaling patch on
`sam3/train/trainer.py` is pinned by full SHA256 in `config/sam301_patch_manifest.json`
and enforced fail-closed by preflight, the launcher, and the training subprocess.
Before formal training (and after any sam301 rebuild) run:

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py verify   # must exit 0 (PATCHED)
```

`status` / `apply` / `revert` are also available; see `docs/SAM301_PATCH_MANAGEMENT.md`.

## Dataset Identity (SAM3 Pre-Annotation, Not Human-Reviewed GT)

The current `data/formal_book_spine_sam3_dataset` split (184 images / 7185 annotations)
is **SAM3's own machine pre-annotation output**, not human-corrected ground truth —
every source COCO's `info.description` says so, and 100% of its polygons have <=8
vertices, consistent with an unedited export. It is registered in
`data_manifests/dataset_identity_registry.json` with `human_reviewed=false` and
`allowed_for_formal_training=false`. Preflight looks this up by resolved annotation
path (never by filename) and blocks `--training-mode formal` and any `max_epochs>1`
smoke run against it. See `docs/E3_DATASET_IDENTITY_ERRATUM.md` for the full
remediation and how to promote a dataset to formal status once it has been
independently human-reviewed.

This guard applies to **every** dataset, including new non-book targets (e.g. a
cable dataset): providing dataset paths and a training prompt is enough for a
smoke run (`max_epochs<=1`), but formal or multi-epoch training additionally
requires registering the dataset in `data_manifests/dataset_identity_registry.json`
with `allowed_for_formal_training=true` after human review. Unregistered datasets
are treated as not reviewed (fail-safe default).

## Checkpoint Export (Trainer Checkpoint -> Inference Checkpoint)

A trainer checkpoint (`checkpoints/checkpoint.pt`) cannot be passed directly to the
inference entry point — `Sam3Adapter` now identifies checkpoint type by real
structure and rejects trainer checkpoints outright with an export hint, because
`sam301/model_builder.py`'s loader silently loads zero weights from one (see
`docs/CHECKPOINT_EXPORT_AND_INFERENCE.md`). Export first:

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

## Checkpoint Evaluation (Validation Selection + Diagnostic Test)

After training, evaluate every unique `checkpoint_N.pt` plus the original
`sam3.pt` baseline on validation first, select the best checkpoint from validation
only, then evaluate the same checkpoints on test as diagnostic-only evidence. Test
metrics never change `best_checkpoint.json` or `inference_best.pt`. The evaluator
saves raw prediction masks, match records, per-instance metrics, GT snapshots, and
visualizations under `<run_dir>/evaluation/{validation,test}/`. UI: "Checkpoint
评估" tab. Docs: `docs/CHECKPOINT_EVALUATION_CN.md`.

```bash
conda run -n sam301 python scripts/evaluate_sam3_checkpoints.py \
  --run-dir /home/book/book01/runs/training/<run_id> --split all --export-best
```

## Authoritative SAM3 Training Config

The authoritative fine-tuning config for this workspace is:

`/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`

Despite the book_spine name, it is the fixed base template for **any** single-target
category: preflight never edits it, and instead renders a per-run
`runtime_config.yaml` that overrides the prompt, dataset paths, checkpoint, and
output/log directories (`dumps/<task_slug>`, `logs/<task_slug>`). To train a
different target (e.g. cable), point train/val at that dataset and set the training
prompt — no YAML editing needed. When `--training-prompt` is omitted, the prompt
falls back to the COCO's first category name.

Use the preflight command to inspect the config and generate the training command without starting training:

```bash
conda run -n sam301 python scripts/training_preflight.py
```

To manually set the SAM3 training text prompt without renaming COCO categories:

```bash
conda run -n sam301 python scripts/training_preflight.py \
  --training-prompt "book spine"
```

The generated training command is (train.py's own `-c` is a Hydra config name inside
`pkg://sam3.train` and cannot load an external YAML path, so the wrapper initializes
Hydra from the run's config directory and then calls the official `sam3.train.train.main()`):

```bash
conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py \
  -c /home/book/book01/runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

The preflight reads the authoritative base config, creates a per-run runtime config under `/home/book/book01/runs/training/<run_id>/config/runtime_config.yaml`, writes `dataset_info.json` and `command.txt`, and reports batch size, GPU count, gradient accumulation, effective batch size, checkpoint, train/val data paths, output directory, COCO category, requested training prompt, resolved training prompt, and prompt source. It does not start SAM3 training.

Default runtime paths:

- checkpoint: `/home/book/sam301/sam3.pt`
- train data: `/home/book/book01/data/book_spine_sam3_dataset/train`
- val data: `/home/book/book01/data/book_spine_sam3_dataset/val`
- training outputs: `/home/book/book01/runs/training/<run_id>`

## Unified Inference

Legacy NPZ mode:

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --legacy-raw-run /home/book/book01/data/dataset_raw/20260622_231208_book_spine \
  --limit 2
```

Real SAM3 mode, one image:

```bash
PYTHONPATH=/home/book/sam301 conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/book_spine_sam3_dataset/test/images \
  --limit 1 \
  --checkpoint /home/book/sam301/sam3.pt \
  --prompt "book spine" \
  --device cuda
```

The real SAM3 command requests CUDA by default. It fails fast when CUDA is unavailable; CPU inference is allowed only with explicit `--device cpu`.

CUDA diagnosis helper:

```bash
bash scripts/check_cuda_environment.sh
```

## CVAT Export

Each inference run writes:

```text
cvat_export/
├── images/
├── annotations/
│   └── instances_default.json
├── polygon/
│   ├── instances_default.json
│   ├── validation_report.json
│   └── polygon_fidelity_report.json
└── rle/
    ├── instances_default.json
    └── validation_report.json
```

Re-export and validate CVAT package without re-running SAM3:

```bash
conda run -n sam301 python scripts/export_cvat_package.py \
  --run-dir /home/book/book01/runs/inference/<run_id> \
  --segmentation-format both \
  --polygon-fidelity
```

Segmentation modes:

- `polygon`: default-compatible CVAT package. It is lossy relative to the final NMS mask.
- `rle`: exact mask export from final NMS bool masks. The local validator checks exact decoded-mask equality.
- `both`: writes separate `polygon/` and `rle/` COCO files so the formats do not overwrite each other.

Only two things are automatically verified: pycocotools can decode the segmentation, and the project validator passes. CVAT actual import and CVAT re-export mask fidelity must be tested manually. For a one-image smoke test, import `cvat_export/rle/instances_default.json` with `cvat_export/images/` into a temporary CVAT task, then export it back and compare against `npz_nms/im_000001.npz`.
