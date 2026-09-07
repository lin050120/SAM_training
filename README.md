# SAM3 Fine-tuning Workflow

> Languages: **English** | [中文](README_CN.md) | [日本語](README_JA.md)

Single-target SAM3 fine-tuning workflow (book spine by default; any new target
such as cable works the same way). The local web UI can switch between Chinese
and Japanese.

- Quick start: [`docs/QUICK_START_CN.md`](docs/QUICK_START_CN.md)（日本語: [`docs/QUICK_START_JA.md`](docs/QUICK_START_JA.md)）
- Full user guide: [`docs/USER_GUIDE_ZH_CN.md`](docs/USER_GUIDE_ZH_CN.md)（日本語: [`docs/USER_GUIDE_JA.md`](docs/USER_GUIDE_JA.md)）
- Program migration guide: [`docs/PROGRAM_MIGRATION_EN.md`](docs/PROGRAM_MIGRATION_EN.md)（中文: [`docs/PROGRAM_MIGRATION_CN.md`](docs/PROGRAM_MIGRATION_CN.md)，日本語: [`docs/PROGRAM_MIGRATION_JA.md`](docs/PROGRAM_MIGRATION_JA.md)）
- Pause/resume training: [`docs/TRAINING_PAUSE_RESUME_EN.md`](docs/TRAINING_PAUSE_RESUME_EN.md) (中文: [`docs/TRAINING_PAUSE_RESUME_CN.md`](docs/TRAINING_PAUSE_RESUME_CN.md), 日本語: [`docs/TRAINING_PAUSE_RESUME_JA.md`](docs/TRAINING_PAUSE_RESUME_JA.md))
- Online training augmentation: [`docs/ONLINE_TRAINING_AUGMENTATION_EN.md`](docs/ONLINE_TRAINING_AUGMENTATION_EN.md) (中文: [`docs/ONLINE_TRAINING_AUGMENTATION_CN.md`](docs/ONLINE_TRAINING_AUGMENTATION_CN.md), 日本語: [`docs/ONLINE_TRAINING_AUGMENTATION_JA.md`](docs/ONLINE_TRAINING_AUGMENTATION_JA.md))

Run all commands below from the `book01` project root. The machine-specific
locations of `book01` and the SAM301 source tree come from the git-ignored
`config/local_paths.json` (see "Moving to a New Machine"); on the original
machine they default to `/home/book/book01` and `/home/book/sam301`.
`<sam301_root>` below means the configured SAM301 source directory.

## Moving to a New Machine

After copying `book01` and the `sam301` source tree to a new computer, run the
migration wizard once from the new location:

```bash
conda run -n sam301 python scripts/migrate_environment.py
```

Two folder dialogs ask for the new `book01` and `sam301` directories; without a
display, pass `--book-root` / `--sam301-root` (add `--dry-run` for a read-only
preview). The wizard validates both trees, writes `config/local_paths.json`,
checks the SAM3 editable install, the trainer patch, and CUDA — asking before
repairing anything, and never touching a trainer file with an UNKNOWN hash —
then writes `config/migration_report.json`. A moved checkout without
`config/local_paths.json` refuses to start and points to this wizard. Details:
[`docs/PROGRAM_MIGRATION_EN.md`](docs/PROGRAM_MIGRATION_EN.md).

## Local Web UI

A local Gradio UI wraps the existing CLI workflow (inference, history browsing,
result viewing, CVAT export, training preflight and orchestration) without
reimplementing any of it. Details: `docs/stage_d_ui.md` and
`docs/stage_e1_training_ui.md`.

```bash
conda run -n sam301 python app.py
```

The training tab supports new runs plus durable pause/resume: preflight generates and validates a
runtime config without starting anything; the start button is gated server-side
and launches the official SAM3 trainer (via `scripts/launch_sam3_training.py`,
which hands the per-run runtime YAML to `sam3.train.train.main()`) only once
preflight passed, the checkpoint/data/runtime YAML all exist, no other training
task is running, CUDA is available, and the user has explicitly confirmed.
Editing any preflight input immediately invalidates the stored preflight result.
After at least one epoch checkpoint exists, pause stops the process and releases
GPU memory; Stage C resumes the same run from its latest complete `checkpoint.pt`.
The same page can add train-only online augmentation with Off, Light, or bounded
Custom settings. Image and mask geometry stays synchronized, while validation,
test, and inference remain unchanged.

Launch from a normal terminal, not a restricted sandbox, so the UI process's
CUDA detection reflects the real GPU visibility. Listens on `127.0.0.1:7860`
only (`share=False`). The UI never silently starts training or falls back to
CPU: when `device=cuda` is requested but CUDA is unavailable, it refuses to
start inference.

## SAM301 Trainer Patch Guard

The SAM301 source tree is not a git checkout; the grad-accum loss-scaling patch
on `sam3/train/trainer.py` is pinned by full SHA256 in
`config/sam301_patch_manifest.json` and enforced fail-closed by preflight, the
launcher, and the training subprocess. Before formal training (and after any
sam301 rebuild) run:

```bash
conda run -n sam301 python scripts/manage_sam301_patch.py verify   # must exit 0 (PATCHED)
```

`status` / `apply` / `revert` are also available; see `docs/SAM301_PATCH_MANAGEMENT.md`.

## Dataset Identity (SAM3 Pre-Annotation, Not Human-Reviewed GT)

The current `data/formal_book_spine_sam3_dataset` split (184 images / 7185
annotations) is **SAM3's own machine pre-annotation output**, not
human-corrected ground truth. It is registered in
`data_manifests/dataset_identity_registry.json` with `human_reviewed=false` and
`allowed_for_formal_training=false`; preflight looks datasets up by resolved
annotation path (never by filename) and blocks `--training-mode formal` and any
`max_epochs>1` run against it. See `docs/E3_DATASET_IDENTITY_ERRATUM.md` for
how to promote a dataset once it has been independently human-reviewed.

This guard applies to **every** dataset, including new targets (e.g. cable):
dataset paths and a training prompt are enough for a smoke run
(`max_epochs<=1`), but formal or multi-epoch training requires registering the
dataset with `allowed_for_formal_training=true` after human review (the UI has
a dataset registration tab; see `docs/DATASET_REGISTRATION_UI_CN.md`).
Unregistered datasets are treated as not reviewed (fail-safe default).

When validation must isolate whole shooting scenes rather than sample at
random, the "图片分组" (image grouping) tab assigns Train/Val per image with an
annotated preview, tags each image with a scene group, and exports to a new
directory without touching the sources; see `docs/DATASET_GROUPING_CN.md`.

## Checkpoint Export (Trainer Checkpoint → Inference Checkpoint)

A trainer checkpoint (`checkpoints/checkpoint.pt`) cannot be passed directly to
the inference entry point: `sam3/model_builder.py`'s loader would silently load
zero weights from it, so `Sam3Adapter` identifies the checkpoint type by real
structure and rejects trainer checkpoints with an export hint (see
`docs/CHECKPOINT_EXPORT_AND_INFERENCE.md`). Export first:

```bash
conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \
  --input <run_dir>/checkpoints/checkpoint.pt \
  --output <run_dir>/checkpoints/inference_model.pt
```

## Checkpoint Evaluation (Validation Selection + Diagnostic Test)

After training, every unique `checkpoint_N.pt` plus the original `sam3.pt`
baseline is evaluated on validation first, the best checkpoint is selected from
validation only, and the same checkpoints are then evaluated on test as
diagnostic-only evidence — test metrics never change `best_checkpoint.json` or
`inference_best.pt`. Raw prediction masks, match records, per-instance metrics,
GT snapshots, and visualizations go to `<run_dir>/evaluation/{validation,test}/`.
UI: "Checkpoint 评估" tab. Docs: `docs/CHECKPOINT_EVALUATION_CN.md`.

```bash
conda run -n sam301 python scripts/evaluate_sam3_checkpoints.py \
  --run-dir runs/training/<run_id> --split all --export-best
```

## Authoritative SAM3 Training Config

The authoritative fine-tuning config is:

`<sam301_root>/sam3/train/configs/book_spine/book_spine_finetune.yaml`

Despite the book_spine name, it is the fixed base template for **any**
single-target category: preflight never edits it and instead renders a per-run
`runtime_config.yaml` that overrides the prompt, dataset paths, checkpoint, and
output/log directories (`dumps/<task_slug>`, `logs/<task_slug>`). To train a
different target (e.g. cable), point train/val at that dataset and set the
training prompt — no YAML editing needed. When `--training-prompt` is omitted,
the prompt falls back to the COCO's first category name.

Preflight inspects the config and generates the training command without
starting training:

```bash
conda run -n sam301 python scripts/training_preflight.py
# optional, sets the SAM3 text prompt without renaming COCO categories:
#   --training-prompt "book spine"
```

It creates `runs/training/<run_id>/config/runtime_config.yaml`, writes
`dataset_info.json` and `command.txt`, and reports batch size, gradient
accumulation, effective batch size, checkpoint and data paths, output
directory, and the resolved training prompt with its source. The generated
training command is:

```bash
conda run -n sam301 python scripts/launch_sam3_training.py \
  -c runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

(train.py's own `-c` only accepts a Hydra config name inside `pkg://sam3.train`,
so the wrapper initializes Hydra from the run's config directory and then calls
the official `sam3.train.train.main()`.)

Default runtime paths:

- checkpoint: `<sam301_root>/sam3.pt`
- train data: `data/book_spine_sam3_dataset/train`
- val data: `data/book_spine_sam3_dataset/val`
- training outputs: `runs/training/<run_id>`

## Unified Inference

Legacy NPZ mode:

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --legacy-raw-run data/dataset_raw/20260622_231208_book_spine \
  --limit 2
```

Real SAM3 mode, one image:

```bash
conda run -n sam301 python scripts/run_unified_inference.py \
  --input-dir data/book_spine_sam3_dataset/test/images \
  --limit 1 \
  --prompt "book spine" \
  --device cuda
```

Real SAM3 mode requests CUDA by default and fails fast when it is unavailable;
CPU inference requires an explicit `--device cpu`. The checkpoint defaults to
`<sam301_root>/sam3.pt` (override with `--checkpoint`).

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

Re-export and validate the CVAT package without re-running SAM3:

```bash
conda run -n sam301 python scripts/export_cvat_package.py \
  --run-dir runs/inference/<run_id> \
  --segmentation-format both \
  --polygon-fidelity
```

Segmentation modes:

- `polygon`: default-compatible CVAT package. It is lossy relative to the final NMS mask.
- `rle`: exact mask export from final NMS bool masks. The local validator checks exact decoded-mask equality.
- `both`: writes separate `polygon/` and `rle/` COCO files so the formats do not overwrite each other.

Only two things are automatically verified: pycocotools can decode the
segmentation, and the project validator passes. Actual CVAT import and CVAT
re-export mask fidelity must be tested manually. For a one-image smoke test,
import `cvat_export/rle/instances_default.json` with `cvat_export/images/` into
a temporary CVAT task, then export it back and compare against
`npz_nms/im_000001.npz`.
