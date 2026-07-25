# Program Migration Guide (English)

This document explains how to deploy the current SAM3 fine-tuning program on another
computer. The GitHub `book01` repository contains the application code, configs,
manifests, and documentation only. Training data, images, weights, historical runs,
the SAM3 source tree, and the conda environment must be prepared separately.

## 1. Automatic Path Configuration

The new computer does not need to use the old directory paths. After preparing
`book01`, `sam301`, and the `sam301` conda environment, run this from the new checkout:

```bash
conda run -n sam301 python scripts/migrate_environment.py
```

Select the new `book01` and `sam301` directories in the two folder dialogs. The
wizard validates the project, SAM3 source, training YAML, BPE file, and `sam3.pt`,
then creates `config/local_paths.json`. It also checks the SAM3 import source, the
trainer patch, and CUDA. It asks before repairing the editable install or applying
the known patch, and never overwrites an UNKNOWN trainer.

The local config and `config/migration_report.json` are ignored by Git. An existing
config is backed up as `config/local_paths.<timestamp>.bak.json`.

Note: a `book01` checkout in a new location refuses to start until the wizard has
created `config/local_paths.json`; it reports a clear error pointing at
`scripts/migrate_environment.py` instead of silently using the old machine's paths.

The same operation is available from the command line:

```bash
conda run -n sam301 python scripts/migrate_environment.py \
  --book-root "/new/path/book01" \
  --sam301-root "/new/path/sam301"
```

Run a read-only preview before applying the migration:

```bash
conda run -n sam301 python scripts/migrate_environment.py \
  --book-root "/new/path/book01" \
  --sam301-root "/new/path/sam301" \
  --dry-run
```

Non-interactive mode requires both paths. Add `--repair-install --apply-patch`
explicitly when those repairs are allowed.

Without a local config, the original paths remain as compatibility defaults:

```text
/home/book/book01
/home/book/sam301
```

Do not edit `core/config.py` manually. The production patch manifest now stores the
trainer path relative to the configured SAM301 root, so it does not need a per-PC edit.

## 2. Clone the Project from GitHub

Use the current branch. It carries generalized single-target training plus the
two SAM3 numerical fixes from 2026-07-24: the bf16 GradScaler underflow and the
Triton focal-loss `gamma=0` backward NaN. Earlier stage branches lack those patch
manifests and crash with all-NaN weights around epoch 6-7:

```bash
git clone -b codex-stage-e8-nan-fix-and-test-loss-20260725 \
  git@github.com:lin050120/SAM_training.git \
  /home/book/book01
```

If the repository already exists:

```bash
cd /home/book/book01
git checkout codex-stage-e8-nan-fix-and-test-loss-20260725
git pull
```

## 3. Prepare the Conda Environment

All commands assume the conda environment:

```text
sam301
```

Export it on the old machine:

```bash
conda env export -n sam301 > sam301_environment.yml
```

Copy the file to the new machine and create the environment:

```bash
conda env create -f sam301_environment.yml
```

Check that Python works:

```bash
conda run -n sam301 python --version
```

## 4. Prepare the SAM3 Source Tree and Weights

The GitHub `book01` repository does not include the SAM3 source tree. Place SAM3 at:

```text
/home/book/sam301
```

At minimum, it must contain:

```text
/home/book/sam301/sam3/
/home/book/sam301/sam3.pt
/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml
```

You can copy `/home/book/sam301` from the old machine, or reinstall SAM3 and then
restore `sam3.pt` and the training YAML required by this project.

## 5. Install SAM3 as an Editable Package

Make the `sam301` environment import SAM3 from `/home/book/sam301`:

The migration wizard performs this check and asks before repairing an incorrect
editable install. Use the commands below only when handling it manually.

```bash
cd /home/book/sam301
conda run -n sam301 pip install -e ".[train,dev]"
```

Verify the import path:

```bash
env -u PYTHONPATH conda run -n sam301 python -c \
  "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())"
```

Expected output:

```text
/home/book/sam301/sam3/__init__.py
```

It must not point to an old directory such as `/home/book/sam3`.

## 6. Apply and Verify the Trainer Patch

This project requires the gradient-accumulation loss-scaling patch on:

```text
/home/book/sam301/sam3/train/trainer.py
```

Training preflight, the launcher, and the training subprocess all enforce this
fail-closed.

The migration wizard applies the patch only when the state is UNPATCHED and the user
confirms. It never overwrites an UNKNOWN trainer.

Run from `/home/book/book01`:

```bash
cd /home/book/book01
conda run -n sam301 python scripts/manage_sam301_patch.py status
conda run -n sam301 python scripts/manage_sam301_patch.py apply
conda run -n sam301 python scripts/manage_sam301_patch.py verify
```

`verify` must exit with code 0 and report PATCHED.  
If `status` is UNKNOWN, do not force overwrite; inspect the SAM3 file first.

## 7. Copy Data, Models, and Historical Runs

Large files and runtime artifacts are excluded by `.gitignore`, for example:

```text
data/
runs/
*.pt
*.jpg
*.png
*.npz
```

Therefore, a GitHub checkout will not contain:

- training datasets
- source images
- historical inference/training runs
- checkpoints / inference models
- `.npz` intermediate files

Copy what you need manually:

```text
/home/book/book01/data/
/home/book/book01/runs/          # only if historical runs are needed
/home/book/sam301/sam3.pt
```

`data_manifests/dataset_identity_registry.json` is tracked by GitHub, but the
registered `annotations_path` files must actually exist on the new machine.

## 8. Dataset Identity Registration

Formal or multi-epoch training requires the dataset to be registered with:

```json
"allowed_for_formal_training": true
```

For a new target such as cable, register it in the UI tab `数据集登记`
(`Dataset Registration`). The dataset should look like:

```text
data/cable_sam3_dataset/
  train/images/
  train/annotations.json
  val/images/
  val/annotations.json
  test/images/              # optional
  test/annotations.json     # optional
```

After registration, go back to the training preflight tab, choose the same train/val
COCO files, and set `training mode` to `formal`.

## 9. Check GPU / CUDA

The new machine needs an NVIDIA GPU, a working driver, and a PyTorch-compatible CUDA
setup.

```bash
conda run -n sam301 python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If this prints `False`, the UI will refuse CUDA inference and real training.

## 10. Launch the UI

```bash
cd /home/book/book01
conda run -n sam301 python app.py
```

Open:

```text
http://127.0.0.1:7860
```

The UI listens on `127.0.0.1` only and is not publicly exposed by default.

## 11. Minimal Acceptance Checks

After migration, run:

```bash
cd /home/book/book01
conda run -n sam301 python scripts/manage_sam301_patch.py verify
env -u PYTHONPATH conda run -n sam301 python -c \
  "from pathlib import Path; import sam3; print(Path(sam3.__file__).resolve())"
conda run -n sam301 pytest -q tests/test_sam301_patch.py tests/test_ui.py::UiImportsTest
```

For a full regression:

```bash
conda run -n sam301 pytest -q tests
```

## 12. Migration Checklist

Downloading the GitHub code is not enough. The new machine also needs:

- the `sam301` conda environment
- the SAM3 source tree at `/home/book/sam301`
- `/home/book/sam301/sam3.pt`
- editable SAM3 installation
- trainer patch verification passing
- training/inference data and model files
- working GPU/CUDA
- local path configuration generated by `scripts/migrate_environment.py`
