# Program Migration Guide (English)

This document explains how to deploy the current SAM3 fine-tuning program on another
computer. The GitHub `book01` repository contains the application code, configs,
manifests, and documentation only. Training data, images, weights, historical runs,
the SAM3 source tree, and the conda environment must be prepared separately.

## 1. Recommended Directory Layout

The safest migration is to keep the same paths on the new machine:

```text
/home/book/book01
/home/book/sam301
```

The key paths are defined in:

```text
/home/book/book01/core/config.py
```

Current values:

```python
BOOK_ROOT = Path("/home/book/book01")
SAM301_ROOT = Path("/home/book/sam301")
```

If the paths are different on the new machine, update at least `core/config.py`.
Also check `config/sam301_patch_manifest.json`, because it records
`/home/book/sam301`; if that path changes, the patch guard may fail unless it is
handled consistently.

## 2. Clone the Project from GitHub

Use the current branch for generalized single-target training:

```bash
git clone -b codex-stage-e5-general-target-training \
  git@github.com:lin050120/SAM_training.git \
  /home/book/book01
```

If the repository already exists:

```bash
cd /home/book/book01
git checkout codex-stage-e5-general-target-training
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
- if paths differ, updated `core/config.py` and patch-manifest-related paths

