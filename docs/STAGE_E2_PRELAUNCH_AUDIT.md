# Stage E2 Prelaunch Audit

Generated: 2026-07-02 19:17 JST
Updated: 2026-07-02 19:46 JST for `sam301` environment migration

This report was generated before starting any real SAM3 training. No trainer process was launched, no GPU training was run, and `/home/book/sam301` was not modified.

## SAM301 Migration Update

- Old preflight run: `2026-07-02_19-17-28`
- Old run status: `superseded_due_to_conda_environment_migration`
- Old run was never launched as a trainer process.
- Old run is not a failed training run.
- Old command used `conda run -n sam3 ...` and must not be used for real training.
- New execution environment: `sam301`
- Expected SAM3 root: `/home/book/sam301`
- Resolved `sam3` import path in `sam301`: `/home/book/sam301/sam3/__init__.py`
- Import guard result for new preflight: passed
- New preflight run: `2026-07-02_19-46-53`
- New runtime YAML: `/home/book/book01/runs/training/2026-07-02_19-46-53/config/runtime_config.yaml`
- New command file: `/home/book/book01/runs/training/2026-07-02_19-46-53/command.txt`
- New command:

```bash
conda run -n sam301 python /home/book/sam301/sam3/train/train.py -c /home/book/book01/runs/training/2026-07-02_19-46-53/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
```

- New preflight generated `dataset_info.json`, `training_config_summary.json`, `command.txt`, and `config/runtime_config.yaml`.
- New run has no `training_summary.json` and no checkpoint files.
- New run has not started real SAM3 training.
- Environment dependency verification:
  - Python: `/home/book/anaconda3/envs/sam301/bin/python`
  - PyTorch: `2.10.0+cu128`
  - CUDA build: `12.8`
  - Gradio: `5.50.0`
  - huggingface_hub: `1.19.0`
  - timm: `1.0.27`
  - `pip check`: reports `decord 0.6.0 is not supported on this platform`
- Codex sandbox GPU visibility:
  - CUDA available: `False`
  - GPU: `NONE`
  - `nvidia-smi`: failed to communicate with NVIDIA driver
  - GPU must still be confirmed from a normal terminal.
- Tests after migration:
  - Targeted: `142 passed, 3 warnings, 15 subtests passed`
  - Full: `144 passed, 5 warnings, 15 subtests passed`
  - Gradio smoke: passed
- No residual process matched `[s]am3/train/train.py|[r]untime_config.yaml`.

## Git State

- Branch: `codex-stage-e2`
- HEAD: `5860bbb Archive Claude review of Codex E1 P2 fixes`
- E1 review state: P1-1 closed, P1-2 closed, P2-2 closed, P2-3 closed, R-1 closed.
- Deferred P3: R-2 remains deferred and does not block E2.
- New P3 from Claude E1 P2 review: R-3, lock-held generator yield on BLOCKED/ERROR launch branches, non-blocking.
- Claude E1 P2 review archived in commit: `5860bbb`

## Environment

- Conda environment: `sam3`
- Python: `3.12.13`
- PyTorch: `2.10.0+cu128`
- torch CUDA build: `12.8`
- `torch.cuda.is_available()`: `False` in the Codex session
- `torch.cuda.device_count()`: `0` in the Codex session
- GPU: `NONE` in the Codex session
- `nvidia-smi`: failed to communicate with the NVIDIA driver in the Codex session
- Free GPU memory: unknown in the Codex session
- Gradio: `5.50.0`
- `sam3` import path in the Codex session: `/home/book/sam3/sam3/__init__.py`
- Expected SAM3 source for this project: `/home/book/sam301`

Warnings:

- Codex sandbox GPU visibility is not sufficient to conclude the machine has no GPU. The user must confirm GPU availability from a normal terminal before approving training.
- The conda environment has editable package `sam3` installed from `/home/book/sam3`, while the requested training script path is `/home/book/sam301/sam3/train/train.py`. Python imports currently resolve `sam3` to `/home/book/sam3/sam3`, not `/home/book/sam301/sam3`. This must be confirmed or fixed before launching real training.

## Checkpoint

- Initial checkpoint: `/home/book/sam301/sam3.pt`
- Exists: yes
- Size: `3450062241` bytes
- Modified: `2026-06-15 19:22:15.022366000 +0900`
- Permissions: `-rw-rw-r--`
- Full SHA256: not computed, per prelaunch scope.

## Dataset

- Train images: `/home/book/book01/data/book_spine_sam3_dataset/train/images`
- Train COCO: `/home/book/book01/data/book_spine_sam3_dataset/train/annotations.json`
- Train image count: `8`
- Train annotation count: `186`
- Val images: `/home/book/book01/data/book_spine_sam3_dataset/val/images`
- Val COCO: `/home/book/book01/data/book_spine_sam3_dataset/val/annotations.json`
- Val image count: `2`
- Val annotation count: `49`
- Missing images: `0`
- COCO category id: `1`
- COCO category: `book spine`
- Requested prompt: `book spine`
- Resolved prompt: `book spine`
- Prompt source: `manual_override`

Input file stats:

- Base YAML: `/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`, size `16570`, modified `2026-07-02 11:52:19.735847166 +0900`
- Train COCO: size `46639`, modified `2026-06-26 16:59:37.793044000 +0900`
- Val COCO: size `12297`, modified `2026-06-26 16:59:37.796044000 +0900`

## Runtime Configuration

- Run id: `2026-07-02_19-17-28`
- Output directory: `/home/book/book01/runs/training/2026-07-02_19-17-28`
- Runtime YAML: `/home/book/book01/runs/training/2026-07-02_19-17-28/config/runtime_config.yaml`
- Command file: `/home/book/book01/runs/training/2026-07-02_19-17-28/command.txt`
- Output root: `/home/book/book01/runs/training`
- Output root canonical validation: passed
- Existing files before training:
  - `config/runtime_config.yaml`
  - `dataset_info.json`
  - `command.txt`
- Empty directories before training:
  - `logs/`
  - `checkpoints/`
- No `training_summary.json` exists.
- No checkpoint exists in the new run directory.

Training parameters:

- `trainer.max_epochs`: `1`
- `scratch.train_batch_size`: `1`
- `scratch.gradient_accumulation_steps`: `4`
- `trainer.gradient_accumulation_steps`: `${scratch.gradient_accumulation_steps}` resolves to `4`
- `num_gpus`: `1` via CLI `--num-gpus 1`
- Effective batch size: `4`
- `scratch.num_train_workers`: `10`
- `scratch.lr_transformer`: `${times:8e-4,${scratch.lr_scale}}`
- Learning rate numeric value: not resolved by project preflight because `times` is a SAM3 custom OmegaConf resolver; expression is preserved for real training.
- `scratch.lr_vision_backbone`: `0.0`
- `scratch.lr_language_backbone`: `0.0`

Runtime paths:

- `paths.dataset_root`: `/home/book/book01/data/book_spine_sam3_dataset`
- `paths.experiment_log_dir`: `/home/book/book01/runs/training/2026-07-02_19-17-28`
- `trainer.model.checkpoint_path`: `/home/book/sam301/sam3.pt`
- `trainer.checkpoint.save_dir`: `/home/book/book01/runs/training/2026-07-02_19-17-28/checkpoints`
- `trainer.logging.log_dir`: `/home/book/book01/runs/training/2026-07-02_19-17-28/logs/book_spine`
- TensorBoard directory: `/home/book/book01/runs/training/2026-07-02_19-17-28/tensorboard`
- Validation dump directory: `/home/book/book01/runs/training/2026-07-02_19-17-28/dumps/book_spine`

Final generated command:

```bash
conda run -n sam3 python /home/book/sam301/sam3/train/train.py -c /home/book/book01/runs/training/2026-07-02_19-17-28/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
```

The command was generated by existing preflight code and saved in `command.txt`. It was not executed.

## Resume Audit

- Trainer resume mechanism: `Trainer.load_checkpoint()` calls `get_resume_checkpoint(self.checkpoint_conf.save_dir)`.
- Resume checkpoint lookup path for this run: `/home/book/book01/runs/training/2026-07-02_19-17-28/checkpoints/checkpoint.pt`
- Current resume checkpoint exists: no
- `trainer.checkpoint.resume_from`: absent/null
- `trainer.checkpoint.model_weight_initializer`: absent/null
- Auto resume from old experiment directory: no evidence found; resume is scoped to the current runtime `trainer.checkpoint.save_dir`.
- Old `/home/book/book/experiments` path in base YAML comments is not present in runtime output paths.
- Initialization checkpoint and resume checkpoint are distinct:
  - Initialization checkpoint: `trainer.model.checkpoint_path=/home/book/sam301/sam3.pt`
  - Resume checkpoint: `trainer.checkpoint.save_dir/checkpoint.pt`
- With an empty new checkpoint directory and `max_epochs=1`, training should start from epoch 0 and train until epoch 1 rather than continuing an old run.

Resume risk conclusion:

- Runtime resume risk is low for this generated run because the run directory is unique and `checkpoints/` is empty.
- Real launch should still be blocked until the SAM3 import path mismatch is resolved or explicitly accepted.

## Checkpoint Save Audit

- `trainer.skip_saving_ckpts`: `false`
- `trainer.checkpoint.save_freq`: `5`
- `trainer.checkpoint.save_list`: absent/default empty
- Save call location: `Trainer.run_train()` calls `self.save_checkpoint(self.epoch + 1)` after each training epoch and before validation.
- Default saved checkpoint name: `checkpoint.pt`
- Additional numbered checkpoint: only when `epoch % save_freq == 0` or epoch is in `save_list`.
- For `max_epochs=1`, expected checkpoint: `checkpoint.pt`
- For `max_epochs=1` and `save_freq=5`, not expected: `checkpoint_1.pt`
- Checkpoint includes model, optimizer, loss state, steps, timing, best meters, and AMP scaler when AMP is enabled.
- Checkpoint output directory: `/home/book/book01/runs/training/2026-07-02_19-17-28/checkpoints`
- Save happens after the train epoch and before final validation.

## Coverage / Overwrite Audit

- Base checkpoint `/home/book/sam301/sam3.pt` should only be read as initialization checkpoint.
- Base YAML should only be read; runtime YAML was written under the new run directory.
- Train/val COCO files should only be read.
- Existing training runs before preflight:
  - `2026-07-02_12-01-48`
  - `2026-07-02_12-03-34`
  - `2026-07-02_12-18-30`
  - `2026-07-02_12-23-41`
  - `2026-07-02_15-57-38`
- New run directory: `2026-07-02_19-17-28`
- The new run directory is unique and contains no prior training output.
- Runtime YAML contains no `/home/book/book/experiments` path and no old run checkpoint path.
- Runtime YAML writes outputs only under `/home/book/book01/runs/training/2026-07-02_19-17-28`.
- No old trainer process was found by `pgrep -af "[s]am3/train/train.py|[r]untime_config.yaml"`.

Overwrite risk conclusion:

- Output/checkpoint overwrite risk is low for this generated run.
- Do not start training until GPU visibility and SAM3 import path are confirmed.

## Warnings

1. Codex sandbox does not see CUDA/GPU. The user must run GPU checks in a normal terminal.
2. `sam3` imports resolve to `/home/book/sam3`, not `/home/book/sam301`, in the current conda environment.
3. Learning rate numeric value was not resolved during project preflight; raw expression is `${times:8e-4,${scratch.lr_scale}}`.
4. COCO category uses `book spine` with a space; this is intentionally decoupled from the prompt/category alias logic and was preserved.

## Errors / Blocking Items Before Real Training

1. GPU availability is unconfirmed in the Codex session.
2. SAM3 import path does not currently point at `/home/book/sam301`; the launch environment must be confirmed or adjusted before executing the command.

Suggested ordinary-terminal checks before approval:

```bash
cd /home/book/book01
nvidia-smi
conda run -n sam3 python -c "import torch, sam3; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print(sam3.__file__)"
PYTHONPATH=/home/book/sam301 conda run -n sam3 python -c "import sam3; print(sam3.__file__)"
```

## Recommendation

- Suggested launch decision after migration: `Yes after normal-terminal GPU confirmation`
- Required before real training:
  - Confirm GPU/CUDA from a normal terminal using `sam301`.
  - Confirm `sam3` resolves to `/home/book/sam301/sam3/__init__.py` with `PYTHONPATH` unset.
  - Reconfirm that `/home/book/sam301/sam3.pt` mtime and size are unchanged immediately before launch.
  - Reconfirm that `/home/book/book01/runs/training/2026-07-02_19-46-53/checkpoints` is still empty.

Do not execute `command.txt` until the user explicitly approves E2 real max_epochs=1 training.
