# E2 Host One-Epoch Training Acceptance

Generated: 2026-07-02 23:51:37 +0900

## Result

Final status: **FAIL**

The host GPU and `sam301` environment passed the startup checks, but the formal launcher invocation failed before importing the project launch module. No SAM3 trainer process was created, no launch token was consumed, no training epoch ran, and no retry was attempted.

## Context

- Project: `/home/book/book01`
- Branch: `codex-stage-e2`
- Start commit: `43c56b02af32447821a0820bcfb094f66472a2b1`
- Approved run: `/home/book/book01/runs/training/2026-07-02_19-46-53`
- Runtime config: `/home/book/book01/runs/training/2026-07-02_19-46-53/config/runtime_config.yaml`
- Command file: `/home/book/book01/runs/training/2026-07-02_19-46-53/command.txt`

Prior Codex `CUDA=False` observations were caused by the Codex agent sandbox not mapping the GPU. Host diagnostics confirmed the NVIDIA driver, CUDA visibility, and `sam301` PyTorch CUDA access were healthy.

## GPU Diagnosis Summary

- GPU: `NVIDIA GeForce RTX 5090`
- Driver: `595.71.05`
- NVIDIA-SMI CUDA version: `13.2`
- PyTorch CUDA build: `12.8`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.device_count()`: `1`
- Startup GPU memory: `870 MiB used / 32607 MiB total / 31220 MiB free`
- GPU utilization at startup: `0%`
- Unknown heavy compute processes: none observed

## Environment Check

- Python: `/home/book/anaconda3/envs/sam301/bin/python`
- Python version: `3.12.13`
- PyTorch: `2.10.0+cu128`
- `sam3` import path: `/home/book/sam301/sam3/__init__.py`
- Expected SAM3 source root: `/home/book/sam301`
- Old `/home/book/sam3` source was not imported.

## Formal Launcher

The project formal launch path is:

- Module/function: `ui.training_preflight_page.start_training(preflight_state, confirmed=True)`
- Process manager: `ui.training_process_manager.training_process_manager`
- Underlying process manager: `ui.process_manager.ProcessManager`

This path consumes the in-memory launch token, acquires the server launch lock, validates the SAM3 import guard, starts the command through `ProcessManager`, captures stdout/stderr, records PID/PGID, and finalizes `training_summary.json` through the server-side lifecycle callback.

No raw `train.py` shell execution was used as a substitute for this launcher.

## Runtime Parameters

- `max_epochs`: `1`
- `train_batch_size`: `1`
- `gradient_accumulation_steps`: `4`
- Effective batch size: `4`
- `num_gpus`: `1`
- Prompt: `book spine`
- `resume_from`: absent / `None`
- `num_workers`: `10`
- Learning rate: base YAML value, unresolved in summary as `null`
- Initial checkpoint: `/home/book/sam301/sam3.pt`

## Dataset

- Train images: `8`
- Train annotations: `186`
- Val images: `2`
- Val annotations: `49`
- Category: `book spine`
- Missing images: `0`

## Base Checkpoint

Before attempted launch:

- Path: `/home/book/sam301/sam3.pt`
- Type: regular file
- Size: `3450062241`
- Mtime: `2026-06-15 19:22:15.022366000 +0900`
- SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

After failed launcher invocation:

- Path: `/home/book/sam301/sam3.pt`
- Type: regular file
- Size: `3450062241`
- Mtime: `2026-06-15 19:22:15.022366000 +0900`
- SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

The base checkpoint was unchanged.

## Intended Training Command

From `command.txt`:

```bash
conda run -n sam301 python /home/book/sam301/sam3/train/train.py -c /home/book/book01/runs/training/2026-07-02_19-46-53/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
```

This trainer command was not executed because the formal launcher driver failed before importing the project launcher.

## Timeline

- Startup checks passed: host GPU visible, `sam301` CUDA available, SAM3 import path correct, no residual training process found.
- A first stdin-based launcher attempt did not enter the project launcher and produced no trainer process or run output.
- A temporary host driver was then invoked to call the project formal launch function.
- The host driver failed immediately at import time before calling `start_training()`.
- Per E2 rules, no retry, code change, environment change, config change, or alternate raw trainer launch was attempted.

## First Substantive Error

Command attempted:

```bash
conda run -n sam301 python /tmp/book01_e2_formal_launcher.py
```

Traceback:

```text
Traceback (most recent call last):
  File "/tmp/book01_e2_formal_launcher.py", line 10, in <module>
    from ui.training_preflight_page import start_training
ModuleNotFoundError: No module named 'ui'

ERROR conda.cli.main_run:execute(127): `conda run python /tmp/book01_e2_formal_launcher.py` failed. (See above for error)
```

The failure occurred before token consumption, before import guard execution in the project launcher, before `ProcessManager.start()`, and before trainer process creation.

## Launch And Process State

- Launch token consumed: no
- Trainer PID: none
- Trainer PGID: none
- Trainer exit code: not applicable
- Completed epochs: `0`
- Automatic retry: no
- Residual training process: none found
- `training_summary.json`: not generated
- Checkpoints: none generated

Run directory after failure contained only prelaunch files:

- `command.txt`
- `config/runtime_config.yaml`
- `dataset_info.json`
- `training_config_summary.json`
- empty `logs/`
- empty `checkpoints/`

## Output Safety

- No checkpoint was written.
- No training stdout/stderr file was written by the trainer.
- No output was observed outside `/home/book/book01/runs/training/2026-07-02_19-46-53`.
- No old run was overwritten.
- No dataset directory was modified.
- No SAM3 source file was modified.
- No Conda environment, driver, or system package was modified.

## Final Assessment

E2 host one-epoch acceptance is **FAIL**, not PASS, because the formal launcher was not successfully entered and the approved `max_epochs=1` training did not start.

This failure is not a CUDA, GPU, SAM3 source binding, dataset, checkpoint, or runtime YAML failure. It is an operator-side launcher invocation failure: the temporary driver was executed from `/tmp` without the project package import path, so Python could not import `ui.training_preflight_page`.

## Backlog Observations

The following non-blocking observations from the prior Claude review remain backlog items and were not modified in this phase:

- Guard code contains unused `relative_to` related dead code.
- Launching the child process while holding the lock keeps the lock for longer than ideal.
- The old run lacks a machine-readable superseded marker.
- Early analysis documents still contain `-n sam3` examples.

## Recommendation

Do not mark E2 complete. A user decision is required before any further attempt, because the approved one-attempt window has been used and the run was not launched successfully through the project formal launcher.
