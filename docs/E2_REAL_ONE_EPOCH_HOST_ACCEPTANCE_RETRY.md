# E2 Host One-Epoch Training Acceptance Retry

Generated: 2026-07-03 00:13:30 +0900

## Result

Final status: **FAIL**

The corrected import context allowed the temporary driver to import and call the project formal launcher, `ui.training_preflight_page.start_training(...)`. The launcher consumed the launch token and started exactly one managed training process through `ProcessManager`, but the trainer exited with code `1` before creating the SAM3 trainer or completing any epoch.

No retry was attempted.

## Baseline

- Project: `/home/book/book01`
- Branch: `codex-stage-e2`
- Start commit: `7235ac3207bf5e9a82bc4a52119a9232e15cbb7e`
- Approved run: `/home/book/book01/runs/training/2026-07-02_19-46-53`
- Runtime YAML: `/home/book/book01/runs/training/2026-07-02_19-46-53/config/runtime_config.yaml`
- Command file: `/home/book/book01/runs/training/2026-07-02_19-46-53/command.txt`

## Import Context Fix

The project module exists at:

- `/home/book/book01/ui/training_preflight_page.py`

The temporary retry driver used the only allowed import-context correction:

```python
sys.path.insert(0, "/home/book/book01")
from ui.training_preflight_page import start_training
```

No formal project code was modified.

## Startup Checks

- Git branch: `codex-stage-e2`
- Git HEAD before launch: `7235ac3207bf5e9a82bc4a52119a9232e15cbb7e`
- Expected untracked file present and preserved: `docs/CLAUDE_REVIEW_OF_SAM301_ENV_MIGRATION.md`
- Residual training processes before launch: none
- Runtime directory had no `training_summary.json`
- `checkpoints/` was empty

Host GPU check before launch:

- NVIDIA driver: `595.71.05`
- NVIDIA-SMI CUDA version: `13.2`
- GPU: `NVIDIA GeForce RTX 5090`
- Memory: `866 MiB used / 32607 MiB total`
- GPU utilization: `0%`
- Unknown heavy compute process: none observed

`sam301` check before launch:

- Python: `/home/book/anaconda3/envs/sam301/bin/python`
- PyTorch: `2.10.0+cu128`
- PyTorch CUDA build: `12.8`
- `torch.cuda.is_available()`: `True`
- CUDA device count: `1`
- CUDA device name: `NVIDIA GeForce RTX 5090`
- `sam3` import path: `/home/book/sam301/sam3/__init__.py`

## Runtime Parameters

Parsed from the runtime YAML:

- `trainer.max_epochs`: `1`
- `scratch.train_batch_size`: `1`
- `scratch.gradient_accumulation_steps`: `4`
- `trainer.gradient_accumulation_steps`: `4`
- Effective batch size: `4`
- CLI `num_gpus`: `1`
- Train prompt: `book spine`
- Val prompt: `book spine`
- `resume_from`: absent / `None`
- Initial checkpoint: `/home/book/sam301/sam3.pt`
- Checkpoint output directory: `/home/book/book01/runs/training/2026-07-02_19-46-53/checkpoints`

Dataset metadata:

- Train images: `8`
- Train annotations: `186`
- Val images: `2`
- Val annotations: `49`
- Category: `book spine`

## Base Checkpoint Integrity

Before launch:

- Path: `/home/book/sam301/sam3.pt`
- Size: `3450062241`
- Mtime: `2026-06-15 19:22:15.022366000 +0900`
- SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

After failed training attempt:

- Path: `/home/book/sam301/sam3.pt`
- Size: `3450062241`
- Mtime: `2026-06-15 19:22:15.022366000 +0900`
- SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

The base checkpoint was unchanged.

## Formal Launcher Execution

- Formal launcher: `ui.training_preflight_page.start_training(preflight_state, confirmed=True)`
- Temporary driver: `/tmp/book01_e2_formal_launcher_retry.py`
- Import fix: `sys.path.insert(0, "/home/book/book01")`
- Launch token consumed: yes
- State `ok` after launch: `False`
- ProcessManager PID: `1432430`
- ProcessManager PGID at start: `1432430`
- Automatic retry: no

Command started by `ProcessManager`:

```bash
conda run -n sam301 python /home/book/sam301/sam3/train/train.py -c /home/book/book01/runs/training/2026-07-02_19-46-53/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
```

Timeline:

- Driver start: `2026-07-03 00:12:31 +0900`
- Process start: `2026-07-03 00:12:35 +0900`
- Running observations: `00:12:35`, `00:12:36`, `00:12:37`
- Process finish: `2026-07-03 00:12:37 +0900`
- Final observed status: `status=failed pid=1432430`
- Duration from summary: `2.8684332370758057` seconds
- Exit code: `1`
- Completed epochs: `0`

## Failure

First substantive error:

```text
hydra.errors.MissingConfigException: Cannot find primary config 'home/book/book01/runs/training/2026-07-02_19-46-53/config/runtime_config.yaml'. Check that it's in your config search path.

Config search path:
    provider=hydra, path=pkg://hydra.conf
    provider=main, path=pkg://sam3.train
    provider=schema, path=structured://
```

Traceback location:

- `training_summary.json` field: `stdout_stderr_tail`
- Launcher observation file: `/home/book/book01/runs/training/2026-07-02_19-46-53/logs/launcher_observation_retry.json`

The trainer reached `/home/book/sam301/sam3/train/train.py`, but failed while Hydra composed the config:

- `/home/book/sam301/sam3/train/train.py:141`
- `cfg = compose(config_name=args.config)`

This happened before a trainer was created, before dataset loading, before checkpoint loading, before epoch execution, and before checkpoint saving.

## Generated Run Artifacts

- `/home/book/book01/runs/training/2026-07-02_19-46-53/training_summary.json`
- `/home/book/book01/runs/training/2026-07-02_19-46-53/logs/launcher_observation_retry.json`

`training_summary.json` summary:

- Status: `failed`
- Exit code: `1`
- Conda environment: `sam301`
- Import guard: passed
- Resolved SAM3 import path: `/home/book/sam301/sam3/__init__.py`
- Discovered checkpoints: `[]`

No checkpoint files were generated.

## Cleanup And Safety

- Residual `train.py` / `torchrun` / current-run process: none found
- GPU after failure: `866 MiB used / 32607 MiB total`, no heavy compute process observed
- Output remained under the current run directory
- No checkpoint was written
- No old run was overwritten
- No dataset directory was modified
- No SAM3 source was modified
- No project code was modified
- No Conda environment was modified
- No system, CUDA, or driver setting was modified

## Conclusion

E2 retry acceptance is **FAIL**.

The formal launcher path itself was reached and the launch token was consumed. The failure is now inside the approved trainer command: Hydra cannot load the absolute runtime YAML path passed with `-c`. Because the run failed after the single approved launch, no further launch, retry, code change, runtime YAML edit, environment change, or raw `train.py` workaround was attempted.
