# Codex Strict Review of Hydra and Gradient Accumulation Fixes

Generated: 2026-07-03

## Verdict

Final verdict: **APPROVE_WITH_FOLLOWUPS**

Two separate conclusions:

- **Current E2 configuration**: approved as a completed one-epoch smoke/acceptance run. The Hydra wrapper and the gradient-accumulation chunking path both ran on the real GPU with the expected run artifacts.
- **General implementation**: not production-complete. The current `gradient_accumulation_steps > 1` path accumulates gradients without scaling the loss by the number of accumulation steps, so it is not mathematically equivalent to a normal larger batch at the same learning rate.

In short: **E2 current configuration approved, implementation not production-complete.**

## Git Baseline

- Project: `/home/book/book01`
- Branch: `codex-stage-e2`
- HEAD reviewed: `8e5b85f9b94753d8c21b6af60567c865bab60473`
- Relevant history:
  - `5363e75 Fix Hydra runtime config launch`
  - `5a9f585 Wire gradient accumulation via chunking collate in runtime YAML`
  - `8f569d5`, `8f8a362`, `8e5b85f` are report commits
- Status before report: only untracked handoff/review docs were present:
  - `docs/CLAUDE_REVIEW_OF_GRAD_ACCUM_FIX.md`
  - `docs/CLAUDE_REVIEW_OF_SAM301_ENV_MIGRATION.md`
  - `docs/HANDOFF_FOR_REVIEW_E2_FIXES.md`
- `git ls-files` check found no tracked `runs/`, `data/`, `experiments/`, images, `.pt`, `.pth`, `.ckpt`, or `.npz`.
- No active `train.py`, `torchrun`, Hydra, or current-run training process was found.

Reviewed diff from `05c33ce..HEAD`:

- `scripts/launch_sam3_training.py`
- `core/config.py`
- `core/training_runner.py`
- `tests/test_e1_training.py`
- `tests/test_ui.py`
- `README.md`
- `docs/stage_e1_training_ui.md`
- Three E2 report documents

No source edit was made during this review.

## Hydra Launch Review

### Official SAM3 `train.py` Semantics

`/home/book/sam301/sam3/train/train.py` calls:

- `initialize_config_module("sam3.train", version_base="1.2")` in `__main__`
- `compose(config_name=args.config)` inside `main(args)`

Therefore `-c` is a Hydra config name in `pkg://sam3.train`, not an arbitrary filesystem YAML path. The previous failure with an absolute runtime YAML path was real.

### Wrapper Behavior

`scripts/launch_sam3_training.py`:

- Resolves the runtime YAML path with `Path.expanduser().resolve()`
- Requires an existing `.yaml` or `.yml` file
- Maps it to `(config_dir, config_name)` using parent directory and stem
- Uses `initialize_config_dir(config_dir=..., version_base="1.2")`
- Calls the official `sam3.train.train.main(args)` for real training
- Registers SAM3 OmegaConf resolvers before the real handoff
- Does not change the model, trainer, dataset, optimizer, checkpoint, or logging semantics

`--validate-only` composes the config through Hydra and checks for `trainer`, `launcher`, and `submitit`. It is not a mock. It does not instantiate the trainer or import torch-heavy training objects.

### Hydra Boundary Checks

Observed by lightweight subprocess probes:

- Existing runtime YAML with spaces in the path: `--validate-only` returned `0`
- Missing config path: returned non-zero with `FileNotFoundError`
- Symlink path to a valid YAML: resolved and validated successfully
- Both real run configs validated successfully:
  - `2026-07-03_00-43-43/config/runtime_config.yaml`
  - `2026-07-03_00-56-32/config/runtime_config.yaml`

The official launcher command now uses:

```bash
conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py -c <runtime_config.yaml> --use-cluster 0 --num-gpus 1
```

The import guard and training subprocess still use `sam301`, `PYTHONPATH=/home/book/sam301[:existing]`, and expected import path `/home/book/sam301/sam3/__init__.py`.

## Gradient Accumulation Data Flow

### SAM3 Trainer Contract

`/home/book/sam301/sam3/train/trainer.py::_run_step()` requires:

- For `gradient_accumulation_steps > 1`, the batch must be a `list`
- `len(batch)` must equal `gradient_accumulation_steps`
- It loops over each micro-batch and calls backward for each
- Optimizer step, scheduler step, AMP scaler update, gradient clipping, and zero-grad happen once per outer dataloader iteration

`/home/book/sam301/sam3/train/data/collator.py::collate_fn_api_with_chunking()` is the official producer of that list. It splits one dataloader batch into `num_chunks` chunks and collates each chunk using `collate_fn_api`.

### book01 Runtime Wiring

`core/training_runner.py::write_runtime_yaml()` now does the necessary config wiring when `effective_accum > 1`:

- `scratch.collate_fn._target_ = sam3.train.data.collator.collate_fn_api_with_chunking`
- `scratch.collate_fn.num_chunks = ${scratch.gradient_accumulation_steps}`
- `trainer.data.train.batch_size = scratch.train_batch_size * scratch.gradient_accumulation_steps`

Static probes confirmed:

| micro batch | accum | train loader batch size | collate | num_chunks | effective batch |
|---:|---:|---:|---|---:|---:|
| 1 | 1 | 1 | plain `collate_fn_api` | none | 1 |
| 1 | 2 | 2 | chunking | 2 | 2 |
| 1 | 4 | 4 | chunking | 4 | 4 |
| 2 | 4 | 8 | chunking | 4 | 8 |

Validation collate remains plain `collate_fn_api`; it was not changed by the training accumulation wiring.

## Real Run Review

### accum=1 Run

Run: `/home/book/book01/runs/training/2026-07-03_00-43-43`

- Summary status: `completed`
- Exit code: `0`
- Duration: `30.443966388702393` seconds
- Command uses `scripts/launch_sam3_training.py`
- Import guard: passed
- SAM3 import path: `/home/book/sam301/sam3/__init__.py`
- Dataset: train `8 images / 186 annotations`, val `2 images / 49 annotations`
- Runtime:
  - `max_epochs=1`
  - `train_batch_size=1`
  - `gradient_accumulation_steps=1`
  - `trainer.data.train.batch_size=1`
  - plain train collate
- Log evidence:
  - `Train Epoch: [0][0/8]`
  - `Trainer/steps_train: 8`
  - `Val Epoch: [0][0/2]`
  - `Trainer/steps_val: 2`
- Checkpoint: `/home/book/book01/runs/training/2026-07-03_00-43-43/checkpoints/checkpoint.pt`
- Checkpoint size: `10081250310` bytes

Conclusion: completed one epoch over 8 train iterations.

### accum=4 Run

Run: `/home/book/book01/runs/training/2026-07-03_00-56-32`

- Summary status: `completed`
- Exit code: `0`
- Duration: `29.709418296813965` seconds
- Command uses `scripts/launch_sam3_training.py`
- Import guard: passed
- SAM3 import path: `/home/book/sam301/sam3/__init__.py`
- Dataset: train `8 images / 186 annotations`, val `2 images / 49 annotations`
- Runtime:
  - `max_epochs=1`
  - `train_batch_size=1`
  - `gradient_accumulation_steps=4`
  - `trainer.data.train.batch_size=4`
  - train collate `collate_fn_api_with_chunking`
  - `num_chunks=4`
  - val collate unchanged
- Log evidence:
  - `Train Epoch: [0][0/2]`
  - `Trainer/steps_train: 8`
  - `Val Epoch: [0][0/2]`
  - `Trainer/steps_val: 2`
  - no `Expected a list of batches` error
  - no Hydra `MissingConfigException`
- Interpretation:
  - 2 outer dataloader iterations
  - each outer batch split into 4 micro-batches
  - total 8 micro-batch `_step()` calls
  - 2 optimizer steps
- Checkpoint: `/home/book/book01/runs/training/2026-07-03_00-56-32/checkpoints/checkpoint.pt`
- Checkpoint size: `10081250310` bytes

Conclusion: chunking and accumulation control flow ran successfully for the current E2 dataset.

### Base Checkpoint And Output Safety

`/home/book/sam301/sam3.pt` after review:

- Size: `3450062241`
- Mtime: `2026-06-15 19:22:15.022366000 +0900`
- SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

No residual training process was found. The generated checkpoints are under their run directories and are not tracked by Git.

## Tests Run

Full suite:

```text
conda run -n sam301 python -m pytest tests/ -v
151 passed, 5 warnings, 15 subtests passed in 30.30s
```

The Gradio startup smoke test is included in that run and passed:

```text
tests/test_ui_smoke.py::UiStartupSmokeTest::test_app_launches_and_serves_http PASSED
```

Repeated new Hydra and gradient accumulation tests:

```text
for i in 1 2 3; do
  conda run -n sam301 python -m pytest \
    tests/test_e1_training.py::HydraLaunchRegressionTest \
    tests/test_e1_training.py::GradAccumWiringTest -v
done
```

Each repeat: `7 passed`.

Additional lightweight boundary probes:

- `--validate-only` with a config path containing spaces: passed
- `--validate-only` with missing config path: failed non-zero
- `--validate-only` through a symlink to a valid config: passed
- `accum=1`, `accum=2`, `accum=4`, `micro_batch_size=2 + accum=4`: static runtime wiring passed
- Tiny dataset length `3`, batch `4`, `drop_last=True`: dataloader length `0`
- Missing `scratch.gradient_accumulation_steps`: reproduced `TypeError`
- Missing `scratch.train_batch_size`: reproduced `TypeError`

No real training was started during this review.

## Findings

### P1

None found.

### P2-1: Gradient accumulation does not scale loss by accumulation steps

Status: confirmed.

Evidence:

- `trainer.py::_run_step()` loops over micro-batches and calls `self.scaler.scale(loss).backward()` for each.
- There is no `loss = loss / accum_steps` or equivalent before backward.
- `train_epoch()` calls `self.scaler.step(self.optim.optimizer)` once per outer dataloader iteration.

Impact:

- The current accum=4 run did perform 8 micro-batch backward passes and 2 optimizer steps.
- However, gradients are summed across 4 micro-batches rather than averaged.
- At unchanged learning rate, this is not equivalent to a true effective batch size of 4; it behaves closer to increasing gradient magnitude by about the accumulation factor, subject to the internal loss normalization.

Severity:

- **P2 for general training quality and learning-rate semantics.**
- Not a process safety issue and not a blocker to saying the E2 smoke path ran.
- It should be fixed before relying on accum>1 for formal longer training, unless the team explicitly chooses sum-style accumulation and adjusts learning rate/docs accordingly.

Recommended layer:

- Prefer fixing at the trainer accumulation layer by dividing the loss by `accum_steps` before backward.
- If SAM3 source must remain untouched, the alternative is to scale the effective learning rate or document sum-style semantics, but that is more error-prone.

### P3-1 / R-4: Missing `scratch.*` keys cause `int(None)` TypeError

Status: confirmed.

Minimal no-training reproduction:

- Base YAML missing `scratch.gradient_accumulation_steps` causes:
  - `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'`
- Base YAML missing `scratch.train_batch_size` causes the same TypeError.

Impact:

- Does not affect the current authoritative book-spine config because both keys exist.
- Affects non-default/future configs and produces an exception instead of a clean preflight error.

Severity: **P3**.

Recommended layer:

- `core.training_runner.write_runtime_yaml()` / preflight validation should detect missing required `scratch.*` keys and return explicit errors before `int(...)`.

### P3-2 / R-5: Dataset smaller than effective batch can produce zero train iterations

Status: confirmed by static code and minimal DataLoader probe.

Evidence:

- Current train loader uses `drop_last=True`.
- PyTorch DataLoader with dataset length `3`, batch size `4`, `drop_last=True` has `len(loader) == 0` and yields no batches.
- `Trainer.train_epoch()` does not reject `iters_per_epoch == 0`.
- `Trainer.run()` calls `save_checkpoint(self.epoch + 1)` after `train_epoch()`.

Impact:

- Current E2 dataset is not affected: 8 train images and effective batch 4 produce 2 outer iterations.
- Future smaller datasets or too-large effective batch sizes could complete an epoch with zero optimizer steps and still write a checkpoint/summary.

Severity: **P3** for current workflow, potentially higher if future automation treats checkpoint existence as proof of training.

Recommended layer:

- Preflight should compute or conservatively estimate `train_images >= train_batch_size * gradient_accumulation_steps * num_gpus` when `drop_last=True`, and block or warn explicitly.
- Summary should record actual optimizer-step count when available.

### Non-Blocking Observation: Historical docs still contain old `sam3` examples

Some historical reports and older docs still contain `conda run -n sam3` or `/home/book/sam3` examples. Current README, stage E1 training UI doc, executable code, and tests use `sam301`. This is not an execution bug, but old docs remain potentially confusing.

## R-4 / R-5 Independent Conclusions

R-4:

- Real: yes.
- Severity: P3.
- Blocks current E2: no.
- Blocks future formal training: only for non-default configs missing required `scratch.*` keys.
- Fix layer: preflight/runtime YAML validation in `core.training_runner`.

R-5:

- Real: yes.
- Severity: P3 for current project, with potential to become more serious if larger automation accepts zero-step checkpoints.
- Blocks current E2: no, because 8 images / effective batch 4 gives 2 outer iterations.
- Blocks future formal training: should be fixed before generalizing to arbitrary datasets/batches.
- Fix layer: preflight dataset/effective-batch validation and/or trainer zero-iteration guard.

## Current E2 Conclusion

Current E2 8-image, batch=1, accum=4 result is credible for:

- Hydra runtime YAML launch through the wrapper
- `sam301` environment and import guard
- real CUDA/GPU execution
- checkpoint creation under the run directory
- chunking collate producing an accumulation-compatible list
- 2 outer train iterations and 8 micro-batch `_step()` calls

Current E2 result is not sufficient proof of:

- mathematically correct effective-batch-4 optimization at unchanged learning rate, because the loss is not divided by accumulation steps.

## General Implementation Conclusion

The implementation is adequate to proceed past E2 as a smoke/acceptance milestone, but it is not robust enough for longer formal training without follow-up.

Before larger formal training:

1. Decide and fix gradient accumulation semantics: average the loss across accumulation steps or explicitly adjust learning rate and documentation for sum-style accumulation.
2. Add clear validation for missing required `scratch.train_batch_size` and `scratch.gradient_accumulation_steps`.
3. Block or loudly warn when `drop_last=True` and dataset size is smaller than effective batch size.
4. Prefer recording actual optimizer-step count in `training_summary.json` or parsed metrics.

Final verdict: **APPROVE_WITH_FOLLOWUPS**.
