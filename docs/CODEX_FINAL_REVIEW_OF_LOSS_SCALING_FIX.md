# Codex Final Review of Loss Scaling and Preflight Fix

Generated: 2026-07-03

## Verdict

Final verdict: **APPROVE_WITH_FOLLOWUPS**

Layered conclusion:

- **Current E2 one-epoch run**: approved. The latest `accum=4` run completed with the patched trainer, correct runtime wiring, completed summary, checkpoint, and unchanged base checkpoint.
- **Formal multi-epoch training**: not yet approved as a reproducible operating procedure. The SAM301 source patch is technically correct, but `/home/book/sam301` is not version-controlled and the project has no patch-application or hash-guard script that fails fast before formal training if the trainer patch is missing.

## Git Baseline

- Project: `/home/book/book01`
- Branch: `codex-stage-e2`
- HEAD reviewed: `a8a877043c91257aa3736af7a8da22e502b2f7e1`
- Relevant commits:
  - `74ec46f Add preflight guards for scratch keys and effective batch size`
  - `81f79d5 Record sam301 trainer grad-accum loss scaling patch with numerics tests`
  - `a8a8770 Document E2 loss scaling and preflight fix acceptance`
- Untracked files present and not touched:
  - `docs/CLAUDE_REVIEW_OF_GRAD_ACCUM_FIX.md`
  - `docs/CLAUDE_REVIEW_OF_SAM301_ENV_MIGRATION.md`
  - `docs/HANDOFF_FOR_REVIEW_E2_FIXES.md`
- No tracked `runs/`, `data/`, `experiments/`, images, `.pt`, `.pth`, `.ckpt`, or `.npz`.
- No active `train.py`, `torchrun`, Hydra, or training process was found.

This review did not modify project source, SAM301 source, runtime configs, run artifacts, Conda, CUDA, or drivers.

## Loss Scaling Review

Reviewed file:

- `/home/book/sam301/sam3/train/trainer.py`

Current trainer SHA256:

```text
bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2
```

The only semantic change in `_run_step()` is the backward input:

```python
backward_loss = loss / accum_steps if accum_steps > 1 else loss
self.scaler.scale(backward_loss).backward()
```

Confirmed:

- `math.isfinite(loss.item())` still checks the raw, unscaled loss.
- `loss_mts[loss_key].update(loss.item(), batch_size)` still logs/meters the raw, unscaled loss.
- Extra loss meters still log raw extra losses.
- `zero_grad` remains once at `_run_step()` entry, i.e. once per outer dataloader iteration.
- `optimizer.step`, `scaler.update`, scheduler stepping, gradient clipping, and gradient logging remain in `train_epoch()` and are still once per outer dataloader iteration.
- `accum=1` path sets `backward_loss is loss`, preserving original behavior.
- `accum=2/4` now computes the gradient of the mean micro-batch loss.
- DDP does not require an additional division by `num_gpus`; standard DDP all-reduce semantics already average gradients across ranks.

Micro-batch size equality:

- book01 runtime wiring sets outer train loader `batch_size = micro_batch_size * accumulation_steps`.
- `collate_fn_api_with_chunking` splits a full outer batch with `batch[i::num_chunks]`.
- With `drop_last=True`, incomplete outer batches are dropped, so configured book01 runs produce equal-sized chunks.
- Non-divisible dataset size is now warned; smaller-than-effective-batch is rejected.

Residual edge:

- Direct, non-book01 configs that set a train loader batch size not divisible by `num_chunks` could still create mean-of-means behavior. The book01 generator does not do this.

## Numerical Test Review

Test file:

- `tests/test_grad_accum_numerics.py`

The tests call the real patched `sam3.train.trainer.Trainer._run_step()` via `Trainer.__new__` with minimal injected attributes. They do not reimplement `_run_step`.

Confirmed test properties:

- Same initial parameters via `copy.deepcopy`.
- Same deterministic float64 data.
- Same optimizer type and learning rate (`torch.optim.SGD`, `lr=0.1`).
- Same MSE loss.
- `batch=4/accum=1` after one optimizer step matches `micro=1/accum=4` after one optimizer step with `rtol=1e-12`, `atol=1e-12`.
- `accum=1` gradients are exactly equal to plain unscaled `loss.backward()`.
- `accum=2` and `accum=4` repeated micro-batch cases verify mean-gradient semantics while logged values remain raw.
- List/length asserts for accumulation batches still fire.
- A source sentinel checks that the patched trainer lines exist.

The `1e-12` tolerance is credible because the test uses CPU float64, a tiny linear model, deterministic inputs, and no real AMP/CUDA nondeterminism.

Deletion/failure sensitivity:

- If the loss scaling line is removed from `/home/book/sam301/sam3/train/trainer.py`, the source sentinel fails.
- The numerical equivalence tests would also fail for the `batch4 accum1 == micro1 accum4` comparison because summed gradients produce a larger update.

## P3 Preflight Guard Review

Reviewed file:

- `core/training_runner.py`

Confirmed:

- Missing `scratch.train_batch_size` and missing `scratch.gradient_accumulation_steps` produce clean preflight errors.
- Direct `write_runtime_yaml()` missing-key calls raise a clear `ValueError`, not `int(None)` `TypeError`.
- UI/core override validation continues to reject non-integer, NaN, fractional, zero, and negative values.
- `effective_batch_size = train_batch_size * gradient_accumulation_steps * num_gpus`.
- If train image count is smaller than effective batch, preflight rejects and runtime YAML is not written.
- If train image count is not divisible by effective batch, preflight emits a warning and allows the run.
- This prevents the known zero-step completed case for the current preflight path.

Limit:

- The guard is preflight-side. A manually authored runtime YAML outside the book01 preflight path could still bypass it, but the formal UI/launcher flow does not.

## Latest Successful E2 Run Review

Run:

- `/home/book/book01/runs/training/2026-07-03_10-17-13`

Command:

```bash
conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py -c /home/book/book01/runs/training/2026-07-03_10-17-13/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
```

Runtime config:

- `trainer.max_epochs = 1`
- `scratch.train_batch_size = 1`
- `scratch.gradient_accumulation_steps = 4`
- `trainer.gradient_accumulation_steps = 4`
- `trainer.data.train.batch_size = 4`
- `scratch.collate_fn._target_ = sam3.train.data.collator.collate_fn_api_with_chunking`
- `scratch.collate_fn.num_chunks = 4`
- `scratch.collate_fn_val._target_ = sam3.train.data.collator.collate_fn_api`
- `trainer.data.train.drop_last = True`
- `trainer.checkpoint.save_dir = /home/book/book01/runs/training/2026-07-03_10-17-13/checkpoints`
- `trainer.model.checkpoint_path = /home/book/sam301/sam3.pt`
- `launcher.gpus_per_node = 1`

Metadata:

- `conda_environment = sam301`
- `resolved_sam3_import_path = /home/book/sam301/sam3/__init__.py`
- `sam3_import_guard_ok = true`
- `hydra_validation_ok = true`
- train images/annotations: `8 / 186`
- val images/annotations: `2 / 49`
- prompt: `book spine`

Summary:

- status: `completed`
- exit code: `0`
- start: `2026-07-03T01:17:19.996182+00:00`
- end: `2026-07-03T01:17:49.650653+00:00`
- duration: `29.654471397399902` seconds
- errors: `[]`
- warnings: `[]`

Log evidence:

- Trainer components set up successfully.
- Components moved to `cuda:0`.
- `Train Epoch: [0][0/2]`.
- Final train stats include `Trainer/steps_train: 8`.
- `Val Epoch: [0][0/2]`.
- Final val stats include `Trainer/steps_val: 2`.

Interpretation:

- 1 epoch completed.
- 2 outer train dataloader iterations.
- 4 micro-batches per outer iteration.
- 8 `_step()` calls total.
- 2 optimizer steps.
- 8 train images covered by full outer batches.

Checkpoint:

- `/home/book/book01/runs/training/2026-07-03_10-17-13/checkpoints/checkpoint.pt`
- Size: `10081250310` bytes
- Non-zero and located under the current run.

Base checkpoint:

```text
/home/book/sam301/sam3.pt size=3450062241 mtime=2026-06-15 19:22:15.022366000 +0900
SHA256=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

No residual process was found. Run artifacts, checkpoint, and logs are not tracked by Git.

## SAM301 Patch Reproducibility

Patch file:

- `/home/book/book01/patches/sam301_trainer_grad_accum_loss_scaling.patch`
- SHA256: `f28588dda179af4133ead23a7ceed2c4615822f420fa83eade282e4775caa85d`

Patch dry-run findings:

- Reverse dry-run against the current trainer succeeds.
- Reversing the patch in a temp copy reconstructs original trainer SHA:
  - `9c9c4159d2d5e799159cf6c855fc9be6f81c607fa2614cf570130da6159e248d`
- Applying the patch to that reconstructed original temp file succeeds and reproduces current trainer SHA:
  - `bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2`
- Reapplying the patch to an already-patched temp file is detected as reversed/previously applied and skips the hunk.

Conclusion:

- The patch file matches the current trainer change.
- The patch is reproducible on the reconstructed original.
- Repeated application is not silently duplicative.

Reproducibility gap:

- `/home/book/sam301` is not a Git repository.
- book01 Git contains the patch and tests, but a plain book01 checkout does not automatically patch `/home/book/sam301`.
- There is no formal patch apply script, patch status script, or launch-time trainer hash guard.
- The source sentinel test catches a missing patch during pytest, but formal training launch does not currently fail fast on trainer hash mismatch.

This is the only substantive blocker I found for approving longer formal training as an operational procedure.

## Tests Run

Full suite:

```text
conda run -n sam301 python -m pytest tests/ -v
160 passed, 9 warnings, 17 subtests passed in 31.59s
```

Gradio smoke:

- Included in the full suite.
- `tests/test_ui_smoke.py::UiStartupSmokeTest::test_app_launches_and_serves_http PASSED`.

Repeated focused tests:

```text
for i in 1 2 3; do
  conda run -n sam301 python -m pytest \
    tests/test_grad_accum_numerics.py \
    tests/test_e1_training.py::PreflightGuardsTest -v
done
```

Each repeat:

```text
9 passed, 4 warnings, 2 subtests passed
```

Hydra validate-only:

```text
hydra validation ok: /home/book/book01/runs/training/2026-07-03_10-17-13/config/runtime_config.yaml
```

Patch checks:

- reverse dry-run: passed
- reconstructed original forward dry-run/apply: passed
- double-apply dry-run: detected already-applied patch and skipped

No real training was started during this review.

## Findings

### P1

None.

### P2-1: SAM301 trainer patch is not enforced at launch time

Status: confirmed.

Impact:

- Current machine state is correct.
- Tests catch patch loss if they are run.
- But formal training can be launched without first proving `/home/book/sam301/sam3/train/trainer.py` still has the expected patched SHA.
- Because SAM301 is outside Git, book01 HEAD alone is insufficient to reconstruct the full runtime state.

Severity:

- P2 for formal multi-epoch training reproducibility and operator safety.
- Not a blocker for accepting the already completed E2 one-epoch run.

Required before formal multi-epoch training:

- Add a patch status/hash guard command or script.
- Add a documented patch apply/reverse procedure.
- Make the formal launcher/preflight verify the expected trainer hash or patch marker before allowing training.
- Prefer bringing `/home/book/sam301` under version control, or otherwise pinning source hashes in an environment manifest.

### P3

None newly found beyond the already-fixed R-4/R-5 class. I found no evidence that the current book01 path still permits those two cases.

## Final Conclusions

### A. Current E2

The latest one-epoch `accum=4` result is trustworthy as an E2 acceptance run:

- Correct environment and import path.
- Correct runtime config.
- Loss scaling patch present.
- 2 outer iterations and 8 micro-batch `_step()` calls.
- Completed status and exit code 0.
- Checkpoint generated.
- Base checkpoint unchanged.
- No residual process.

### B. Formal Multi-Epoch Training

Not approved yet as a reproducible procedure.

The remaining blocker is not the training math or P3 preflight guards; it is the unmanaged SAM301 source patch. Formal training should not start until the launcher/preflight or an explicit operator checklist verifies the trainer SHA/patch state.

### C. Reproducibility

Current patch management is good enough for review evidence, but not enough for reliable environment reconstruction.

Required before formal training:

1. Add a `scripts/check_sam301_trainer_patch.py` or equivalent hash guard.
2. Record expected current and original trainer SHA256 in a committed manifest.
3. Add an idempotent patch apply/status script, or make patch status verification a preflight hard error.
4. Document environment reconstruction from a clean `sam301` source tree.
5. Prefer placing `/home/book/sam301` under Git or vendoring a pinned source revision.

Final verdict: **APPROVE_WITH_FOLLOWUPS**.
