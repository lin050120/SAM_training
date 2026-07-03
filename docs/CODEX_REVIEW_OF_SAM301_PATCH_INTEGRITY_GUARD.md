# Codex Review of SAM301 Patch Integrity Guard

Date: 2026-07-03

Reviewer: Codex

Scope: strict read-only review of the SAM301 trainer loss-scaling patch integrity guard at book01 commit `02ca75fcc751e30a7c70aa27f976be8f3bd8e655`.

No code, manifest, patch, SAM301 source, Conda environment, run directory, checkpoint, or runtime YAML was modified during this review.

## Verdict

Final judgment: **APPROVE_WITH_FOLLOWUPS**

Current patch management and the three-layer training guard are sufficient to allow a new one-epoch acceptance run, assuming the run is created from this reviewed Git commit with a clean worktree and the live trainer hash remains the reviewed patched hash.

This does not yet approve unattended formal multi-epoch training. Before formal multi-epoch training, the project should add stronger traceability metadata and tighten manifest path confinement.

## Git Baseline

- Branch: `codex-stage-e2`
- HEAD: `02ca75fcc751e30a7c70aa27f976be8f3bd8e655`
- HEAD summary: `02ca75f Add SAM301 patch integrity guard`
- Worktree status at review start:
  - Untracked review/handoff documents only:
    - `docs/CLAUDE_REVIEW_OF_GRAD_ACCUM_FIX.md`
    - `docs/CLAUDE_REVIEW_OF_SAM301_ENV_MIGRATION.md`
    - `docs/HANDOFF_FOR_REVIEW_E2_FIXES.md`

Reviewed diff from `HEAD~1..HEAD`:

- `README.md`
- `config/sam301_patch_manifest.json`
- `core/sam301_patch.py`
- `core/training_runner.py`
- `docs/SAM301_PATCH_MANAGEMENT.md`
- `docs/stage_e1_training_ui.md`
- `scripts/launch_sam3_training.py`
- `scripts/manage_sam301_patch.py`
- `tests/test_sam301_patch.py`
- `ui/training_preflight_page.py`

No tracked `runs/`, `data/`, `experiments/`, `test_pic/`, `.pt`, `.pth`, `.ckpt`, or `.npz` files were found.

No residual `sam3/train/train.py`, `book_spine_finetune`, `runs/training`, `torchrun`, or `hydra` training processes were found.

## Manifest and Hashes

Manifest:

- Path: `/home/book/book01/config/sam301_patch_manifest.json`
- Manifest SHA256: `cc55ef49343969cf471c58090cb65f85a8fbe9cfa7424899e65bf6a2aa22b604`
- Target: `/home/book/sam301/sam3/train/trainer.py`
- Patch: `/home/book/book01/patches/sam301_trainer_grad_accum_loss_scaling.patch`
- Patch SHA256: `f28588dda179af4133ead23a7ceed2c4615822f420fa83eade282e4775caa85d`
- Expected original trainer SHA256: `9c9c4159d2d5e799159cf6c855fc9be6f81c607fa2614cf570130da6159e248d`
- Expected patched trainer SHA256: `bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2`

Live trainer:

- Path: `/home/book/sam301/sam3/train/trainer.py`
- Independent SHA256: `bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2`
- Status: `PATCHED`

The live trainer exactly matches the manifest's patched hash.

## Patch Dry-Run Review

Dry-run checks were performed on temporary copies only. The real `/home/book/sam301/sam3/train/trainer.py` was not applied, reverted, or rewritten.

Results:

- Reverse dry-run against the temporary patched trainer succeeded.
- Reverting the temporary patched trainer produced SHA256 `9c9c4159d2d5e799159cf6c855fc9be6f81c607fa2614cf570130da6159e248d`.
- Forward dry-run against the reconstructed temporary original trainer succeeded.
- Applying forward patch to the temporary original trainer produced SHA256 `bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2`.
- Repeating forward apply against an already patched temporary trainer was detected by `patch` as reversed/previously applied and failed rather than duplicating the change.

This confirms the saved patch is consistent with both manifest hashes and is not silently re-applicable.

## Patch Manager Semantics

Reviewed script: `/home/book/book01/scripts/manage_sam301_patch.py`

State handling:

- `PATCHED`: actual trainer hash equals manifest `patched_sha256`.
- `UNPATCHED`: actual trainer hash equals manifest `original_sha256`.
- `UNKNOWN`: target exists but hash matches neither expected state.
- `MISSING`: target file does not exist.

Read-only real-tree commands:

- `status`: returned `PATCHED`.
- `verify`: returned success for the current patched tree.
- JSON form works when `--json` is supplied before the subcommand.

Safety behavior:

- `UNKNOWN` fails closed for verify/apply/revert.
- `MISSING` fails closed.
- Repeated apply is refused when state is already `PATCHED`.
- Repeated revert is refused when state is already `UNPATCHED`.
- Tampered patch content is rejected through patch-file SHA verification before patch execution.
- The script has no force mode.

Apply/revert recovery:

- The manager performs a dry-run before modifying.
- It backs up the target file before invoking `patch`.
- On patch command failure or post-apply hash mismatch, it restores the backup.

Atomicity limitation:

- The actual patch operation modifies the target file in place. Detected failures are restored, but an OS crash, process kill, or power loss during the `patch` invocation could still leave a partially modified file. This is acceptable for the reviewed one-epoch gate because later hash guards fail closed, but it should be documented or hardened before relying on the patch tool in automated environment rebuilds.

## Path Confinement Review

Positive checks:

- The current manifest target is `/home/book/sam301/sam3/train/trainer.py`.
- `core/sam301_patch.py` rejects targets under `/home/book/sam3`, preventing the old source tree from being used as the patch target.
- No current execution target points to `/home/book/sam3`.

Gap:

- The loader does not enforce that `target_file` is under `expected_sam3_root`.
- A temporary-manifest probe showed that targets such as `/tmp/probe/trainer.py` and `/home/book/sam301_evil/trainer.py` would be accepted if the trusted manifest were modified accordingly.
- The patch path is resolved from the manifest but is not explicitly constrained to the book01 `patches/` directory.

This is not a current-run blocker because the manifest is tracked in Git and the reviewed manifest is correct, but the manifest path model should be tightened before formal long-running training.

## Training Guard Chain

The intended chain is:

`preflight guard -> runtime YAML/token -> launcher guard -> token consumption -> ProcessManager/Popen -> wrapper guard -> official SAM3 main`

### Preflight Guard

Reviewed file: `/home/book/book01/core/training_runner.py`

`inspect_training_config(..., prepare_runtime=True)` calls `verify_patched_for_training()` before writing runtime YAML. If the guard fails, it appends an error and skips runtime file generation and launch-token creation.

Conclusion: preflight guard is correctly placed before runtime YAML and token creation.

### Launcher Guard and Token Ordering

Reviewed file: `/home/book/book01/ui/training_preflight_page.py`

`_consume_preflight_for_launch()` calls `verify_patched_for_training()` before adding the token to `_consumed_preflight_tokens` and before marking the state as consumed. Tests assert that a failing guard does not consume the token and does not call `manager.start`.

Conclusion: launcher guard is correctly placed before token consumption and before child process creation.

### ProcessManager and Popen

The launcher calls the existing training process manager only after preflight state validation, token validation, output confinement checks, and patch verification. No bypassing `ProcessManager` was introduced in this diff.

Conclusion: ProcessManager/Popen remains downstream of the token and patch guards.

### Wrapper Guard

Reviewed file: `/home/book/book01/scripts/launch_sam3_training.py`

`run_training()` calls `_verify_sam301_patch_or_die()` before importing Hydra and before importing `sam3.train.train.main`. Static order check confirmed:

- guard call occurs before `from hydra import initialize_config_dir`;
- guard call occurs before `from sam3.train.train import main as sam3_train_main`.

The wrapper locates the manifest via `Path(__file__).resolve().parent.parent / "config" / "sam301_patch_manifest.json"`, so it is independent of current working directory and robust to invoking the script through a symlink to the reviewed file.

Conclusion: wrapper guard executes before official training code is imported or run.

### TOCTOU

The three guard points reduce the practical time-of-check/time-of-use window:

- preflight catches a bad trainer before creating launchable state;
- launcher catches replacement after preflight but before token consumption;
- wrapper catches replacement after child process launch but before SAM3 training code import.

A very narrow residual window remains after wrapper guard and before Python loads every downstream module from disk. Eliminating that would require stronger deployment controls such as immutable source tree, container image, filesystem snapshot, or signed artifact workflow. For the current local one-epoch acceptance workflow, this is not blocking.

## Trust Boundary

The manifest and guard code are in the same Git repository. If an attacker or process can modify both the manifest and guard code and the operator runs from that modified commit, hash checking cannot provide independent authenticity.

For this project stage, a pinned reviewed Git commit, clean working tree, and independent SHA verification are sufficient. Formal training should additionally record:

- book01 commit;
- manifest SHA256;
- patch SHA256;
- trainer.py SHA256;
- SAM301 source root;
- SAM301 import path.

These values should be written into preflight metadata and final `training_summary.json` so a future checkpoint can be traced to the exact local patch state.

## Tests Run

Full CPU-only test suite:

- Command: `conda run -n sam301 python -m pytest tests/ -v`
- Result: `170 passed, 9 warnings, 17 subtests passed in 31.46s`

Patch test repeats:

- Command: `conda run -n sam301 python -m pytest tests/test_sam301_patch.py -v`
- Result 1: `10 passed`
- Result 2: `10 passed`
- Result 3: `10 passed`

Gradio smoke:

- Covered by full suite: `tests/test_ui_smoke.py::UiStartupSmokeTest::test_app_launches_and_serves_http PASSED`

Hydra validate-only:

- Command: `conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py -c /home/book/book01/runs/training/2026-07-03_10-17-13/config/runtime_config.yaml --validate-only`
- Result: `hydra validation ok`

Real-tree patch status/verify:

- `scripts/manage_sam301_patch.py status`: `PATCHED`
- `scripts/manage_sam301_patch.py verify`: success

Patch dry-run/reverse dry-run:

- Performed only on temporary copies.
- Forward and reverse directions matched the manifest hashes.

No training was started.

## Findings

### P1

None.

### P2

None.

### P3-1: Manifest Target Is Not Constrained to expected_sam3_root

The current manifest is correct, and `/home/book/sam3` is explicitly rejected. However, the manifest loader does not require `target_file` to be under `/home/book/sam301` using a canonical `relative_to` check. A modified manifest could point at `/tmp` or a sibling such as `/home/book/sam301_evil`.

Impact: not blocking for the reviewed commit, but weakens the manifest as a safety boundary.

Recommended fix before formal multi-epoch training: require resolved target paths to be under `expected_sam3_root`, and constrain patch files to the book01 `patches/` directory unless an explicit audited exception exists.

### P3-2: Apply/Revert Is Best-Effort, Not Crash-Atomic

The manager backs up the file and restores on detected failure, but `patch` modifies the target in place. A hard kill or power loss during patching could leave a partially modified trainer.

Impact: not blocking for training launch because all training guards hash-check and fail closed. It matters for automated environment rebuild and operator recovery.

Recommended fix: document this limitation, or implement patching through a temporary reconstructed file plus atomic replace after hash verification.

### P3-3: Patch Integrity Metadata Is Not Yet Persisted in Training Results

The guard verifies live state, but formal training results should retain the provenance values needed to audit a checkpoint later: book01 commit, manifest SHA, patch SHA, trainer SHA, and SAM301 source root/import path.

Impact: not blocking for a new one-epoch run, but important before formal multi-epoch training where checkpoints become durable artifacts.

Recommended fix: write these values into preflight metadata and final `training_summary.json`.

### Non-Blocking Observation: CLI JSON Flag Placement

`--json` is a global argparse option and must be placed before the subcommand, e.g. `manage_sam301_patch.py --json status`. `status --json` fails. This is acceptable but should be documented precisely.

## Conclusions

### A. Current Patch Management

Credible for this stage. The live trainer is patched, hashes match independently, the saved patch round-trips to the expected original and patched hashes on temporary copies, and status/verify fail closed for non-matching states.

### B. Guard for New One-Epoch Run

Approved. The preflight, launcher, and wrapper guards are placed in the right order and prevent runtime YAML/token creation, token consumption, or trainer startup when the patch state is not `PATCHED`.

### C. Formal Training Readiness

Not yet approved for unattended formal multi-epoch training. The next one-epoch guard validation run is acceptable, but before formal multi-epoch training the project should tighten manifest path confinement and persist patch provenance metadata into run outputs.

### D. Required Before Formal Multi-Epoch Training

1. Enforce `target_file` under `expected_sam3_root` with canonical `Path.resolve()` plus `relative_to`.
2. Constrain `patch_file` to the reviewed book01 patches directory or document the manifest as a fully trusted input.
3. Record book01 commit, manifest SHA256, patch SHA256, trainer SHA256, SAM301 source root, and import path in preflight and `training_summary.json`.
4. Document or harden apply/revert crash atomicity.
5. Clarify `--json` command usage in patch-management documentation.

## Final Decision

**APPROVE_WITH_FOLLOWUPS**

- P1 findings: 0
- P2 findings: 0
- P3 findings: 3
- Non-blocking observations: 1
- New one-epoch run: approved
- Formal multi-epoch training: not approved until the formal-training followups above are completed or explicitly waived
