# E2 Patch Provenance and Atomicity Acceptance

Date: 2026-07-03

Branch: `codex-stage-e2`

Code commit under test: `13556674850d6c1dc995fbb899d613b82feaf681`

Final result: **FAIL**

Reason: the three SAM301 patch-integrity P3 fixes were implemented and passed CPU-only verification, but the single approved one-epoch real training run exited with code `1` before training because PyTorch distributed initialization could not bind its selected TCPStore port.

No retry was attempted.

## P3 Fixes

### P3-1: Canonical manifest target confinement

Fixed in `core/sam301_patch.py`.

Rules:

- `target_file` is expanded and resolved canonically.
- It must remain under manifest `expected_sam3_root`, currently `/home/book/sam301`.
- It must not resolve under `/home/book/sam3`.
- `..` escape and symlink escape are rejected by resolved-path relationship checks.

The wrapper subprocess guard in `scripts/launch_sam3_training.py` performs the same stdlib-only target confinement check before importing Hydra or SAM3 training code.

### P3-2: Canonical patch path confinement

Fixed in `core/sam301_patch.py`.

Rules:

- Real patch files must resolve under `/home/book/book01/patches`.
- External absolute paths are rejected.
- Symlinks that resolve outside the patch root are rejected.
- The patch file SHA256 must match the manifest before verify/apply/revert/training guard success.

The test-only manifest path supports an explicit temp `expected_patch_root`; the real manifest defaults to `/home/book/book01/patches`.

### P3-3: Atomic apply/revert

Fixed in `scripts/manage_sam301_patch.py`.

Implementation:

- Acquire a same-directory lock file.
- Copy the current trainer to a same-directory temporary file.
- Apply or reverse-apply the patch to the temporary file.
- Verify final SHA256 before replacing the real trainer.
- Preserve original file mode.
- fsync the temporary file.
- Use `os.replace()` for atomic replacement.
- fsync the parent directory.
- On any detected failure, the original trainer remains at its original hash.
- `UNKNOWN` remains fail-closed and is never overwritten.

No apply or revert was executed against the real `/home/book/sam301/sam3/train/trainer.py` in this acceptance run.

## Provenance

Implemented in `core/sam301_patch.py`, `core/training_runner.py`, `ui/training_preflight_page.py`, and `ui/training_process_manager.py`.

Fields recorded:

- `book01_git_commit`
- `book01_git_dirty`
- `book01_git_status_short`
- `manifest_path`
- `manifest_sha256`
- `patch_path`
- `patch_sha256`
- `expected_patch_sha256`
- `trainer_path`
- `trainer_sha256`
- `expected_trainer_patched_sha256`
- `expected_trainer_original_sha256`
- `trainer_patch_state`
- `patch_guard_ok`
- `patch_guard_error`
- `sam301_root`
- `sam3_import_path`
- `python_executable`
- `runtime_yaml_path`
- `runtime_yaml_sha256`

Saved locations:

- `provenance.json`
- `dataset_info.json` as `training_provenance`
- `training_config_summary.json` as `training_provenance`
- `training_summary.json` as `training_provenance`

The launcher recalculated provenance immediately before `ProcessManager.start`; it did not only copy preflight metadata.

## Tests

Patch/provenance tests:

- `conda run -n sam301 python -m pytest tests/test_sam301_patch.py -v`
- Result after implementation: `18 passed`
- Repeats: `18 passed`, `18 passed`, `18 passed`

Real SAM301 read-only status:

- `conda run -n sam301 python scripts/manage_sam301_patch.py status`
- Result: `PATCHED`

Real SAM301 read-only verify:

- `conda run -n sam301 python scripts/manage_sam301_patch.py verify`
- Result: `ok: True`

Hydra validate-only:

- `conda run -n sam301 python scripts/launch_sam3_training.py -c /home/book/book01/runs/training/2026-07-03_10-17-13/config/runtime_config.yaml --validate-only`
- Result: `hydra validation ok`

Full CPU-only pytest:

- First sandbox run: `177 passed`, `1 failed`
- Failure cause: sandbox denied local TCP socket creation in Gradio smoke (`PermissionError: [Errno 1] Operation not permitted`)
- Re-run with host permissions: `178 passed, 9 warnings, 17 subtests passed in 31.68s`
- Gradio smoke: passed in the host-permission run

No tracked run, checkpoint, data, or log files were added.

## Real One-Epoch Run

Run directory:

`/home/book/book01/runs/training/2026-07-03_12-34-30`

Command:

```bash
conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py -c /home/book/book01/runs/training/2026-07-03_12-34-30/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
```

Formal launcher:

`ui.training_preflight_page.start_training(...)`

Runtime parameters:

- `max_epochs`: `1`
- `scratch.train_batch_size`: `1`
- `scratch.gradient_accumulation_steps`: `4`
- trainer gradient accumulation: `4`
- train dataloader batch size: `4`
- collate function: `sam3.train.data.collator.collate_fn_api_with_chunking`
- `num_chunks`: `4`
- effective batch size: `4`
- `num_gpus`: `1`
- prompt: `book spine`
- `resume_from`: `None`

Preflight artifacts:

- `config/runtime_config.yaml`
- `dataset_info.json`
- `training_config_summary.json`
- `provenance.json`
- `command.txt`

Launcher:

- PID: `1514844`
- PGID: `1514844` while running
- Start time: `2026-07-03T03:35:14.753528+00:00`
- End time: `2026-07-03T03:35:17.850254+00:00`
- Duration: `3.096726179122925` seconds
- Exit code: `1`
- Summary status: `failed`

Failure:

First substantive error:

```text
torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 34508, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use
```

Traceback location:

- `training_summary.json` field `stdout_stderr_tail`
- temporary launcher event log `/tmp/book01_e2_training_events.json`

The trainer did not reach a complete epoch. It failed during distributed backend initialization before dataset loading, optimizer steps, or checkpoint writing.

## Summary and Checkpoint

Training summary:

`/home/book/book01/runs/training/2026-07-03_12-34-30/training_summary.json`

Summary status:

`failed`

Exit code:

`1`

Checkpoint files:

None.

Checkpoint result:

No valid checkpoint was generated.

## sam3.pt Integrity

Before training:

`/home/book/sam301/sam3.pt`

SHA256:

`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

After training:

`/home/book/sam301/sam3.pt`

SHA256:

`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`

Result:

Unchanged.

SAM301 trainer SHA256:

`bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2`

## Residual Process Check

Post-run checks found no residual processes matching:

- `sam3/train/train.py`
- `book_spine_finetune`
- `runs/training`
- `torchrun`
- `hydra`

Post-run GPU state:

- GPU: NVIDIA GeForce RTX 5090
- Driver: `595.71.05`
- Total memory: `32607 MiB`
- Used memory: `836 MiB`
- Free memory: `31254 MiB`

## Final Conclusion

P3 fixes: implemented and verified.

Real one-epoch training: **FAIL**.

Failure is not a patch-integrity/provenance failure. Patch guard and provenance behaved correctly. The run failed because PyTorch distributed initialization selected a port already in use.

No retry was attempted, per task boundary.

Status:

`NOT_READY_FOR_FORMAL_TRAINING_PREPARATION`
