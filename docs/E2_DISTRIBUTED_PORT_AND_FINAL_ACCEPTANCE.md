# E2 Distributed Port and Final One-Epoch Acceptance

Date: 2026-07-03

Branch: `codex-stage-e2`

Final result: **PASS**

Status: `READY_FOR_FORMAL_TRAINING_PREPARATION`

## Baseline

- Project: `/home/book/book01`
- SAM301 source: `/home/book/sam301`
- Conda environment: `sam301`
- Previous failed run: `/home/book/book01/runs/training/2026-07-03_12-34-30`
- Previous failure: `torch.distributed.DistNetworkError`, TCPStore could not bind port `34508`
- Previous failed run was not reused.

## Root Cause

SAM3 official `train.py` chooses the local distributed TCPStore port from:

```python
main_port = random.randint(submitit_conf.port_range[0], submitit_conf.port_range[1])
single_node_runner(cfg, main_port)
```

Then `single_proc_run()` force-sets:

```python
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(main_port)
```

and the trainer later calls `torch.distributed.init_process_group()` through the env rendezvous path.

Port `34508` was not hardcoded by book01. It was randomly selected by SAM3 from the broad `submitit.port_range` in the runtime config. There was no pre-bind availability check, so the random choice could collide with an existing listener.

`ss -ltnp | grep ':34508'` after the fact showed no current listener, so the exact owner at failure time could not be attributed and no process was killed.

## Code Fix

Code commits:

- `967cb14 Allocate per-run distributed training port`
- `f42d6c5 Record launcher distributed port metadata`

Files modified:

- `core/training_runner.py`
- `core/sam301_patch.py`
- `ui/training_preflight_page.py`
- `ui/training_process_manager.py`
- `tests/test_e1_training.py`

Fix mechanism:

1. book01 allocates a bindable localhost TCP port per run.
2. The runtime YAML is written with `submitit.port_range: [port, port]`.
3. The formal launcher re-checks and re-allocates the port inside the launch lock immediately before token consumption and `ProcessManager.start`.
4. If port allocation fails, launch is blocked before token consumption and before child process creation.
5. The selected `MASTER_ADDR` and `MASTER_PORT` are added to the child process environment.
6. Provenance, `training_config_summary.json`, and `training_summary.json` record the actual launch port.
7. No SAM301 source file was modified.

Residual TOCTOU note:

The selected port must be released before SAM3 can bind it, so a tiny OS-level check/use window remains. The selection and Popen are now adjacent under the launcher lock, and the run-level port is no longer a broad random SAM3 choice.

## Tests

New and related tests repeated:

- `HydraLaunchRegressionTest`
- `StartTrainingOneTimePreflightTest`
- `tests/test_sam301_patch.py`

Results:

- Repeat 1: `25 passed`
- Repeat 2: `37 passed, 2 subtests passed`
- Repeat 3: `37 passed, 2 subtests passed`

Full test suite:

- Command: `conda run -n sam301 python -m pytest tests/ -v`
- Result: `182 passed, 9 warnings, 17 subtests passed in 32.49s`
- Gradio smoke: passed

Hydra validate-only:

- Result: passed

SAM301 patch status:

- `status`: `PATCHED`
- `verify`: `ok: True`

## New E2 Run

Run directory:

`/home/book/book01/runs/training/2026-07-03_12-48-21`

Command:

```bash
conda run -n sam301 python /home/book/book01/scripts/launch_sam3_training.py -c /home/book/book01/runs/training/2026-07-03_12-48-21/config/runtime_config.yaml --use-cluster 0 --num-gpus 1
```

Formal launcher:

`ui.training_preflight_page.start_training(...)`

Runtime parameters:

- `max_epochs`: `1`
- `train_batch_size`: `1`
- `gradient_accumulation_steps`: `4`
- effective batch size: `4`
- `num_gpus`: `1`
- prompt: `book spine`
- `resume_from`: `None`

Distributed port:

- `MASTER_ADDR`: `localhost`
- `MASTER_PORT`: `58803`
- Runtime YAML `submitit.port_range`: `[58803, 58803]`
- `provenance.json`: `58803`
- `training_config_summary.json`: `58803`
- `training_summary.json`: `58803`

## Acceptance Result

Training summary:

`/home/book/book01/runs/training/2026-07-03_12-48-21/training_summary.json`

Result:

- status: `completed`
- exit code: `0`
- start: `2026-07-03T03:48:57.232725+00:00`
- end: `2026-07-03T03:49:26.398500+00:00`
- duration: `29.165775299072266` seconds

Training evidence:

- train log: `Train Epoch: [0][0/2]`
- train stats: `Trainer/epoch = 0`
- train stats: `Trainer/steps_train = 8`
- validation stats: `Trainer/steps_val = 2`
- This confirms one epoch over 2 outer train iterations.
- With `gradient_accumulation_steps=4`, 2 outer iterations correspond to 8 micro-batches and 2 optimizer steps.

Checkpoint:

- Path: `/home/book/book01/runs/training/2026-07-03_12-48-21/checkpoints/checkpoint.pt`
- Size: `10081250310` bytes
- Non-zero: yes

## Integrity Checks

Initial checkpoint:

- Path: `/home/book/sam301/sam3.pt`
- Before SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- After SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- Result: unchanged

SAM301 trainer:

- Path: `/home/book/sam301/sam3/train/trainer.py`
- SHA256: `bcf5d8d6970ffd8fda609c92b14f9beda475576180262d5e5483523020609ec2`
- Patch state: `PATCHED`

Residual processes:

- No residual `sam3/train/train.py`, `book_spine_finetune`, `runs/training`, `torchrun`, or `hydra` processes found.

Residual port:

- `ss -ltnp | grep ':58803'` returned no listener.

GPU after run:

- NVIDIA GeForce RTX 5090
- Used memory: `830 MiB`
- Free memory: `31260 MiB`

## Notes

- No automatic retry was performed.
- No `/home/book/sam3` modifications were made.
- No Conda, CUDA, driver, or SAM301 source changes were made.
- Run outputs, logs, summaries, and checkpoints were not added to Git.

## Final Conclusion

The TCPStore port collision is fixed at the book01 orchestration layer. A new one-epoch, accum=4 E2 run completed successfully through the formal launcher with run-level port provenance, a valid checkpoint, unchanged base checkpoint, and no residual processes or listening port.

`READY_FOR_FORMAL_TRAINING_PREPARATION`
