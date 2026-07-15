# Pausing and Resuming SAM3 Training

## What This Feature Does

This is a durable pause. It stops the training process, releases GPU memory, and later
resumes the same run from `checkpoints/checkpoint.pt`. It works across UI and machine
restarts.

SAM3 updates `checkpoint.pt` after every completed epoch. The checkpoint contains the
model, optimizer, epoch, step counters, loss state, and AMP scaler. If training is paused
mid-epoch, work performed after the latest checkpoint is discarded.

## Pause

1. Open the `训练预检` training tab.
2. Confirm that `<run_dir>/checkpoints/checkpoint.pt` appears in the monitor.
3. Click `暂停训练并释放显存`.
4. Wait for status `paused`. The training process has exited and GPU memory is free.

The UI refuses to pause before the first complete checkpoint exists. If
`checkpoint.pt.tmp` is being written, wait for checkpoint saving to finish and try again.

## Resume

1. Open `阶段 C: 从最近完整 Checkpoint 恢复` on the same page.
2. Click `刷新可恢复 run`.
3. Select the original training run.
4. Check `恢复预检信息`: `resumable` must be true and the data paths, prompt, maximum
   epochs, and `resume_checkpoint` must be correct.
5. Select the resume confirmation checkbox and click `恢复训练`.

Resume reuses the original run's runtime YAML, output directories, and `checkpoint.pt`,
while allocating a fresh distributed port. Do not put a trainer checkpoint in the
`initial checkpoint` field; that field initializes a new run and does not restore the
optimizer or epoch.

## Server-Side Guards

Before resuming, the server verifies that:

- the run, runtime YAML, and a non-empty `checkpoint.pt` exist;
- the original base `sam3.pt` and BPE files still exist;
- the runtime YAML still saves checkpoints inside the selected run;
- train/validation paths exist and the current dataset registry still permits the
  original training mode;
- CUDA capacity satisfies the original run;
- the SAM301 trainer patch and SAM3 import path are valid;
- no live process is already using the same runtime YAML;
- a completed run is not resumed accidentally.

The top level of `training_summary.json` describes the latest launch. Its `attempts`
array retains the initial launch, pauses, and all resume attempts in the same run.

## Limitations

- An incomplete epoch is not preserved.
- Pausing is unavailable before the first checkpoint.
- Resume does not change max epochs, batch size, learning rate, prompt, or dataset.
  Create a new preflight and run when those settings must change.
- Source changes are not hot-loaded into an already running UI. Let the current training
  finish before restarting the UI; closing the old UI stops the training process it owns.
