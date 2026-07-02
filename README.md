# Book Spine SAM3 Workflow

## Authoritative SAM3 Training Config

The default book-spine fine-tuning config for this workspace is:

`/home/book/sam301/sam3/train/configs/book_spine/book_spine_finetune.yaml`

Use the preflight command to inspect the config and generate the training command without starting training:

```bash
conda run -n sam3 python scripts/training_preflight.py
```

To manually set the SAM3 training text prompt without renaming COCO categories:

```bash
conda run -n sam3 python scripts/training_preflight.py \
  --training-prompt "book spine"
```

The generated training command is:

```bash
conda run -n sam3 python /home/book/sam301/sam3/train/train.py \
  -c /home/book/book01/runs/training/<run_id>/config/runtime_config.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

The preflight reads the authoritative base config, creates a per-run runtime config under `/home/book/book01/runs/training/<run_id>/config/runtime_config.yaml`, writes `dataset_info.json` and `command.txt`, and reports batch size, GPU count, gradient accumulation, effective batch size, checkpoint, train/val data paths, output directory, COCO category, requested training prompt, resolved training prompt, and prompt source. It does not start SAM3 training.

Default runtime paths:

- checkpoint: `/home/book/sam301/sam3.pt`
- train data: `/home/book/book01/data/book_spine_sam3_dataset/train`
- val data: `/home/book/book01/data/book_spine_sam3_dataset/val`
- training outputs: `/home/book/book01/runs/training/<run_id>`

## Unified Inference

Legacy NPZ mode:

```bash
conda run -n sam3 python scripts/run_unified_inference.py \
  --legacy-raw-run /home/book/book01/data/dataset_raw/20260622_231208_book_spine \
  --limit 2
```

Real SAM3 mode, one image:

```bash
PYTHONPATH=/home/book/sam301 conda run -n sam3 python scripts/run_unified_inference.py \
  --input-dir /home/book/book01/data/book_spine_sam3_dataset/test/images \
  --limit 1 \
  --checkpoint /home/book/sam301/sam3.pt \
  --prompt "book spine" \
  --device cuda
```

The real SAM3 command requests CUDA by default. It fails fast when CUDA is unavailable; CPU inference is allowed only with explicit `--device cpu`.

CUDA diagnosis helper:

```bash
bash scripts/check_cuda_environment.sh
```

## CVAT Export

Each inference run writes:

```text
cvat_export/
├── images/
├── annotations/
│   └── instances_default.json
├── polygon/
│   ├── instances_default.json
│   ├── validation_report.json
│   └── polygon_fidelity_report.json
└── rle/
    ├── instances_default.json
    └── validation_report.json
```

Re-export and validate CVAT package without re-running SAM3:

```bash
conda run -n sam3 python scripts/export_cvat_package.py \
  --run-dir /home/book/book01/runs/inference/<run_id> \
  --segmentation-format both \
  --polygon-fidelity
```

Segmentation modes:

- `polygon`: default-compatible CVAT package. It is lossy relative to the final NMS mask.
- `rle`: exact mask export from final NMS bool masks. The local validator checks exact decoded-mask equality.
- `both`: writes separate `polygon/` and `rle/` COCO files so the formats do not overwrite each other.

Only two things are automatically verified: pycocotools can decode the segmentation, and the project validator passes. CVAT actual import and CVAT re-export mask fidelity must be tested manually. For a one-image smoke test, import `cvat_export/rle/instances_default.json` with `cvat_export/images/` into a temporary CVAT task, then export it back and compare against `npz_nms/im_000001.npz`.
