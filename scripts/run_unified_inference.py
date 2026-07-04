from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DEFAULT_SAM3_CHECKPOINT
from core.inference_run import migrate_legacy_raw_run, run_sam3_image_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a unified inference run directory.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--legacy-raw-run", type=Path, help="Existing ft_01 dataset_raw run directory.")
    mode.add_argument("--input-dir", type=Path, help="Input image directory for real SAM3 inference.")
    parser.add_argument("--output-root", type=Path, default=Path("/home/book/book01/runs"))
    parser.add_argument("--book-root", type=Path, default=Path("/home/book/book01"))
    parser.add_argument("--sam3-root", type=Path, default=Path("/home/book/sam301"))
    parser.add_argument("--checkpoint", "--model-path", dest="checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--prompt", default="book spine")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--dtype-mode", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5)
    parser.add_argument("--nms-metric", choices=["iou", "iomin"], default="iou")
    parser.add_argument("--nms-mode", choices=["suppress", "merge"], default="suppress")
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--category-name", default="book_spine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.legacy_raw_run:
        run_dir = migrate_legacy_raw_run(
            legacy_run_dir=args.legacy_raw_run,
            output_root=args.output_root,
            book_root=args.book_root,
            sam3_root=args.sam3_root,
            limit=args.limit,
            nms_iou_thresh=args.nms_iou_thresh,
            nms_metric=args.nms_metric,
            nms_mode=args.nms_mode,
            min_area=args.min_area,
            category_name=args.category_name,
        )
    else:
        run_dir = run_sam3_image_directory(
            input_dir=args.input_dir,
            output_root=args.output_root,
            book_root=args.book_root,
            sam3_root=args.sam3_root,
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            score_threshold=args.score_threshold,
            confidence_threshold=args.confidence_threshold,
            dtype_mode=args.dtype_mode,
            device=args.device,
            limit=args.limit,
            nms_iou_thresh=args.nms_iou_thresh,
            nms_metric=args.nms_metric,
            nms_mode=args.nms_mode,
            min_area=args.min_area,
            category_name=args.category_name,
        )
    print(run_dir)


if __name__ == "__main__":
    main()
