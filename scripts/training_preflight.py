from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DEFAULT_BOOK_SPINE_FINETUNE_CONFIG
from core.training_runner import format_preflight, inspect_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SAM3 training config and generate command without training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_BOOK_SPINE_FINETUNE_CONFIG)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--train-images", type=Path, default=None)
    parser.add_argument("--train-annotations", type=Path, default=None)
    parser.add_argument("--val-images", type=Path, default=None)
    parser.add_argument("--val-annotations", type=Path, default=None)
    parser.add_argument("--training-prompt", default=None)
    parser.add_argument("--max-epochs", type=float, default=None)
    parser.add_argument("--train-batch-size", type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--allow-external-output", action="store_true")
    parser.add_argument("--no-prepare-runtime", action="store_true", help="Only inspect; do not create runtime YAML/run dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "config_path": args.config,
        "num_gpus": args.num_gpus,
        "initial_checkpoint": args.initial_checkpoint,
        "train_images": args.train_images,
        "train_annotations": args.train_annotations,
        "val_images": args.val_images,
        "val_annotations": args.val_annotations,
        "training_prompt": args.training_prompt,
        "max_epochs": args.max_epochs,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "allow_external_output": args.allow_external_output,
        "prepare_runtime": not args.no_prepare_runtime,
    }
    if args.output_root is not None:
        kwargs["output_root"] = args.output_root
    preflight = inspect_training_config(**kwargs)
    print(format_preflight(preflight))


if __name__ == "__main__":
    main()
