from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import DEFAULT_DATASET_ROOT, DEFAULT_TRAINING_PROMPT
from core.dataset_split_builder import DatasetBuildConfig, build_training_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build SAM3 train/val/test dataset: annotation pool -> train/val, test dir -> test."
    )
    parser.add_argument("--annotation-pool-dir", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--category-name", default=DEFAULT_TRAINING_PROMPT)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = build_training_dataset(
            DatasetBuildConfig(
                annotation_pool_dir=args.annotation_pool_dir,
                test_dir=args.test_dir,
                output_dir=args.output_dir,
                category_name=args.category_name,
                val_ratio=args.val_ratio,
                seed=args.seed,
                overwrite=args.overwrite,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"dataset build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
