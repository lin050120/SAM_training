"""CLI: evaluate all trainer checkpoints of a run on the registered validation split.

Usage:
    conda run -n sam301 python scripts/evaluate_sam3_checkpoints.py \\
        --run-dir /home/book/book01/runs/training/<run_id>

Defaults (validation paths, prompt) are read from the run's runtime_config.yaml and
training_config_summary.json plus config/checkpoint_evaluation.yaml; every flag
below overrides them explicitly. See docs/CHECKPOINT_EVALUATION_CN.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="training run directory")
    parser.add_argument("--val-annotations", default=None, help="override validation COCO annotations path")
    parser.add_argument("--val-images", default=None, help="override validation images directory")
    parser.add_argument("--checkpoints", nargs="*", default=None, help="only evaluate these checkpoint file names")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None, help="SAM3 processor confidence threshold (pre-NMS proposal cut)")
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--boundary-tolerance", type=int, default=None, help="boundary F1 tolerance in px (default 2)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-baseline", action="store_true", help="skip evaluating the original sam3.pt baseline")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--export-best", action="store_true", help="export the best trainer checkpoint to checkpoints/inference_best.pt")
    parser.add_argument("--force", action="store_true", help="ignore cached per-checkpoint results")
    parser.add_argument("--max-images", type=int, default=None, help="limit validation images (smoke runs only)")
    parser.add_argument("--smoke", action="store_true", help="mark this evaluation as a smoke run (not an official ranking)")
    parser.add_argument("--json", action="store_true", help="print the machine-readable summary")
    args = parser.parse_args(argv)

    from core.checkpoint_evaluation.evaluator import config_from_run, evaluate_run

    overrides: dict = {}
    if args.val_annotations:
        overrides["val_annotations"] = Path(args.val_annotations)
    if args.val_images:
        overrides["val_images"] = Path(args.val_images)
    if args.checkpoints:
        overrides["checkpoint_names"] = list(args.checkpoints)
    if args.prompt is not None:
        overrides["prompt"] = args.prompt
    if args.score_threshold is not None:
        overrides["score_threshold"] = args.score_threshold
    if args.confidence_threshold is not None:
        overrides["confidence_threshold"] = args.confidence_threshold
    if args.min_area is not None:
        overrides["min_area"] = args.min_area
    if args.boundary_tolerance is not None:
        overrides["boundary_tolerance_px"] = args.boundary_tolerance
    if args.no_baseline:
        overrides["include_baseline"] = False
    if args.baseline_checkpoint:
        overrides["baseline_checkpoint"] = Path(args.baseline_checkpoint)
    overrides["device"] = args.device
    overrides["export_best"] = bool(args.export_best)
    overrides["force"] = bool(args.force)
    overrides["smoke"] = bool(args.smoke)
    if args.max_images is not None:
        overrides["max_images"] = args.max_images
        overrides["smoke"] = True  # a truncated validation set is never an official ranking

    try:
        config = config_from_run(Path(args.run_dir), **overrides)
    except Exception as exc:
        print(f"configuration failed: {exc}", file=sys.stderr)
        return 2

    summary = evaluate_run(config)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"status: {summary['status']}")
        if summary.get("reason"):
            print(f"reason: {summary['reason']}")
        best = summary.get("best")
        if best:
            print(
                f"best: {best['checkpoint_name']} (epoch {best['epoch']}) "
                f"mean_iou_all_gt={best['mean_iou_all_gt']}"
            )
            print(f"improved over baseline: {summary.get('finetuned_improved_over_baseline')}")
        print(f"details: {summary.get('files', {}).get('checkpoint_metrics_csv')}")
    if summary["status"] in {"blocked", "failed"}:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
