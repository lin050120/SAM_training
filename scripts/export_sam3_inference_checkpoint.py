"""CLI: export a SAM3 trainer checkpoint into a self-contained inference checkpoint.

See docs/CHECKPOINT_EXPORT_AND_INFERENCE.md for the full explanation of why trainer
checkpoints cannot be passed directly to the existing inference entry point, and
what the exported file format contains.

Usage:
    conda run -n sam301 python scripts/export_sam3_inference_checkpoint.py \\
        --input <trainer_checkpoint.pt> --output <inference_model.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.checkpoint_export import export_inference_checkpoint  # noqa: E402
from core.config import DEFAULT_SAM3_CHECKPOINT  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="path to a trainer checkpoint.pt")
    parser.add_argument("--output", required=True, help="path to write the inference checkpoint")
    parser.add_argument(
        "--base-checkpoint",
        default=str(DEFAULT_SAM3_CHECKPOINT),
        help=f"base checkpoint to diff against (default: {DEFAULT_SAM3_CHECKPOINT})",
    )
    parser.add_argument("--no-base-diff", action="store_true", help="skip the base-checkpoint diff/identity check")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting an existing output file")
    parser.add_argument("--dataset-identity-json", default=None, help="optional JSON string recorded into metadata")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON result")
    args = parser.parse_args(argv)

    dataset_identity = json.loads(args.dataset_identity_json) if args.dataset_identity_json else None

    try:
        result = export_inference_checkpoint(
            trainer_checkpoint_path=args.input,
            output_path=args.output,
            base_checkpoint_path=None if args.no_base_diff else args.base_checkpoint,
            overwrite=args.overwrite,
            dataset_identity=dataset_identity,
        )
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "output_path": result.output_path,
        "output_sha256": result.output_sha256,
        "matched_tensors": result.mapping_result.matched_tensors,
        "matched_parameters": result.mapping_result.matched_parameters,
        "coverage_ratio": result.mapping_result.coverage_ratio,
        "missing_keys": result.mapping_result.missing,
        "unexpected_keys": result.mapping_result.unexpected,
        "base_diff": result.metadata.get("base_diff"),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"exported: {result.output_path}")
        print(f"sha256: {result.output_sha256}")
        print(
            f"matched_tensors={result.mapping_result.matched_tensors} "
            f"coverage={result.mapping_result.coverage_ratio:.4f} "
            f"missing={len(result.mapping_result.missing)} "
            f"unexpected={len(result.mapping_result.unexpected)}"
        )
        if payload["base_diff"]:
            print(f"base_diff: {payload['base_diff']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
