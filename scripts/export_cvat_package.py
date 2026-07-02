from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cvat_export import export_cvat_package, polygon_fidelity_report, write_nms_pair_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-export and validate a CVAT COCO package from an inference run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--segmentation-format", choices=["polygon", "rle", "both"], default="polygon")
    parser.add_argument("--zip", action="store_true", help="Also create cvat_export.zip.")
    parser.add_argument("--polygon-fidelity", action="store_true", help="Measure polygon conversion fidelity against final NMS masks.")
    parser.add_argument("--nms-review", action="store_true", help="Write NMS review visualization for a kept/removed pair.")
    parser.add_argument("--image-name", default="im_000001.png")
    parser.add_argument("--kept-source-id", type=int, default=3)
    parser.add_argument("--removed-source-id", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_cvat_package(args.run_dir, make_zip=args.zip, segmentation_format=args.segmentation_format)
    output = {"export": report}
    if args.polygon_fidelity:
        output["polygon_fidelity"] = polygon_fidelity_report(args.run_dir)
    if args.nms_review:
        output["nms_review"] = write_nms_pair_review(
            args.run_dir,
            image_name=args.image_name,
            kept_source_id=args.kept_source_id,
            removed_source_id=args.removed_source_id,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
