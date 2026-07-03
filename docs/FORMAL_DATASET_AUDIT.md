# Formal Dataset Audit
Date: 2026-07-03T05:41:15.560632+00:00

## Source
Formal candidate source: six CVAT polygon COCO exports in `data/cvat_import_polygon_*_book_spine_iou0.5_book_spine.json`, mapped to matching `data/dataset_raw/<timestamp>_book_spine/images` directories. Existing `data/book_spine_sam3_dataset` is a 22-image smoke split and is not used as the formal source.

## Totals
- Images: 184
- Annotations: 7185
- Categories: {(1, 'book_spine'): 6}
- Missing images: 0
- Empty images: 0
- Extremely small masks (`area_ratio < 0.001`): 286

## Source Files
- `data/cvat_import_polygon_20260622_231208_book_spine_iou0.5_book_spine.json`: group `20260622_231208`, split counts {'val': 12}, images 12, annotations 253, raw images `data/dataset_raw/20260622_231208_book_spine/images`
- `data/cvat_import_polygon_20260622_231750_book_spine_iou0.5_book_spine.json`: group `20260622_231750`, split counts {'test': 4}, images 4, annotations 61, raw images `data/dataset_raw/20260622_231750_book_spine/images`
- `data/cvat_import_polygon_20260623_183600_book_spine_iou0.5_book_spine.json`: group `20260623_183600`, split counts {'test': 4}, images 4, annotations 61, raw images `data/dataset_raw/20260623_183600_book_spine/images`
- `data/cvat_import_polygon_20260623_233120_book_spine_iou0.5_book_spine.json`: group `20260623_233120`, split counts {'train': 98, 'test': 4}, images 102, annotations 4466, raw images `data/dataset_raw/20260623_233120_book_spine/images`
- `data/cvat_import_polygon_20260626_161819_book_spine_iou0.5_book_spine.json`: group `20260626_161819`, split counts {'train': 10}, images 10, annotations 235, raw images `data/dataset_raw/20260626_161819_book_spine/images`
- `data/cvat_import_polygon_20260630_180628_book_spine_iou0.5_book_spine.json`: group `20260630_180628`, split counts {'train': 52}, images 52, annotations 2109, raw images `data/dataset_raw/20260630_180628_book_spine/images`

## Split
No project-level formal split ratio was found. The selected split is duplicate-aware: byte-identical image clusters are not allowed to cross split, and train image count is divisible by the E3 effective batch size 4 so drop_last does not drop training images. Seed: `20260703`.
- train: images 160, annotations 6749, groups {'20260623_233120': 98, '20260626_161819': 10, '20260630_180628': 52}, avg anns/image 42.18, median 45.0
- val: images 12, annotations 253, groups {'20260622_231208': 12}, avg anns/image 21.08, median 20.5
- test: images 12, annotations 183, groups {'20260622_231750': 4, '20260623_183600': 4, '20260623_233120': 4}, avg anns/image 15.25, median 14.5

## Quality Checks
- Duplicate image IDs within source groups: 0
- Duplicate annotation IDs within source groups: 0
- Duplicate filenames within source groups: {}
- Exact duplicate image bytes: 44 hash group(s)
- Exact duplicate image groups crossing split: 0
- Average-hash near duplicate pairs: 731 total, 0 across splits
- Mask area ratio: min 0.00011332947530864197, median 0.011618923611111111, max 0.10943793402777778

## Problems
- No missing images, duplicate IDs, invalid polygons, zero-area masks, bbox out-of-bounds, pycocotools decode failures, exact duplicate cross-split leakage, or train-count divisibility issue was detected.

## Warnings
- exact duplicate image bytes found: 92 files in 44 groups; duplicate clusters are kept within one split

## Generated Artifacts
- `data_manifests/formal_dataset_manifest.json`
- `data_manifests/formal_split_manifest.json`
- Ignored split COCO files under `data/formal_book_spine_sam3_dataset/{train,val,test}/annotations.json`
- Training/validation image root for SAM3: `/home/book/book01/data/dataset_raw`
