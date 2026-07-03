# E3 Dataset Identity Erratum

Generated: 2026-07-03

**This is an erratum, not a replacement.** `docs/E3_FORMAL_DATASET_AND_ONE_EPOCH_ACCEPTANCE.md`, `docs/FORMAL_DATASET_AUDIT.md`, `data_manifests/formal_dataset_manifest.json`, and `data_manifests/formal_split_manifest.json` are **unmodified** — their content and SHA256 hashes are exactly what they were before this document was written. This erratum corrects how those documents' results should be *interpreted*.

## The correction

The 184-image / 7185-annotation dataset those documents call "the formal dataset" is **not human-reviewed ground truth**. It is SAM3's own machine pre-annotation output:

- All six source COCO files (`data/cvat_import_polygon_*_book_spine_iou0.5_book_spine.json`) carry `info.description == "book_spine SAM3 pre-annotation (polygon ~8pts, NMS)"`.
- 100% of the 7185 annotation polygons have <=8 vertices — consistent with an unedited machine export, not manual CVAT correction (a human correcting a spine boundary would not consistently stay under 8 points).

This was identified in `docs/FABLE5_FULL_INDEPENDENT_PROJECT_AUDIT.md`, finding **F-P1-1**, and is now recorded machine-readably in `data_manifests/dataset_identity_registry.json` (dataset_id `formal_book_spine_sam3_dataset_2026_07_03_smoke`).

## What this means for the E3 one-epoch acceptance

- **`docs/E3_FORMAL_DATASET_AND_ONE_EPOCH_ACCEPTANCE.md` remains a valid end-to-end pipeline / smoke-test acceptance.** The Hydra wrapper, gradient-accumulation loss scaling, SAM301 patch guard, port allocation, provenance recording, and checkpoint generation all really ran, on the real GPU, and produced the artifacts that report describes. None of that is in question.
- **It does NOT constitute a formal training result.** Training on the model's own pre-annotation is self-distillation: it cannot demonstrate improvement on the failure modes this project actually cares about (comic/complex-pattern misclassification, tilted spines, thin spines, adjacent-spine bleeding, mask over-expansion), because those failure modes are baked into the labels themselves.
- **`bbox AP=0.838` / `AP50=0.977` are NOT evidence of fine-tuning quality.** They measure agreement with SAM3's own pre-annotation, which is close to tautological after one epoch of fine-tuning starting from the same model. They are also bbox-only metrics; no mask AP/IoU was computed, so they say nothing about book-spine mask boundary quality even on their own (biased) terms.

## Machine-readable record

`data_manifests/dataset_identity_registry.json` — new file, records (independently recomputed this session, not copied from any prior report):

- `annotation_source: "sam3_machine_preannotation"`, `human_reviewed: false`, `independently_corrected_gt: false`
- `allowed_for_formal_training: false`, `allowed_for_model_evaluation: false`, `max_epochs_without_human_review: 1`
- `image_file_count: 184`, `unique_image_count: 136`, `annotation_count: 7185`, `exact_duplicate_group_count: 44`
- per-split file/unique-image/annotation counts (train 160/120/6749, val 12/12/253, test 12/4/183)
- pointers + SHA256 to `formal_dataset_manifest.json` and `formal_split_manifest.json` (unchanged)

## Enforcement (fail-closed, not just a warning)

`core/dataset_identity.py::resolve_dataset_identity()` looks up a dataset by the **resolved path** of its train/val annotation files against the registry — never by filename. Unmatched datasets default to `human_reviewed=false` (fail-safe, not fail-open).

`core/training_runner.py::inspect_training_config()` now takes a `training_mode` parameter (`"smoke"` default, or `"formal"`) and enforces, before any runtime YAML is written or launch token issued:

- `training_mode="formal"` against a dataset with `allowed_for_formal_training=false` -> **rejected**.
- `max_epochs` greater than the dataset's `max_epochs_without_human_review` (1, for this dataset) -> **rejected**, in either mode.
- One-epoch smoke runs continue to work exactly as before.

The UI (`ui/training_preflight_page.py`) shows a "training mode" selector (default `smoke`) and prints a prominent warning banner whenever the resolved dataset is not `human_reviewed`. `dataset_identity` and `training_mode` are written into every new run's `dataset_info.json` and `training_config_summary.json` (a pre-existing historical run, e.g. `2026-07-03_14-42-27`, predates this field and does not have it — that is expected, not a bug).

## How to promote a dataset to formal status

1. Have a human independently re-annotate or correct the polygons (in CVAT or otherwise) for these images, or acquire a new, human-reviewed batch.
2. Re-export COCO files and regenerate `data_manifests/formal_dataset_manifest.json` / `formal_split_manifest.json` from the corrected data (this itself requires the data-audit/split tooling identified as missing in finding F-P2-1 — out of scope for this erratum).
3. Add a **new** entry to `data_manifests/dataset_identity_registry.json` with a new `dataset_id`, `human_reviewed: true`, `independently_corrected_gt: true`, and `allowed_for_formal_training: true` only after independent verification that the correction actually happened (do not just flip the flag).
4. Only then does `training_mode="formal"` and `max_epochs>1` become available for that dataset.
