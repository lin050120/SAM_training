"""Checkpoint evaluation orchestrator.

Current flow:
  validation guard -> evaluate baseline + unique trainer checkpoints on validation
  -> select best checkpoint from validation only -> optional inference_best export
  -> evaluate baseline + unique trainer checkpoints on test as diagnostic-only
  -> save raw predictions, GT snapshot, match records, per-image/per-instance CSV,
     visualizations, split audit, and atomic training_summary.json integration.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.checkpoint_evaluation import EVALUATOR_VERSION
from core.checkpoint_evaluation.artifacts import (
    INSTANCE_COLUMNS,
    build_instance_rows,
    checkpoint_key,
    write_csv,
    write_gt_snapshot,
    write_match_record,
    write_raw_predictions,
    write_visualizations,
)
from core.checkpoint_evaluation.checkpoint_loader import (
    CheckpointCandidate,
    discover_checkpoints,
    load_model_for_candidate,
)
from core.checkpoint_evaluation.dataset_audit import SplitPaths, audit_dataset_splits, sha256_of_path
from core.checkpoint_evaluation.dataset_loader import (
    ValidationGuardResult,
    check_validation_guard,
    load_validation_ground_truth,
)
from core.checkpoint_evaluation.mask_matching import match_instances
from core.checkpoint_evaluation.metrics import (
    DEFAULT_BOUNDARY_TOLERANCE_PX,
    aggregate_metrics,
    compute_image_metrics,
)
from core.checkpoint_evaluation.report_writer import (
    atomic_write_json,
    atomic_write_text,
    update_training_summary,
    write_checkpoint_metrics,
    write_per_image_metrics,
)
from core.checkpoint_evaluation.selector import (
    SELECTION_METRIC,
    SELECTION_RULE,
    TIE_TOLERANCE,
    select_best,
)
from core.checkpoint_export import sha256_of_file
from core.config import BOOK_ROOT, DEFAULT_SAM3_CHECKPOINT, SAM301_ROOT
from core.dataset_identity import resolve_split_identity

AUTO_EVALUATE_AFTER_TRAINING_DEFAULT = False
DEFAULTS_CONFIG_PATH = BOOK_ROOT / "config" / "checkpoint_evaluation.yaml"
MATCH_THRESHOLD = 0.5


@dataclass
class EvaluationConfig:
    run_dir: Path
    val_annotations: Path
    val_images: Path
    train_annotations: Path | None
    test_annotations: Path | None = None
    test_images: Path | None = None
    prompt: str = "book spine"
    score_threshold: float = 0.3
    confidence_threshold: float = 0.05
    min_area: int = 200
    dtype_mode: str = "bf16"
    device: str = "cuda"
    boundary_tolerance_px: int = DEFAULT_BOUNDARY_TOLERANCE_PX
    include_baseline: bool = True
    baseline_checkpoint: Path = DEFAULT_SAM3_CHECKPOINT
    checkpoint_names: list[str] | None = None
    max_images: int | None = None
    force: bool = False
    smoke: bool = False
    export_best: bool = False
    split: str = "all"  # validation | test | all

    def metric_affecting_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "score_threshold": self.score_threshold,
            "confidence_threshold": self.confidence_threshold,
            "min_area": self.min_area,
            "dtype_mode": self.dtype_mode,
            "device": self.device,
            "boundary_tolerance_px": self.boundary_tolerance_px,
            "max_images": self.max_images,
        }

    def config_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.metric_affecting_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


def load_defaults() -> dict[str, Any]:
    if DEFAULTS_CONFIG_PATH.is_file():
        from omegaconf import OmegaConf

        return dict(OmegaConf.to_container(OmegaConf.load(DEFAULTS_CONFIG_PATH)))
    return {}


def _default_test_paths(train_annotations: Path | None) -> tuple[Path | None, Path | None]:
    if train_annotations is None:
        return None, None
    dataset_root = Path(train_annotations).expanduser().resolve(strict=False).parents[1]
    test_ann = dataset_root / "test" / "annotations.json"
    test_img = dataset_root / "test" / "images"
    if test_ann.is_file() and test_img.is_dir():
        return test_ann, test_img
    return None, None


def _split_images_dir_from_annotations(annotations: Path, split_names: set[str]) -> Path | None:
    """Return the canonical sibling images dir for a COCO split, when present."""
    annotations = Path(annotations).expanduser().resolve(strict=False)
    split_dir = annotations.parent
    if split_dir.name.lower() not in split_names:
        return None
    images_dir = split_dir / "images"
    return images_dir if images_dir.is_dir() else None


def _resolve_default_validation_images(val_annotations: Path, configured_val_images: Path) -> Path:
    """Use the validation split folder as the default evaluation image root.

    Runtime configs from older experiments can point validation images at a broad
    raw-image pool. Checkpoint selection should be tied to the validation split:
    val/annotations.json plus val/images when that canonical folder exists.
    Explicit --val-images overrides are respected by config_from_run().
    """
    sibling_val_images = _split_images_dir_from_annotations(val_annotations, {"val", "validation"})
    return sibling_val_images or Path(configured_val_images)


def config_from_run(run_dir: Path, **overrides: Any) -> EvaluationConfig:
    run_dir = Path(run_dir).expanduser().resolve(strict=False)
    runtime_yaml = run_dir / "config" / "runtime_config.yaml"
    val_ann = overrides.pop("val_annotations", None)
    val_img = overrides.pop("val_images", None)
    explicit_val_images = val_img is not None
    test_ann = overrides.pop("test_annotations", None)
    test_img = overrides.pop("test_images", None)
    train_ann = None
    prompt = None
    if runtime_yaml.is_file():
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(runtime_yaml)
        train_ann = OmegaConf.select(cfg, "trainer.data.train.dataset.ann_file")
        if val_ann is None:
            val_ann = OmegaConf.select(cfg, "trainer.data.val.dataset.ann_file")
        if val_img is None:
            val_img = OmegaConf.select(cfg, "trainer.data.val.dataset.img_folder")
    summary_path = run_dir / "training_config_summary.json"
    if summary_path.is_file():
        try:
            prompt = json.loads(summary_path.read_text(encoding="utf-8")).get("resolved_training_prompt")
        except (OSError, json.JSONDecodeError):
            prompt = None
    if val_ann is None or val_img is None:
        raise ValueError(
            f"cannot determine validation dataset for {run_dir}: runtime_config.yaml missing "
            "or lacks trainer.data.val.dataset paths; pass --val-annotations/--val-images explicitly"
        )
    val_ann_path = Path(str(val_ann))
    val_img_path = Path(str(val_img))
    if not explicit_val_images:
        val_img_path = _resolve_default_validation_images(val_ann_path, val_img_path)
    if test_ann is None or test_img is None:
        inferred_test_ann, inferred_test_img = _default_test_paths(Path(str(train_ann)) if train_ann else None)
        test_ann = test_ann or inferred_test_ann
        test_img = test_img or inferred_test_img

    defaults = load_defaults()
    kwargs: dict[str, Any] = {
        "run_dir": run_dir,
        "val_annotations": val_ann_path,
        "val_images": val_img_path,
        "train_annotations": Path(str(train_ann)) if train_ann else None,
        "test_annotations": Path(str(test_ann)) if test_ann else None,
        "test_images": Path(str(test_img)) if test_img else None,
    }
    if prompt:
        kwargs["prompt"] = str(prompt)
    for key in (
        "prompt", "score_threshold", "confidence_threshold", "min_area", "dtype_mode",
        "boundary_tolerance_px", "include_baseline",
    ):
        if key in defaults and key not in overrides:
            kwargs[key] = defaults[key]
    kwargs.update(overrides)
    if "baseline_checkpoint" in kwargs and kwargs["baseline_checkpoint"] is not None:
        kwargs["baseline_checkpoint"] = Path(kwargs["baseline_checkpoint"])
    return EvaluationConfig(**kwargs)


def sam3_source_hash() -> str:
    parts = []
    for rel in ("sam3/train/trainer.py", "sam3/model_builder.py", "sam3/model/sam3_image_processor.py"):
        path = SAM301_ROOT / rel
        parts.append(sha256_of_file(path) if path.is_file() else f"missing:{rel}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class _Logger:
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = log_path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._handle.write(f"[{stamp}] {message}\n")

    def close(self) -> None:
        self._handle.close()


def _release_gpu(*objects: Any) -> None:
    import torch

    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _split_guard(
    split: str,
    annotations: Path,
    train_annotations: Path | None,
    val_annotations: Path | None,
) -> ValidationGuardResult:
    if split == "validation":
        return check_validation_guard(annotations, train_annotations)
    warnings: list[str] = []
    resolved = Path(annotations).expanduser().resolve(strict=False)
    if not resolved.is_file():
        return ValidationGuardResult(False, f"test annotations file does not exist: {resolved}", [], None)
    for forbidden_name, forbidden in (("training", train_annotations), ("validation", val_annotations)):
        if forbidden is not None and resolved == Path(forbidden).expanduser().resolve(strict=False):
            return ValidationGuardResult(
                False,
                f"test annotations path equals the {forbidden_name} annotations path; diagnostic test must be separate",
                [],
                None,
            )
    identity = resolve_split_identity("test", resolved)
    if not identity.matched:
        return ValidationGuardResult(False, f"test dataset is not registered: {identity.warning} (path: {resolved})", [], identity.to_dict())
    if not identity.human_reviewed:
        return ValidationGuardResult(
            False,
            f"test dataset {identity.dataset_id!r} is not human-reviewed ({identity.annotation_source}); refusing checkpoint diagnostics",
            [],
            identity.to_dict(),
        )
    if not identity.allowed_for_model_evaluation:
        warnings.append(
            f"dataset {identity.dataset_id!r} has allowed_for_model_evaluation=false; test metrics are diagnostic-only, "
            "do not affect validation best selection, and should not be quoted as final blind-test quality."
        )
    return ValidationGuardResult(True, None, warnings, identity.to_dict())


def _candidate_rows_for_alias(candidate: CheckpointCandidate) -> dict[str, Any]:
    return {
        "checkpoint_name": candidate.name,
        "checkpoint_path": str(candidate.path),
        "epoch": candidate.epoch,
        "sha256": candidate.sha256,
        "size_bytes": candidate.size_bytes,
        "is_baseline": candidate.is_baseline,
        "evaluation_status": "alias",
        "error_message": f"byte-identical alias of {candidate.alias_of}",
        "cache_hit": False,
    }


def _evaluate_one_checkpoint(
    split: str,
    split_dir: Path,
    candidate: CheckpointCandidate,
    ground_truth: list,
    config: EvaluationConfig,
    log,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from core.sam3_adapter import Sam3Adapter
    from core.checkpoint_evaluation.metrics import ImageMetrics

    ckpt_key = checkpoint_key(candidate.name, candidate.is_baseline)
    row: dict[str, Any] = {
        "split": split,
        "checkpoint_name": candidate.name,
        "checkpoint_path": str(candidate.path),
        "epoch": candidate.epoch,
        "sha256": candidate.sha256,
        "size_bytes": candidate.size_bytes,
        "is_baseline": candidate.is_baseline,
        "cache_hit": False,
        "error_message": "",
    }
    model = None
    adapter = None
    try:
        t0 = time.time()
        model, load_info = load_model_for_candidate(candidate, device=config.device)
        row["epoch"] = candidate.epoch
        row["checkpoint_type"] = load_info.get("checkpoint_type")
        adapter = Sam3Adapter.from_model(
            model,
            confidence_threshold=config.confidence_threshold,
            dtype_mode=config.dtype_mode,
            device=config.device,
            checkpoint_label=str(candidate.path),
        )
        log(f"{split}/{candidate.name}: model loaded ({load_info.get('checkpoint_type')}) in {time.time()-t0:.1f}s")
        per_image_metrics = []
        per_image_rows: list[dict[str, Any]] = []
        per_instance_rows: list[dict[str, Any]] = []
        human_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        inference_seconds = 0.0
        params = {
            "prompt": config.prompt,
            "score_threshold": config.score_threshold,
            "confidence_threshold": config.confidence_threshold,
            "min_area": config.min_area,
            "dtype_mode": config.dtype_mode,
            "device": config.device,
            "boundary_tolerance_px": config.boundary_tolerance_px,
        }
        for gt in ground_truth:
            raw_npz = None
            match_record = None
            vis_paths: dict[str, str] = {}
            try:
                t1 = time.time()
                instances = adapter.predict(
                    gt.path,
                    prompt=config.prompt,
                    score_threshold=config.score_threshold,
                    min_area=config.min_area,
                )
                elapsed = time.time() - t1
                inference_seconds += elapsed
                pred_masks = [instances.masks[i] for i in range(instances.count)]
                match = match_instances(gt.masks, pred_masks)
                raw_npz, _meta = write_raw_predictions(split_dir, split, ckpt_key, gt, instances, params, elapsed)
                match_record = write_match_record(split_dir, split, ckpt_key, candidate.name, gt, pred_masks, match, raw_npz)
                vis_paths = write_visualizations(split_dir, split, ckpt_key, gt, pred_masks, match)
                image_metrics = compute_image_metrics(
                    gt.file_name, gt.masks, pred_masks, match, config.boundary_tolerance_px
                )
                per_instance_rows.extend(build_instance_rows(split, candidate.name, gt, pred_masks, match, MATCH_THRESHOLD))
            except Exception as exc:  # noqa: BLE001
                log(f"{split}/{candidate.name}: image {gt.file_name} FAILED: {exc!r}")
                image_metrics = ImageMetrics(
                    image_name=gt.file_name,
                    gt_count=len(gt.masks),
                    prediction_count=0,
                    matching_method="none",
                    error=f"{type(exc).__name__}: {exc}",
                )
            per_image_metrics.append(image_metrics)
            row_for_image = {"split": split, "checkpoint_name": candidate.name, "image_id": gt.image_id, **image_metrics.to_row()}
            if raw_npz:
                row_for_image["raw_prediction_path"] = str(raw_npz)
            if match_record:
                row_for_image["match_record_path"] = str(match_record)
            per_image_rows.append(row_for_image)
            human_rows.append(
                {
                    "split": split,
                    "checkpoint": candidate.name,
                    "image_id": gt.image_id,
                    "image_path": str(gt.path),
                    "gt_count": len(gt.masks),
                    "prediction_count": image_metrics.prediction_count,
                    "mean_iou": image_metrics.to_row().get("mean_iou_all_gt"),
                    "minimum_iou": min(image_metrics.gt_ious) if image_metrics.gt_ious else "",
                    "miss_count": image_metrics.gt_count - image_metrics.matched_count_iou_50,
                    "fp_count": image_metrics.false_positive_count,
                    "area_ratio": float(sum(image_metrics.area_ratios) / len(image_metrics.area_ratios)) if image_metrics.area_ratios else "",
                    "gt_overlay_path": vis_paths.get("gt_overlay", ""),
                    "prediction_overlay_path": vis_paths.get("prediction_overlay", ""),
                    "matched_overlay_path": vis_paths.get("matched_overlay", ""),
                    "raw_prediction_path": str(raw_npz) if raw_npz else "",
                    "match_record_path": str(match_record) if match_record else "",
                    "review_status": "",
                    "reviewer_notes": "",
                }
            )
            if image_metrics.error or image_metrics.false_positive_count or image_metrics.gt_count - image_metrics.matched_count_iou_50:
                failure_rows.append({**human_rows[-1], "reason": image_metrics.error or "miss_or_false_positive"})
        aggregated = aggregate_metrics(per_image_metrics, config.boundary_tolerance_px)
        row.update(aggregated)
        row["inference_time_seconds"] = round(inference_seconds, 3)
        failed = aggregated["failed_image_count"]
        if failed == 0:
            row["evaluation_status"] = "completed"
        elif failed < len(per_image_metrics):
            row["evaluation_status"] = "partial"
            row["error_message"] = f"{failed}/{len(per_image_metrics)} images failed"
        else:
            row["evaluation_status"] = "failed"
            row["error_message"] = "all images failed"
        return row, per_image_rows, per_instance_rows, human_rows, failure_rows
    except Exception as exc:  # noqa: BLE001
        log(f"{split}/{candidate.name}: FAILED: {exc!r}\n{traceback.format_exc(limit=5)}")
        row["evaluation_status"] = "failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        return row, [], [], [], []
    finally:
        _release_gpu(adapter, model)


def _cache_key(split: str, candidate: CheckpointCandidate, dataset_sha: str, image_hash: str, config_sha: str, sam3_hash: str) -> str:
    payload = "|".join([split, candidate.sha256, dataset_sha, image_hash, config_sha, EVALUATOR_VERSION, sam3_hash])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _image_hash_for_ground_truth(ground_truth: list) -> str:
    parts = []
    for gt in ground_truth:
        try:
            parts.append(f"{gt.image_id}:{sha256_of_path(gt.path)}")
        except OSError:
            parts.append(f"{gt.image_id}:missing:{gt.path}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _write_split_artifacts(
    split_dir: Path,
    rows: list[dict[str, Any]],
    per_image_rows: list[dict[str, Any]],
    per_instance_rows: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> None:
    write_checkpoint_metrics(split_dir, rows)
    write_per_image_metrics(split_dir, per_image_rows)
    write_csv(split_dir / "per_instance_metrics.csv", per_instance_rows, INSTANCE_COLUMNS)
    human_columns = [
        "split", "checkpoint", "image_id", "image_path", "gt_count", "prediction_count",
        "mean_iou", "minimum_iou", "miss_count", "fp_count", "area_ratio",
        "gt_overlay_path", "prediction_overlay_path", "matched_overlay_path",
        "raw_prediction_path", "match_record_path", "review_status", "reviewer_notes",
    ]
    index_path = split_dir / "human_review_index.csv"
    if index_path.is_file():
        try:
            import csv
            from io import StringIO

            existing_text = index_path.read_text(encoding="utf-8")
            existing = list(csv.DictReader(StringIO(existing_text)))
            notes = {
                (r.get("checkpoint"), r.get("image_id")): (r.get("review_status", ""), r.get("reviewer_notes", ""))
                for r in existing
            }
            for row in human_rows:
                key = (row.get("checkpoint"), str(row.get("image_id")))
                if key in notes:
                    row["review_status"], row["reviewer_notes"] = notes[key]
        except Exception:
            pass
    write_csv(index_path, human_rows, human_columns)
    write_csv(split_dir / "human_review_index.auto.csv", human_rows, human_columns)
    write_csv(split_dir / "failure_cases.csv", failure_rows, human_columns + ["reason"])


def _evaluate_split(
    split: str,
    annotations: Path,
    images_dir: Path,
    config: EvaluationConfig,
    candidates: list[CheckpointCandidate],
    sam3_hash: str,
    log,
) -> dict[str, Any]:
    evaluation_dir = Path(config.run_dir) / "evaluation"
    split_dir = evaluation_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    guard = _split_guard(split, annotations, config.train_annotations, config.val_annotations)
    for warning in guard.warnings:
        log(f"WARNING {split}: {warning}")
    if not guard.ok:
        summary = {
            "split": split,
            "status": "blocked",
            "reason": guard.reason,
            "dataset_path": str(annotations),
            "dataset_identity": guard.identity,
        }
        if split == "validation":
            atomic_write_json(split_dir.parent / "best_checkpoint.json", {"status": "blocked", "reason": guard.reason})
        atomic_write_json(split_dir / "evaluation_summary.json", summary)
        return summary

    dataset_sha = sha256_of_file(annotations)
    ground_truth = load_validation_ground_truth(annotations, images_dir, max_images=config.max_images)
    image_hash = _image_hash_for_ground_truth(ground_truth)
    write_gt_snapshot(split_dir, split, ground_truth, annotations, dataset_sha)
    config_sha = config.config_sha256()
    cache_dir = evaluation_dir / "cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"{split}: {len(ground_truth)} images, {sum(len(g.masks) for g in ground_truth)} GT instances sha={dataset_sha[:12]}")

    rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    per_instance_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.discovery_error:
            rows.append(
                {
                    "split": split,
                    "checkpoint_name": candidate.name,
                    "checkpoint_path": str(candidate.path),
                    "epoch": candidate.epoch,
                    "sha256": candidate.sha256,
                    "is_baseline": candidate.is_baseline,
                    "evaluation_status": "failed",
                    "error_message": candidate.discovery_error,
                    "cache_hit": False,
                }
            )
            continue
        if candidate.alias_of:
            log(f"{split}/{candidate.name}: alias of {candidate.alias_of}, not re-evaluated")
            rows.append(_candidate_rows_for_alias(candidate))
            continue
        key = _cache_key(split, candidate, dataset_sha, image_hash, config_sha, sam3_hash)
        cache_path = cache_dir / f"{key}.json"
        if cache_path.is_file() and not config.force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                row = cached["row"]
                row.update({"cache_hit": True, "checkpoint_name": candidate.name, "checkpoint_path": str(candidate.path)})
                rows.append(row)
                per_image_rows.extend(cached.get("per_image", []))
                per_instance_rows.extend(cached.get("per_instance", []))
                human_rows.extend(cached.get("human_review", []))
                failure_rows.extend(cached.get("failure_cases", []))
                log(f"{split}/{candidate.name}: cache hit ({key[:12]})")
                continue
            except (json.JSONDecodeError, KeyError) as exc:
                log(f"{split}/{candidate.name}: cache unreadable ({exc}); re-evaluating")
        row, image_rows, instance_rows, review_rows, fail_rows = _evaluate_one_checkpoint(
            split, split_dir, candidate, ground_truth, config, log
        )
        rows.append(row)
        per_image_rows.extend(image_rows)
        per_instance_rows.extend(instance_rows)
        human_rows.extend(review_rows)
        failure_rows.extend(fail_rows)
        if row.get("evaluation_status") in {"completed", "partial"}:
            atomic_write_json(
                cache_path,
                {
                    "row": row,
                    "per_image": image_rows,
                    "per_instance": instance_rows,
                    "human_review": review_rows,
                    "failure_cases": fail_rows,
                    "cache_key_inputs": {
                        "split": split,
                        "checkpoint_sha256": candidate.sha256,
                        "annotations_sha256": dataset_sha,
                        "image_hash": image_hash,
                        "eval_config_sha256": config_sha,
                        "evaluator_version": EVALUATOR_VERSION,
                        "sam3_source_hash": sam3_hash,
                    },
                },
            )
        log(
            f"{split}/{candidate.name}: {row.get('evaluation_status')} "
            f"mean_iou_all_gt={row.get('mean_iou_all_gt')} bf1={row.get('mean_boundary_f1_all_gt')}"
        )

    _write_split_artifacts(split_dir, rows, per_image_rows, per_instance_rows, human_rows, failure_rows)
    evaluated = [r for r in rows if not r.get("is_baseline") and r.get("evaluation_status") != "alias"]
    n_completed = sum(1 for r in evaluated if r.get("evaluation_status") == "completed")
    n_failed = sum(1 for r in evaluated if r.get("evaluation_status") == "failed")
    status = "completed" if n_completed and n_failed == 0 else ("partial" if n_completed else "failed")
    summary = {
        "split": split,
        "status": status,
        "dataset_path": str(annotations),
        "images_dir": str(images_dir),
        "annotations_sha256": dataset_sha,
        "image_hash": image_hash,
        "dataset_identity": guard.identity,
        "guard_warnings": guard.warnings,
        "image_count": len(ground_truth),
        "gt_count": sum(len(g.masks) for g in ground_truth),
        "checkpoint_count": len(evaluated),
        "completed_count": n_completed,
        "failed_count": n_failed,
        "files": {
            "checkpoint_metrics_csv": str(split_dir / "checkpoint_metrics.csv"),
            "checkpoint_metrics_json": str(split_dir / "checkpoint_metrics.json"),
            "per_image_metrics_csv": str(split_dir / "per_image_metrics.csv"),
            "per_instance_metrics_csv": str(split_dir / "per_instance_metrics.csv"),
            "human_review_index_csv": str(split_dir / "human_review_index.csv"),
            "gt_snapshot_json": str(split_dir / "gt_snapshot.json"),
            "raw_predictions_dir": str(split_dir / "raw_predictions"),
            "match_records_dir": str(split_dir / "match_records"),
            "visualizations_dir": str(split_dir / "visualizations"),
        },
    }
    atomic_write_json(split_dir / "evaluation_summary.json", summary)
    return summary


def _discover_candidates(config: EvaluationConfig) -> list[CheckpointCandidate]:
    candidates = discover_checkpoints(config.run_dir)
    if config.checkpoint_names:
        wanted = set(config.checkpoint_names)
        candidates = [c for c in candidates if c.name in wanted]
    if config.include_baseline and Path(config.baseline_checkpoint).is_file():
        candidates.append(
            CheckpointCandidate(
                name=Path(config.baseline_checkpoint).name + " (baseline)",
                path=Path(config.baseline_checkpoint),
                epoch=None,
                sha256=sha256_of_file(config.baseline_checkpoint),
                size_bytes=Path(config.baseline_checkpoint).stat().st_size,
                is_baseline=True,
            )
        )
    return candidates


def _write_validation_legacy_files(evaluation_dir: Path) -> None:
    validation_dir = evaluation_dir / "validation"
    for name in ("checkpoint_metrics.json", "checkpoint_metrics.csv", "per_image_metrics.csv"):
        src = validation_dir / name
        if src.is_file():
            atomic_write_text(evaluation_dir / name, src.read_text(encoding="utf-8"))
    summary = validation_dir / "evaluation_summary.json"
    if summary.is_file():
        atomic_write_text(evaluation_dir / "validation_summary.json", summary.read_text(encoding="utf-8"))


def _write_evaluation_config(config: EvaluationConfig, evaluation_dir: Path, sam3_hash: str, started_at: str) -> None:
    from omegaconf import OmegaConf

    record = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
        "evaluator_version": EVALUATOR_VERSION,
        "sam3_source_hash": sam3_hash,
        "eval_config_sha256": config.config_sha256(),
        "matching_method": "hungarian_max_total_iou",
        "match_threshold": MATCH_THRESHOLD,
        "started_at": started_at,
        "test_warning": (
            "best checkpoint is selected by validation only; test results are diagnostic-only and do not affect model selection"
        ),
    }
    atomic_write_text(evaluation_dir / "evaluation_config.yaml", OmegaConf.to_yaml(OmegaConf.create(record)))


def _select_and_write_best(config: EvaluationConfig, evaluation_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None, bool | None]:
    metrics_path = evaluation_dir / "validation" / "checkpoint_metrics.json"
    rows = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else []
    selection = select_best(rows)
    baseline_row = next((r for r in rows if r.get("is_baseline")), None)
    best = selection.get("best")
    improved = None
    if best is not None and baseline_row is not None and baseline_row.get("evaluation_status") == "completed":
        base_iou = baseline_row.get("mean_iou_all_gt")
        if isinstance(base_iou, (int, float)) and isinstance(best.get("mean_iou_all_gt"), (int, float)):
            improved = best["mean_iou_all_gt"] > base_iou
    best_payload = {
        "status": selection["status"],
        "best_checkpoint": best.get("checkpoint_path") if best else None,
        "best_checkpoint_name": best.get("checkpoint_name") if best else None,
        "best_epoch": best.get("epoch") if best else None,
        "selection_metric": SELECTION_METRIC,
        "selection_rule": SELECTION_RULE,
        "tie_tolerance": TIE_TOLERANCE,
        "selection_reason": selection.get("reason"),
        "mean_iou_all_gt": best.get("mean_iou_all_gt") if best else None,
        "mean_boundary_f1_all_gt": best.get("mean_boundary_f1_all_gt") if best else None,
        "miss_rate_iou_50": best.get("miss_rate_iou_50") if best else None,
        "false_positive_per_image": best.get("false_positive_per_image") if best else None,
        "validation_dataset": str(config.val_annotations),
        "evaluated_checkpoints": [r["checkpoint_name"] for r in rows],
        "baseline_checkpoint": str(config.baseline_checkpoint) if config.include_baseline else None,
        "baseline_metrics": baseline_row,
        "finetuned_improved_over_baseline": improved,
        "matching_method": "hungarian_max_total_iou",
        "smoke": config.smoke,
        "test_does_not_affect_model_selection": True,
    }
    atomic_write_json(evaluation_dir / "best_checkpoint.json", best_payload)
    return best_payload, best, improved


def _write_test_comparison(evaluation_dir: Path, val_best: dict[str, Any] | None) -> dict[str, Any] | None:
    metrics_path = evaluation_dir / "test" / "checkpoint_metrics.json"
    if not metrics_path.is_file():
        return None
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    eligible = [
        r for r in rows
        if r.get("evaluation_status") == "completed" and not r.get("is_baseline") and isinstance(r.get("mean_iou_all_gt"), (int, float))
    ]
    top = max(eligible, key=lambda r: r["mean_iou_all_gt"]) if eligible else None
    payload = {
        "diagnostic_only": True,
        "does_not_affect_model_selection": True,
        "warning": (
            "best checkpoint is decided by validation set only. Because all checkpoints are displayed on test, "
            "this test should no longer be treated as a fully blind final test set."
        ),
        "val_selected_checkpoint": val_best.get("checkpoint_name") if val_best else None,
        "val_selected_epoch": val_best.get("epoch") if val_best else None,
        "test_highest_metric_checkpoint": top.get("checkpoint_name") if top else None,
        "test_highest_metric_epoch": top.get("epoch") if top else None,
        "test_highest_metric_mean_iou_all_gt": top.get("mean_iou_all_gt") if top else None,
    }
    atomic_write_json(evaluation_dir / "test_checkpoint_comparison.json", payload)
    return payload


def evaluate_run(config: EvaluationConfig) -> dict[str, Any]:
    run_dir = Path(config.run_dir)
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    log = _Logger(evaluation_dir / "evaluation.log")
    started_at = datetime.now(timezone.utc).isoformat()
    sam3_hash = sam3_source_hash()
    _write_evaluation_config(config, evaluation_dir, sam3_hash, started_at)
    log(f"=== evaluation start (evaluator {EVALUATOR_VERSION}, split={config.split}, smoke={config.smoke}) ===")

    candidates = _discover_candidates(config)
    if not candidates:
        summary = {
            "status": "blocked",
            "reason": f"no checkpoint files found under {run_dir / 'checkpoints'}",
            "started_at": started_at,
            "run_dir": str(run_dir),
        }
        atomic_write_json(evaluation_dir / "evaluation_summary.json", summary)
        update_training_summary(run_dir, {"status": "blocked", "reason": summary["reason"]})
        log.close()
        return summary
    log("candidates: " + ", ".join(f"{c.name}(e={c.epoch},alias={c.alias_of})" for c in candidates))

    validation_summary = None
    test_summary = None
    best_payload = None
    best = None
    improved = None
    export_status: dict[str, Any] = {"requested": bool(config.export_best)}

    try:
        split_names = {config.split}
        if config.split == "all":
            split_names = {"validation", "test"}
        if "validation" in split_names:
            validation_summary = _evaluate_split(
                "validation", config.val_annotations, config.val_images, config, candidates, sam3_hash, log
            )
            if validation_summary["status"] not in {"blocked", "failed"}:
                best_payload, best, improved = _select_and_write_best(config, evaluation_dir)
                _write_validation_legacy_files(evaluation_dir)
                if config.export_best and best is not None:
                    try:
                        from core.checkpoint_export import export_inference_checkpoint

                        output_path = run_dir / "checkpoints" / "inference_best.pt"
                        result = export_inference_checkpoint(
                            trainer_checkpoint_path=best["checkpoint_path"],
                            output_path=output_path,
                            base_checkpoint_path=config.baseline_checkpoint,
                            overwrite=True,
                            dataset_identity=validation_summary.get("dataset_identity"),
                        )
                        export_status.update(
                            {
                                "status": "completed",
                                "output_path": result.output_path,
                                "output_sha256": result.output_sha256,
                                "source_checkpoint": best["checkpoint_path"],
                                "source_sha256": best.get("sha256"),
                                "exported_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        export_status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                        log(f"export best FAILED (evaluation results unaffected): {exc!r}")
        if "test" in split_names:
            if config.test_annotations is None or config.test_images is None:
                test_summary = {
                    "split": "test",
                    "status": "blocked",
                    "reason": "test dataset was not found; pass --test-annotations/--test-images or add sibling test/ split",
                }
                atomic_write_json(evaluation_dir / "test" / "evaluation_summary.json", test_summary)
            elif config.split == "all" and (validation_summary or {}).get("status") in {"blocked", "failed"}:
                test_summary = {
                    "split": "test",
                    "status": "blocked",
                    "reason": "validation failed, so the full validation->test flow stopped before test",
                }
                atomic_write_json(evaluation_dir / "test" / "evaluation_summary.json", test_summary)
            else:
                test_summary = _evaluate_split(
                    "test", config.test_annotations, config.test_images, config, candidates, sam3_hash, log
                )
                _write_test_comparison(evaluation_dir, best)
        split_paths = []
        if config.train_annotations:
            split_paths.append(SplitPaths("train", config.train_annotations, config.train_annotations.parent / "images"))
        split_paths.append(SplitPaths("validation", config.val_annotations, config.val_images))
        if config.test_annotations and config.test_images:
            split_paths.append(SplitPaths("test", config.test_annotations, config.test_images))
        audit = audit_dataset_splits(split_paths, evaluation_dir)
    finally:
        log(f"=== evaluation end ===")
        log.close()

    statuses = [s.get("status") for s in (validation_summary, test_summary) if s]
    if statuses and all(s == "completed" for s in statuses):
        status = "completed"
    elif any(s == "completed" for s in statuses):
        status = "partial"
    elif statuses and all(s == "blocked" for s in statuses):
        status = "blocked"
    else:
        status = "failed" if statuses else "blocked"

    summary = {
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "smoke": config.smoke,
        "run_dir": str(run_dir),
        "validation": validation_summary,
        "test": {
            **(test_summary or {"status": "not_requested"}),
            "does_not_affect_model_selection": True,
        },
        "best": {k: best.get(k) for k in ("checkpoint_name", "checkpoint_path", "epoch", "mean_iou_all_gt")} if best else None,
        "selection_reason": (best_payload or {}).get("selection_reason"),
        "finetuned_improved_over_baseline": improved,
        "export_best": export_status,
        "dataset_split_audit": str(evaluation_dir / "dataset_split_audit.json"),
        "files": {
            "evaluation_config_yaml": str(evaluation_dir / "evaluation_config.yaml"),
            "best_checkpoint_json": str(evaluation_dir / "best_checkpoint.json"),
            "validation_summary_json": str(evaluation_dir / "validation" / "evaluation_summary.json"),
            "test_summary_json": str(evaluation_dir / "test" / "evaluation_summary.json"),
            "test_checkpoint_comparison_json": str(evaluation_dir / "test_checkpoint_comparison.json"),
            "dataset_split_audit_json": str(evaluation_dir / "dataset_split_audit.json"),
            "dataset_split_audit_csv": str(evaluation_dir / "dataset_split_audit.csv"),
            "log": str(evaluation_dir / "evaluation.log"),
        },
    }
    if (validation_summary or {}).get("reason"):
        summary["reason"] = validation_summary["reason"]
    atomic_write_json(evaluation_dir / "evaluation_summary.json", summary)
    training_block = {
        "status": status,
        "evaluation_directory": str(evaluation_dir),
        "best_checkpoint": (best_payload or {}).get("best_checkpoint"),
        "best_epoch": (best_payload or {}).get("best_epoch"),
        "selection_metric": SELECTION_METRIC,
        "summary_path": str(evaluation_dir / "evaluation_summary.json"),
        "validation": {
            "status": (validation_summary or {}).get("status"),
            "dataset_path": str(config.val_annotations),
            "summary_path": str(evaluation_dir / "validation" / "evaluation_summary.json"),
            "best_checkpoint": (best_payload or {}).get("best_checkpoint"),
            "best_epoch": (best_payload or {}).get("best_epoch"),
        },
        "test": {
            "status": (test_summary or {}).get("status", "not_requested"),
            "dataset_path": str(config.test_annotations) if config.test_annotations else None,
            "summary_path": str(evaluation_dir / "test" / "evaluation_summary.json"),
            "evaluated_checkpoints": [
                r.get("checkpoint_name")
                for r in json.loads((evaluation_dir / "test" / "checkpoint_metrics.json").read_text(encoding="utf-8"))
            ] if (evaluation_dir / "test" / "checkpoint_metrics.json").is_file() else [],
            "does_not_affect_model_selection": True,
            "val_selected_checkpoint": (best_payload or {}).get("best_checkpoint"),
            "test_highest_metric_checkpoint_diagnostic_only": (
                json.loads((evaluation_dir / "test_checkpoint_comparison.json").read_text(encoding="utf-8")).get("test_highest_metric_checkpoint")
                if (evaluation_dir / "test_checkpoint_comparison.json").is_file() else None
            ),
        },
    }
    error = update_training_summary(run_dir, training_block)
    if error:
        summary["training_summary_update_error"] = error
        atomic_write_json(evaluation_dir / "evaluation_summary.json", summary)
    return summary
