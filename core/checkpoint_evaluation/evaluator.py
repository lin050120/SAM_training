"""Checkpoint evaluation orchestrator.

Pipeline per run directory:
  validation guard (registered human-reviewed val split, never train data)
  -> checkpoint discovery (epoch-sorted, checkpoint.pt alias dedupe, sha256)
  -> per checkpoint: fresh model, fixed inference conditions, per-image
     prediction -> Hungarian one-to-one matching -> mask metrics
  -> hierarchical best selection (selector.py)
  -> reports under <run_dir>/evaluation/ + atomic training_summary.json block
  -> optional export of the best trainer checkpoint to inference_best.pt

Fixed inference conditions (identical for every checkpoint, recorded in
evaluation_config.yaml): validation images+GT, prompt, score/confidence thresholds,
min area, dtype mode, device, mask post-processing (all inherited from the single
shared Sam3Adapter.predict path), SAM3 source hash and evaluator version.

Cache: <run_dir>/evaluation/cache/<key>.json where key =
sha256(checkpoint_sha256 | val_annotations_sha256 | eval_config_sha256 |
EVALUATOR_VERSION | sam3_source_hash). File names are never trusted as cache keys.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.checkpoint_evaluation import EVALUATOR_VERSION
from core.checkpoint_evaluation.checkpoint_loader import (
    CheckpointCandidate,
    discover_checkpoints,
    load_model_for_candidate,
)
from core.checkpoint_evaluation.dataset_loader import (
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

# First-stage default: evaluation is manual-trigger only. When this is flipped to
# true (config/checkpoint_evaluation.yaml), the training flow may call
# evaluate_run() after a successful formal training whose validation guard passes.
AUTO_EVALUATE_AFTER_TRAINING_DEFAULT = False

DEFAULTS_CONFIG_PATH = BOOK_ROOT / "config" / "checkpoint_evaluation.yaml"


@dataclass
class EvaluationConfig:
    run_dir: Path
    val_annotations: Path
    val_images: Path
    train_annotations: Path | None
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

    def metric_affecting_dict(self) -> dict[str, Any]:
        """The subset of the config that can change metric OUTCOMES (cache key input).

        run_dir/force/export_best/include_baseline/checkpoint_names do not change a
        single checkpoint's numbers and are excluded on purpose.
        """
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


def config_from_run(run_dir: Path, **overrides: Any) -> EvaluationConfig:
    """Build the evaluation config with defaults read from the run itself."""
    run_dir = Path(run_dir).expanduser().resolve(strict=False)
    runtime_yaml = run_dir / "config" / "runtime_config.yaml"
    val_ann = overrides.pop("val_annotations", None)
    val_img = overrides.pop("val_images", None)
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
    defaults = load_defaults()
    kwargs: dict[str, Any] = {
        "run_dir": run_dir,
        "val_annotations": Path(str(val_ann)),
        "val_images": Path(str(val_img)),
        "train_annotations": Path(str(train_ann)) if train_ann else None,
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
    """Pin the SAM3 source state relevant to evaluation outcomes."""
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


def _evaluate_one_checkpoint(
    candidate: CheckpointCandidate,
    ground_truth: list,
    config: EvaluationConfig,
    log,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Returns (checkpoint_row, per_image_rows). Never raises for per-checkpoint failures."""
    from core.sam3_adapter import Sam3Adapter

    row: dict[str, Any] = {
        "checkpoint_name": candidate.name,
        "checkpoint_path": str(candidate.path),
        "epoch": candidate.epoch,
        "sha256": candidate.sha256,
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
        adapter = Sam3Adapter.from_model(
            model,
            confidence_threshold=config.confidence_threshold,
            dtype_mode=config.dtype_mode,
            device=config.device,
            checkpoint_label=str(candidate.path),
        )
        log(f"{candidate.name}: model loaded ({load_info.get('checkpoint_type')}) in {time.time()-t0:.1f}s")
        per_image_metrics = []
        inference_seconds = 0.0
        for gt in ground_truth:
            try:
                t1 = time.time()
                instances = adapter.predict(
                    gt.path,
                    prompt=config.prompt,
                    score_threshold=config.score_threshold,
                    min_area=config.min_area,
                )
                inference_seconds += time.time() - t1
                pred_masks = [instances.masks[i] for i in range(instances.count)]
                match = match_instances(gt.masks, pred_masks)
                image_metrics = compute_image_metrics(
                    gt.file_name, gt.masks, pred_masks, match, config.boundary_tolerance_px
                )
            except Exception as exc:  # noqa: BLE001 - single-image failure is recorded, not fatal
                log(f"{candidate.name}: image {gt.file_name} FAILED: {exc!r}")
                from core.checkpoint_evaluation.metrics import ImageMetrics

                image_metrics = ImageMetrics(
                    image_name=gt.file_name,
                    gt_count=len(gt.masks),
                    prediction_count=0,
                    matching_method="none",
                    error=f"{type(exc).__name__}: {exc}",
                )
            per_image_metrics.append(image_metrics)
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
        per_image_rows = [
            {"checkpoint_name": candidate.name, **m.to_row()} for m in per_image_metrics
        ]
        return row, per_image_rows
    except Exception as exc:  # noqa: BLE001 - checkpoint-level failure is recorded, not fatal
        log(f"{candidate.name}: FAILED: {exc!r}\n{traceback.format_exc(limit=5)}")
        row["evaluation_status"] = "failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        return row, []
    finally:
        _release_gpu(adapter, model)


def evaluate_run(config: EvaluationConfig) -> dict[str, Any]:
    run_dir = Path(config.run_dir)
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = evaluation_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    log = _Logger(evaluation_dir / "evaluation.log")
    started_at = datetime.now(timezone.utc).isoformat()
    log(f"=== evaluation start (evaluator {EVALUATOR_VERSION}, smoke={config.smoke}) ===")

    def _finish(summary: dict[str, Any]) -> dict[str, Any]:
        atomic_write_json(evaluation_dir / "evaluation_summary.json", summary)
        summary_block = {
            "status": summary["status"],
            "evaluation_directory": str(evaluation_dir),
            "best_checkpoint": (summary.get("best") or {}).get("checkpoint_path"),
            "best_epoch": (summary.get("best") or {}).get("epoch"),
            "selection_metric": SELECTION_METRIC,
            "summary_path": str(evaluation_dir / "evaluation_summary.json"),
            "smoke": config.smoke,
        }
        if summary.get("reason"):
            summary_block["reason"] = summary["reason"]
        error = update_training_summary(run_dir, summary_block)
        if error:
            log(f"training_summary update FAILED (evaluation results unaffected): {error}")
            summary["training_summary_update_error"] = error
            atomic_write_json(evaluation_dir / "evaluation_summary.json", summary)
        log(f"=== evaluation end: {summary['status']} ===")
        log.close()
        return summary

    # ---- validation guard (fail closed) ----
    guard = check_validation_guard(config.val_annotations, config.train_annotations)
    for warning in guard.warnings:
        log(f"WARNING: {warning}")
    if not guard.ok:
        log(f"BLOCKED: {guard.reason}")
        blocked = {
            "status": "blocked",
            "reason": guard.reason,
            "validation_dataset": str(config.val_annotations),
            "dataset_identity": guard.identity,
            "started_at": started_at,
            "smoke": config.smoke,
        }
        atomic_write_json(evaluation_dir / "best_checkpoint.json", {"status": "blocked", "reason": guard.reason})
        return _finish(blocked)

    # ---- fixed inputs ----
    val_sha = sha256_of_file(config.val_annotations)
    sam3_hash = sam3_source_hash()
    config_sha = config.config_sha256()
    ground_truth = load_validation_ground_truth(
        config.val_annotations, config.val_images, max_images=config.max_images
    )
    log(
        f"validation: {len(ground_truth)} images, "
        f"{sum(len(g.masks) for g in ground_truth)} GT instances "
        f"(dataset {guard.identity.get('dataset_id') if guard.identity else '?'} sha {val_sha[:12]})"
    )

    # evaluation_config.yaml — every fixed condition, for the record
    from omegaconf import OmegaConf

    config_record = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
        "evaluator_version": EVALUATOR_VERSION,
        "sam3_source_hash": sam3_hash,
        "val_annotations_sha256": val_sha,
        "eval_config_sha256": config_sha,
        "matching_method": "hungarian_max_total_iou",
        "dataset_identity": guard.identity,
        "guard_warnings": guard.warnings,
        "started_at": started_at,
    }
    atomic_write_text(evaluation_dir / "evaluation_config.yaml", OmegaConf.to_yaml(OmegaConf.create(config_record)))

    # ---- checkpoint discovery ----
    candidates = discover_checkpoints(run_dir)
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
    if not candidates:
        return _finish(
            {
                "status": "blocked",
                "reason": f"no checkpoint files found under {run_dir / 'checkpoints'}",
                "started_at": started_at,
                "smoke": config.smoke,
            }
        )
    log("candidates: " + ", ".join(f"{c.name}(e={c.epoch},alias={c.alias_of})" for c in candidates))

    # ---- evaluate ----
    rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    cancelled = False
    try:
        for candidate in candidates:
            if candidate.discovery_error:
                rows.append(
                    {
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
                log(f"{candidate.name}: alias of {candidate.alias_of} (same sha256), not re-evaluated")
                rows.append(
                    {
                        "checkpoint_name": candidate.name,
                        "checkpoint_path": str(candidate.path),
                        "epoch": candidate.epoch,
                        "sha256": candidate.sha256,
                        "is_baseline": False,
                        "evaluation_status": "alias",
                        "error_message": f"byte-identical alias of {candidate.alias_of}",
                        "cache_hit": False,
                    }
                )
                continue
            cache_key = hashlib.sha256(
                "|".join([candidate.sha256, val_sha, config_sha, EVALUATOR_VERSION, sam3_hash]).encode("utf-8")
            ).hexdigest()
            cache_path = cache_dir / f"{cache_key}.json"
            if cache_path.is_file() and not config.force:
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    row = cached["row"]
                    row["cache_hit"] = True
                    # identity fields may legitimately differ (renamed file): refresh them
                    row["checkpoint_name"] = candidate.name
                    row["checkpoint_path"] = str(candidate.path)
                    rows.append(row)
                    per_image_rows.extend(cached.get("per_image", []))
                    log(f"{candidate.name}: cache hit ({cache_key[:12]})")
                    continue
                except (json.JSONDecodeError, KeyError) as exc:
                    log(f"{candidate.name}: cache unreadable ({exc}); re-evaluating")
            row, image_rows = _evaluate_one_checkpoint(candidate, ground_truth, config, log)
            rows.append(row)
            per_image_rows.extend(image_rows)
            if row["evaluation_status"] in {"completed", "partial"}:
                atomic_write_json(cache_path, {"row": row, "per_image": image_rows, "cache_key_inputs": {
                    "checkpoint_sha256": candidate.sha256,
                    "val_annotations_sha256": val_sha,
                    "eval_config_sha256": config_sha,
                    "evaluator_version": EVALUATOR_VERSION,
                    "sam3_source_hash": sam3_hash,
                }})
            log(
                f"{candidate.name}: {row['evaluation_status']} "
                f"mean_iou_all_gt={row.get('mean_iou_all_gt')} "
                f"bf1={row.get('mean_boundary_f1_all_gt')} miss={row.get('miss_rate_iou_50')}"
            )
    except KeyboardInterrupt:
        cancelled = True
        log("INTERRUPTED — writing partial results")

    write_checkpoint_metrics(evaluation_dir, rows)
    write_per_image_metrics(evaluation_dir, per_image_rows)

    # ---- selection ----
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
        "validation_manifest_sha256": val_sha,
        "evaluation_config_sha256": config_sha,
        "evaluated_checkpoints": [r["checkpoint_name"] for r in rows],
        "baseline_checkpoint": str(config.baseline_checkpoint) if config.include_baseline else None,
        "baseline_metrics": baseline_row,
        "finetuned_improved_over_baseline": improved,
        "matching_method": "hungarian_max_total_iou",
        "smoke": config.smoke,
        "guard_warnings": guard.warnings,
    }
    atomic_write_json(evaluation_dir / "best_checkpoint.json", best_payload)

    # ---- optional export of best ----
    export_status: dict[str, Any] = {"requested": bool(config.export_best)}
    if config.export_best and best is not None:
        try:
            from core.checkpoint_export import export_inference_checkpoint

            output_path = run_dir / "checkpoints" / "inference_best.pt"
            result = export_inference_checkpoint(
                trainer_checkpoint_path=best["checkpoint_path"],
                output_path=output_path,
                base_checkpoint_path=config.baseline_checkpoint,
                overwrite=True,
                dataset_identity=guard.identity,
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
            log(f"export best -> {result.output_path}")
        except Exception as exc:  # noqa: BLE001 - export failure must not invalidate evaluation
            export_status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            log(f"export best FAILED (evaluation results unaffected): {exc!r}")

    # ---- final status ----
    evaluated = [r for r in rows if not r.get("is_baseline") and r.get("evaluation_status") != "alias"]
    n_completed = sum(1 for r in evaluated if r["evaluation_status"] == "completed")
    n_failed = sum(1 for r in evaluated if r["evaluation_status"] == "failed")
    if cancelled:
        status = "cancelled"
    elif selection["status"] == "blocked":
        status = "failed" if n_completed == 0 else "partial"
    elif n_failed or any(r["evaluation_status"] == "partial" for r in evaluated):
        status = "partial"
    else:
        status = "completed"

    summary = {
        "status": status,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "smoke": config.smoke,
        "run_dir": str(run_dir),
        "validation_dataset": str(config.val_annotations),
        "validation_manifest_sha256": val_sha,
        "dataset_identity": guard.identity,
        "guard_warnings": guard.warnings,
        "checkpoint_count": len(evaluated),
        "completed_count": n_completed,
        "failed_count": n_failed,
        "best": {k: best.get(k) for k in ("checkpoint_name", "checkpoint_path", "epoch", "mean_iou_all_gt")} if best else None,
        "selection_reason": selection.get("reason"),
        "finetuned_improved_over_baseline": improved,
        "export_best": export_status,
        "files": {
            "checkpoint_metrics_csv": str(evaluation_dir / "checkpoint_metrics.csv"),
            "checkpoint_metrics_json": str(evaluation_dir / "checkpoint_metrics.json"),
            "per_image_metrics_csv": str(evaluation_dir / "per_image_metrics.csv"),
            "best_checkpoint_json": str(evaluation_dir / "best_checkpoint.json"),
            "evaluation_config_yaml": str(evaluation_dir / "evaluation_config.yaml"),
            "log": str(evaluation_dir / "evaluation.log"),
        },
    }
    return _finish(summary)
