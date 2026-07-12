"""Unit tests for the checkpoint evaluation module (CPU-only, synthetic masks)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.checkpoint_evaluation.mask_matching import iou_matrix, match_instances  # noqa: E402
from core.checkpoint_evaluation.metrics import (  # noqa: E402
    aggregate_metrics,
    boundary_f1,
    compute_image_metrics,
)
from core.checkpoint_evaluation.selector import select_best  # noqa: E402
from core.npz_io import InstanceSet  # noqa: E402


def _rect(h, w, y0, y1, x0, x1):
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class MaskMatchingTest(unittest.TestCase):
    def test_identical_masks_iou_1(self) -> None:
        gt = [_rect(50, 50, 10, 30, 10, 30)]
        pred = [_rect(50, 50, 10, 30, 10, 30)]
        result = match_instances(gt, pred)
        self.assertEqual(len(result.matches), 1)
        self.assertAlmostEqual(result.matches[0][2], 1.0)

    def test_disjoint_masks_iou_0_and_not_matched(self) -> None:
        gt = [_rect(50, 50, 0, 10, 0, 10)]
        pred = [_rect(50, 50, 40, 50, 40, 50)]
        result = match_instances(gt, pred)
        self.assertEqual(result.matches, [])
        self.assertEqual(result.unmatched_gt, [0])
        self.assertEqual(result.unmatched_pred, [0])

    def test_one_prediction_cannot_match_two_gt(self) -> None:
        # one big prediction overlapping both GT boxes: exactly one GT may match
        gt = [_rect(60, 60, 0, 20, 0, 60), _rect(60, 60, 30, 50, 0, 60)]
        pred = [_rect(60, 60, 0, 50, 0, 60)]
        result = match_instances(gt, pred)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(len(result.unmatched_gt), 1)
        self.assertEqual(result.unmatched_pred, [])

    def test_fewer_predictions_than_gt_leaves_unmatched_gt(self) -> None:
        gt = [_rect(50, 50, 0, 10, 0, 10), _rect(50, 50, 20, 30, 20, 30)]
        pred = [_rect(50, 50, 0, 10, 0, 10)]
        result = match_instances(gt, pred)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.unmatched_gt, [1])

    def test_more_predictions_than_gt_yields_false_positives(self) -> None:
        gt = [_rect(50, 50, 0, 10, 0, 10)]
        pred = [_rect(50, 50, 0, 10, 0, 10), _rect(50, 50, 20, 30, 20, 30)]
        result = match_instances(gt, pred)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.unmatched_pred, [1])

    def test_hungarian_maximizes_total_iou_not_greedy(self) -> None:
        # greedy by best-first would pair (gt0,pred0)=0.5 leaving (gt1,pred1)=0.1;
        # hungarian picks (gt0,pred1)=0.45 + (gt1,pred0)=0.4 = higher total.
        matrix = np.array([[0.5, 0.45], [0.4, 0.1]])
        gt = [_rect(10, 10, 0, 5, 0, 5), _rect(10, 10, 5, 10, 5, 10)]
        pred = [_rect(10, 10, 0, 5, 0, 5), _rect(10, 10, 5, 10, 5, 10)]
        from unittest import mock

        with mock.patch("core.checkpoint_evaluation.mask_matching.iou_matrix", return_value=matrix):
            result = match_instances(gt, pred)
        pairs = {(gi, pi) for gi, pi, _ in result.matches}
        self.assertEqual(pairs, {(0, 1), (1, 0)})

    def test_iou_matrix_empty_inputs(self) -> None:
        self.assertEqual(iou_matrix([], []).shape, (0, 0))
        self.assertEqual(iou_matrix([_rect(5, 5, 0, 2, 0, 2)], []).shape, (1, 0))


class MetricsTest(unittest.TestCase):
    def _image(self, gt, pred):
        return compute_image_metrics("img.png", gt, pred, match_instances(gt, pred))

    def test_mean_iou_all_gt_counts_missed_gt_as_zero(self) -> None:
        gt = [_rect(50, 50, 0, 10, 0, 10), _rect(50, 50, 20, 30, 20, 30)]
        pred = [_rect(50, 50, 0, 10, 0, 10)]  # second GT missed
        m = self._image(gt, pred)
        agg = aggregate_metrics([m], boundary_tolerance_px=2)
        self.assertAlmostEqual(agg["mean_iou_all_gt"], 0.5)  # (1.0 + 0.0) / 2
        self.assertAlmostEqual(agg["mean_iou_matched_only"], 1.0)  # inflated by construction
        self.assertEqual(agg["missed_gt_count"], 1)
        self.assertAlmostEqual(agg["miss_rate_iou_50"], 0.5)

    def test_false_positive_counting(self) -> None:
        gt = [_rect(50, 50, 0, 10, 0, 10)]
        pred = [_rect(50, 50, 0, 10, 0, 10), _rect(50, 50, 30, 40, 30, 40), _rect(50, 50, 42, 48, 42, 48)]
        m = self._image(gt, pred)
        agg = aggregate_metrics([m], boundary_tolerance_px=2)
        self.assertEqual(agg["false_positive_count"], 2)
        self.assertAlmostEqual(agg["false_positive_per_image"], 2.0)
        self.assertAlmostEqual(agg["precision_iou_50"], 1 / 3)
        self.assertAlmostEqual(agg["recall_iou_50"], 1.0)

    def test_area_ratio(self) -> None:
        gt = [_rect(100, 100, 10, 50, 10, 50)]        # 40x40 = 1600
        pred = [_rect(100, 100, 8, 52, 8, 52)]        # 44x44 = 1936 (dilated ~1.21x)
        m = self._image(gt, pred)
        agg = aggregate_metrics([m], boundary_tolerance_px=2)
        self.assertAlmostEqual(agg["mean_area_ratio"], 1936 / 1600)
        self.assertEqual(agg["pred_larger_than_gt_rate"], 1.0)

    def test_boundary_f1_identical_and_shifted(self) -> None:
        gt = _rect(100, 100, 20, 60, 20, 60)
        self.assertAlmostEqual(boundary_f1(gt, gt, 2), 1.0)
        shifted_1px = _rect(100, 100, 21, 61, 20, 60)
        self.assertGreater(boundary_f1(gt, shifted_1px, 2), 0.95)  # within 2px tolerance
        far = _rect(100, 100, 70, 99, 70, 99)
        self.assertLess(boundary_f1(gt, far, 2), 0.05)

    def test_unmatched_gt_boundary_f1_is_zero(self) -> None:
        gt = [_rect(50, 50, 0, 10, 0, 10)]
        m = self._image(gt, [])
        agg = aggregate_metrics([m], boundary_tolerance_px=2)
        self.assertEqual(agg["mean_boundary_f1_all_gt"], 0.0)

    def test_empty_prediction_does_not_crash(self) -> None:
        gt = [_rect(50, 50, 0, 10, 0, 10)]
        m = self._image(gt, [])
        agg = aggregate_metrics([m], boundary_tolerance_px=2)
        self.assertEqual(agg["mean_iou_all_gt"], 0.0)
        self.assertIsNone(agg["mean_iou_matched_only"])
        self.assertIsNone(agg["mean_area_ratio"])

    def test_empty_gt_is_handled(self) -> None:
        m = self._image([], [_rect(50, 50, 0, 10, 0, 10)])
        agg = aggregate_metrics([m], boundary_tolerance_px=2)
        self.assertIsNone(agg["mean_iou_all_gt"])  # no GT instances -> undefined, not fake 0
        self.assertEqual(agg["false_positive_count"], 1)
        self.assertIsNone(agg["miss_rate_iou_50"])


def _candidate(name, epoch, iou, bf1=0.8, miss=0.1, fp=1.0, status="completed", baseline=False):
    return {
        "checkpoint_name": name,
        "checkpoint_path": f"/x/{name}",
        "epoch": epoch,
        "evaluation_status": status,
        "mean_iou_all_gt": iou,
        "mean_boundary_f1_all_gt": bf1,
        "miss_rate_iou_50": miss,
        "false_positive_per_image": fp,
        "is_baseline": baseline,
    }


class SelectorTest(unittest.TestCase):
    def test_highest_mean_iou_wins(self) -> None:
        result = select_best([_candidate("a", 5, 0.70), _candidate("b", 10, 0.80)])
        self.assertEqual(result["best"]["checkpoint_name"], "b")

    def test_boundary_f1_breaks_close_tie(self) -> None:
        result = select_best([
            _candidate("a", 5, 0.800, bf1=0.90),
            _candidate("b", 10, 0.803, bf1=0.70),
        ])
        self.assertEqual(result["best"]["checkpoint_name"], "a")
        self.assertIn("tie-break 1", result["reason"])

    def test_miss_rate_breaks_remaining_tie(self) -> None:
        result = select_best([
            _candidate("a", 5, 0.80, bf1=0.80, miss=0.20),
            _candidate("b", 10, 0.80, bf1=0.80, miss=0.05),
        ])
        self.assertEqual(result["best"]["checkpoint_name"], "b")

    def test_false_positive_breaks_remaining_tie(self) -> None:
        result = select_best([
            _candidate("a", 5, 0.80, bf1=0.80, miss=0.1, fp=3.0),
            _candidate("b", 10, 0.80, bf1=0.80, miss=0.1, fp=0.5),
        ])
        self.assertEqual(result["best"]["checkpoint_name"], "b")

    def test_earlier_epoch_wins_full_tie(self) -> None:
        result = select_best([
            _candidate("late", 20, 0.80),
            _candidate("early", 5, 0.80),
        ])
        self.assertEqual(result["best"]["checkpoint_name"], "early")

    def test_failed_checkpoints_are_excluded(self) -> None:
        result = select_best([
            _candidate("broken", 20, 0.99, status="failed"),
            _candidate("ok", 5, 0.70),
        ])
        self.assertEqual(result["best"]["checkpoint_name"], "ok")

    def test_baseline_is_never_auto_selected(self) -> None:
        result = select_best([
            _candidate("baseline", None, 0.99, baseline=True),
            _candidate("ok", 5, 0.70),
        ])
        self.assertEqual(result["best"]["checkpoint_name"], "ok")

    def test_no_eligible_checkpoint_yields_blocked_not_fake_best(self) -> None:
        result = select_best([
            _candidate("broken", 20, 0.99, status="failed"),
            _candidate("baseline", None, 0.99, baseline=True),
        ])
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["best"])

    def test_nan_iou_is_excluded(self) -> None:
        result = select_best([_candidate("nan", 5, float("nan"))])
        self.assertEqual(result["status"], "blocked")


class ValidationGuardTest(unittest.TestCase):
    def test_unregistered_val_set_is_blocked(self) -> None:
        from core.checkpoint_evaluation.dataset_loader import check_validation_guard

        with tempfile.TemporaryDirectory() as tmp:
            val = Path(tmp) / "annotations.json"
            val.write_text("{}")
            result = check_validation_guard(val, None)
        self.assertFalse(result.ok)
        self.assertIn("not registered", result.reason)

    def test_val_equal_to_train_is_blocked(self) -> None:
        from core.checkpoint_evaluation.dataset_loader import check_validation_guard

        with tempfile.TemporaryDirectory() as tmp:
            same = Path(tmp) / "annotations.json"
            same.write_text("{}")
            result = check_validation_guard(same, same)
        self.assertFalse(result.ok)
        self.assertIn("training data", result.reason)

    def test_missing_val_file_is_blocked(self) -> None:
        from core.checkpoint_evaluation.dataset_loader import check_validation_guard

        result = check_validation_guard(Path("/nonexistent/annotations.json"), None)
        self.assertFalse(result.ok)
        self.assertIn("does not exist", result.reason)

    def test_machine_preannotation_val_set_is_blocked(self) -> None:
        """The registered pre-annotation dataset's val split must be refused."""
        from core.checkpoint_evaluation.dataset_loader import check_validation_guard
        from core.config import BOOK_ROOT

        preann_val = BOOK_ROOT / "data" / "formal_book_spine_sam3_dataset" / "val" / "annotations.json"
        if not preann_val.exists():
            self.skipTest("pre-annotation val split not present")
        result = check_validation_guard(preann_val, None)
        self.assertFalse(result.ok)
        self.assertIn("not human-reviewed", result.reason)

    def test_registered_human_corrected_val_set_passes_with_eval_warning(self) -> None:
        from core.checkpoint_evaluation.dataset_loader import check_validation_guard
        from core.config import BOOK_ROOT

        val = BOOK_ROOT / "data" / "book_spine_sam3_dataset" / "val" / "annotations.json"
        train = BOOK_ROOT / "data" / "book_spine_sam3_dataset" / "train" / "annotations.json"
        if not val.exists():
            self.skipTest("human-corrected val split not present")
        result = check_validation_guard(val, train)
        self.assertTrue(result.ok, result.reason)
        self.assertTrue(any("allowed_for_model_evaluation=false" in w for w in result.warnings))


class EvaluationConfigFromRunTest(unittest.TestCase):
    def _write_runtime_config(
        self,
        run_dir: Path,
        train_images: Path,
        train_annotations: Path,
        val_images: Path,
        val_annotations: Path,
    ) -> None:
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "runtime_config.yaml").write_text(
            "\n".join(
                [
                    "trainer:",
                    "  data:",
                    "    train:",
                    "      dataset:",
                    f"        img_folder: {train_images}",
                    f"        ann_file: {train_annotations}",
                    "    val:",
                    "      dataset:",
                    f"        img_folder: {val_images}",
                    f"        ann_file: {val_annotations}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def test_default_validation_images_resolve_to_val_split_folder(self) -> None:
        from core.checkpoint_evaluation.evaluator import config_from_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "book_spine_sam3_dataset"
            train_images = dataset / "train" / "images"
            val_images = dataset / "val" / "images"
            raw_images = root / "dataset_raw"
            for path in (train_images, val_images, raw_images):
                path.mkdir(parents=True)
            train_annotations = dataset / "train" / "annotations.json"
            val_annotations = dataset / "val" / "annotations.json"
            train_annotations.write_text("{}", encoding="utf-8")
            val_annotations.write_text("{}", encoding="utf-8")
            run_dir = root / "run"
            self._write_runtime_config(
                run_dir,
                train_images=raw_images,
                train_annotations=train_annotations,
                val_images=raw_images,
                val_annotations=val_annotations,
            )

            config = config_from_run(run_dir)

        self.assertEqual(config.val_annotations, val_annotations)
        self.assertEqual(config.val_images, val_images)

    def test_explicit_val_images_override_is_respected(self) -> None:
        from core.checkpoint_evaluation.evaluator import config_from_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "book_spine_sam3_dataset"
            train_images = dataset / "train" / "images"
            val_images = dataset / "val" / "images"
            override_images = root / "manual_val_images"
            for path in (train_images, val_images, override_images):
                path.mkdir(parents=True)
            train_annotations = dataset / "train" / "annotations.json"
            val_annotations = dataset / "val" / "annotations.json"
            train_annotations.write_text("{}", encoding="utf-8")
            val_annotations.write_text("{}", encoding="utf-8")
            run_dir = root / "run"
            self._write_runtime_config(
                run_dir,
                train_images=train_images,
                train_annotations=train_annotations,
                val_images=train_images,
                val_annotations=val_annotations,
            )

            config = config_from_run(run_dir, val_images=override_images)

        self.assertEqual(config.val_images, override_images)

    def test_run_summary_prompt_beats_defaults_file(self) -> None:
        import json as _json

        from core.checkpoint_evaluation.evaluator import config_from_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "cable_dataset"
            train_images = dataset / "train" / "images"
            val_images = dataset / "val" / "images"
            for path in (train_images, val_images):
                path.mkdir(parents=True)
            train_annotations = dataset / "train" / "annotations.json"
            val_annotations = dataset / "val" / "annotations.json"
            train_annotations.write_text("{}", encoding="utf-8")
            val_annotations.write_text("{}", encoding="utf-8")
            run_dir = root / "run"
            self._write_runtime_config(
                run_dir,
                train_images=train_images,
                train_annotations=train_annotations,
                val_images=val_images,
                val_annotations=val_annotations,
            )
            (run_dir / "training_config_summary.json").write_text(
                _json.dumps({"resolved_training_prompt": "cable"}), encoding="utf-8"
            )

            config = config_from_run(run_dir)
            overridden = config_from_run(run_dir, prompt="manual override")

        # The run's recorded prompt must survive the defaults file (which pins
        # "book spine"); an explicit override must beat both.
        self.assertEqual(config.prompt, "cable")
        self.assertEqual(overridden.prompt, "manual override")


class ReportWriterTest(unittest.TestCase):
    def test_training_summary_update_is_atomic_and_preserves_fields(self) -> None:
        from core.checkpoint_evaluation.report_writer import update_training_summary

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            original = {"status": "completed", "exit_code": 0, "custom": [1, 2, 3]}
            (run_dir / "training_summary.json").write_text(json.dumps(original))
            error = update_training_summary(run_dir, {"status": "completed", "best_epoch": 15})
            self.assertIsNone(error)
            updated = json.loads((run_dir / "training_summary.json").read_text())
            self.assertEqual(updated["status"], "completed")
            self.assertEqual(updated["custom"], [1, 2, 3])
            self.assertEqual(updated["checkpoint_evaluation"]["best_epoch"], 15)

    def test_training_summary_update_failure_does_not_corrupt(self) -> None:
        from core.checkpoint_evaluation.report_writer import update_training_summary

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "training_summary.json").write_text("NOT JSON {")
            error = update_training_summary(run_dir, {"status": "completed"})
            self.assertIsNotNone(error)
            self.assertEqual((run_dir / "training_summary.json").read_text(), "NOT JSON {")


class CheckpointDiscoveryTest(unittest.TestCase):
    def test_numeric_epoch_sorting_and_alias_dedupe(self) -> None:
        from core.checkpoint_evaluation.checkpoint_loader import discover_checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            ckpt_dir.mkdir()
            # write distinct contents; checkpoint.pt identical to checkpoint_20.pt
            (ckpt_dir / "checkpoint_5.pt").write_bytes(b"five")
            (ckpt_dir / "checkpoint_10.pt").write_bytes(b"ten")
            (ckpt_dir / "checkpoint_20.pt").write_bytes(b"twenty")
            (ckpt_dir / "checkpoint.pt").write_bytes(b"twenty")
            candidates = discover_checkpoints(Path(tmp))
        names = [c.name for c in candidates]
        # numeric sort: 5 < 10 < 20 (string sort would put 10 first)
        self.assertEqual(names[:3], ["checkpoint_5.pt", "checkpoint_10.pt", "checkpoint_20.pt"])
        latest = candidates[-1]
        self.assertEqual(latest.name, "checkpoint.pt")
        self.assertEqual(latest.alias_of, "checkpoint_20.pt")
        self.assertEqual(latest.epoch, 20)

    def test_unique_latest_checkpoint_is_kept_as_candidate(self) -> None:
        from core.checkpoint_evaluation.checkpoint_loader import discover_checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            ckpt_dir.mkdir()
            (ckpt_dir / "checkpoint_5.pt").write_bytes(b"five")
            (ckpt_dir / "checkpoint.pt").write_bytes(b"unique-latest")
            candidates = discover_checkpoints(Path(tmp))
        latest = candidates[-1]
        self.assertEqual(latest.name, "checkpoint.pt")
        self.assertIsNone(latest.alias_of)


class SplitEvaluationArtifactTest(unittest.TestCase):
    def test_cache_key_is_split_specific(self) -> None:
        from core.checkpoint_evaluation.checkpoint_loader import CheckpointCandidate
        from core.checkpoint_evaluation.evaluator import _cache_key

        candidate = CheckpointCandidate(
            name="checkpoint_5.pt", path=Path("/x/checkpoint_5.pt"), epoch=5,
            sha256="abc", size_bytes=10,
        )
        val_key = _cache_key("validation", candidate, "dataset", "images", "config", "sam3")
        test_key = _cache_key("test", candidate, "dataset", "images", "config", "sam3")
        self.assertNotEqual(val_key, test_key)

    def test_test_metrics_do_not_change_validation_selector(self) -> None:
        validation_rows = [
            _candidate("checkpoint_5.pt", 5, 0.80, bf1=0.80),
            _candidate("checkpoint_20.pt", 20, 0.90, bf1=0.80),
        ]
        test_rows = [
            _candidate("checkpoint_5.pt", 5, 0.99, bf1=0.99),
            _candidate("checkpoint_20.pt", 20, 0.10, bf1=0.10),
        ]
        self.assertEqual(select_best(validation_rows)["best"]["checkpoint_name"], "checkpoint_20.pt")
        self.assertEqual(select_best(test_rows)["best"]["checkpoint_name"], "checkpoint_5.pt")
        # The production flow calls select_best only on validation rows; this
        # assertion documents the intended separation.
        self.assertNotEqual(
            select_best(validation_rows)["best"]["checkpoint_name"],
            select_best(test_rows)["best"]["checkpoint_name"],
        )

    def test_raw_prediction_npz_can_recompute_iou(self) -> None:
        from core.checkpoint_evaluation.artifacts import load_raw_prediction_masks, write_raw_predictions
        from core.checkpoint_evaluation.dataset_loader import ImageGroundTruth

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "img.png"
            image_path.write_bytes(b"placeholder")
            gt = ImageGroundTruth(
                image_id=1,
                file_name="img.png",
                path=image_path,
                width=20,
                height=20,
                masks=[_rect(20, 20, 2, 10, 2, 10)],
                annotation_ids=[101],
            )
            pred = _rect(20, 20, 2, 10, 2, 10)
            instances = InstanceSet(
                masks=np.stack([pred]),
                scores=np.array([0.9], dtype=np.float32),
                bboxes=np.array([[2, 2, 8, 8]], dtype=np.int32),
                instance_ids=np.array([1], dtype=np.int32),
                extra={},
            )
            npz_path, _meta = write_raw_predictions(
                Path(tmp), "validation", "checkpoint_5", gt, instances, {"prompt": "book spine"}, 0.01
            )
            loaded = load_raw_prediction_masks(npz_path)
            self.assertAlmostEqual(iou_matrix(gt.masks, loaded)[0, 0], 1.0)

    def test_per_instance_rows_include_unmatched_gt_and_prediction(self) -> None:
        from core.checkpoint_evaluation.artifacts import build_instance_rows
        from core.checkpoint_evaluation.dataset_loader import ImageGroundTruth

        gt = ImageGroundTruth(
            image_id=1,
            file_name="img.png",
            path=Path("/tmp/img.png"),
            width=30,
            height=30,
            masks=[_rect(30, 30, 0, 5, 0, 5), _rect(30, 30, 10, 15, 10, 15)],
            annotation_ids=[1, 2],
        )
        preds = [_rect(30, 30, 20, 25, 20, 25)]
        match = match_instances(gt.masks, preds)
        rows = build_instance_rows("validation", "checkpoint_5.pt", gt, preds, match, 0.5)
        self.assertTrue(any(row["miss"] is True and row["gt_instance_id"] == 1 for row in rows))
        self.assertTrue(any(row["false_positive"] is True and row["pred_instance_id"] == 1 for row in rows))

    def test_match_record_serializes_iou_matrix_and_assignment(self) -> None:
        from core.checkpoint_evaluation.artifacts import write_match_record
        from core.checkpoint_evaluation.dataset_loader import ImageGroundTruth

        with tempfile.TemporaryDirectory() as tmp:
            gt = ImageGroundTruth(
                image_id=1,
                file_name="img.png",
                path=Path(tmp) / "img.png",
                width=20,
                height=20,
                masks=[_rect(20, 20, 1, 6, 1, 6)],
                annotation_ids=[1],
            )
            preds = [_rect(20, 20, 1, 6, 1, 6)]
            match = match_instances(gt.masks, preds)
            raw = Path(tmp) / "pred.npz"
            raw.write_bytes(b"x")
            path = write_match_record(Path(tmp), "validation", "checkpoint_5", "checkpoint_5.pt", gt, preds, match, raw)
            data = json.loads(path.read_text())
            self.assertEqual(data["iou_matrix"], [[1.0]])
            self.assertEqual(data["accepted_matches"][0]["gt_index"], 0)

    def test_visualization_overlays_use_distinct_instance_colors(self) -> None:
        import cv2

        from core.checkpoint_evaluation.artifacts import write_visualizations
        from core.checkpoint_evaluation.dataset_loader import ImageGroundTruth
        from core.checkpoint_evaluation.mask_matching import match_instances

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "img.png"
            cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
            gt_masks = [
                _rect(20, 20, 1, 6, 1, 6),
                _rect(20, 20, 10, 15, 10, 15),
            ]
            pred_masks = [
                _rect(20, 20, 1, 6, 1, 6),
                _rect(20, 20, 10, 15, 10, 15),
            ]
            gt = ImageGroundTruth(
                image_id=1,
                file_name="img.png",
                path=image_path,
                width=20,
                height=20,
                masks=gt_masks,
                annotation_ids=[1, 2],
            )
            paths = write_visualizations(Path(tmp), "validation", "checkpoint_5", gt, pred_masks, match_instances(gt_masks, pred_masks))
            gt_overlay = cv2.imread(paths["gt_overlay"], cv2.IMREAD_COLOR)
            pred_overlay = cv2.imread(paths["prediction_overlay"], cv2.IMREAD_COLOR)

        self.assertIsNotNone(gt_overlay)
        self.assertIsNotNone(pred_overlay)
        self.assertNotEqual(tuple(gt_overlay[2, 2].tolist()), tuple(gt_overlay[12, 12].tolist()))
        self.assertNotEqual(tuple(pred_overlay[2, 2].tolist()), tuple(pred_overlay[12, 12].tolist()))

    def test_registered_test_split_identity_is_diagnostic_only(self) -> None:
        from core.dataset_identity import resolve_split_identity
        from core.config import BOOK_ROOT

        test_ann = BOOK_ROOT / "data" / "book_spine_sam3_dataset" / "test" / "annotations.json"
        if not test_ann.exists():
            self.skipTest("registered diagnostic test split not present")
        identity = resolve_split_identity("test", test_ann)
        self.assertTrue(identity.matched)
        self.assertTrue(identity.human_reviewed)
        self.assertFalse(identity.allowed_for_model_evaluation)


if __name__ == "__main__":
    unittest.main()


class EvaluatorBlockedPathTest(unittest.TestCase):
    """Integration: a run whose validation set is machine pre-annotation must be
    blocked before any checkpoint is loaded, with honest blocked outputs written.
    Uses a throwaway temp run dir — no historical run is touched. No GPU needed:
    the guard fires before any model loading."""

    def test_preannotation_val_blocks_evaluation_and_writes_blocked_outputs(self) -> None:
        from core.checkpoint_evaluation.evaluator import EvaluationConfig, evaluate_run
        from core.config import BOOK_ROOT

        preann_val = BOOK_ROOT / "data" / "formal_book_spine_sam3_dataset" / "val" / "annotations.json"
        if not preann_val.exists():
            self.skipTest("pre-annotation val split not present")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "fake_run"
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "checkpoints" / "checkpoint_5.pt").write_bytes(b"not a real checkpoint")
            (run_dir / "training_summary.json").write_text(json.dumps({"status": "completed"}))
            config = EvaluationConfig(
                run_dir=run_dir,
                val_annotations=preann_val,
                val_images=BOOK_ROOT / "data" / "dataset_raw",
                train_annotations=None,
                device="cpu",
                include_baseline=False,
            )
            summary = evaluate_run(config)
            self.assertEqual(summary["status"], "blocked")
            self.assertIn("not human-reviewed", summary["reason"])
            best = json.loads((run_dir / "evaluation" / "best_checkpoint.json").read_text())
            self.assertEqual(best["status"], "blocked")
            self.assertNotIn("best_checkpoint", {k: v for k, v in best.items() if v and k == "best_checkpoint"})
            training_summary = json.loads((run_dir / "training_summary.json").read_text())
            self.assertEqual(training_summary["checkpoint_evaluation"]["status"], "blocked")
            self.assertEqual(training_summary["status"], "completed")  # original field intact
