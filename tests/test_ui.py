from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BOOK_ROOT, DEFAULT_SAM3_CHECKPOINT, DEFAULT_TRAINING_RUN_ROOT

REAL_RUN_DIR = BOOK_ROOT / "runs" / "inference" / "2026-07-02_12-28-40"
INCOMPLETE_RUN_DIR = BOOK_ROOT / "runs" / "inference" / "2026-07-02_12-10-36"
INFERENCE_RUNS_ROOT = BOOK_ROOT / "runs" / "inference"
TEST_IMAGES_DIR = BOOK_ROOT / "data" / "book_spine_sam3_dataset" / "test" / "images"


class UiImportsTest(unittest.TestCase):
    def test_import_all_ui_modules(self) -> None:
        import ui.cvat_page  # noqa: F401
        import ui.dataset_registry_page  # noqa: F401
        import ui.history_page  # noqa: F401
        import ui.inference_page  # noqa: F401
        import ui.process_manager  # noqa: F401
        import ui.results_page  # noqa: F401
        import ui.run_reader  # noqa: F401
        import ui.training_preflight_page  # noqa: F401
        import ui.ui_utils  # noqa: F401

    def test_import_app(self) -> None:
        import app

        self.assertIsNotNone(app.demo)


class UiLanguageSwitchTest(unittest.TestCase):
    def test_every_chinese_ui_string_has_japanese_translation(self) -> None:
        import re

        import gradio as gr

        import app
        from ui import i18n

        demo = app.build_app()
        han = re.compile(r"[一-鿿]")
        missed: list[tuple[str, str, str]] = []
        for block in demo.blocks.values():
            if isinstance(block, gr.Radio) and getattr(block, "label", "") == "Language / 语言 / 言語":
                continue  # the switch itself is intentionally trilingual
            for attr in ("label", "info", "placeholder"):
                text = getattr(block, attr, None)
                if isinstance(text, str) and han.search(text) and text.strip() not in i18n._JA_NORMALIZED:
                    missed.append((type(block).__name__, attr, text[:60]))
            if isinstance(block, (gr.Markdown, gr.Button)):
                text = getattr(block, "value", None)
                # The CUDA banner is runtime data (detection results), not chrome.
                if isinstance(text, str) and han.search(text) and "CUDA 检测" not in text:
                    if text.strip() not in i18n._JA_NORMALIZED:
                        missed.append((type(block).__name__, "value", text[:60]))
            for label, _value in getattr(block, "choices", None) or []:
                if isinstance(label, str) and han.search(label) and label.strip() not in i18n._JA_NORMALIZED:
                    missed.append((type(block).__name__, "choice", label[:60]))
        self.assertEqual(missed, [])

    def test_switch_translates_and_restores(self) -> None:
        import app

        demo = app.build_app()
        updates_ja = demo.i18n_switch("ja")
        updates_zh = demo.i18n_switch("zh")
        self.assertEqual(len(updates_ja), len(updates_zh))
        self.assertGreater(len(updates_ja), 50)
        ja_labels = {u.get("label") for u in updates_ja if isinstance(u, dict)}
        for expected in ("推論タスク設定", "学習プリフライト", "データセット登録", "プリフライト状態"):
            self.assertIn(expected, ja_labels)
        self.assertIn("训练预检", {u.get("label") for u in updates_zh if isinstance(u, dict)})
        ja_choice_updates = [u["choices"] for u in updates_ja if isinstance(u, dict) and "choices" in u]
        self.assertTrue(
            any(("formal（人手レビュー済み GT が必要）", "formal") in choices for choices in ja_choice_updates)
        )


@unittest.skipUnless(REAL_RUN_DIR.exists(), "real inference run fixture not present")
class RunReaderRealRunTest(unittest.TestCase):
    def test_summarize_run_reads_real_fields(self) -> None:
        from ui.run_reader import summarize_run

        summary = summarize_run(REAL_RUN_DIR)
        self.assertEqual(summary.run_id, "2026-07-02_12-28-40")
        self.assertEqual(summary.input_images, 1)
        self.assertEqual(summary.success_images, 1)
        self.assertEqual(summary.raw_instances, 20)
        self.assertEqual(summary.nms_instances, 19)
        self.assertEqual(summary.coco_annotations, 19)
        self.assertIn("polygon", summary.segmentation_formats)
        self.assertIn("rle", summary.segmentation_formats)

    def test_list_image_names(self) -> None:
        from ui.run_reader import list_image_names

        names = list_image_names(REAL_RUN_DIR)
        self.assertEqual(names, ["im_000001.png"])

    def test_load_instance_table(self) -> None:
        from ui.run_reader import load_instance_table

        rows, err = load_instance_table(REAL_RUN_DIR, "im_000001.png")
        self.assertIsNone(err)
        self.assertEqual(len(rows), 19)
        self.assertTrue(all("source_instance_id" in row for row in rows))

    def test_find_nms_review_images(self) -> None:
        from ui.run_reader import find_nms_review_images

        reviews = find_nms_review_images(REAL_RUN_DIR, "im_000001.png")
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["kept_instance_id"], 3)
        self.assertEqual(reviews[0]["removed_instance_id"], 7)
        self.assertIsNotNone(reviews[0]["image_path"])

    def test_acceptance_report_readable(self) -> None:
        from ui.ui_utils import safe_read_json

        data, err = safe_read_json(REAL_RUN_DIR / "acceptance_report.json")
        self.assertIsNone(err)
        # ok is expected False here: the acceptance script checks polygon segmentation
        # against the exact NMS mask, and polygon COCO is a documented lossy conversion
        # (see docs/stage_b2_real_inference.md). This is not a bug to fix in stage D1.
        self.assertIn("ok", data)
        self.assertIn("errors", data)

    def test_polygon_fidelity_readable(self) -> None:
        from ui.ui_utils import safe_read_json

        data, err = safe_read_json(REAL_RUN_DIR / "cvat_export" / "polygon" / "polygon_fidelity_report.json")
        self.assertIsNone(err)
        self.assertEqual(data.get("instance_count"), 19)

    def test_rle_validation_readable(self) -> None:
        from ui.ui_utils import safe_read_json

        data, err = safe_read_json(REAL_RUN_DIR / "cvat_export" / "rle" / "validation_report.json")
        self.assertIsNone(err)
        self.assertEqual(data.get("exact_mask_matches"), 19)
        self.assertEqual(data.get("exact_mask_total"), 19)

    def test_list_inference_runs_includes_all_real_runs(self) -> None:
        from ui.run_reader import list_inference_runs

        summaries = list_inference_runs(INFERENCE_RUNS_ROOT)
        run_ids = {s.run_id for s in summaries}
        self.assertIn("2026-07-02_12-28-40", run_ids)


@unittest.skipUnless(INCOMPLETE_RUN_DIR.exists(), "legacy incomplete run fixture not present")
class RunReaderOldRunMissingFieldsTest(unittest.TestCase):
    """2026-07-02_12-10-36 has no acceptance_report.json, no rle export, no run_summary.json."""

    def test_summarize_old_run_does_not_crash(self) -> None:
        from ui.run_reader import summarize_run

        summary = summarize_run(INCOMPLETE_RUN_DIR)
        self.assertEqual(summary.input_images, 2)
        self.assertNotIn("rle", summary.segmentation_formats)

    def test_extra_reports_show_unavailable_not_crash(self) -> None:
        from ui.results_page import load_extra_reports

        acceptance, validation, polygon_fidelity, rle_validation, legacy = load_extra_reports("2026-07-02_12-10-36")
        self.assertNotIn("Traceback", acceptance)
        self.assertTrue("not available" in acceptance or "file not found" in acceptance)
        self.assertTrue("not available" in rle_validation or "file not found" in rle_validation)


class RunReaderEmptyRunTest(unittest.TestCase):
    def test_summarize_empty_directory_does_not_crash(self) -> None:
        from ui.run_reader import summarize_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "empty_run"
            run_dir.mkdir()
            summary = summarize_run(run_dir)
            self.assertTrue(summary.warnings)
            self.assertEqual(summary.input_images, 0)

    def test_list_inference_runs_survives_one_bad_run(self) -> None:
        from ui.run_reader import list_inference_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "good_run").mkdir()
            (root / "good_run" / "run_config.json").write_text("{not valid json", encoding="utf-8")
            (root / "another_run").mkdir()
            summaries = list_inference_runs(root)
            self.assertEqual(len(summaries), 2)


class InferenceCommandBuilderTest(unittest.TestCase):
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            input_dir=str(TEST_IMAGES_DIR),
            checkpoint=str(DEFAULT_SAM3_CHECKPOINT),
            prompt="book spine",
            device="cuda",
            score_threshold=0.3,
            confidence_threshold=0.05,
            dtype_mode="bf16",
            nms_iou_thresh=0.5,
            nms_metric="iou",
            nms_mode="suppress",
            min_area=200,
            category_name="book_spine",
            output_root=str(BOOK_ROOT / "runs"),
            limit=None,
            cuda_available=True,
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_command_is_a_list_of_strings(self) -> None:
        from ui.inference_page import build_inference_command

        command, errors, warnings = build_inference_command(**self._base_kwargs())
        self.assertEqual(errors, [])
        self.assertIsInstance(command, list)
        self.assertTrue(all(isinstance(part, str) for part in command))
        self.assertIn("--device", command)
        self.assertIn("--model-path", command)
        self.assertIn("cuda", command)
        self.assertNotIn("&&", " ".join(command))

    def test_cuda_requested_but_unavailable_fails_fast_no_fallback(self) -> None:
        from ui.inference_page import build_inference_command

        command, errors, warnings = build_inference_command(**self._base_kwargs(cuda_available=False))
        self.assertIsNone(command)
        self.assertTrue(any("CUDA" in e or "cuda" in e for e in errors))
        # must fail fast, not silently switch the requested device to cpu itself
        self.assertNotIn("falling back", " ".join(errors).lower())
        self.assertNotIn("using cpu instead", " ".join(errors).lower())

    def test_missing_input_dir_blocks_command(self) -> None:
        from ui.inference_page import build_inference_command

        command, errors, warnings = build_inference_command(**self._base_kwargs(input_dir="/no/such/dir/at/all"))
        self.assertIsNone(command)
        self.assertTrue(errors)

    def test_missing_checkpoint_blocks_command(self) -> None:
        from ui.inference_page import build_inference_command

        command, errors, warnings = build_inference_command(**self._base_kwargs(checkpoint="/no/such/checkpoint.pt"))
        self.assertIsNone(command)
        self.assertTrue(errors)

    def test_cpu_device_with_cuda_unavailable_is_allowed_explicitly(self) -> None:
        from ui.inference_page import build_inference_command

        command, errors, warnings = build_inference_command(**self._base_kwargs(device="cpu", cuda_available=False))
        self.assertEqual(errors, [])
        self.assertIsInstance(command, list)
        self.assertIn("cpu", command)


class InferenceModelFieldSyncTest(unittest.TestCase):
    def test_discovered_model_updates_display_fields_and_checkpoint(self) -> None:
        from ui.inference_page import sync_ui_model_fields

        selected = "/tmp/run/checkpoints/inference_checkpoint_35.pt"

        directory, filename, absolute, checkpoint = sync_ui_model_fields(
            "discovered",
            selected,
            str(DEFAULT_SAM3_CHECKPOINT.parent),
            DEFAULT_SAM3_CHECKPOINT.name,
            str(DEFAULT_SAM3_CHECKPOINT),
        )

        self.assertEqual(directory, "/tmp/run/checkpoints")
        self.assertEqual(filename, "inference_checkpoint_35.pt")
        self.assertEqual(absolute, selected)
        self.assertEqual(checkpoint, selected)

    def test_default_model_updates_display_fields_and_checkpoint(self) -> None:
        from ui.inference_page import sync_ui_model_fields

        directory, filename, absolute, checkpoint = sync_ui_model_fields(
            "default",
            "/tmp/run/checkpoints/inference_checkpoint_35.pt",
            "/tmp/run/checkpoints",
            "inference_checkpoint_35.pt",
            "/tmp/run/checkpoints/inference_checkpoint_35.pt",
        )

        self.assertEqual(directory, str(DEFAULT_SAM3_CHECKPOINT.parent))
        self.assertEqual(filename, DEFAULT_SAM3_CHECKPOINT.name)
        self.assertEqual(absolute, str(DEFAULT_SAM3_CHECKPOINT))
        self.assertEqual(checkpoint, str(DEFAULT_SAM3_CHECKPOINT))


class OptionalLimitParsingTest(unittest.TestCase):
    """Gradio delivers None/''/NaN for empty numeric fields; the adapter must not
    mistake empty for 'limit set to an invalid value'."""

    def _command_for_limit(self, limit):
        from ui.inference_page import build_inference_command

        return build_inference_command(
            input_dir=str(TEST_IMAGES_DIR),
            checkpoint=str(DEFAULT_SAM3_CHECKPOINT),
            prompt="book spine",
            device="cpu",
            score_threshold=0.3,
            confidence_threshold=0.05,
            dtype_mode="bf16",
            nms_iou_thresh=0.5,
            nms_metric="iou",
            nms_mode="suppress",
            min_area=200,
            category_name="book_spine",
            output_root=str(BOOK_ROOT / "runs"),
            limit=limit,
            cuda_available=False,
        )

    def test_none_means_all_images_no_limit_flag(self) -> None:
        command, errors, _ = self._command_for_limit(None)
        self.assertEqual(errors, [])
        self.assertNotIn("--limit", command)

    def test_empty_string_means_all_images_no_limit_flag(self) -> None:
        command, errors, _ = self._command_for_limit("")
        self.assertEqual(errors, [])
        self.assertNotIn("--limit", command)

    def test_whitespace_string_means_all_images(self) -> None:
        command, errors, _ = self._command_for_limit("   ")
        self.assertEqual(errors, [])
        self.assertNotIn("--limit", command)

    def test_positive_int_adds_limit_flag(self) -> None:
        command, errors, _ = self._command_for_limit(1)
        self.assertEqual(errors, [])
        self.assertIn("--limit", command)
        self.assertEqual(command[command.index("--limit") + 1], "1")

    def test_positive_int_string_adds_limit_flag(self) -> None:
        command, errors, _ = self._command_for_limit("10")
        self.assertEqual(errors, [])
        self.assertEqual(command[command.index("--limit") + 1], "10")

    def test_whole_float_is_accepted_as_int(self) -> None:
        command, errors, _ = self._command_for_limit(10.0)
        self.assertEqual(errors, [])
        self.assertEqual(command[command.index("--limit") + 1], "10")

    def test_zero_fails_validation(self) -> None:
        command, errors, _ = self._command_for_limit(0)
        self.assertIsNone(command)
        self.assertTrue(any("limit" in e for e in errors))

    def test_negative_fails_validation(self) -> None:
        command, errors, _ = self._command_for_limit(-1)
        self.assertIsNone(command)
        self.assertTrue(any("limit" in e for e in errors))

    def test_fractional_fails_validation(self) -> None:
        command, errors, _ = self._command_for_limit(1.5)
        self.assertIsNone(command)
        self.assertTrue(any("limit" in e for e in errors))

    def test_nan_fails_validation(self) -> None:
        command, errors, _ = self._command_for_limit(float("nan"))
        self.assertIsNone(command)
        self.assertTrue(any("limit" in e for e in errors))

    def test_invalid_text_fails_validation(self) -> None:
        command, errors, _ = self._command_for_limit("abc")
        self.assertIsNone(command)
        self.assertTrue(any("limit" in e for e in errors))

    def test_parser_unit_cases(self) -> None:
        from ui.ui_utils import parse_optional_positive_int

        self.assertEqual(parse_optional_positive_int(None), (None, None))
        self.assertEqual(parse_optional_positive_int(""), (None, None))
        self.assertEqual(parse_optional_positive_int(" "), (None, None))
        self.assertEqual(parse_optional_positive_int(1), (1, None))
        self.assertEqual(parse_optional_positive_int("2"), (2, None))
        self.assertEqual(parse_optional_positive_int(10.0), (10, None))
        for bad in [0, -1, 1.5, float("nan"), float("inf"), "abc", True]:
            value, error = parse_optional_positive_int(bad)
            self.assertIsNone(value, f"expected rejection for {bad!r}")
            self.assertIsNotNone(error, f"expected error message for {bad!r}")

    def test_nan_threshold_fields_fail_validation_not_crash(self) -> None:
        from ui.inference_page import build_inference_command

        command, errors, _ = build_inference_command(
            input_dir=str(TEST_IMAGES_DIR),
            checkpoint=str(DEFAULT_SAM3_CHECKPOINT),
            prompt="book spine",
            device="cpu",
            score_threshold=float("nan"),
            confidence_threshold=float("nan"),
            dtype_mode="bf16",
            nms_iou_thresh=float("nan"),
            nms_metric="iou",
            nms_mode="suppress",
            min_area=float("nan"),
            category_name="book_spine",
            output_root=str(BOOK_ROOT / "runs"),
            limit=None,
            cuda_available=False,
        )
        self.assertIsNone(command)
        for field in ["score_threshold", "confidence_threshold", "nms_iou_thresh", "min_area"]:
            self.assertTrue(any(field in e for e in errors), f"missing validation error for {field}")


@unittest.skipUnless(REAL_RUN_DIR.exists(), "real inference run fixture not present")
class NmsRemovedPairsTest(unittest.TestCase):
    def test_real_run_lists_pair_3_vs_7(self) -> None:
        from ui.run_reader import list_nms_removed_pairs

        pairs = list_nms_removed_pairs(REAL_RUN_DIR)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["file_name"], "im_000001.png")
        self.assertEqual(pairs[0]["kept_source_id"], 3)
        self.assertEqual(pairs[0]["removed_source_id"], 7)

    def test_cvat_page_pair_choices_for_real_run(self) -> None:
        from ui.cvat_page import _decode_pair, _pair_choices

        choices = _pair_choices(REAL_RUN_DIR.name)
        self.assertEqual(len(choices), 1)
        label, encoded = choices[0]
        self.assertIn("kept 3", label)
        self.assertIn("removed 7", label)
        self.assertEqual(_decode_pair(encoded), ("im_000001.png", 3, 7))

    def test_decode_pair_rejects_garbage(self) -> None:
        from ui.cvat_page import _decode_pair

        self.assertIsNone(_decode_pair(""))
        self.assertIsNone(_decode_pair("no separators here"))
        self.assertIsNone(_decode_pair("a|b|c"))

    def test_existing_reviews_autoloaded_for_real_run(self) -> None:
        from ui.cvat_page import load_existing_reviews

        text = load_existing_reviews(REAL_RUN_DIR.name)
        self.assertIn("3", text)
        self.assertIn("7", text)
        self.assertIn("instance_3_vs_7", text)

    def test_pairs_empty_for_run_without_manifest(self) -> None:
        from ui.run_reader import list_nms_removed_pairs

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_nms_removed_pairs(Path(tmp)), [])


class RenderInstanceRowsTest(unittest.TestCase):
    def test_missing_source_instance_id_shows_unavailable(self) -> None:
        from ui.results_page import render_instance_rows

        rows = render_instance_rows(
            [
                {"annotation_id": 1, "source_instance_id": None, "score": 0.9, "bbox": [0, 0, 1, 1], "area": 1},
                {"annotation_id": 2, "source_instance_id": 5, "score": 0.8, "bbox": [1, 1, 2, 2], "area": 4},
            ]
        )
        self.assertEqual(rows[0][1], "unavailable")
        self.assertEqual(rows[1][1], 5)


class ProcessManagerTest(unittest.TestCase):
    """Only drives trivial python sleep/print commands, never the real SAM3 script."""

    def test_start_stop_lifecycle_with_dummy_command(self) -> None:
        from ui.process_manager import ProcessManager

        manager = ProcessManager()
        self.assertFalse(manager.is_running())
        manager.start(["python3", "-c", "import time; [print(i) or time.sleep(0.05) for i in range(50)]"])
        self.assertTrue(manager.is_running())
        manager.stop(timeout=5.0)
        log_text, state = manager.snapshot()
        self.assertFalse(state.running)
        self.assertTrue(state.stopped_by_user)

    def test_second_start_while_running_raises(self) -> None:
        from ui.process_manager import ProcessManager

        manager = ProcessManager()
        manager.start(["python3", "-c", "import time; time.sleep(2)"])
        try:
            with self.assertRaises(RuntimeError):
                manager.start(["python3", "-c", "print(1)"])
        finally:
            manager.stop(timeout=5.0)

    def test_child_runs_in_its_own_process_group(self) -> None:
        import os

        from ui.process_manager import ProcessManager

        manager = ProcessManager()
        manager.start(["python3", "-c", "import time; time.sleep(10)"])
        try:
            child_pid = manager._process.pid
            # start_new_session makes the child a session/group leader: pgid == pid,
            # and distinct from the test process's own group.
            self.assertEqual(os.getpgid(child_pid), child_pid)
            self.assertNotEqual(os.getpgid(child_pid), os.getpgid(0))
        finally:
            manager.stop(timeout=5.0)

    def test_stop_kills_grandchild_too(self) -> None:
        import os
        import time

        from ui.process_manager import ProcessManager

        # Parent spawns a grandchild sleeper and prints its pid, then sleeps.
        parent_code = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        manager = ProcessManager()
        manager.start(["python3", "-c", parent_code])
        grandchild_pid = None
        deadline = time.time() + 10
        while time.time() < deadline:
            log_text, _ = manager.snapshot()
            lines = [line for line in log_text.splitlines() if line.strip().isdigit()]
            if lines:
                grandchild_pid = int(lines[0])
                break
            time.sleep(0.1)
        self.assertIsNotNone(grandchild_pid, "grandchild pid never appeared in the log")

        manager.stop(timeout=5.0)

        # SIGTERM to the group should take the grandchild down with the parent.
        deadline = time.time() + 5
        grandchild_alive = True
        while time.time() < deadline:
            try:
                os.kill(grandchild_pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                grandchild_alive = False
                break
        self.assertFalse(grandchild_alive, f"grandchild {grandchild_pid} survived stop()")
        log_text, state = manager.snapshot()
        self.assertTrue(state.stopped_by_user)
        self.assertFalse(state.running)
        # captured stdout stays available after the stop
        self.assertIn(str(grandchild_pid), log_text)

    def test_shutdown_stops_active_task(self) -> None:
        from ui.process_manager import ProcessManager

        manager = ProcessManager()
        manager.start(["python3", "-c", "import time; time.sleep(30)"])
        self.assertTrue(manager.is_running())
        manager.shutdown()
        _, state = manager.snapshot()
        self.assertFalse(state.running)

    def test_shutdown_with_no_active_task_is_noop(self) -> None:
        from ui.process_manager import ProcessManager

        manager = ProcessManager()
        manager.shutdown()  # must not raise
        self.assertFalse(manager.is_running())


@unittest.skipUnless(REAL_RUN_DIR.exists(), "real inference run fixture not present")
class CvatExportCallTest(unittest.TestCase):
    """Exercises core.cvat_export against a temp copy so the real run is never mutated."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "run_copy"
        shutil.copytree(REAL_RUN_DIR, self.run_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_polygon_export(self) -> None:
        from core.cvat_export import export_cvat_package

        report = export_cvat_package(self.run_dir, segmentation_format="polygon")
        self.assertTrue(report["ok"])
        legacy_path = self.run_dir / "cvat_export" / "annotations" / "instances_default.json"
        data = json.loads(legacy_path.read_text(encoding="utf-8"))
        vertex_counts = [len(poly) // 2 for ann in data["annotations"] for poly in ann["segmentation"]]
        self.assertTrue(vertex_counts)
        self.assertLessEqual(max(vertex_counts), 8)

    def test_polygon_zip_uses_simplified_annotations_entrypoint(self) -> None:
        from core.cvat_export import export_cvat_package

        report = export_cvat_package(self.run_dir, segmentation_format="polygon", make_zip=True)
        self.assertTrue(report["ok"])
        zip_path = Path(report["zip_path"])
        self.assertTrue(zip_path.exists())
        with zipfile.ZipFile(zip_path) as zf:
            self.assertIn("annotations/instances_default.json", zf.namelist())
            data = json.loads(zf.read("annotations/instances_default.json").decode("utf-8"))
        vertex_counts = [len(poly) // 2 for ann in data["annotations"] for poly in ann["segmentation"]]
        self.assertTrue(vertex_counts)
        self.assertLessEqual(max(vertex_counts), 8)

    def test_rle_export(self) -> None:
        from core.cvat_export import export_cvat_package

        report = export_cvat_package(self.run_dir, segmentation_format="rle")
        self.assertTrue(report["ok"])
        self.assertEqual(report["exact_mask_matches"], report["exact_mask_total"])

    def test_both_export(self) -> None:
        from core.cvat_export import export_cvat_package

        report = export_cvat_package(self.run_dir, segmentation_format="both")
        self.assertTrue(report["ok"])
        self.assertIn("polygon", report["modes"])
        self.assertIn("rle", report["modes"])

    def test_cvat_page_wrapper_does_not_raise(self) -> None:
        from ui.cvat_page import run_cvat_export

        # run_id won't resolve under INFERENCE_RUNS_ROOT, so this should return a
        # clean error string, not raise.
        result_text = run_cvat_export(
            run_id=self.run_dir.name,
            segmentation_format="polygon",
            make_zip=False,
            do_polygon_fidelity=False,
            do_nms_review=False,
            nms_pair="",
        )
        self.assertIn("ERROR", result_text)
        self.assertNotIn("Traceback", result_text)


class TrainingPreflightCallTest(unittest.TestCase):
    def test_inspect_training_config_does_not_start_training(self) -> None:
        from core.training_runner import inspect_training_config

        output_root = DEFAULT_TRAINING_RUN_ROOT / f"_test_ui_preflight_{id(self)}"
        try:
            preflight = inspect_training_config(
                training_prompt="book spine",
                max_epochs=1,
                output_root=output_root,
                prepare_runtime=True,
            )
            # No training process is ever spawned by this call; only files are written.
            self.assertTrue(preflight.config_exists)
            self.assertTrue(output_root.exists())
            if not preflight.errors:
                self.assertIsNotNone(preflight.runtime_config_path)
                self.assertTrue(Path(preflight.runtime_config_path).exists())
                # Training launches through the book01 Hydra wrapper (train.py's -c is a
                # pkg://sam3.train config name and cannot load the per-run YAML path).
                self.assertIn("launch_sam3_training.py", " ".join(preflight.command))
                self.assertIn("--num-gpus", preflight.command)
        finally:
            if output_root.exists():
                shutil.rmtree(output_root)

    def test_training_page_wrapper_does_not_raise(self) -> None:
        from ui.training_preflight_page import run_training_preflight
        from core.config import DEFAULT_BOOK_SPINE_FINETUNE_CONFIG, DEFAULT_SAM3_CHECKPOINT

        with tempfile.TemporaryDirectory() as tmp:
            result_text, state, status = run_training_preflight(
                config_path=str(DEFAULT_BOOK_SPINE_FINETUNE_CONFIG),
                train_images="",
                train_annotations="",
                val_images="",
                val_annotations="",
                checkpoint=str(DEFAULT_SAM3_CHECKPOINT),
                training_prompt="book spine",
                output_root=str(Path(tmp) / "training_runs"),
                max_epochs="",
                train_batch_size="",
                gradient_accumulation_steps="",
                learning_rate="",
                num_workers="",
                num_gpus="1",
            )
            self.assertNotIn("Traceback", result_text)


class PathValidationAndUnicodeTest(unittest.TestCase):
    def test_check_input_path_rejects_missing(self) -> None:
        from ui.ui_utils import check_input_path

        self.assertIsNotNone(check_input_path("/definitely/not/a/real/path"))

    def test_check_input_path_accepts_existing_dir(self) -> None:
        from ui.ui_utils import check_input_path

        self.assertIsNone(check_input_path(str(BOOK_ROOT), must_exist=True, must_be_dir=True))

    def test_filename_with_space(self) -> None:
        from ui.run_reader import list_image_names

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = [{"file_name": "book spine photo 1.png", "status": "ok"}]
            import json

            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            names = list_image_names(run_dir)
            self.assertEqual(names, ["book spine photo 1.png"])

    def test_filename_with_chinese(self) -> None:
        from ui.run_reader import list_image_names

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = [{"file_name": "书脊照片一.png", "status": "ok"}]
            import json

            (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            names = list_image_names(run_dir)
            self.assertEqual(names, ["书脊照片一.png"])

    def test_filename_with_japanese(self) -> None:
        from ui.run_reader import list_image_names

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = [{"file_name": "本棚の写真一.png", "status": "ok"}]
            import json

            (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            names = list_image_names(run_dir)
            self.assertEqual(names, ["本棚の写真一.png"])

    def test_summarize_run_with_unicode_and_space_paths(self) -> None:
        from ui.run_reader import summarize_run

        with tempfile.TemporaryDirectory() as base:
            run_dir = Path(base) / "run 実行 中文"
            run_dir.mkdir()
            import json

            (run_dir / "run_config.json").write_text(json.dumps({"created_at": "now", "prompt": "book spine"}), encoding="utf-8")
            (run_dir / "manifest.json").write_text(
                json.dumps([{"file_name": "図書 の 書脊.png", "status": "ok", "raw_prediction_count": 1, "nms_instance_count": 1}]),
                encoding="utf-8",
            )
            (run_dir / "errors.json").write_text("[]", encoding="utf-8")
            summary = summarize_run(run_dir)
            self.assertEqual(summary.input_images, 1)
            self.assertEqual(summary.success_images, 1)


if __name__ == "__main__":
    unittest.main()
