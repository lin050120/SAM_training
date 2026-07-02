from __future__ import annotations

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BOOK_ROOT, DEFAULT_BOOK_SPINE_DATASET_ROOT, DEFAULT_BOOK_SPINE_FINETUNE_CONFIG, DEFAULT_SAM3_CHECKPOINT

REAL_CONFIG_EXISTS = DEFAULT_BOOK_SPINE_FINETUNE_CONFIG.exists()
REAL_CHECKPOINT_EXISTS = DEFAULT_SAM3_CHECKPOINT.exists()
REAL_DATASET_EXISTS = (DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "annotations.json").exists()
_TRAINING_FIXTURES_AVAILABLE = REAL_CONFIG_EXISTS and REAL_CHECKPOINT_EXISTS and REAL_DATASET_EXISTS


def _scratch_output_root(tag: str) -> Path:
    """A temp output root inside runs/ so path-inside-workspace checks pass, always
    cleaned up by the caller."""
    return BOOK_ROOT / "runs" / f"_test_e1_{tag}_{int(time.time() * 1000)}"


@unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
class RuntimeYamlOverrideTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output_root = _scratch_output_root("override")

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def test_overrides_are_written_to_runtime_yaml(self) -> None:
        from omegaconf import OmegaConf

        from core.training_runner import inspect_training_config

        preflight = inspect_training_config(
            max_epochs=3,
            train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=0.0001,
            num_workers=4,
            training_prompt="book spine",
            output_root=self.output_root,
            prepare_runtime=True,
        )
        self.assertEqual(preflight.errors, [])
        self.assertEqual(preflight.max_epochs, 3)
        self.assertEqual(preflight.train_batch_size, 2)
        self.assertEqual(preflight.gradient_accumulation_steps, 8)
        self.assertEqual(preflight.learning_rate, 0.0001)
        self.assertEqual(preflight.num_workers, 4)
        self.assertEqual(preflight.effective_batch_size, 2 * 1 * 8)

        cfg = OmegaConf.load(preflight.runtime_config_path)
        self.assertEqual(OmegaConf.select(cfg, "trainer.max_epochs"), 3)
        self.assertEqual(OmegaConf.select(cfg, "scratch.train_batch_size"), 2)
        self.assertEqual(OmegaConf.select(cfg, "scratch.gradient_accumulation_steps"), 8)
        self.assertEqual(OmegaConf.select(cfg, "scratch.num_train_workers"), 4)
        self.assertAlmostEqual(OmegaConf.select(cfg, "scratch.lr_transformer"), 0.0001)
        # val workers and the frozen backbone LRs must stay untouched
        self.assertEqual(OmegaConf.select(cfg, "scratch.num_val_workers"), 0)
        self.assertEqual(OmegaConf.select(cfg, "scratch.lr_vision_backbone"), 0.0)
        self.assertEqual(OmegaConf.select(cfg, "scratch.lr_language_backbone"), 0.0)

    def test_no_overrides_fall_back_to_base_yaml(self) -> None:
        from core.training_runner import inspect_training_config

        preflight = inspect_training_config(
            training_prompt="book spine",
            output_root=self.output_root,
            prepare_runtime=True,
        )
        self.assertEqual(preflight.errors, [])
        self.assertEqual(preflight.requested_max_epochs, None)
        self.assertEqual(preflight.requested_train_batch_size, None)
        self.assertEqual(preflight.requested_gradient_accumulation_steps, None)
        self.assertEqual(preflight.requested_learning_rate, None)
        self.assertEqual(preflight.requested_num_workers, None)
        # base YAML values, per docs/training_path_audit.md
        self.assertEqual(preflight.train_batch_size, 1)
        self.assertEqual(preflight.gradient_accumulation_steps, 4)
        self.assertEqual(preflight.effective_batch_size, 4)
        self.assertEqual(preflight.max_epochs, 20)
        self.assertEqual(preflight.num_workers, 10)

    def test_nan_override_is_rejected_and_not_written(self) -> None:
        from omegaconf import OmegaConf

        from core.training_runner import inspect_training_config

        preflight = inspect_training_config(
            learning_rate=float("nan"),
            training_prompt="book spine",
            output_root=self.output_root,
            prepare_runtime=True,
        )
        self.assertTrue(any("learning_rate" in e for e in preflight.errors))
        # prepare_runtime is skipped once errors exist, so no runtime YAML is written
        self.assertIsNone(preflight.runtime_config_path) if preflight.runtime_config_path is None else self.assertFalse(
            Path(preflight.runtime_config_path).exists()
        )

    def test_max_epochs_validation(self) -> None:
        from core.training_runner import inspect_training_config

        for bad in [0, -1, 1.5, float("nan")]:
            preflight = inspect_training_config(
                max_epochs=bad, output_root=self.output_root, prepare_runtime=False
            )
            self.assertTrue(any("max_epochs" in e for e in preflight.errors), f"expected error for {bad!r}")

    def test_train_batch_size_validation(self) -> None:
        from core.training_runner import inspect_training_config

        for bad in [0, -2, 1.5]:
            preflight = inspect_training_config(
                train_batch_size=bad, output_root=self.output_root, prepare_runtime=False
            )
            self.assertTrue(any("train_batch_size" in e for e in preflight.errors), f"expected error for {bad!r}")

    def test_gradient_accumulation_validation(self) -> None:
        from core.training_runner import inspect_training_config

        for bad in [0, -4, 2.5]:
            preflight = inspect_training_config(
                gradient_accumulation_steps=bad, output_root=self.output_root, prepare_runtime=False
            )
            self.assertTrue(
                any("gradient_accumulation_steps" in e for e in preflight.errors), f"expected error for {bad!r}"
            )

    def test_learning_rate_validation(self) -> None:
        from core.training_runner import inspect_training_config

        for bad in [0, -0.001, float("nan"), float("inf")]:
            preflight = inspect_training_config(
                learning_rate=bad, output_root=self.output_root, prepare_runtime=False
            )
            self.assertTrue(any("learning_rate" in e for e in preflight.errors), f"expected error for {bad!r}")

    def test_num_workers_validation(self) -> None:
        from core.training_runner import inspect_training_config

        for bad in [-1, 2.5]:
            preflight = inspect_training_config(
                num_workers=bad, output_root=self.output_root, prepare_runtime=False
            )
            self.assertTrue(any("num_workers" in e for e in preflight.errors), f"expected error for {bad!r}")

    def test_num_workers_zero_is_valid(self) -> None:
        # 0 is PyTorch DataLoader's "load in the main process" — a real, common
        # value, not an invalid one (the base YAML's own val split already uses it).
        from core.training_runner import inspect_training_config

        preflight = inspect_training_config(
            num_workers=0, output_root=self.output_root, prepare_runtime=False
        )
        self.assertFalse(any("num_workers" in e for e in preflight.errors), preflight.errors)

    def test_valid_overrides_produce_no_errors(self) -> None:
        from core.training_runner import inspect_training_config

        preflight = inspect_training_config(
            max_epochs=1, train_batch_size=1, gradient_accumulation_steps=1, learning_rate=1e-5, num_workers=2,
            training_prompt="book spine", output_root=self.output_root, prepare_runtime=False,
        )
        self.assertEqual(preflight.errors, [])


class OptionalTrainingFieldParserTest(unittest.TestCase):
    def test_positive_float_parser_unit_cases(self) -> None:
        from ui.ui_utils import parse_optional_positive_float

        self.assertEqual(parse_optional_positive_float(None), (None, None))
        self.assertEqual(parse_optional_positive_float(""), (None, None))
        self.assertEqual(parse_optional_positive_float("  "), (None, None))
        self.assertEqual(parse_optional_positive_float(0.0001), (0.0001, None))
        self.assertEqual(parse_optional_positive_float("1e-5"), (1e-5, None))
        for bad in [0, -1.0, float("nan"), float("inf"), "abc", True]:
            value, error = parse_optional_positive_float(bad)
            self.assertIsNone(value, f"expected rejection for {bad!r}")
            self.assertIsNotNone(error)

    def test_positive_int_parser_allow_zero(self) -> None:
        from ui.ui_utils import parse_optional_positive_int

        value, error = parse_optional_positive_int(0)
        self.assertIsNone(value)
        self.assertIsNotNone(error)

        value, error = parse_optional_positive_int(0, allow_zero=True)
        self.assertEqual(value, 0)
        self.assertIsNone(error)

        value, error = parse_optional_positive_int(-1, allow_zero=True)
        self.assertIsNone(value)
        self.assertIsNotNone(error)


@unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
class TrainingPageStageATest(unittest.TestCase):
    def setUp(self) -> None:
        self.output_root = _scratch_output_root("page_stage_a")

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root)

    def _run(self, **overrides):
        from ui.training_preflight_page import run_training_preflight

        kwargs = dict(
            config_path=str(DEFAULT_BOOK_SPINE_FINETUNE_CONFIG),
            train_images=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "images"),
            train_annotations=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "annotations.json"),
            val_images=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "images"),
            val_annotations=str(DEFAULT_BOOK_SPINE_DATASET_ROOT / "val" / "annotations.json"),
            checkpoint=str(DEFAULT_SAM3_CHECKPOINT),
            training_prompt="book spine",
            output_root=str(self.output_root),
            max_epochs="", train_batch_size="", gradient_accumulation_steps="",
            learning_rate="", num_workers="", num_gpus="1",
        )
        kwargs.update(overrides)
        return self._run_kwargs(**kwargs)

    @staticmethod
    def _run_kwargs(**kwargs):
        from ui.training_preflight_page import run_training_preflight

        return run_training_preflight(**kwargs)

    def test_successful_preflight_sets_ok_state(self) -> None:
        result_text, state, status = self._run()
        self.assertTrue(state["ok"], result_text)
        self.assertIn("预检通过", status)
        self.assertTrue(Path(state["run_dir"]).exists())
        self.assertTrue(Path(state["runtime_yaml"]).exists())
        self.assertIsInstance(state["command"], list)
        self.assertTrue(all(isinstance(part, str) for part in state["command"]))

    def test_invalid_numeric_field_blocks_preflight_without_running_it(self) -> None:
        result_text, state, status = self._run(max_epochs="not-a-number")
        self.assertFalse(state["ok"])
        self.assertIn("VALIDATION FAILED", result_text)
        # must not have created a run directory for an unparseable request
        self.assertFalse((self.output_root).exists())

    def test_num_workers_zero_through_page_is_accepted(self) -> None:
        result_text, state, status = self._run(num_workers="0")
        self.assertTrue(state["ok"], result_text)
        self.assertIn('"num_workers": 0', result_text)

    def test_num_gpus_validation(self) -> None:
        result_text, state, status = self._run(num_gpus="0")
        self.assertFalse(state["ok"])
        self.assertIn("num_gpus", result_text)

    def test_invalidate_preflight_resets_state(self) -> None:
        from ui.training_preflight_page import invalidate_preflight

        _, state, _ = self._run()
        self.assertTrue(state["ok"])
        new_state, status_text = invalidate_preflight("anything")
        self.assertFalse(new_state["ok"])
        self.assertIsNone(new_state["run_dir"])
        self.assertIn("失效", status_text)


class ValidateCanStartTrainingTest(unittest.TestCase):
    """Pure-function gate; no Gradio, no subprocess."""

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            preflight_ok=True,
            run_dir="/tmp/does-not-matter",
            runtime_yaml="/tmp/does-not-matter",
            checkpoint="/tmp/does-not-matter",
            train_images="/tmp/does-not-matter",
            train_annotations="/tmp/does-not-matter",
            val_images="/tmp/does-not-matter",
            val_annotations="/tmp/does-not-matter",
            confirmed=True,
            already_running=False,
            cuda_available=True,
            requested_num_gpus=1,
            cuda_device_count=1,
        )
        kwargs.update(overrides)
        return kwargs

    def test_preflight_not_ok_blocks(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        reasons = validate_can_start_training(**self._base_kwargs(preflight_ok=False))
        self.assertTrue(any("预检" in r for r in reasons))

    def test_missing_confirmation_blocks(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        with tempfile.TemporaryDirectory() as tmp:
            reasons = validate_can_start_training(
                **self._base_kwargs(
                    confirmed=False, run_dir=tmp, runtime_yaml=str(Path(tmp) / "x"), checkpoint=str(Path(tmp) / "y")
                )
            )
        self.assertTrue(any("确认框" in r for r in reasons))

    def test_missing_runtime_yaml_blocks(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        with tempfile.TemporaryDirectory() as tmp:
            reasons = validate_can_start_training(
                **self._base_kwargs(run_dir=tmp, runtime_yaml=str(Path(tmp) / "missing.yaml"))
            )
        self.assertTrue(any("runtime YAML" in r for r in reasons))

    def test_missing_checkpoint_blocks(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "runtime.yaml"
            yaml_path.write_text("x: 1")
            reasons = validate_can_start_training(
                **self._base_kwargs(run_dir=tmp, runtime_yaml=str(yaml_path), checkpoint=str(Path(tmp) / "missing.pt"))
            )
        self.assertTrue(any("checkpoint" in r for r in reasons))

    def test_cuda_unavailable_fails_fast(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        reasons = validate_can_start_training(**self._base_kwargs(cuda_available=False, cuda_device_count=0))
        self.assertTrue(any("CUDA" in r for r in reasons))

    def test_insufficient_gpu_count_blocks(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        reasons = validate_can_start_training(**self._base_kwargs(requested_num_gpus=4, cuda_device_count=1))
        self.assertTrue(any("num_gpus" in r for r in reasons))

    def test_already_running_blocks(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        reasons = validate_can_start_training(**self._base_kwargs(already_running=True))
        self.assertTrue(any("已有一个训练任务" in r for r in reasons))

    def test_all_conditions_met_returns_empty(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "runtime.yaml"
            yaml_path.write_text("x: 1")
            ckpt_path = Path(tmp) / "ckpt.pt"
            ckpt_path.write_text("fake")
            reasons = validate_can_start_training(
                **self._base_kwargs(
                    run_dir=tmp,
                    runtime_yaml=str(yaml_path),
                    checkpoint=str(ckpt_path),
                    train_images=tmp,
                    train_annotations=tmp,
                    val_images=tmp,
                    val_annotations=tmp,
                )
            )
        self.assertEqual(reasons, [])


class TrainingProcessManagerTest(unittest.TestCase):
    """Only drives fake python commands, never SAM3's train.py."""

    def setUp(self) -> None:
        from ui.process_manager import ProcessManager

        # Fresh instance per test instead of the shared singleton, so tests don't
        # interfere with each other.
        self.manager = ProcessManager()

    def tearDown(self) -> None:
        if self.manager.is_running():
            self.manager.stop(timeout=5.0)

    def test_pid_and_pgid_reported(self) -> None:
        self.manager.start(["python3", "-c", "import time; time.sleep(10)"])
        self.assertIsNotNone(self.manager.pid)
        self.assertEqual(self.manager.pgid, self.manager.pid)

    def test_realtime_stdout_capture(self) -> None:
        self.manager.start(["python3", "-c", "import time\nfor i in range(5):\n print(f'line {i}', flush=True)\n time.sleep(0.05)"])
        time.sleep(0.4)
        log_text, _ = self.manager.snapshot()
        self.assertIn("line", log_text)

    def test_status_completed(self) -> None:
        from ui.training_process_manager import training_status_label

        self.manager.start(["python3", "-c", "print('ok')"])
        deadline = time.time() + 5
        while self.manager.is_running() and time.time() < deadline:
            time.sleep(0.05)
        _, state = self.manager.snapshot()
        self.assertEqual(training_status_label(state), "completed")
        self.assertEqual(state.returncode, 0)

    def test_status_failed(self) -> None:
        from ui.training_process_manager import training_status_label

        self.manager.start(["python3", "-c", "import sys; sys.exit(3)"])
        deadline = time.time() + 5
        while self.manager.is_running() and time.time() < deadline:
            time.sleep(0.05)
        _, state = self.manager.snapshot()
        self.assertEqual(training_status_label(state), "failed")
        self.assertEqual(state.returncode, 3)

    def test_status_cancelled(self) -> None:
        from ui.training_process_manager import training_status_label

        self.manager.start(["python3", "-c", "import time; time.sleep(30)"])
        self.manager.stop(timeout=5.0)
        _, state = self.manager.snapshot()
        self.assertEqual(training_status_label(state), "cancelled")

    def test_second_start_while_running_is_rejected(self) -> None:
        self.manager.start(["python3", "-c", "import time; time.sleep(5)"])
        with self.assertRaises(RuntimeError):
            self.manager.start(["python3", "-c", "print(1)"])

    def test_stop_kills_grandchild_process_group(self) -> None:
        import os

        parent_code = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        self.manager.start(["python3", "-c", parent_code])
        grandchild_pid = None
        deadline = time.time() + 10
        while time.time() < deadline:
            log_text, _ = self.manager.snapshot()
            lines = [line for line in log_text.splitlines() if line.strip().isdigit()]
            if lines:
                grandchild_pid = int(lines[0])
                break
            time.sleep(0.1)
        self.assertIsNotNone(grandchild_pid)

        self.manager.stop(timeout=5.0)

        deadline = time.time() + 5
        alive = True
        while time.time() < deadline:
            try:
                os.kill(grandchild_pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                alive = False
                break
        self.assertFalse(alive, "grandchild survived stop() — orphan risk")


class FinalizeTrainingSummaryTest(unittest.TestCase):
    def test_summary_content_and_checkpoint_discovery(self) -> None:
        from ui.process_manager import ProcessManager
        from ui.training_process_manager import finalize_training_summary

        manager = ProcessManager()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "checkpoints" / "epoch_1.pt").write_text("fake")

            import ui.training_process_manager as tpm

            original = tpm.training_process_manager
            tpm.training_process_manager = manager
            try:
                manager.start(["python3", "-c", "print('done')"])
                deadline = time.time() + 5
                while manager.is_running() and time.time() < deadline:
                    time.sleep(0.05)
                summary = finalize_training_summary(run_dir, ["fake", "cmd"], "runtime.yaml", "/fake/ckpt.pt")
            finally:
                tpm.training_process_manager = original

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(summary["command"], ["fake", "cmd"])
        self.assertEqual(len(summary["discovered_checkpoint_files"]), 1)
        self.assertIn("epoch_1.pt", summary["discovered_checkpoint_files"][0])

    def test_failed_status_records_error(self) -> None:
        from ui.process_manager import ProcessManager
        from ui.training_process_manager import finalize_training_summary

        manager = ProcessManager()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run1"
            run_dir.mkdir()

            import ui.training_process_manager as tpm

            original = tpm.training_process_manager
            tpm.training_process_manager = manager
            try:
                manager.start(["python3", "-c", "import sys; sys.exit(1)"])
                deadline = time.time() + 5
                while manager.is_running() and time.time() < deadline:
                    time.sleep(0.05)
                summary = finalize_training_summary(run_dir, ["fake"], None, None)
                self.assertTrue((run_dir / "training_summary.json").exists())
            finally:
                tpm.training_process_manager = original

        self.assertEqual(summary["status"], "failed")
        self.assertTrue(summary["errors"])
        self.assertEqual(summary["discovered_checkpoint_files"], [])


class TrainingMetricParsingTest(unittest.TestCase):
    def test_parses_common_fields_when_present(self) -> None:
        from ui.training_process_manager import parse_training_metrics

        log = "epoch 3 iter 120 loss 0.4521 lr 8e-05\nsome unrelated line\nmemory 12345MB"
        metrics = parse_training_metrics(log)
        self.assertEqual(metrics["epoch"], "3")
        self.assertEqual(metrics["iteration"], "120")
        self.assertEqual(metrics["loss"], "0.4521")

    def test_missing_fields_report_unavailable_not_crash(self) -> None:
        from ui.training_process_manager import parse_training_metrics

        metrics = parse_training_metrics("totally unrelated log content with no numbers")
        self.assertEqual(metrics["epoch"], "unavailable")
        self.assertEqual(metrics["gpu_memory"], "unavailable")

    def test_empty_log_does_not_crash(self) -> None:
        from ui.training_process_manager import parse_training_metrics

        metrics = parse_training_metrics("")
        self.assertTrue(all(v == "unavailable" for v in metrics.values()))


class TrainingRunHistoryRobustnessTest(unittest.TestCase):
    def test_training_run_missing_summary_shows_placeholder_status(self) -> None:
        from ui.run_reader import list_training_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_without_summary").mkdir()
            rows = list_training_runs(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "preflight_only_or_unknown")
        self.assertIsNone(rows[0]["training_summary"])

    def test_training_run_with_corrupt_dataset_info_does_not_crash(self) -> None:
        from ui.run_reader import list_training_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "bad_run"
            run_dir.mkdir()
            (run_dir / "dataset_info.json").write_text("{not valid json", encoding="utf-8")
            rows = list_training_runs(root)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["dataset_info"])

    def test_training_run_with_summary_reads_status(self) -> None:
        from ui.run_reader import list_training_runs
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "good_run"
            run_dir.mkdir()
            (run_dir / "training_summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            rows = list_training_runs(root)
        self.assertEqual(rows[0]["status"], "completed")


@unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
class UnicodeAndSpacePathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_output_root_with_space_and_unicode(self) -> None:
        from core.training_runner import inspect_training_config

        output_root = BOOK_ROOT / "runs" / "训练 输出 テスト"
        try:
            preflight = inspect_training_config(
                training_prompt="book spine",
                output_root=output_root,
                prepare_runtime=True,
            )
            self.assertEqual(preflight.errors, [])
            self.assertTrue(Path(preflight.run_dir).exists())
        finally:
            if output_root.exists():
                shutil.rmtree(output_root)

    def test_validate_can_start_training_with_unicode_space_paths(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        run_dir = self.base / "run 実行 中文"
        run_dir.mkdir()
        yaml_path = run_dir / "runtime.yaml"
        yaml_path.write_text("x: 1")
        ckpt_path = self.base / "check point 模型.pt"
        ckpt_path.write_text("fake")
        reasons = validate_can_start_training(
            preflight_ok=True,
            run_dir=str(run_dir),
            runtime_yaml=str(yaml_path),
            checkpoint=str(ckpt_path),
            train_images=str(self.base),
            train_annotations=str(self.base),
            val_images=str(self.base),
            val_annotations=str(self.base),
            confirmed=True,
            already_running=False,
            cuda_available=True,
            requested_num_gpus=1,
            cuda_device_count=1,
        )
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
