from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    BOOK_ROOT,
    DEFAULT_BOOK_SPINE_DATASET_ROOT,
    DEFAULT_BOOK_SPINE_FINETUNE_CONFIG,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_CONDA_ENV,
    EXPECTED_SAM3_INIT,
    EXPECTED_SAM3_PACKAGE_DIR,
    EXPECTED_SAM3_ROOT,
    DEFAULT_TRAINING_RUN_ROOT,
)

REAL_CONFIG_EXISTS = DEFAULT_BOOK_SPINE_FINETUNE_CONFIG.exists()
REAL_CHECKPOINT_EXISTS = DEFAULT_SAM3_CHECKPOINT.exists()
REAL_DATASET_EXISTS = (DEFAULT_BOOK_SPINE_DATASET_ROOT / "train" / "annotations.json").exists()
_TRAINING_FIXTURES_AVAILABLE = REAL_CONFIG_EXISTS and REAL_CHECKPOINT_EXISTS and REAL_DATASET_EXISTS


def _scratch_output_root(tag: str) -> Path:
    """A temp output root inside runs/training so output-root confinement passes, always
    cleaned up by the caller."""
    return DEFAULT_TRAINING_RUN_ROOT / f"_test_e1_{tag}_{int(time.time() * 1000)}"


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

        # effective batch must stay <= the real 8-image train set: the R-5 preflight
        # guard now (correctly) rejects effective batches larger than the dataset,
        # and that rejection has its own dedicated test in PreflightGuardsTest.
        preflight = inspect_training_config(
            max_epochs=3,
            train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=0.0001,
            num_workers=4,
            training_prompt="book spine",
            output_root=self.output_root,
            prepare_runtime=True,
        )
        self.assertEqual(preflight.errors, [])
        self.assertEqual(preflight.max_epochs, 3)
        self.assertEqual(preflight.train_batch_size, 2)
        self.assertEqual(preflight.gradient_accumulation_steps, 4)
        self.assertEqual(preflight.learning_rate, 0.0001)
        self.assertEqual(preflight.num_workers, 4)
        self.assertEqual(preflight.effective_batch_size, 2 * 1 * 4)

        cfg = OmegaConf.load(preflight.runtime_config_path)
        self.assertEqual(OmegaConf.select(cfg, "trainer.max_epochs"), 3)
        self.assertEqual(OmegaConf.select(cfg, "scratch.train_batch_size"), 2)
        self.assertEqual(OmegaConf.select(cfg, "scratch.gradient_accumulation_steps"), 4)
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


class Sam301EnvironmentMigrationTest(unittest.TestCase):
    def test_default_environment_and_expected_root(self) -> None:
        self.assertEqual(DEFAULT_CONDA_ENV, "sam301")
        self.assertEqual(EXPECTED_SAM3_ROOT, Path("/home/book/sam301"))
        self.assertEqual(EXPECTED_SAM3_PACKAGE_DIR, Path("/home/book/sam301/sam3"))
        self.assertEqual(EXPECTED_SAM3_INIT, Path("/home/book/sam301/sam3/__init__.py"))

    def test_training_command_uses_sam301_environment(self) -> None:
        from core.training_runner import inspect_training_config

        preflight = inspect_training_config(output_root=DEFAULT_TRAINING_RUN_ROOT, prepare_runtime=False)
        self.assertEqual(preflight.conda_environment, "sam301")
        self.assertEqual(preflight.command[:4], ["conda", "run", "-n", "sam301"])
        self.assertEqual(preflight.expected_sam3_root, "/home/book/sam301")

    def test_training_subprocess_env_prefixes_expected_root_and_preserves_existing(self) -> None:
        from core.training_runner import training_subprocess_env

        original = {"PYTHONPATH": "/tmp/custom", "OTHER": "1"}
        env = training_subprocess_env(original)
        self.assertEqual(env["PYTHONPATH"], "/home/book/sam301:/tmp/custom")
        self.assertEqual(original["PYTHONPATH"], "/tmp/custom")
        self.assertEqual(env["OTHER"], "1")

    def test_validate_sam3_import_path_accepts_expected_init(self) -> None:
        from core.training_runner import validate_sam3_import_path

        self.assertIsNone(validate_sam3_import_path("/home/book/sam301/sam3/__init__.py"))

    def test_validate_sam3_import_path_rejects_old_and_external_paths(self) -> None:
        from core.training_runner import validate_sam3_import_path

        for actual in ["/home/book/sam3/sam3/__init__.py", "/tmp/sam3/__init__.py", "", None]:
            with self.subTest(actual=actual):
                self.assertIsNotNone(validate_sam3_import_path(actual))

    def test_import_guard_command_uses_sam301_and_is_cwd_independent(self) -> None:
        from core.training_runner import build_sam3_import_guard_command

        command = build_sam3_import_guard_command()
        self.assertEqual(command[:4], ["conda", "run", "-n", "sam301"])
        self.assertIn("Path(sam3.__file__).resolve()", command[-1])

    def test_run_sam3_import_guard_rejects_failures_and_bad_output(self) -> None:
        import subprocess

        from core import training_runner

        with mock.patch.object(training_runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "boom")):
            result = training_runner.run_sam3_import_guard()
            self.assertFalse(result["ok"])
            self.assertIn("exited", result["error"])

        with mock.patch.object(training_runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "not-json\n", "")):
            result = training_runner.run_sam3_import_guard()
            self.assertFalse(result["ok"])
            self.assertIn("parseable JSON", result["error"])

        bad = json.dumps({"python": "/env/bin/python", "sam3": "/home/book/sam3/sam3/__init__.py"})
        with mock.patch.object(training_runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, bad + "\n", "")):
            result = training_runner.run_sam3_import_guard()
            self.assertFalse(result["ok"])
            self.assertIn("/home/book/sam3", result["error"])

    def test_run_sam3_import_guard_accepts_expected_output(self) -> None:
        import subprocess

        from core import training_runner

        good = json.dumps({"python": "/home/book/anaconda3/envs/sam301/bin/python", "sam3": str(EXPECTED_SAM3_INIT)})
        with mock.patch.object(training_runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "warning\n" + good + "\n", "")):
            result = training_runner.run_sam3_import_guard()
            self.assertTrue(result["ok"])
            self.assertEqual(result["sam3"], str(EXPECTED_SAM3_INIT))

    def test_preflight_records_sam301_import_metadata_and_config_summary(self) -> None:
        from core import training_runner

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            output_root = Path(tmp)
            fake_guard = {
                "ok": True,
                "sam3": str(EXPECTED_SAM3_INIT),
                "python": "/home/book/anaconda3/envs/sam301/bin/python",
                "error": None,
            }
            with mock.patch.object(training_runner, "run_sam3_import_guard", return_value=fake_guard):
                preflight = training_runner.inspect_training_config(
                    training_prompt="book spine",
                    output_root=output_root,
                    prepare_runtime=True,
                    collect_import_metadata=True,
                )
            self.assertEqual(preflight.errors, [])
            self.assertEqual(preflight.conda_environment, "sam301")
            self.assertEqual(preflight.resolved_sam3_import_path, str(EXPECTED_SAM3_INIT))
            run_dir = Path(preflight.run_dir)
            command_text = (run_dir / "command.txt").read_text(encoding="utf-8")
            self.assertIn("conda run -n sam301", command_text)
            config_summary = json.loads((run_dir / "training_config_summary.json").read_text(encoding="utf-8"))
            dataset_info = json.loads((run_dir / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(config_summary["conda_environment"], "sam301")
            self.assertEqual(dataset_info["resolved_sam3_import_path"], str(EXPECTED_SAM3_INIT))


def _load_launcher_module():
    import importlib.util

    path = BOOK_ROOT / "scripts" / "launch_sam3_training.py"
    spec = importlib.util.spec_from_file_location("launch_sam3_training_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HydraLaunchRegressionTest(unittest.TestCase):
    """Regression tests for the E2 Hydra launch failure.

    train.py resolves -c inside pkg://sam3.train, so an absolute runtime YAML path
    passed as its config name can never compose (MissingConfigException). Training
    must go through scripts/launch_sam3_training.py, which initializes Hydra from
    the runtime YAML's own directory.
    """

    def test_resolve_config_target_maps_yaml_path_to_dir_and_stem(self) -> None:
        launcher = _load_launcher_module()
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "config" / "runtime_config.yaml"
            yaml_path.parent.mkdir()
            yaml_path.write_text("x: 1")
            config_dir, config_name = launcher.resolve_config_target(yaml_path)
            self.assertEqual(config_dir, str(yaml_path.parent.resolve()))
            self.assertEqual(config_name, "runtime_config")
        with self.assertRaises(FileNotFoundError):
            launcher.resolve_config_target("/definitely/not/there/runtime_config.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            not_yaml = Path(tmp) / "runtime_config.json"
            not_yaml.write_text("{}")
            with self.assertRaises(ValueError):
                launcher.resolve_config_target(not_yaml)

    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
    def test_training_command_uses_wrapper_not_train_py_as_hydra_entry(self) -> None:
        from core.config import DEFAULT_SAM3_TRAIN_SCRIPT, DEFAULT_TRAINING_LAUNCHER
        from core.training_runner import inspect_training_config

        preflight = inspect_training_config(output_root=DEFAULT_TRAINING_RUN_ROOT, prepare_runtime=False)
        command = preflight.command
        self.assertEqual(command[:4], ["conda", "run", "-n", "sam301"])
        self.assertEqual(command[4:6], ["python", str(DEFAULT_TRAINING_LAUNCHER)])
        # regression: train.py must never receive a filesystem path as its Hydra config name
        self.assertNotIn(str(DEFAULT_SAM3_TRAIN_SCRIPT), command)
        config_value = command[command.index("-c") + 1]
        self.assertTrue(config_value.endswith(".yaml"))

    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
    def test_generated_runtime_yaml_composes_via_hydra_and_validate_only_subprocess_passes(self) -> None:
        import subprocess

        from core.training_runner import (
            build_hydra_validation_command,
            inspect_training_config,
            training_subprocess_env,
        )
        from omegaconf import OmegaConf

        launcher = _load_launcher_module()
        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            preflight = inspect_training_config(
                training_prompt="book spine",
                max_epochs=1,
                train_batch_size=1,
                gradient_accumulation_steps=4,
                output_root=Path(tmp),
                prepare_runtime=True,
                collect_import_metadata=False,
            )
            self.assertEqual(preflight.errors, [])
            runtime_yaml = Path(preflight.runtime_config_path)

            # In-process: the exact compose the launch path performs must succeed.
            cfg = launcher.compose_runtime_config(runtime_yaml)
            self.assertEqual(OmegaConf.select(cfg, "trainer.max_epochs"), 1)
            self.assertEqual(OmegaConf.select(cfg, "scratch.train_batch_size"), 1)
            self.assertEqual(OmegaConf.select(cfg, "scratch.gradient_accumulation_steps"), 4)
            self.assertIsNotNone(OmegaConf.select(cfg, "launcher.experiment_log_dir"))

            # Real subprocess in the sam301 env, same command preflight uses.
            command = build_hydra_validation_command(runtime_yaml)
            self.assertEqual(command[:4], ["conda", "run", "-n", "sam301"])
            completed = subprocess.run(
                command,
                cwd=str(BOOK_ROOT),
                env=training_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("hydra validation ok", completed.stdout)

            missing = build_hydra_validation_command(Path(tmp) / "no_such_config.yaml")
            failed = subprocess.run(
                missing,
                cwd=str(BOOK_ROOT),
                env=training_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            self.assertNotEqual(failed.returncode, 0)

    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
    def test_preflight_writes_per_run_distributed_port_range(self) -> None:
        from core import training_runner
        from omegaconf import OmegaConf

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            with mock.patch.object(training_runner, "allocate_distributed_port", side_effect=[41001, 41002]):
                first = training_runner.inspect_training_config(
                    training_prompt="book spine",
                    max_epochs=1,
                    train_batch_size=1,
                    gradient_accumulation_steps=4,
                    output_root=Path(tmp),
                    prepare_runtime=True,
                    collect_import_metadata=False,
                )
                second = training_runner.inspect_training_config(
                    training_prompt="book spine",
                    max_epochs=1,
                    train_batch_size=1,
                    gradient_accumulation_steps=4,
                    output_root=Path(tmp),
                    prepare_runtime=True,
                    collect_import_metadata=False,
                )
            first_cfg = OmegaConf.load(first.runtime_config_path)
            second_cfg = OmegaConf.load(second.runtime_config_path)
            self.assertEqual(OmegaConf.select(first_cfg, "submitit.port_range"), [41001, 41001])
            self.assertEqual(OmegaConf.select(second_cfg, "submitit.port_range"), [41002, 41002])
            self.assertNotEqual(
                first.training_provenance["distributed"]["master_port"],
                second.training_provenance["distributed"]["master_port"],
            )
            self.assertNotEqual(first.training_provenance["distributed"]["master_port"], 34508)

    def test_port_allocator_skips_listening_port(self) -> None:
        from core.training_runner import allocate_distributed_port, is_tcp_port_available

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            occupied = sock.getsockname()[1]
            self.assertFalse(is_tcp_port_available(occupied))
            allocated = allocate_distributed_port()
            self.assertNotEqual(allocated, occupied)
            self.assertTrue(is_tcp_port_available(allocated))

    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
    def test_hydra_validation_failure_blocks_preflight(self) -> None:
        from core import training_runner

        fake_guard = {
            "ok": True,
            "sam3": str(EXPECTED_SAM3_INIT),
            "python": "/home/book/anaconda3/envs/sam301/bin/python",
            "error": None,
        }
        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            with mock.patch.object(training_runner, "run_sam3_import_guard", return_value=fake_guard):
                with mock.patch.object(
                    training_runner,
                    "run_hydra_config_validation",
                    return_value={"ok": False, "error": "boom", "returncode": 1},
                ):
                    preflight = training_runner.inspect_training_config(
                        training_prompt="book spine",
                        output_root=Path(tmp),
                        prepare_runtime=True,
                        collect_import_metadata=True,
                    )
        self.assertTrue(any("hydra config validation failed" in e for e in preflight.errors))

    def test_hydra_validation_command_and_import_guard_stay_on_sam301(self) -> None:
        from core.config import DEFAULT_TRAINING_LAUNCHER
        from core.training_runner import build_hydra_validation_command, build_sam3_import_guard_command

        command = build_hydra_validation_command(Path("/x/config/runtime_config.yaml"))
        self.assertEqual(command[:4], ["conda", "run", "-n", "sam301"])
        self.assertEqual(command[4:6], ["python", str(DEFAULT_TRAINING_LAUNCHER)])
        self.assertIn("--validate-only", command)
        guard_command = build_sam3_import_guard_command()
        self.assertEqual(guard_command[:4], ["conda", "run", "-n", "sam301"])
        self.assertEqual(EXPECTED_SAM3_INIT, Path("/home/book/sam301/sam3/__init__.py"))


@unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
class GradAccumWiringTest(unittest.TestCase):
    """trainer._run_step requires a list of exactly accum_steps micro-batches when
    gradient_accumulation_steps > 1; the runtime YAML must wire
    collate_fn_api_with_chunking and scale the train DataLoader batch_size."""

    def _preflight(self, accum: int, tmp: str):
        from core.training_runner import inspect_training_config

        return inspect_training_config(
            training_prompt="book spine",
            max_epochs=1,
            train_batch_size=1,
            gradient_accumulation_steps=accum,
            output_root=Path(tmp),
            prepare_runtime=True,
            collect_import_metadata=False,
        )

    def test_accum_4_wires_chunking_collate_and_scaled_train_batch_size(self) -> None:
        from omegaconf import OmegaConf

        launcher = _load_launcher_module()
        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            preflight = self._preflight(4, tmp)
            self.assertEqual(preflight.errors, [])
            cfg = OmegaConf.load(preflight.runtime_config_path)
            self.assertEqual(
                OmegaConf.select(cfg, "scratch.collate_fn._target_"),
                "sam3.train.data.collator.collate_fn_api_with_chunking",
            )
            self.assertEqual(OmegaConf.select(cfg, "scratch.collate_fn.num_chunks"), 4)
            self.assertTrue(OmegaConf.select(cfg, "scratch.collate_fn._partial_"))
            self.assertEqual(OmegaConf.select(cfg, "scratch.collate_fn.dict_key"), "all")
            self.assertEqual(OmegaConf.select(cfg, "trainer.data.train.batch_size"), 4)
            self.assertEqual(OmegaConf.select(cfg, "scratch.train_batch_size"), 1)
            self.assertEqual(OmegaConf.select(cfg, "trainer.gradient_accumulation_steps"), 4)
            # val side must stay on the plain collator (no accumulation on val)
            self.assertEqual(
                OmegaConf.select(cfg, "scratch.collate_fn_val._target_"),
                "sam3.train.data.collator.collate_fn_api",
            )
            # and the wired YAML must still compose through the real launch path
            composed = launcher.compose_runtime_config(preflight.runtime_config_path)
            self.assertEqual(OmegaConf.select(composed, "scratch.collate_fn.num_chunks"), 4)
            self.assertEqual(preflight.effective_batch_size, 4)

    def test_accum_1_leaves_collate_and_train_batch_size_untouched(self) -> None:
        from omegaconf import OmegaConf

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            preflight = self._preflight(1, tmp)
            self.assertEqual(preflight.errors, [])
            cfg = OmegaConf.load(preflight.runtime_config_path)
            self.assertEqual(
                OmegaConf.select(cfg, "scratch.collate_fn._target_"),
                "sam3.train.data.collator.collate_fn_api",
            )
            self.assertNotIn("num_chunks", cfg.scratch.collate_fn)
            self.assertEqual(OmegaConf.select(cfg, "trainer.data.train.batch_size"), 1)
            self.assertEqual(preflight.effective_batch_size, 1)


class PreflightGuardsTest(unittest.TestCase):
    """R-4: missing scratch keys must give clean errors, never int(None) TypeError.
    R-5: train set smaller than the effective batch must be rejected (drop_last=True
    would otherwise 'complete' an epoch with zero optimizer steps)."""

    def test_missing_scratch_keys_produce_clean_preflight_error(self) -> None:
        from core.training_runner import inspect_training_config

        with tempfile.TemporaryDirectory() as tmp:
            bad_config = Path(tmp) / "no_scratch.yaml"
            bad_config.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")
            preflight = inspect_training_config(
                config_path=bad_config,
                prepare_runtime=False,
            )
        self.assertTrue(
            any("scratch.train_batch_size is missing" in e for e in preflight.errors),
            preflight.errors,
        )
        self.assertTrue(
            any("scratch.gradient_accumulation_steps is missing" in e for e in preflight.errors),
            preflight.errors,
        )

    def test_write_runtime_yaml_missing_keys_raises_clean_valueerror(self) -> None:
        from core.training_runner import write_runtime_yaml

        with tempfile.TemporaryDirectory() as tmp:
            bad_config = Path(tmp) / "no_scratch.yaml"
            bad_config.write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")
            paths = {
                "initial_checkpoint": Path(tmp) / "ckpt.pt",
                "bpe_path": Path(tmp) / "bpe.gz",
                "train_images": Path(tmp),
                "train_annotations": Path(tmp) / "t.json",
                "val_images": Path(tmp),
                "val_annotations": Path(tmp) / "v.json",
            }
            with self.assertRaises(ValueError) as ctx:
                write_runtime_yaml(bad_config, Path(tmp) / "out" / "runtime.yaml", paths, Path(tmp) / "run")
        self.assertIn("scratch.train_batch_size", str(ctx.exception))

    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
    def test_train_smaller_than_effective_batch_is_rejected(self) -> None:
        from core.training_runner import inspect_training_config

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            # real train set has 8 images; effective = 1 x 16 x 1 = 16 > 8
            preflight = inspect_training_config(
                training_prompt="book spine",
                max_epochs=1,
                train_batch_size=1,
                gradient_accumulation_steps=16,
                output_root=Path(tmp),
                prepare_runtime=True,
                collect_import_metadata=False,
            )
            self.assertTrue(
                any("smaller than the effective batch size" in e for e in preflight.errors),
                preflight.errors,
            )
            # runtime YAML must not be written for a rejected run
            self.assertFalse(Path(preflight.runtime_config_path).exists())

    @unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
    def test_non_divisible_effective_batch_warns_but_allows(self) -> None:
        from core.training_runner import inspect_training_config

        with tempfile.TemporaryDirectory(dir=DEFAULT_TRAINING_RUN_ROOT) as tmp:
            # 8 images, effective = 3 -> 2 full outer batches, 2 images dropped
            preflight = inspect_training_config(
                training_prompt="book spine",
                max_epochs=1,
                train_batch_size=1,
                gradient_accumulation_steps=3,
                output_root=Path(tmp),
                prepare_runtime=True,
                collect_import_metadata=False,
            )
            self.assertEqual(preflight.errors, [])
            self.assertTrue(
                any("not divisible by the effective batch size" in w for w in preflight.warnings),
                preflight.warnings,
            )
            self.assertTrue(Path(preflight.runtime_config_path).exists())


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
        self.assertIsNotNone(state["launch_token"])
        self.assertFalse(state["consumed"])

    def test_repeated_preflight_creates_new_launch_token_run_dir_and_runtime_yaml(self) -> None:
        result_text_1, state_1, _ = self._run()
        result_text_2, state_2, _ = self._run()
        self.assertTrue(state_1["ok"], result_text_1)
        self.assertTrue(state_2["ok"], result_text_2)
        self.assertNotEqual(state_1["launch_token"], state_2["launch_token"])
        self.assertNotEqual(state_1["run_dir"], state_2["run_dir"])
        self.assertNotEqual(state_1["runtime_yaml"], state_2["runtime_yaml"])

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
        import ui.training_process_manager as tpm

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "runtime.yaml"
            yaml_path.write_text("x: 1")
            ckpt_path = Path(tmp) / "ckpt.pt"
            ckpt_path.write_text("fake")
            with mock.patch.object(tpm, "validate_training_run_path", return_value=None):
                reasons = tpm.validate_can_start_training(
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

    def test_consumed_preflight_blocks(self) -> None:
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
                    preflight_consumed=True,
                )
            )
        self.assertTrue(any("已经被启动消费" in r for r in reasons))

    def test_existing_training_artifacts_block_run_dir_reuse(self) -> None:
        import ui.training_process_manager as tpm

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            yaml_path = run_dir / "runtime.yaml"
            yaml_path.write_text("x: 1")
            ckpt_path = run_dir / "initial.pt"
            ckpt_path.write_text("fake")
            (run_dir / "checkpoints").mkdir()
            (run_dir / "checkpoints" / "epoch_1.pt").write_text("old")
            with mock.patch.object(tpm, "validate_training_run_path", return_value=None):
                reasons = tpm.validate_can_start_training(
                    **self._base_kwargs(
                        run_dir=str(run_dir),
                        runtime_yaml=str(yaml_path),
                        checkpoint=str(ckpt_path),
                        train_images=tmp,
                        train_annotations=tmp,
                        val_images=tmp,
                        val_annotations=tmp,
                    )
                )
        self.assertTrue(any("checkpoint" in r and "拒绝复用" in r for r in reasons))

    def test_existing_training_summary_blocks_run_dir_reuse(self) -> None:
        import ui.training_process_manager as tpm

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            yaml_path = run_dir / "runtime.yaml"
            yaml_path.write_text("x: 1")
            ckpt_path = run_dir / "initial.pt"
            ckpt_path.write_text("fake")
            (run_dir / "training_summary.json").write_text("{}")
            with mock.patch.object(tpm, "validate_training_run_path", return_value=None):
                reasons = tpm.validate_can_start_training(
                    **self._base_kwargs(
                        run_dir=str(run_dir),
                        runtime_yaml=str(yaml_path),
                        checkpoint=str(ckpt_path),
                        train_images=tmp,
                        train_annotations=tmp,
                        val_images=tmp,
                        val_annotations=tmp,
                    )
                )
        self.assertTrue(any("training_summary.json" in r and "拒绝复用" in r for r in reasons))


class StartTrainingOneTimePreflightTest(unittest.TestCase):
    """Exercises start_training with fake commands only; never starts SAM3."""

    def setUp(self) -> None:
        from types import SimpleNamespace

        from ui.process_manager import ProcessManager
        import ui.training_preflight_page as tpp
        import ui.training_process_manager as tpm

        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.manager = ProcessManager()
        self.tpp = tpp
        self.tpm = tpm
        self.original_tpp_manager = tpp.training_process_manager
        self.original_tpm_manager = tpm.training_process_manager
        self.original_detect_cuda = tpp.detect_cuda
        self.original_sleep = tpp.time.sleep
        self.original_validate_training_run_path = tpm.validate_training_run_path
        self.original_verify_sam3_import_for_training = tpp.verify_sam3_import_for_training
        self.original_allocate_distributed_port = tpp.allocate_distributed_port
        self.original_configure_runtime_distributed_port = tpp.configure_runtime_distributed_port
        self._next_port = 43000
        tpp.training_process_manager = self.manager
        tpm.training_process_manager = self.manager
        tpm.validate_training_run_path = lambda _path: None
        tpp.verify_sam3_import_for_training = lambda env=None: {
            "ok": True,
            "sam3": str(EXPECTED_SAM3_INIT),
            "expected": str(EXPECTED_SAM3_INIT),
            "conda_environment": DEFAULT_CONDA_ENV,
            "effective_pythonpath": env.get("PYTHONPATH") if env else None,
        }
        def fake_allocate(_master_addr="localhost"):
            self._next_port += 1
            return self._next_port

        def fake_configure(runtime_config, master_port, master_addr="localhost"):
            return {
                "master_addr": master_addr,
                "master_port": int(master_port),
                "port_range": [int(master_port), int(master_port)],
                "runtime_config": str(runtime_config),
            }

        tpp.allocate_distributed_port = fake_allocate
        tpp.configure_runtime_distributed_port = fake_configure
        tpp.detect_cuda = lambda: SimpleNamespace(available=True, device_count=1)
        tpp.time.sleep = lambda _seconds: None
        tpp._consumed_preflight_tokens.clear()

    def tearDown(self) -> None:
        if self.manager.is_running():
            self.manager.stop(timeout=1.0)
        if self.manager._reader_thread is not None:
            self.manager._reader_thread.join(timeout=2.0)
        self.tpp.training_process_manager = self.original_tpp_manager
        self.tpm.training_process_manager = self.original_tpm_manager
        self.tpp.detect_cuda = self.original_detect_cuda
        self.tpp.time.sleep = self.original_sleep
        self.tpm.validate_training_run_path = self.original_validate_training_run_path
        self.tpp.verify_sam3_import_for_training = self.original_verify_sam3_import_for_training
        self.tpp.allocate_distributed_port = self.original_allocate_distributed_port
        self.tpp.configure_runtime_distributed_port = self.original_configure_runtime_distributed_port
        self.tpp._consumed_preflight_tokens.clear()
        self.tmp.cleanup()

    def _state(self, run_name: str = "run1", command: list[str] | None = None) -> dict:
        import uuid

        run_dir = self.base / run_name
        run_dir.mkdir(parents=True)
        runtime_yaml = run_dir / "config" / "runtime_config.yaml"
        runtime_yaml.parent.mkdir()
        runtime_yaml.write_text("x: 1")
        checkpoint = self.base / f"{run_name}_initial.pt"
        checkpoint.write_text("fake")
        train_images = self.base / f"{run_name}_train_images"
        val_images = self.base / f"{run_name}_val_images"
        train_images.mkdir()
        val_images.mkdir()
        train_annotations = self.base / f"{run_name}_train.json"
        val_annotations = self.base / f"{run_name}_val.json"
        train_annotations.write_text("{}")
        val_annotations.write_text("{}")
        (run_dir / "checkpoints").mkdir()
        (run_dir / "logs").mkdir()
        return {
            "ok": True,
            "run_dir": str(run_dir),
            "runtime_yaml": str(runtime_yaml),
            "checkpoint": str(checkpoint),
            "command": command or [sys.executable, "-c", "print('completed fake training')"],
            "train_images": str(train_images),
            "train_annotations": str(train_annotations),
            "val_images": str(val_images),
            "val_annotations": str(val_annotations),
            "num_gpus": 1,
            "launch_token": uuid.uuid4().hex,
            "consumed": False,
        }

    def _run_to_end(self, state: dict, confirmed: bool = True) -> list[tuple[str, str, str, str]]:
        return list(self.tpp.start_training(state, confirmed))

    def test_first_start_succeeds_and_second_same_preflight_is_rejected(self) -> None:
        state = self._state()
        first = self._run_to_end(state)
        self.assertIn("status=completed", first[-1][0])
        self.assertTrue((Path(state["run_dir"]) / "training_summary.json").exists())

        second = self._run_to_end(state)
        self.assertIn("BLOCKED", second[0][0])
        self.assertIn("已经被启动消费", second[0][0])

    def test_completed_failed_and_cancelled_states_do_not_allow_old_preflight_reuse(self) -> None:
        cases = [
            ("completed", [sys.executable, "-c", "print('ok')"]),
            ("failed", [sys.executable, "-c", "import sys; sys.exit(2)"]),
        ]
        for expected, command in cases:
            with self.subTest(expected=expected):
                state = self._state(expected, command)
                first = self._run_to_end(state)
                self.assertIn(f"status={expected}", first[-1][0])
                second = self._run_to_end(state)
                self.assertIn("BLOCKED", second[0][0])
                self.assertIn("已经被启动消费", second[0][0])

        cancelled = self._state("cancelled", [sys.executable, "-c", "import time; time.sleep(30)"])
        gen = self.tpp.start_training(cancelled, True)
        first_status = next(gen)[0]
        self.assertIn("status=running", first_status)
        self.manager.stop(timeout=1.0)
        rest = list(gen)
        self.assertIn("status=cancelled", rest[-1][0])
        second = self._run_to_end(cancelled)
        self.assertIn("BLOCKED", second[0][0])
        self.assertIn("已经被启动消费", second[0][0])

    def test_new_preflight_after_consumption_can_start_with_new_run_dir_and_runtime_yaml(self) -> None:
        old_state = self._state("old_run")
        self._run_to_end(old_state)
        new_state = self._state("new_run")
        new_result = self._run_to_end(new_state)

        self.assertIn("status=completed", new_result[-1][0])
        self.assertNotEqual(old_state["run_dir"], new_state["run_dir"])
        self.assertNotEqual(old_state["runtime_yaml"], new_state["runtime_yaml"])
        self.assertEqual(new_state["runtime_yaml"], (Path(new_state["run_dir"]) / "config" / "runtime_config.yaml").as_posix())

    def test_concurrent_double_start_only_one_request_succeeds(self) -> None:
        state = self._state("concurrent", [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"])
        statuses: list[str] = []

        def start_once() -> None:
            gen = self.tpp.start_training(state, True)
            statuses.append(next(gen)[0])

        threads = [threading.Thread(target=start_once), threading.Thread(target=start_once)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
        if self.manager.is_running():
            self.manager.stop(timeout=1.0)

        self.assertEqual(sum("status=running" in status for status in statuses), 1, statuses)
        self.assertEqual(sum("BLOCKED" in status for status in statuses), 1, statuses)
        self.assertTrue(any("已经被启动消费" in status for status in statuses))

    def test_existing_artifacts_and_parameter_invalidation_block_start(self) -> None:
        state = self._state("artifact")
        (Path(state["run_dir"]) / "checkpoints" / "old.pt").write_text("old")
        blocked = self._run_to_end(state)
        self.assertIn("BLOCKED", blocked[0][0])
        self.assertIn("checkpoint", blocked[0][0])

        invalid_state, _ = self.tpp.invalidate_preflight("changed")
        blocked = self._run_to_end(invalid_state)
        self.assertIn("BLOCKED", blocked[0][0])
        self.assertIn("预检", blocked[0][0])

    def test_confirmation_required_before_preflight_is_consumed(self) -> None:
        state = self._state("confirm")
        blocked = self._run_to_end(state, confirmed=False)
        self.assertIn("BLOCKED", blocked[0][0])
        self.assertIn("确认框", blocked[0][0])
        self.assertFalse(state["consumed"])

    def test_start_failure_consumes_preflight_and_does_not_leave_manager_running(self) -> None:
        state = self._state("start_failure", ["/definitely/not/a/real/executable"])
        first = self._run_to_end(state)
        self.assertIn("ERROR: failed to start training process", first[0][0])
        self.assertTrue(state["consumed"])
        self.assertFalse(self.manager.is_running())

        second = self._run_to_end(state)
        self.assertIn("BLOCKED", second[0][0])
        self.assertIn("已经被启动消费", second[0][0])

    def test_import_guard_failure_rejects_without_starting_trainer(self) -> None:
        self.tpp.verify_sam3_import_for_training = lambda env=None: {
            "ok": False,
            "sam3": "/home/book/sam3/sam3/__init__.py",
            "expected": str(EXPECTED_SAM3_INIT),
            "error": "sam3 import resolved to old source",
        }
        with mock.patch.object(self.manager, "start", wraps=self.manager.start) as start_mock:
            state = self._state("guard_failure")
            result = self._run_to_end(state)
        self.assertIn("ERROR: SAM3 import guard failed", result[0][0])
        start_mock.assert_not_called()
        self.assertTrue(state["consumed"])

    def test_command_txt_and_actual_start_command_can_match_sam301(self) -> None:
        command = ["conda", "run", "-n", DEFAULT_CONDA_ENV, "python", "-c", "print('ok')"]
        state = self._state("sam301_command", command)
        (Path(state["run_dir"]) / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        with mock.patch.object(self.manager, "start", wraps=self.manager.start) as start_mock:
            result = self._run_to_end(state)
        self.assertIn("status=completed", result[-1][0])
        started_command = start_mock.call_args.args[0]
        self.assertEqual(started_command, command)
        self.assertEqual(started_command[:4], ["conda", "run", "-n", "sam301"])
        self.assertEqual((Path(state["run_dir"]) / "command.txt").read_text(encoding="utf-8").strip(), " ".join(started_command))

    def test_summary_records_conda_environment_and_import_metadata(self) -> None:
        state = self._state("summary_env")
        result = self._run_to_end(state)
        self.assertIn("status=completed", result[-1][0])
        summary = json.loads((Path(state["run_dir"]) / "training_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["conda_environment"], "sam301")
        self.assertEqual(summary["expected_sam3_root"], "/home/book/sam301")
        self.assertEqual(summary["resolved_sam3_import_path"], str(EXPECTED_SAM3_INIT))
        self.assertTrue(summary["sam3_import_guard_ok"])
        self.assertTrue(summary["effective_pythonpath"].startswith("/home/book/sam301"))

    def test_distributed_port_failure_blocks_without_consuming_or_spawning(self) -> None:
        state = self._state("port_failure")
        self.tpp.allocate_distributed_port = mock.Mock(side_effect=RuntimeError("no bindable port"))
        with mock.patch.object(self.manager, "start", wraps=self.manager.start) as start_mock:
            result = self._run_to_end(state)
        self.assertIn("BLOCKED", result[0][0])
        self.assertIn("distributed port allocation failed", result[0][0])
        self.assertFalse(state["consumed"])
        self.assertNotIn(state["launch_token"], self.tpp._consumed_preflight_tokens)
        start_mock.assert_not_called()

    def test_normal_start_passes_distributed_port_to_child_env_and_summary(self) -> None:
        state = self._state("port_env")
        self.tpp.allocate_distributed_port = mock.Mock(return_value=42017)
        captured_env: dict[str, str] = {}
        original_start = self.manager.start

        def start_spy(command, cwd=None, env=None, on_finish=None):
            captured_env.update(env or {})
            return original_start(command, cwd=cwd, env=env, on_finish=on_finish)

        with mock.patch.object(self.manager, "start", side_effect=start_spy) as start_mock:
            result = self._run_to_end(state)
        self.assertIn("status=completed", result[-1][0])
        start_mock.assert_called_once()
        self.assertEqual(captured_env["MASTER_ADDR"], "localhost")
        self.assertEqual(captured_env["MASTER_PORT"], "42017")
        self.assertEqual(state["distributed"]["master_port"], 42017)
        summary = json.loads((Path(state["run_dir"]) / "training_summary.json").read_text(encoding="utf-8"))
        provenance = json.loads((Path(state["run_dir"]) / "provenance.json").read_text(encoding="utf-8"))
        config_summary = json.loads((Path(state["run_dir"]) / "training_config_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["distributed"]["master_port"], 42017)
        self.assertEqual(summary["training_provenance"]["distributed"]["master_port"], 42017)
        self.assertEqual(provenance["distributed"]["master_port"], 42017)
        self.assertEqual(config_summary["distributed"]["master_port"], 42017)
        self.assertEqual(config_summary["training_provenance"]["distributed"]["master_port"], 42017)


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

    def test_training_process_manager_shutdown_is_registered_with_atexit_once(self) -> None:
        import importlib
        from unittest import mock

        import ui.training_process_manager as tpm

        callbacks = []
        tpm._TRAINING_SHUTDOWN_REGISTERED = False
        with mock.patch("atexit.register", side_effect=callbacks.append):
            reloaded = importlib.reload(tpm)
        self.assertEqual(len(callbacks), 1)

        class FakeManager:
            def __init__(self) -> None:
                self.called = False

            def shutdown(self) -> None:
                self.called = True

        fake = FakeManager()
        reloaded.training_process_manager = fake
        callbacks[0]()
        self.assertTrue(fake.called)

        callbacks.clear()
        with mock.patch("atexit.register", side_effect=callbacks.append):
            importlib.reload(reloaded)
        self.assertEqual(callbacks, [])

    def test_shutdown_is_idempotent_and_preserves_log_and_cancelled_state(self) -> None:
        self.manager.start([sys.executable, "-c", "import time; print('tail log', flush=True); time.sleep(30)"])
        deadline = time.time() + 5
        while time.time() < deadline:
            log_text, _ = self.manager.snapshot()
            if "tail log" in log_text:
                break
            time.sleep(0.05)
        self.manager.shutdown()
        self.manager.shutdown()
        log_text, state = self.manager.snapshot()
        self.assertIn("tail log", log_text)
        self.assertFalse(state.running)
        self.assertTrue(state.stopped_by_user)

    def test_sigterm_success_does_not_need_sigkill(self) -> None:
        import signal
        from unittest import mock

        original_signal_group = self.manager._signal_group
        signals = []

        def record_and_signal(process, sig):
            signals.append(sig)
            original_signal_group(process, sig)

        self.manager.start([sys.executable, "-c", "import time; time.sleep(30)"])
        with mock.patch.object(self.manager, "_signal_group", side_effect=record_and_signal):
            self.manager.stop(timeout=2.0)
        self.assertIn(signal.SIGTERM, signals)
        self.assertNotIn(signal.SIGKILL, signals)

    def test_sigterm_timeout_uses_sigkill(self) -> None:
        import signal
        from unittest import mock

        original_signal_group = self.manager._signal_group
        signals = []

        def record_and_signal(process, sig):
            signals.append(sig)
            original_signal_group(process, sig)

        code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(30)"
        self.manager.start([sys.executable, "-c", code])
        deadline = time.time() + 5
        while time.time() < deadline:
            log_text, _ = self.manager.snapshot()
            if "ready" in log_text:
                break
            time.sleep(0.05)
        with mock.patch.object(self.manager, "_signal_group", side_effect=record_and_signal):
            self.manager.stop(timeout=0.2)
        self.assertIn(signal.SIGTERM, signals)
        self.assertIn(signal.SIGKILL, signals)

    def test_inference_and_training_managers_do_not_interfere(self) -> None:
        from ui.process_manager import ProcessManager

        inference_manager = ProcessManager()
        training_manager = ProcessManager()
        inference_manager.start([sys.executable, "-c", "import time; time.sleep(30)"])
        training_manager.start([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            training_manager.shutdown()
            self.assertFalse(training_manager.is_running())
            self.assertTrue(inference_manager.is_running())
        finally:
            inference_manager.shutdown()

    def test_start_failure_rolls_back_running_state(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.manager.start(["/definitely/not/a/real/executable"])
        self.assertFalse(self.manager.is_running())
        self.manager.start([sys.executable, "-c", "print('after failure')"])
        deadline = time.time() + 5
        while self.manager.is_running() and time.time() < deadline:
            time.sleep(0.05)
        log_text, state = self.manager.snapshot()
        self.assertIn("after failure", log_text)
        self.assertEqual(state.returncode, 0)


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


class TrainingSummaryBackgroundFinalizationTest(unittest.TestCase):
    """Server-side process completion writes summary without any UI polling."""

    def setUp(self) -> None:
        from ui.process_manager import ProcessManager

        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.manager = ProcessManager()

    def tearDown(self) -> None:
        if self.manager.is_running():
            self.manager.shutdown()
        if self.manager._reader_thread is not None:
            self.manager._reader_thread.join(timeout=2.0)
        self.tmp.cleanup()

    def _run_dir(self, name: str) -> Path:
        run_dir = self.base / name
        (run_dir / "checkpoints").mkdir(parents=True)
        return run_dir

    def _wait_for_summary(self, run_dir: Path) -> dict:
        summary_path = run_dir / "training_summary.json"
        deadline = time.time() + 5
        while time.time() < deadline:
            if summary_path.exists():
                return json.loads(summary_path.read_text(encoding="utf-8"))
            time.sleep(0.05)
        self.fail(f"summary was not written: {summary_path}")

    def _start_with_callback(self, run_dir: Path, command: list[str]) -> None:
        from ui.training_process_manager import make_training_summary_callback

        self.manager.start(
            command,
            on_finish=make_training_summary_callback(run_dir, command, str(run_dir / "config.yaml"), "/fake/ckpt.pt"),
        )

    def test_completed_summary_is_written_without_ui_polling(self) -> None:
        run_dir = self._run_dir("completed")
        (run_dir / "checkpoints" / "epoch_1.pt").write_text("fake")
        command = [
            sys.executable,
            "-c",
            "import sys; print('stdout tail', flush=True); sys.stderr.write('stderr tail\\n'); sys.stderr.flush()",
        ]
        self._start_with_callback(run_dir, command)
        summary = self._wait_for_summary(run_dir)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["exit_code"], 0)
        self.assertGreaterEqual(summary["duration_seconds"], 0)
        self.assertEqual(len(summary["discovered_checkpoint_files"]), 1)
        self.assertIn("epoch_1.pt", summary["discovered_checkpoint_files"][0])
        self.assertIn("stdout tail", summary["stdout_stderr_tail"])
        self.assertIn("stderr tail", summary["stdout_stderr_tail"])
        self.assertFalse(self.manager._reader_thread.is_alive())
        self.assertIsNone(self.manager._on_finish)

    def test_failed_summary_is_written_without_ui_polling_and_no_checkpoint_is_invented(self) -> None:
        run_dir = self._run_dir("failed")
        command = [sys.executable, "-c", "import sys; print('failing', flush=True); sys.exit(7)"]
        self._start_with_callback(run_dir, command)
        summary = self._wait_for_summary(run_dir)

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["exit_code"], 7)
        self.assertEqual(summary["discovered_checkpoint_files"], [])
        self.assertTrue(summary["errors"])
        self.assertIn("failing", summary["stdout_stderr_tail"])

    def test_cancelled_summary_is_written_from_shutdown_path(self) -> None:
        run_dir = self._run_dir("cancelled")
        command = [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"]
        self._start_with_callback(run_dir, command)
        deadline = time.time() + 5
        while time.time() < deadline:
            log_text, _ = self.manager.snapshot()
            if "ready" in log_text:
                break
            time.sleep(0.05)

        self.manager.shutdown()
        summary = self._wait_for_summary(run_dir)
        self.assertEqual(summary["status"], "cancelled")
        self.assertNotEqual(summary["exit_code"], 0)
        self.assertIn("ready", summary["stdout_stderr_tail"])

    def test_summary_finalization_is_idempotent_and_json_stays_valid_under_concurrent_calls(self) -> None:
        from ui.process_manager import ProcessState
        from ui.training_process_manager import finalize_training_summary

        run_dir = self._run_dir("concurrent_finalize")
        state = ProcessState(
            running=False,
            returncode=0,
            command=["fake"],
            started_at=time.time(),
            finished_at=time.time(),
        )
        results = []

        def finalize_once() -> None:
            results.append(finalize_training_summary(run_dir, ["fake"], None, None, log_text="tail", state=state))

        threads = [threading.Thread(target=finalize_once), threading.Thread(target=finalize_once)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(list(run_dir.glob(".training_summary.json.*.tmp")), [])

    def test_training_run_history_reads_disk_summary_after_session_state_is_gone(self) -> None:
        from ui.run_reader import list_training_runs

        run_dir = self._run_dir("history")
        command = [sys.executable, "-c", "print('done')"]
        self._start_with_callback(run_dir, command)
        self._wait_for_summary(run_dir)

        rows = list_training_runs(self.base)
        self.assertEqual(rows[0]["run_id"], "history")
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["training_summary"]["status"], "completed")

    def test_normal_atomic_summary_write_leaves_no_temp_file(self) -> None:
        from ui.process_manager import ProcessState
        from ui.training_process_manager import finalize_training_summary

        run_dir = self._run_dir("atomic")
        state = ProcessState(running=False, returncode=0, started_at=time.time(), finished_at=time.time())
        finalize_training_summary(run_dir, ["fake"], None, None, log_text="tail", state=state)

        self.assertTrue((run_dir / "training_summary.json").exists())
        self.assertEqual(list(run_dir.glob(".training_summary.json.*.tmp")), [])
        json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))


@unittest.skipUnless(_TRAINING_FIXTURES_AVAILABLE, "real base config/checkpoint/dataset not present")
class TrainingOutputConfinementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.allowed_root = self.base / "runs" / "training"
        self.patcher = mock.patch("core.training_runner.DEFAULT_TRAINING_RUN_ROOT", self.allowed_root)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        self.tmp.cleanup()

    def _inspect(self, output_root: Path, prepare_runtime: bool = True):
        from core.training_runner import inspect_training_config

        return inspect_training_config(
            training_prompt="book spine",
            output_root=output_root,
            prepare_runtime=prepare_runtime,
        )

    def test_canonical_output_root_is_allowed_and_creates_run_under_root(self) -> None:
        preflight = self._inspect(self.allowed_root)
        self.assertEqual(preflight.errors, [])
        self.assertTrue(Path(preflight.run_dir).resolve(strict=False).relative_to(self.allowed_root.resolve(strict=False)))
        self.assertTrue(Path(preflight.runtime_config_path).exists())

    def test_allowed_subdirectory_with_space_and_japanese_is_allowed(self) -> None:
        output_root = self.allowed_root / "sub dir 日本語"
        preflight = self._inspect(output_root)
        self.assertEqual(preflight.errors, [])
        Path(preflight.run_dir).resolve(strict=False).relative_to(output_root.resolve(strict=False))

    def test_relative_path_inside_allowed_root_is_allowed(self) -> None:
        relative = self.allowed_root / "relative_ok" / ".." / "relative_ok"
        preflight = self._inspect(relative)
        self.assertEqual(preflight.errors, [])
        Path(preflight.run_dir).resolve(strict=False).relative_to(self.allowed_root.resolve(strict=False))

    def test_illegal_output_roots_are_rejected_without_creating_them(self) -> None:
        illegal_paths = [
            self.base / "runs" / "training_evil",
            self.allowed_root / ".." / ".." / "data",
            self.base / "tmp_training",
            Path("/home/book/sam301"),
            Path("/home/book/book"),
            DEFAULT_BOOK_SPINE_DATASET_ROOT,
            DEFAULT_SAM3_CHECKPOINT.parent,
        ]
        for path in illegal_paths:
            with self.subTest(path=str(path)):
                existed_before = path.exists()
                preflight = self._inspect(path)
                self.assertTrue(any("Training output must remain under" in e for e in preflight.errors))
                if not existed_before:
                    self.assertFalse(path.exists(), f"invalid output path was created: {path}")

    def test_symlink_escape_is_rejected(self) -> None:
        outside = self.base / "outside"
        outside.mkdir()
        self.allowed_root.mkdir(parents=True)
        link = self.allowed_root / "link_to_outside"
        link.symlink_to(outside, target_is_directory=True)

        preflight = self._inspect(link)
        self.assertTrue(any("Training output must remain under" in e for e in preflight.errors))
        self.assertEqual(list(outside.iterdir()), [])

    def test_direct_core_api_rejects_external_output_root(self) -> None:
        preflight = self._inspect(Path("/tmp/training"), prepare_runtime=False)
        self.assertTrue(any("Training output must remain under" in e for e in preflight.errors))

    def test_empty_string_and_non_path_output_roots_are_rejected(self) -> None:
        from core.training_runner import inspect_training_config

        empty = inspect_training_config(training_prompt="book spine", output_root="", prepare_runtime=False)
        self.assertTrue(any("Training output must remain under" in e for e in empty.errors))

        non_path = inspect_training_config(training_prompt="book spine", output_root=float("nan"), prepare_runtime=False)
        self.assertTrue(any("output_root is not a valid path" in e for e in non_path.errors))

    def test_start_validation_rejects_tampered_runtime_yaml_outside_run_dir(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        run_dir = self.allowed_root / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "checkpoints").mkdir()
        runtime_yaml = self.base / "outside_runtime.yaml"
        runtime_yaml.write_text("x: 1")
        checkpoint = self.base / "initial.pt"
        checkpoint.write_text("fake")
        reasons = validate_can_start_training(
            preflight_ok=True,
            run_dir=str(run_dir),
            runtime_yaml=str(runtime_yaml),
            checkpoint=str(checkpoint),
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
        self.assertTrue(any("runtime YAML" in r and "Training output must remain under" in r for r in reasons))

    def test_existing_summary_or_checkpoint_in_run_dir_blocks_start(self) -> None:
        from ui.training_process_manager import validate_can_start_training

        for marker in ["summary", "checkpoint"]:
            with self.subTest(marker=marker):
                run_dir = self.allowed_root / marker
                run_dir.mkdir(parents=True)
                (run_dir / "checkpoints").mkdir(exist_ok=True)
                runtime_yaml = run_dir / "config" / "runtime_config.yaml"
                runtime_yaml.parent.mkdir()
                runtime_yaml.write_text("x: 1")
                checkpoint = self.base / f"{marker}.pt"
                checkpoint.write_text("fake")
                if marker == "summary":
                    (run_dir / "training_summary.json").write_text("{}")
                else:
                    (run_dir / "checkpoints" / "epoch.pt").write_text("fake")
                reasons = validate_can_start_training(
                    preflight_ok=True,
                    run_dir=str(run_dir),
                    runtime_yaml=str(runtime_yaml),
                    checkpoint=str(checkpoint),
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
                self.assertTrue(any("拒绝复用" in r for r in reasons))


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

        output_root = DEFAULT_TRAINING_RUN_ROOT / "训练 输出 テスト"
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

        run_dir = DEFAULT_TRAINING_RUN_ROOT / f"run 実行 中文 {int(time.time() * 1000)}"
        run_dir.mkdir()
        try:
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
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)


if __name__ == "__main__":
    unittest.main()
