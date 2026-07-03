"""Tests for the P1-B checkpoint export/adapter fix (F-P1-2 remediation).

Root cause: sam301/model_builder.py::_load_checkpoint() filters state dict keys with
`if "detector" in k`, which matches 0 of a trainer checkpoint's 1134 keys (they have
no "detector." substring at all — see core/checkpoint_export.py module docstring for
the full empirical derivation). These tests exercise the book01-side exporter/loader
that bypasses that buggy filter entirely, using synthetic small state dicts for the
key-mapping unit tests (fast, no GPU) and one real-checkpoint integration test gated
on fixture availability.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import checkpoint_export as ce  # noqa: E402
from core.config import BOOK_ROOT, DEFAULT_SAM3_CHECKPOINT  # noqa: E402

REAL_TRAINER_CHECKPOINT = BOOK_ROOT / "runs" / "training" / "2026-07-03_14-42-27" / "checkpoints" / "checkpoint.pt"
_REAL_FIXTURES_AVAILABLE = REAL_TRAINER_CHECKPOINT.exists() and DEFAULT_SAM3_CHECKPOINT.exists()


def _write_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))


def _make_trainer_checkpoint(path: Path, model_sd: dict, epoch: int = 1) -> None:
    _write_torch(
        path,
        {
            "model": model_sd,
            "optimizer": {"state": {}, "param_groups": []},
            "epoch": epoch,
            "loss": {},
            "steps": {"train": 10, "val": 2},
            "time_elapsed": 1.0,
            "best_meter_values": {},
            "scaler": {"scale": 1.0},
        },
    )


def _make_base_checkpoint(path: Path, flat_detector_sd: dict) -> None:
    _write_torch(path, flat_detector_sd)


def _make_inference_checkpoint(path: Path, model_sd: dict, metadata: dict | None = None) -> None:
    _write_torch(
        path,
        {
            "format": ce.INFERENCE_FORMAT,
            "format_version": ce.INFERENCE_FORMAT_VERSION,
            "model": model_sd,
            "metadata": metadata or {},
        },
    )


class CheckpointIdentificationTest(unittest.TestCase):
    def test_trainer_checkpoint_is_identified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            _make_trainer_checkpoint(path, {"backbone.x": torch.zeros(2)})
            identity = ce.identify_checkpoint(path)
        self.assertEqual(identity.type, ce.CHECKPOINT_TYPE_TRAINER)
        self.assertEqual(identity.detail["n_tensors"], 1)

    def test_base_checkpoint_is_identified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sam3.pt"
            _make_base_checkpoint(path, {"detector.backbone.x": torch.zeros(2)})
            identity = ce.identify_checkpoint(path)
        self.assertEqual(identity.type, ce.CHECKPOINT_TYPE_BASE)

    def test_inference_checkpoint_is_identified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_model.pt"
            _make_inference_checkpoint(path, {"backbone.x": torch.zeros(2)})
            identity = ce.identify_checkpoint(path)
        self.assertEqual(identity.type, ce.CHECKPOINT_TYPE_INFERENCE)

    def test_missing_file_is_identified(self) -> None:
        identity = ce.identify_checkpoint("/definitely/not/a/real/path.pt")
        self.assertEqual(identity.type, ce.CHECKPOINT_TYPE_MISSING)

    def test_arbitrary_dict_is_unknown_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "random.pt"
            _write_torch(path, {"some_key": "not a tensor at all"})
            identity = ce.identify_checkpoint(path)
        self.assertEqual(identity.type, ce.CHECKPOINT_TYPE_UNKNOWN)

    def test_type_is_never_guessed_from_filename(self) -> None:
        """A file literally named checkpoint.pt whose real structure is base-style
        must be classified as BASE, not TRAINER, proving classification reads content."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            _make_base_checkpoint(path, {"detector.x": torch.zeros(1)})
            identity = ce.identify_checkpoint(path)
        self.assertEqual(identity.type, ce.CHECKPOINT_TYPE_BASE)


class KeyMappingTest(unittest.TestCase):
    def test_identity_mapping_matches_all_keys(self) -> None:
        source = {"backbone.x": torch.zeros(3), "transformer.y": torch.ones(2)}
        target = {"backbone.x": torch.zeros(3), "transformer.y": torch.ones(2)}
        result = ce.build_key_mapping(source, target)
        self.assertEqual(result.chosen_prefix, "")
        self.assertEqual(result.matched_tensors, 2)
        self.assertEqual(result.coverage_ratio, 1.0)
        self.assertEqual(result.missing, [])
        self.assertEqual(result.unexpected, [])

    def test_prefix_stripping_mapping_is_chosen_when_it_matches_better(self) -> None:
        source = {"detector.backbone.x": torch.zeros(3)}
        target = {"backbone.x": torch.zeros(3)}
        result = ce.build_key_mapping(source, target)
        self.assertEqual(result.chosen_prefix, "detector.")
        self.assertEqual(result.mapping, {"detector.backbone.x": "backbone.x"})

    def test_matched_zero_raises(self) -> None:
        source = {"totally.unrelated.key": torch.zeros(1)}
        target = {"backbone.x": torch.zeros(1)}
        with self.assertRaises(ValueError):
            ce.build_key_mapping(source, target)

    def test_shape_mismatch_is_reported_not_matched(self) -> None:
        source = {"backbone.x": torch.zeros(3)}
        target = {"backbone.x": torch.zeros(5)}
        result = ce.build_key_mapping(source, target)
        self.assertEqual(result.matched_tensors, 0)
        self.assertEqual(len(result.shape_mismatch), 1)
        self.assertEqual(result.shape_mismatch[0][0], "backbone.x")

    def test_no_two_source_keys_map_to_the_same_target(self) -> None:
        # "" and "module." prefixes would BOTH resolve "module.x" -> different
        # targets only if ambiguous; construct a genuine conflict: two source keys
        # that, under the SAME chosen prefix, collide on one target key. Since the
        # mapping is built per-source-key -> unique target, a real collision needs
        # two distinct source keys stripping to the identical target under one
        # prefix, which cannot happen with a plain dict (keys are unique) — so we
        # instead verify the conflict guard fires when we hand-construct that case
        # via monkeypatching the prefix candidate list to include a lossy transform.
        with mock.patch.object(ce, "CANDIDATE_SOURCE_PREFIXES", ("", "module.", "collide.")):
            source = {"module.x": torch.zeros(1), "collide.x": torch.zeros(1)}
            target = {"x": torch.zeros(1)}
            # both candidates map ALL of their matching keys to "x"; "module." and
            # "collide." each individually produce a 1-key mapping to the same
            # target with DIFFERENT source keys -> ambiguous, must refuse to guess.
            with self.assertRaises(ValueError):
                ce.build_key_mapping(source, target)

    def test_critical_module_zero_matched_is_visible_in_result(self) -> None:
        source = {"backbone.x": torch.zeros(1)}
        target = {"backbone.x": torch.zeros(1), "transformer.y": torch.zeros(1)}
        result = ce.build_key_mapping(source, target)
        self.assertEqual(result.critical_modules["transformer"], {"matched": 0, "total": 1})
        self.assertEqual(result.critical_modules["backbone"], {"matched": 1, "total": 1})


class ExportRejectionTest(unittest.TestCase):
    """Fail-loud export-time rejections, using synthetic checkpoints and a
    monkeypatched fresh-model builder so these run fast without loading real SAM3."""

    def _patch_fresh_model(self, target_sd: dict):
        fake_model = mock.Mock()
        fake_model.state_dict.return_value = target_sd
        return mock.patch.object(ce, "_fresh_model", return_value=fake_model)

    def test_trainer_type_required_for_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "sam3.pt"
            _make_base_checkpoint(base_path, {"detector.x": torch.zeros(1)})
            with self.assertRaises(ValueError):
                ce.export_inference_checkpoint(base_path, Path(tmp) / "out.pt", base_checkpoint_path=None)

    def test_low_coverage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_path = Path(tmp) / "checkpoint.pt"
            _make_trainer_checkpoint(trainer_path, {"backbone.x": torch.zeros(1)})
            target_sd = {f"backbone.p{i}": torch.zeros(1) for i in range(10)}
            target_sd["backbone.x"] = torch.zeros(1)
            with self._patch_fresh_model(target_sd):
                with self.assertRaises(ValueError) as ctx:
                    ce.export_inference_checkpoint(
                        trainer_path, Path(tmp) / "out.pt", base_checkpoint_path=None
                    )
        self.assertIn("coverage ratio", str(ctx.exception))

    def test_shape_mismatch_is_rejected(self) -> None:
        # 100 matching keys + 1 shape-mismatched key keeps coverage_ratio (100/101 ~
        # 0.99) above MIN_COVERAGE_RATIO, so this exercises the shape-mismatch check
        # specifically rather than tripping the (separately tested) coverage check.
        with tempfile.TemporaryDirectory() as tmp:
            trainer_path = Path(tmp) / "checkpoint.pt"
            source_sd = {f"backbone.ok{i}": torch.zeros(2) for i in range(100)}
            source_sd["backbone.x"] = torch.zeros(3)
            _make_trainer_checkpoint(trainer_path, source_sd)
            target_sd = {f"backbone.ok{i}": torch.zeros(2) for i in range(100)}
            target_sd["backbone.x"] = torch.zeros(5)
            with self._patch_fresh_model(target_sd):
                with self.assertRaises(ValueError) as ctx:
                    ce.export_inference_checkpoint(
                        trainer_path, Path(tmp) / "out.pt", base_checkpoint_path=None
                    )
        self.assertIn("shape mismatch", str(ctx.exception).lower())

    def test_critical_module_missing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_path = Path(tmp) / "checkpoint.pt"
            _make_trainer_checkpoint(trainer_path, {"backbone.x": torch.zeros(1)})
            target_sd = {"backbone.x": torch.zeros(1), "transformer.y": torch.zeros(1)}
            with self._patch_fresh_model(target_sd):
                with self.assertRaises(ValueError) as ctx:
                    ce.export_inference_checkpoint(
                        trainer_path, Path(tmp) / "out.pt", base_checkpoint_path=None
                    )
        self.assertIn("transformer", str(ctx.exception))

    def test_export_never_overwrites_source_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_path = Path(tmp) / "checkpoint.pt"
            _make_trainer_checkpoint(trainer_path, {"backbone.x": torch.zeros(1)})
            target_sd = {"backbone.x": torch.zeros(1)}
            with self._patch_fresh_model(target_sd):
                with self.assertRaises(ValueError):
                    ce.export_inference_checkpoint(trainer_path, trainer_path, base_checkpoint_path=None)

    def test_export_refuses_to_overwrite_base_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_path = Path(tmp) / "checkpoint.pt"
            base_path = Path(tmp) / "sam3.pt"
            _make_trainer_checkpoint(trainer_path, {"backbone.x": torch.zeros(1)})
            _make_base_checkpoint(base_path, {"detector.backbone.x": torch.zeros(1)})
            target_sd = {"backbone.x": torch.zeros(1)}
            with self._patch_fresh_model(target_sd):
                with self.assertRaises(ValueError):
                    ce.export_inference_checkpoint(
                        trainer_path, base_path, base_checkpoint_path=base_path
                    )

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_path = Path(tmp) / "checkpoint.pt"
            out_path = Path(tmp) / "inference_model.pt"
            out_path.write_text("existing")
            _make_trainer_checkpoint(trainer_path, {"backbone.x": torch.zeros(1)})
            target_sd = {"backbone.x": torch.zeros(1)}
            with self._patch_fresh_model(target_sd):
                with self.assertRaises(FileExistsError):
                    ce.export_inference_checkpoint(trainer_path, out_path, base_checkpoint_path=None)

    def test_identical_to_base_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_path = Path(tmp) / "checkpoint.pt"
            base_path = Path(tmp) / "sam3.pt"
            same_tensor = torch.randn(4)
            _make_trainer_checkpoint(trainer_path, {"backbone.x": same_tensor.clone()})
            _make_base_checkpoint(base_path, {"detector.backbone.x": same_tensor.clone()})
            target_sd = {"backbone.x": torch.zeros(4)}
            with self._patch_fresh_model(target_sd):
                with self.assertRaises(ValueError) as ctx:
                    ce.export_inference_checkpoint(
                        trainer_path, Path(tmp) / "out.pt", base_checkpoint_path=base_path
                    )
        self.assertIn("BIT-IDENTICAL", str(ctx.exception))

    def test_changed_from_base_is_accepted_and_metadata_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer_path = Path(tmp) / "checkpoint.pt"
            base_path = Path(tmp) / "sam3.pt"
            out_path = Path(tmp) / "out.pt"
            _make_trainer_checkpoint(trainer_path, {"backbone.x": torch.ones(4)}, epoch=3)
            _make_base_checkpoint(base_path, {"detector.backbone.x": torch.zeros(4)})
            target_sd = {"backbone.x": torch.zeros(4)}
            with self._patch_fresh_model(target_sd):
                # self-verification inside export also calls _fresh_model via
                # load_inference_checkpoint -> patch must remain active through that call.
                with mock.patch.object(ce, "load_inference_checkpoint", return_value=(mock.Mock(), {})):
                    result = ce.export_inference_checkpoint(
                        trainer_path, out_path, base_checkpoint_path=base_path,
                        dataset_identity={"human_reviewed": False},
                    )
            self.assertEqual(result.mapping_result.matched_tensors, 1)
            self.assertEqual(result.metadata["base_diff"]["changed_tensors"], 1)
            self.assertEqual(result.metadata["base_diff"]["identical_tensors"], 0)
            self.assertEqual(result.metadata["epoch"], 3)
            self.assertEqual(result.metadata["dataset_identity"], {"human_reviewed": False})
            self.assertIn("key_mapping_version", result.metadata)
            self.assertTrue(Path(result.output_path).exists())
            loaded = torch.load(result.output_path, weights_only=False)
            self.assertEqual(loaded["format"], ce.INFERENCE_FORMAT)
            self.assertIn("model", loaded)
            self.assertIn("metadata", loaded)
            self.assertNotIn("optimizer", loaded)
            self.assertNotIn("scaler", loaded)


class InferenceLoaderRejectionTest(unittest.TestCase):
    def test_trainer_checkpoint_is_rejected_with_export_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            _make_trainer_checkpoint(path, {"backbone.x": torch.zeros(1)})
            with self.assertRaises(ValueError) as ctx:
                ce.load_inference_checkpoint(path)
        self.assertIn("export_sam3_inference_checkpoint.py", str(ctx.exception))

    def test_base_checkpoint_is_rejected_with_alternate_path_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sam3.pt"
            _make_base_checkpoint(path, {"detector.x": torch.zeros(1)})
            with self.assertRaises(ValueError) as ctx:
                ce.load_inference_checkpoint(path)
        self.assertIn("build_sam3_image_model", str(ctx.exception))

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ce.load_inference_checkpoint("/definitely/not/a/real/path.pt")

    def test_zero_parameter_inference_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference_model.pt"
            _make_inference_checkpoint(path, {})
            fake_model = mock.Mock()
            fake_model.load_state_dict.return_value = ([], [])
            with mock.patch.object(ce, "_fresh_model", return_value=fake_model):
                with self.assertRaises(RuntimeError) as ctx:
                    ce.load_inference_checkpoint(path)
        self.assertIn("zero parameters", str(ctx.exception))


@unittest.skipUnless(_REAL_FIXTURES_AVAILABLE, "real trainer checkpoint / base sam3.pt not present")
class RealCheckpointIntegrationTest(unittest.TestCase):
    """Uses the actual formal-dataset one-epoch trainer checkpoint. Builds a REAL
    fresh SAM3 model (CPU) and does a real strict load — no mocking of torch or the
    model. This is slower (~model construction time) but is the authoritative proof
    that the export/loader round-trips against the real architecture."""

    def test_real_trainer_checkpoint_identity_and_key_structure(self) -> None:
        identity = ce.identify_checkpoint(REAL_TRAINER_CHECKPOINT)
        self.assertEqual(identity.type, ce.CHECKPOINT_TYPE_TRAINER)
        self.assertEqual(identity.detail["n_tensors"], 1134)

    def test_real_base_checkpoint_identity(self) -> None:
        identity = ce.identify_checkpoint(DEFAULT_SAM3_CHECKPOINT)
        self.assertEqual(identity.type, ce.CHECKPOINT_TYPE_BASE)

    def test_export_and_strict_load_into_fresh_model_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "inference_model.pt"
            result = ce.export_inference_checkpoint(
                REAL_TRAINER_CHECKPOINT, out_path, base_checkpoint_path=DEFAULT_SAM3_CHECKPOINT
            )
            self.assertEqual(result.mapping_result.matched_tensors, 1134)
            self.assertEqual(result.mapping_result.coverage_ratio, 1.0)
            self.assertEqual(result.mapping_result.missing, [])
            self.assertEqual(result.mapping_result.unexpected, [])
            self.assertGreater(result.metadata["base_diff"]["changed_tensors"], 0)

            # fresh model, real strict load, independent of the export's internal
            # self-verification
            model, metadata = ce.load_inference_checkpoint(out_path)
            self.assertEqual(metadata["key_mapping_version"], ce.KEY_MAPPING_VERSION)

            # source tensor vs loaded tensor: sample a real changed tensor and
            # confirm the loaded model's parameter is bit-identical to what was
            # exported (not some other default-initialized value).
            trainer_ckpt = torch.load(REAL_TRAINER_CHECKPOINT, map_location="cpu", weights_only=False)
            sample_key = result.metadata["base_diff"]["changed_examples"][0]
            expected_tensor = trainer_ckpt["model"][sample_key]
            loaded_tensor = dict(model.named_parameters())[sample_key] if sample_key in dict(
                model.named_parameters()
            ) else dict(model.state_dict())[sample_key]
            self.assertTrue(torch.equal(expected_tensor, loaded_tensor.detach()))

    def test_deleting_the_mapping_would_be_caught(self) -> None:
        """Corrupt the source state dict to simulate a broken/emptied mapping and
        confirm export refuses (proves the guard isn't a no-op)."""
        trainer_ckpt = torch.load(REAL_TRAINER_CHECKPOINT, map_location="cpu", weights_only=False)
        corrupted = dict(trainer_ckpt)
        corrupted["model"] = {f"nonsense.{k}": v for k, v in trainer_ckpt["model"].items()}
        with tempfile.TemporaryDirectory() as tmp:
            corrupted_path = Path(tmp) / "checkpoint.pt"
            torch.save(corrupted, str(corrupted_path))
            with self.assertRaises(ValueError):
                ce.export_inference_checkpoint(
                    corrupted_path, Path(tmp) / "out.pt", base_checkpoint_path=DEFAULT_SAM3_CHECKPOINT
                )


if __name__ == "__main__":
    unittest.main()
