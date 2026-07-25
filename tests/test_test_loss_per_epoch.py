"""Regression tests for the per-epoch test-split loss.

`LossTracingTrainer.run_val` runs the normal validation pass and then computes
the test loss with the same resident model and criterion. These tests drive the
real methods with stubbed surroundings -- no reimplementation of the logic under
test -- and cover the properties that matter operationally:

* the test split is derived from `data.val` (so an existing run picks it up on
  resume with no config change), and is skipped when it does not exist;
* the reported number is the sample-weighted mean over the whole split;
* a failure in the test pass cannot stop training;
* the pass restores RNG state, so the online augmentation stream is untouched;
* the UI merges these live points over the offline recomputation.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.training_loss_trace import LossTracingTrainer  # noqa: E402


class _Datapoint:
    def __init__(self, batch_size: int) -> None:
        self.img_batch = list(range(batch_size))
        self.find_targets = [object()]


class _Model:
    def __init__(self) -> None:
        self.training = True
        self.calls = 0

    def __call__(self, datapoint):
        self.calls += 1
        return {"stage": datapoint}

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    @staticmethod
    def back_convert(target):
        return target


class _Dataset:
    """Yields fresh batch dicts each epoch (the trainer pops keys out of them)."""

    def __init__(self, sizes_and_losses) -> None:
        self.sizes_and_losses = sizes_and_losses
        self.epochs_requested: list[int] = []

    def get_loader(self, epoch):
        self.epochs_requested.append(epoch)
        return [
            {"book_spine": _Datapoint(size)} for size, _loss in self.sizes_and_losses
        ]


def _criterion_for(sizes_and_losses, *, consume_rng: bool = False, fail: bool = False):
    """Return core_loss per batch, matched positionally to the loader order."""
    remaining = list(sizes_and_losses)

    def criterion(find_stages, find_targets):
        if fail:
            raise RuntimeError("criterion exploded")
        if consume_rng:
            torch.rand(1)
        _size, loss = remaining.pop(0)
        return {
            "core_loss": torch.tensor(loss),
            "loss_ce": torch.tensor(loss / 2.0),
        }

    return criterion


def _make_trainer(tmp_path: Path, dataset, criterion, *, epoch: int = 7):
    """A LossTracingTrainer with only the attributes the test pass touches."""
    trainer = object.__new__(LossTracingTrainer)
    trainer._test_dataset_resolved = True
    trainer._test_dataset = dataset
    trainer._test_stats_path = tmp_path / "test_stats.json"
    trainer._test_annotations_path = None
    trainer._test_annotations_sha256 = None
    trainer.model = _Model()
    trainer.device = torch.device("cpu")
    trainer.optim_conf = None
    trainer.epoch = epoch
    trainer.distributed_rank = 0
    trainer.logger = SimpleNamespace(log_dict=lambda record, epoch: None)
    trainer.loss = {"book_spine": criterion}
    return trainer


class TestLossDerivationTest(unittest.TestCase):
    def test_swap_split_dir_rewrites_the_last_matching_segment(self) -> None:
        swap = LossTracingTrainer._swap_split_dir
        self.assertEqual(
            swap("/data/val/val/annotations.json", "val", "test"),
            "/data/val/test/annotations.json",
        )
        self.assertIsNone(swap("/data/train/images", "val", "test"))
        self.assertIsNone(swap(None, "val", "test"))

    def _trainer_with_val_conf(self, root: Path):
        trainer = object.__new__(LossTracingTrainer)
        trainer._test_loss_data_conf = None
        trainer.data_conf = OmegaConf.create(
            {
                "val": {
                    "_target_": "sam3.train.data.torch_dataset.TorchDataset",
                    "batch_size": 1,
                    "dataset": {
                        "ann_file": str(root / "val" / "annotations.json"),
                        "img_folder": str(root / "val" / "images"),
                    },
                }
            }
        )
        return trainer

    def test_test_split_is_derived_from_val_when_it_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("val", "test"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "annotations.json").write_text("{}", encoding="utf-8")
            conf = self._trainer_with_val_conf(root)._resolve_test_loader_conf()
            self.assertIsNotNone(conf)
            self.assertEqual(
                conf.dataset.ann_file, str(root / "test" / "annotations.json")
            )
            self.assertEqual(conf.dataset.img_folder, str(root / "test" / "images"))
            # Everything else must be inherited from the val loader untouched.
            self.assertEqual(conf.batch_size, 1)

    def test_missing_test_split_disables_the_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "val" / "images").mkdir(parents=True)
            (root / "val" / "annotations.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(self._trainer_with_val_conf(root)._resolve_test_loader_conf())


class TestLossEvaluationTest(unittest.TestCase):
    def test_reports_sample_weighted_mean_over_the_split(self) -> None:
        # Two batches of unequal size: 3 samples at 6.0 and 1 sample at 2.0.
        sizes_and_losses = [(3, 6.0), (1, 2.0)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = _Dataset(sizes_and_losses)
            trainer = _make_trainer(
                tmp_path, dataset, _criterion_for(sizes_and_losses), epoch=7
            )

            trainer._run_test_loss_eval()

            lines = (tmp_path / "test_stats.json").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            # A plain mean would be 4.0; the weighted mean is (6*3 + 2*1)/4.
            self.assertAlmostEqual(record["Losses/test_book_spine_loss"], 5.0)
            self.assertAlmostEqual(record["Losses/test_book_spine_core_loss"], 5.0)
            self.assertAlmostEqual(record["Losses/test_book_spine_loss_ce"], 2.5)
            self.assertEqual(record["Trainer/epoch"], 7)
            self.assertEqual(record["Trainer/test_samples"], 4)
            self.assertEqual(dataset.epochs_requested, [7])

    def test_each_point_records_the_test_set_it_was_measured_on(self) -> None:
        # A mid-project test-set change must be visible in the record rather
        # than silently producing two incomparable curves.
        sizes_and_losses = [(1, 4.0)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotations = tmp_path / "annotations.json"
            annotations.write_text('{"images": []}', encoding="utf-8")
            expected = LossTracingTrainer._sha256_of_file(annotations)

            trainer = _make_trainer(
                tmp_path, _Dataset(sizes_and_losses), _criterion_for(sizes_and_losses)
            )
            trainer._test_annotations_path = str(annotations)
            trainer._test_annotations_sha256 = expected

            trainer._run_test_loss_eval()

            record = json.loads(
                (tmp_path / "test_stats.json").read_text().splitlines()[0]
            )
            self.assertEqual(record["Data/test_annotations"], str(annotations))
            self.assertEqual(record["Data/test_annotations_sha256"], expected)
            self.assertEqual(len(expected), 64)

            # Editing the file must change the digest, so the two are separable.
            annotations.write_text('{"images": [1]}', encoding="utf-8")
            self.assertNotEqual(LossTracingTrainer._sha256_of_file(annotations), expected)

    def test_unreadable_annotations_digest_is_none_not_a_crash(self) -> None:
        self.assertIsNone(LossTracingTrainer._sha256_of_file("/nonexistent/ann.json"))

    def test_model_is_returned_to_training_mode(self) -> None:
        sizes_and_losses = [(1, 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(
                Path(tmp), _Dataset(sizes_and_losses), _criterion_for(sizes_and_losses)
            )
            self.assertTrue(trainer.model.training)
            trainer._run_test_loss_eval()
            self.assertTrue(trainer.model.training)

    def test_rng_state_is_restored(self) -> None:
        sizes_and_losses = [(1, 1.0), (1, 3.0)]
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_trainer(
                Path(tmp),
                _Dataset(sizes_and_losses),
                _criterion_for(sizes_and_losses, consume_rng=True),
            )
            torch.manual_seed(1234)
            before = torch.get_rng_state().clone()

            trainer._run_test_loss_eval()

            self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_failure_never_propagates_to_training(self) -> None:
        sizes_and_losses = [(1, 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trainer = _make_trainer(
                tmp_path,
                _Dataset(sizes_and_losses),
                _criterion_for(sizes_and_losses, fail=True),
            )

            trainer._run_test_loss_eval()  # must not raise

            self.assertFalse((tmp_path / "test_stats.json").exists())
            self.assertTrue(trainer.model.training)

    def test_disabled_when_no_test_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trainer = _make_trainer(tmp_path, None, None)
            trainer._test_dataset = None
            trainer._run_test_loss_eval()
            self.assertFalse((tmp_path / "test_stats.json").exists())

    def test_run_val_runs_validation_then_the_test_pass(self) -> None:
        order: list[str] = []

        class _Trainer(LossTracingTrainer):
            def _run_test_loss_eval(self):
                order.append("test")

        trainer = object.__new__(_Trainer)
        # Stand in for the base Trainer.run_val that super() will reach.
        original = LossTracingTrainer.__mro__[1].run_val
        try:
            LossTracingTrainer.__mro__[1].run_val = lambda self: order.append("val")
            trainer.run_val()
        finally:
            LossTracingTrainer.__mro__[1].run_val = original

        self.assertEqual(order, ["val", "test"])


class TestLossCurveMergeTest(unittest.TestCase):
    def test_live_test_stats_override_the_offline_recomputation(self) -> None:
        from ui.training_process_manager import read_training_loss_curves

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "logs" / "book_spine").mkdir(parents=True)
            offline = run_dir / "evaluation" / "test_loss"
            offline.mkdir(parents=True)
            offline.joinpath("checkpoint_loss.json").write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {"epoch": 5, "training_epoch": 4, "loss": 9.5, "status": "completed"},
                            {"epoch": 9, "training_epoch": 8, "loss": 9.2, "status": "completed"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            # Epoch 8 also has a live value; epoch 9 is live-only.
            run_dir.joinpath("logs", "book_spine", "test_stats.json").write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"Losses/test_book_spine_loss": 8.8, "Trainer/epoch": 8},
                        {"Losses/test_book_spine_loss": 8.5, "Trainer/epoch": 9},
                    )
                ),
                encoding="utf-8",
            )

            points = read_training_loss_curves(run_dir)

        self.assertEqual(
            points["test"],
            [
                {"epoch": 4, "loss": 9.5, "split": "test"},
                {"epoch": 8, "loss": 8.8, "split": "test"},
                {"epoch": 9, "loss": 8.5, "split": "test"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
