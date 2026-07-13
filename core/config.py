from __future__ import annotations

from pathlib import Path

from core.machine_config import load_machine_paths


MACHINE_PATHS = load_machine_paths()
BOOK_ROOT = MACHINE_PATHS.book_root
SAM301_ROOT = MACHINE_PATHS.sam301_root

DEFAULT_TASK_SLUG = "book_spine"
DEFAULT_CATEGORY_NAME = "book_spine"
DEFAULT_TRAINING_PROMPT = "book spine"

DEFAULT_SAM3_TRAIN_SCRIPT = SAM301_ROOT / "sam3" / "train" / "train.py"
# train.py's -c is a Hydra config *name* inside pkg://sam3.train, not a filesystem
# path; per-run runtime YAMLs live outside that package, so training is launched
# through this book01-side wrapper (initialize_config_dir + official main()).
DEFAULT_TRAINING_LAUNCHER = BOOK_ROOT / "scripts" / "launch_sam3_training.py"
DEFAULT_BOOK_SPINE_FINETUNE_CONFIG = (
    SAM301_ROOT / "sam3" / "train" / "configs" / "book_spine" / "book_spine_finetune.yaml"
)
DEFAULT_SAM3_CHECKPOINT = SAM301_ROOT / "sam3.pt"
DEFAULT_SAM3_BPE_PATH = SAM301_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_BOOK_SPINE_DATASET_ROOT = BOOK_ROOT / "data" / "book_spine_sam3_dataset"
DEFAULT_DATASET_ROOT = DEFAULT_BOOK_SPINE_DATASET_ROOT
DEFAULT_TRAINING_RUN_ROOT = BOOK_ROOT / "runs" / "training"
DEFAULT_CONDA_ENV = "sam301"
EXPECTED_SAM3_ROOT = SAM301_ROOT
EXPECTED_SAM3_PACKAGE_DIR = EXPECTED_SAM3_ROOT / "sam3"
EXPECTED_SAM3_INIT = EXPECTED_SAM3_PACKAGE_DIR / "__init__.py"
