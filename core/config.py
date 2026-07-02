from __future__ import annotations

from pathlib import Path


BOOK_ROOT = Path("/home/book/book01")
SAM301_ROOT = Path("/home/book/sam301")

DEFAULT_SAM3_TRAIN_SCRIPT = SAM301_ROOT / "sam3" / "train" / "train.py"
DEFAULT_BOOK_SPINE_FINETUNE_CONFIG = (
    SAM301_ROOT / "sam3" / "train" / "configs" / "book_spine" / "book_spine_finetune.yaml"
)
DEFAULT_SAM3_CHECKPOINT = SAM301_ROOT / "sam3.pt"
DEFAULT_SAM3_BPE_PATH = SAM301_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_BOOK_SPINE_DATASET_ROOT = BOOK_ROOT / "data" / "book_spine_sam3_dataset"
DEFAULT_TRAINING_RUN_ROOT = BOOK_ROOT / "runs" / "training"
DEFAULT_CONDA_ENV = "sam301"
EXPECTED_SAM3_ROOT = SAM301_ROOT
EXPECTED_SAM3_PACKAGE_DIR = EXPECTED_SAM3_ROOT / "sam3"
EXPECTED_SAM3_INIT = EXPECTED_SAM3_PACKAGE_DIR / "__init__.py"
