from __future__ import annotations

from sam3.train.data.sam3_image_dataset import Sam3ImageDataset


class RepeatedSam3ImageDataset(Sam3ImageDataset):
    """Repeat source indices while generating fresh online transforms per fetch."""

    def __init__(self, *args, repeat_factor: int = 1, **kwargs):
        if int(repeat_factor) != repeat_factor or not 1 <= int(repeat_factor) <= 10:
            raise ValueError("repeat_factor must be a whole number between 1 and 10")
        super().__init__(*args, **kwargs)
        self.repeat_factor = int(repeat_factor)
        self._source_length = super().__len__()
        if self._source_length < 1:
            raise ValueError("cannot repeat an empty SAM3 image dataset")
        self._fetching_source_item = False

    def __len__(self) -> int:
        if getattr(self, "_fetching_source_item", False):
            return self._source_length
        return self._source_length * self.repeat_factor

    def __getitem__(self, idx):
        source_idx = int(idx) % self._source_length
        self._fetching_source_item = True
        try:
            # The parent retries failed samples with modulo len(self). While it is
            # fetching, __len__ therefore reports the source length, not the
            # externally repeated length.
            return super().__getitem__(source_idx)
        finally:
            self._fetching_source_item = False
