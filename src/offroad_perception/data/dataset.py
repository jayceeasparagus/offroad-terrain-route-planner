"""PyTorch dataset for RELLIS RGB segmentation."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .rellis import RellisFrame
from .taxonomy import Taxonomy, remap_label_ids


class RellisSegmentationDataset(Dataset[dict[str, Any]]):
    """Load RGB images and remapped five-class masks."""

    def __init__(
        self,
        frames: list[RellisFrame],
        taxonomy: Taxonomy,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.frames = frames
        self.taxonomy = taxonomy
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame = self.frames[index]
        image = Image.open(frame.image_path).convert("RGB")
        label = Image.open(frame.label_path)

        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            label = label.resize((width, height), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        label_array = np.asarray(label)
        if label_array.ndim == 3:
            label_array = label_array[..., 0]

        return {
            "image": torch.from_numpy(image_array).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(remap_label_ids(label_array, self.taxonomy).astype(np.int64)),
            "frame_index": frame.frame_index,
            "image_path": str(frame.image_path),
        }
