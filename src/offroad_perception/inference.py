"""Reusable single-frame inference for the compact terrain segmenter."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .models import CompactUNet


@dataclass(frozen=True)
class SegmentationPrediction:
    """Semantic prediction, confidence, and all per-class probabilities."""

    class_ids: np.ndarray
    confidence: np.ndarray
    probabilities: np.ndarray


def resolve_device(requested: str = "auto") -> torch.device:
    """Choose a PyTorch device, making unavailable CUDA requests explicit."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def load_compact_unet_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    num_classes: int = 5,
    base_channels: int = 16,
) -> tuple[CompactUNet, dict[str, object]]:
    """Load the compact U-Net format produced by the training notebook."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")

    model = CompactUNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


class SegmentationPredictor:
    """Apply the trained model using the same RGB scaling used during training."""

    def __init__(
        self,
        model: CompactUNet,
        device: torch.device,
        image_size: tuple[int, int] = (320, 512),
    ) -> None:
        self.model = model
        self.device = device
        self.image_size = image_size

    def predict_path(self, image_path: str | Path) -> SegmentationPrediction:
        """Read an RGB image and return a full-resolution prediction."""
        with Image.open(image_path) as image:
            return self.predict_image(image.convert("RGB"))

    def predict_image(self, image: Image.Image) -> SegmentationPrediction:
        """Segment a PIL image and resize probabilities back to its original size."""
        rgb = image.convert("RGB")
        original_width, original_height = rgb.size
        model_height, model_width = self.image_size
        resized = rgb.resize((model_width, model_height), Image.Resampling.BILINEAR)

        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            probabilities = F.interpolate(
                probabilities,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

        confidence, class_ids = probabilities.max(dim=1)
        return SegmentationPrediction(
            class_ids=class_ids[0].cpu().numpy().astype(np.uint8),
            confidence=confidence[0].cpu().numpy().astype(np.float32),
            probabilities=probabilities[0].cpu().numpy().astype(np.float32),
        )
