from pathlib import Path

import numpy as np
import torch
from PIL import Image

from offroad_perception.inference import (
    SegmentationPredictor,
    load_compact_unet_checkpoint,
    resolve_device,
)
from offroad_perception.models import CompactUNet


def test_predictor_restores_original_image_size() -> None:
    model = CompactUNet(base_channels=4, num_classes=5)
    predictor = SegmentationPredictor(
        model,
        torch.device("cpu"),
        image_size=(32, 48),
    )
    image = Image.fromarray(np.full((25, 41, 3), 120, dtype=np.uint8))
    prediction = predictor.predict_image(image)

    assert prediction.class_ids.shape == (25, 41)
    assert prediction.confidence.shape == (25, 41)
    assert prediction.probabilities.shape == (5, 25, 41)
    assert np.allclose(prediction.probabilities.sum(axis=0), 1.0, atol=1e-5)


def test_checkpoint_loader_uses_training_checkpoint_format(tmp_path: Path) -> None:
    source_model = CompactUNet(base_channels=4, num_classes=5)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {"epoch": 3, "model_state_dict": source_model.state_dict()},
        checkpoint_path,
    )

    model, checkpoint = load_compact_unet_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
        num_classes=5,
        base_channels=4,
    )

    assert checkpoint["epoch"] == 3
    assert not model.training


def test_device_selection_rejects_unknown_device() -> None:
    try:
        resolve_device("tpu")
    except ValueError as error:
        assert "auto, cpu, cuda" in str(error)
    else:
        raise AssertionError("Unknown device should raise ValueError")
