import torch

from offroad_perception.models import CompactUNet


def test_compact_unet_preserves_resolution() -> None:
    model = CompactUNet(base_channels=8, num_classes=5)
    output = model(torch.randn(2, 3, 64, 96))
    assert output.shape == (2, 5, 64, 96)


def test_compact_unet_has_trainable_parameters() -> None:
    model = CompactUNet(base_channels=8)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
