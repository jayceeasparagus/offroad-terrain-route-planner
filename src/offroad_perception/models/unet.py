"""Small U-Net used as the initial segmentation baseline."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CompactUNet(nn.Module):
    """Compact U-Net that preserves the input spatial resolution."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, self._resize_like(e3, d3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._resize_like(e2, d2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._resize_like(e1, d1)], dim=1))
        return self.head(d1)

    @staticmethod
    def _resize_like(source: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)

