"""Small U-Net for two-channel Sentinel-1 flood segmentation.

All layers are initialized locally; this model downloads and loads no weights.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FloodUNet(nn.Module):
    """Encoder-decoder U-Net initialized from scratch for VV/VH SAR inputs."""

    def __init__(self, in_channels: int = 2, classes: int = 2, base_channels: int = 24):
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.bottleneck = ConvBlock(widths[2], widths[3])
        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2)
        self.dec3 = ConvBlock(widths[2] * 2, widths[2])
        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2)
        self.dec2 = ConvBlock(widths[1] * 2, widths[1])
        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2)
        self.dec1 = ConvBlock(widths[0] * 2, widths[0])
        self.head = nn.Conv2d(widths[0], classes, kernel_size=1)

        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _join(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Odd-sized inputs can produce a one-pixel mismatch after down/up sampling.
        if up.shape[-2:] != skip.shape[-2:]:
            up = nn.functional.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, up), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._join(self.up3(b), e3))
        d2 = self.dec2(self._join(self.up2(d3), e2))
        d1 = self.dec1(self._join(self.up1(d2), e1))
        return self.head(d1)


def create_unet_model(in_channels: int = 2, classes: int = 2, base_channels: int = 24) -> FloodUNet:
    """Create a compact, randomly initialized model (no pretrained backbone)."""
    return FloodUNet(in_channels=in_channels, classes=classes, base_channels=base_channels)


if __name__ == "__main__":
    model = create_unet_model()
    print(f"Model: {model.__class__.__name__}; parameters: {sum(p.numel() for p in model.parameters()):,}")
