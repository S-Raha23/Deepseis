"""
Fault segmentation head, in the spirit of FaultSeg3D (Wu, Liang, Shi & Fomel,
2019): an encoder-decoder CNN trained on synthetic fault labels to predict a
per-pixel fault probability map.

Two variants are provided:

* :class:`FaultSegNet2D` — operates per-slice (2D). This is what the demo
  pipeline actually trains/runs, since the sandbox works with 2D inlines.
* :class:`FaultSegNet3D` — a Conv3d version of the same architecture, for
  when a full 3D SEG-Y volume is available (drop-in replacement; same API).

If ``faultseg.pretrained_weights_path`` is set in the config, real
FaultSeg3D weights are loaded instead of training from scratch — matching
the spec's "wire in FaultSeg3D pretrained model" (Sec. 5 MVP). Without
weights, the net trains on the synthetic fault labels the data module
already produces, which is enough to demonstrate a real, measured
noisy-vs-denoised Dice/PR delta (Sec. 9), not a canned number.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaultSegNet2D(nn.Module):
    """2D U-Net fault segmentation head. Output: fault probability map in [0, 1]."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, depth: int = 3) -> None:
        super().__init__()
        chs = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.encoders.append(_ConvBlock2D(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock2D(chs[-2], chs[-1])

        self.upsamplers = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                          for i in reversed(range(depth))])
        self.decoders = nn.ModuleList([_ConvBlock2D(chs[i] * 2, chs[i]) for i in reversed(range(depth))])

        self.out_conv = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = dec(torch.cat([h, skip], dim=1))

        return torch.sigmoid(self.out_conv(h))


class _ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaultSegNet3D(nn.Module):
    """Volumetric (Conv3d) version of :class:`FaultSegNet2D` for full SEG-Y cubes.

    Same encoder-decoder topology as the original FaultSeg3D paper's
    3D U-Net; use this once you have a real inline x crossline x samples
    volume rather than single 2D slices.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16, depth: int = 3) -> None:
        super().__init__()
        chs = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.encoders.append(_ConvBlock3D(prev, c))
            prev = c
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = _ConvBlock3D(chs[-2], chs[-1])

        self.upsamplers = nn.ModuleList([nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2)
                                          for i in reversed(range(depth))])
        self.decoders = nn.ModuleList([_ConvBlock3D(chs[i] * 2, chs[i]) for i in reversed(range(depth))])

        self.out_conv = nn.Conv3d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-3:] != skip.shape[-3:]:
                h = nn.functional.interpolate(h, size=skip.shape[-3:], mode="nearest")
            h = dec(torch.cat([h, skip], dim=1))

        return torch.sigmoid(self.out_conv(h))


def load_pretrained_or_none(model: nn.Module, weights_path: str | None, device: str = "cpu") -> bool:
    """Try to load pretrained weights; returns True on success, False if unavailable.

    Real FaultSeg3D checkpoints are typically Keras/TF (.hdf5) in the
    original repo; if you convert one to a PyTorch state_dict matching this
    architecture, point ``weights_path`` at it and it'll be used directly.
    """
    if not weights_path:
        return False
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        return True
    except FileNotFoundError:
        return False
