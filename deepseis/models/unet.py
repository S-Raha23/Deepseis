"""
U-Net / ResU-Net backbone for the self-supervised denoiser.

Nothing exotic architecturally — the self-supervision comes entirely from
*how it's trained* (blind-spot masking, Sec. 3.1) and *what it's trained
with* (the fault-preservation loss, Sec. 3.2), not from a specialised
architecture. A small residual U-Net is enough, and it keeps the model
light enough to train on CPU for the demo (Sec. 12: "runs on a single
consumer GPU").
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DenoiserUNet(nn.Module):
    """A small residual U-Net: predicts a clean(er) image from a (blind-spotted) noisy input."""

    def __init__(self, in_channels: int = 1, base_channels: int = 24, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        chs = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.down_blocks = nn.ModuleList([ResBlock(chs[i]) for i in range(depth)])
        self.downsamplers = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(depth)])

        self.bottleneck = ResBlock(chs[-1])

        self.upsamplers = nn.ModuleList([Up(chs[i + 1], chs[i]) for i in reversed(range(depth))])
        self.up_blocks = nn.ModuleList([ResBlock(chs[i]) for i in reversed(range(depth))])
        self.merge_convs = nn.ModuleList([nn.Conv2d(chs[i] * 2, chs[i], 1) for i in reversed(range(depth))])

        self.head = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        skips = []
        for res, down in zip(self.down_blocks, self.downsamplers):
            h = res(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, res, merge, skip in zip(self.upsamplers, self.up_blocks, self.merge_convs, reversed(skips)):
            h = up(h)
            # handle any off-by-one size mismatch from odd input dims
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = merge(torch.cat([h, skip], dim=1))
            h = res(h)

        return self.head(h)
