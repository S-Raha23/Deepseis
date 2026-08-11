"""
Facies segmentation (stretch goal, spec Sec. 3.4 / Sec. 5).

A small multi-class CNN that classifies each pixel into a facies/lithology
class, trained on the synthetic facies labels the data module produces (or
on F3 interpretation labels if you supply them). Same encoder-decoder shape
as the fault head, swapped to a softmax multi-class output.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from deepseis.models.faultseg import _ConvBlock2D


class FaciesNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 3, base_channels: int = 16, depth: int = 3) -> None:
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

        self.out_conv = nn.Conv2d(chs[0], n_classes, 1)

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

        return self.out_conv(h)  # raw logits; use softmax/cross-entropy outside
