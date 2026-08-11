"""
Masked reconstruction loss — the base term of ``L_total`` (spec Sec. 3.2).

Evaluated *only* at the blind-spot locations: the network is scored on how
well it predicted the true (noisy) value at pixels it was never shown,
which is what forces it to learn structure rather than memorize noise.
"""
from __future__ import annotations

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error computed only where ``mask`` is True/1.

    Shapes: pred, target -> (B, 1, H, W); mask -> (B, 1, H, W) in {0, 1}.
    """
    mask = mask.float()
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom
