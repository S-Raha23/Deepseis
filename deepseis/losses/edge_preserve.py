"""
Edge / structure-preservation loss — ``lambda_edge * L_edge_preservation``
in the spec's ``L_total`` (Sec. 3.2), inspired by the DPN2N line of work on
retaining high-frequency fault information through denoising.

Plain masked-MSE reconstruction (``reconstruction.py``) has no opinion about
*where* it's wrong — it will happily let a hard-edged fault discontinuity
blur into a smooth gradient, because on average that reduces MSE against
noisy targets. This term specifically penalizes the denoiser for softening
strong local gradients (candidate fault/edge structure) relative to the
input, regardless of what happens in smooth, low-gradient (likely
noise-dominated) regions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def sobel_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """|grad(x)| via Sobel filters. x: (B, 1, H, W) -> (B, 1, H, W)."""
    kx = _SOBEL_X.to(x.device, x.dtype)
    ky = _SOBEL_Y.to(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def edge_preservation_loss(denoised: torch.Tensor, reference: torch.Tensor,
                            edge_quantile: float = 0.85) -> torch.Tensor:
    """Penalize gradient-magnitude loss at the reference's strongest-edge pixels.

    ``reference`` is the (noisy) input the network is trying to denoise —
    its strongest gradients are our best available proxy for "here's
    probably a fault or a sharp bed boundary, not noise" without needing any
    labels. We only compare gradients at those high-gradient locations
    (top ``edge_quantile``), so smoothing of low-gradient, likely-noisy
    texture elsewhere is *not* penalized — exactly what we want, since that
    smoothing is the whole point of the denoiser.
    """
    grad_ref = sobel_gradient_magnitude(reference)
    grad_den = sobel_gradient_magnitude(denoised)

    with torch.no_grad():
        threshold = torch.quantile(grad_ref.flatten(1), edge_quantile, dim=1).view(-1, 1, 1, 1)
        edge_mask = (grad_ref >= threshold).float()

    diff2 = (grad_den - grad_ref) ** 2 * edge_mask
    denom = edge_mask.sum().clamp_min(1.0)
    return diff2.sum() / denom
