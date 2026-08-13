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


def _smooth3(x: torch.Tensor) -> torch.Tensor:
    """3x3 box blur — just enough to stop single-pixel noise spikes reading as edges."""
    k = torch.full((1, 1, 3, 3), 1.0 / 9.0, device=x.device, dtype=x.dtype)
    return F.conv2d(x, k, padding=1)


def edge_preservation_loss(denoised: torch.Tensor, reference: torch.Tensor,
                            edge_quantile: float = 0.85) -> torch.Tensor:
    """Penalize *weakening* of structural edges, located on a de-spiked reference.

    ``reference`` is the noisy input, which creates two problems the earlier
    formulation walked into:

    1. **Where the edges are.** Selecting the top-``edge_quantile`` gradient
       pixels of the raw noisy input selects noise spikes as readily as faults —
       a single hot sample has a large Sobel response. The mask is therefore
       built from a lightly smoothed copy, so the pixels it protects are ones
       with spatial support, i.e. plausible structure rather than one loud
       sample. The gradients being *compared* are still the unsmoothed ones.

    2. **Which direction is an error.** A symmetric ``(grad_den - grad_ref)^2``
       punishes the denoiser equally for softening a fault and for failing to
       reproduce a noise spike's sharpness — and, because noisy gradients are
       inflated by the noise, its cheapest way to reduce the term is to push
       gradient magnitude *up*. Measured on F3 that showed up as an output with
       higher energy than its input (std 1.25 against 1.00) and worse signal
       leakage with this loss ON than OFF, the opposite of the intent.

       Only the deficit is penalized now: ``relu(grad_ref - grad_den)``. Losing a
       strong edge is punished; reducing an over-sharp noisy one is free, which
       is exactly the asymmetry "preserve faults, remove noise" describes.
    """
    grad_den = sobel_gradient_magnitude(denoised)

    with torch.no_grad():
        # Both the edge locations AND the magnitude to hold are taken from the
        # smoothed reference. Using the raw noisy gradient as the target is what
        # drove the output's energy above its input's: noise inflates gradient
        # magnitude everywhere, so "don't fall below the noisy gradient" is an
        # instruction to keep the noise. The smoothed gradient is a far better
        # stand-in for the underlying structural amplitude.
        grad_struct = sobel_gradient_magnitude(_smooth3(reference))
        threshold = torch.quantile(grad_struct.flatten(1), edge_quantile, dim=1).view(-1, 1, 1, 1)
        edge_mask = (grad_struct >= threshold).float()

    deficit = torch.relu(grad_struct - grad_den)
    denom = edge_mask.sum().clamp_min(1.0)
    return ((deficit ** 2) * edge_mask).sum() / denom
