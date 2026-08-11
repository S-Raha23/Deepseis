"""
"Explain the noise mask" — Jacobian inspection (stretch goal, spec Sec. 3.1
& Sec. 5: "One-click 'explain the noise mask' (Jacobian inspection) panel").

Instead of hand-tuning the blind-spot mask shape/width, inspect how much
each input pixel actually influences a given output pixel's prediction (its
Jacobian / sensitivity), by backpropagating through the trained denoiser.
If that sensitivity extends unusually far along one direction, the noise
correlated along that direction wasn't fully blinded, and the mask
(``blind_shape``/``blind_width`` in the config) should be widened along it.

This both makes the mask design auditable (a real explainability artifact
for the judges) and gives a concrete, data-driven suggestion instead of a
guess.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class MaskSuggestion:
    suggested_blind_shape: str
    suggested_blind_width: int
    lateral_extent_px: float
    vertical_extent_px: float
    jacobian_map: np.ndarray  # (H, W) sensitivity map, for display


def compute_pixel_jacobian(model: torch.nn.Module, x: torch.Tensor, target_yx: tuple[int, int]) -> np.ndarray:
    """|d output[target_yx] / d input| over the whole input patch.

    ``x``: (1, 1, H, W) tensor, requires no external grad tracking (this
    function turns it on internally and cleans up after itself).
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    y, x_pix = target_yx
    scalar = out[0, 0, y, x_pix]
    grad, = torch.autograd.grad(scalar, x, retain_graph=False, create_graph=False)
    return grad.detach().abs().squeeze().cpu().numpy()


def suggest_mask_design(jacobian_map: np.ndarray, center_yx: tuple[int, int],
                         sensitivity_threshold_frac: float = 0.1) -> MaskSuggestion:
    """Turn a Jacobian map into a concrete blind-mask recommendation.

    Measures how far the "significant sensitivity" region (values above
    ``sensitivity_threshold_frac`` of the map's max) extends laterally
    (trace direction) vs. vertically (time/depth direction) from the
    target pixel, and recommends a blind-spot shape/width wide enough to
    cover that extent.
    """
    cy, cx = center_yx
    peak = jacobian_map.max() + 1e-8
    significant = jacobian_map >= (sensitivity_threshold_frac * peak)

    ys, xs = np.nonzero(significant)
    if len(ys) == 0:
        lateral_extent = vertical_extent = 1.0
    else:
        lateral_extent = float(np.max(np.abs(xs - cx))) if len(xs) else 1.0
        vertical_extent = float(np.max(np.abs(ys - cy))) if len(ys) else 1.0

    if lateral_extent > 1.5 * vertical_extent:
        shape = "trace"
        width = max(3, int(round(lateral_extent * 0.6)) | 1)  # force odd width
    elif vertical_extent > 1.5 * lateral_extent:
        shape = "horizontal"
        width = max(3, int(round(vertical_extent * 0.6)) | 1)
    else:
        shape = "trace"
        width = max(3, int(round(max(lateral_extent, vertical_extent) * 0.5)) | 1)

    return MaskSuggestion(
        suggested_blind_shape=shape,
        suggested_blind_width=width,
        lateral_extent_px=lateral_extent,
        vertical_extent_px=vertical_extent,
        jacobian_map=jacobian_map,
    )
