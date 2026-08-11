"""
Noise2Void blind-spot masking (Krull, Buchholz & Jug, 2019).

Core idea: pick a small random subset of pixels per patch, hide their true
value from the network (by replacing them with a value copied from a nearby
pixel), and train the network to predict the *original* value at exactly
those masked locations from everything else in the receptive field.

Because i.i.d. random noise at the masked pixel is statistically
independent of its neighbours, the only way the network can do well at this
task is to learn the underlying *coherent* signal — it physically cannot
learn to predict noise it has never been shown at that location. That's the
whole self-supervised trick, and it's why no clean training pairs are
needed (spec Sec. 3.1).
"""
from __future__ import annotations

import numpy as np


def generate_mask(shape: tuple[int, int], mask_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of pixels to blind, ~mask_fraction of the patch."""
    n = int(round(shape[0] * shape[1] * mask_fraction))
    n = max(n, 1)
    mask = np.zeros(shape, dtype=bool)
    flat_idx = rng.choice(shape[0] * shape[1], size=n, replace=False)
    mask.flat[flat_idx] = True
    return mask


def apply_blind_spot(patch: np.ndarray, mask: np.ndarray, neighborhood_radius: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Replace masked pixels with the value of a random nearby pixel (the "UPS" strategy).

    This is what makes it a *blind spot*: the network's input literally does
    not contain the true value at the masked location, only a plausible
    stand-in from its neighbourhood, so it must infer the value from context.
    """
    h, w = patch.shape
    out = patch.copy()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        dy = rng.integers(-neighborhood_radius, neighborhood_radius + 1)
        dx = rng.integers(-neighborhood_radius, neighborhood_radius + 1)
        ny = int(np.clip(y + dy, 0, h - 1))
        nx = int(np.clip(x + dx, 0, w - 1))
        out[y, x] = patch[ny, nx]
    return out


def make_training_pair(patch: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (masked_input, mask, target) for one patch.

    ``target`` is just the original (noisy) patch — the loss is only ever
    evaluated at ``mask`` locations (see ``losses/reconstruction.py``), so
    the network is never told the ground truth, only asked to reconstruct
    the observed noisy value at pixels it wasn't shown.
    """
    n2v_cfg = cfg["masking"]["n2v"]
    mask = generate_mask(patch.shape, n2v_cfg["mask_fraction"], rng)
    masked_input = apply_blind_spot(patch, mask, n2v_cfg["neighborhood_radius"], rng)
    return masked_input, mask, patch
