"""
Structured Noise2Void (StructN2V) — for *coherent* noise (Liu, Birnie &
Alkhalifah, 2022).

Plain Noise2Void's assumption breaks down for coherent noise: ground roll,
linear noise, and acquisition footprint are correlated across neighbouring
traces/samples, so a network could "cheat" and reconstruct the masked
pixel's noise component straight from its neighbours instead of learning
the geology.

The fix is to blind a whole *region* shaped like the noise's own
correlation footprint — not just the single target pixel — so that no
pixel within the noise's correlation length is available to copy from.
For ground roll (which is trace-correlated, i.e. smoothly varying laterally
at fixed time), that means blinding a vertical stripe (``blind_shape:
"trace"``); other coherent noise geometries use a horizontal or diagonal
stripe instead.
"""
from __future__ import annotations

import numpy as np


def _blind_region_indices(center: tuple[int, int], shape: tuple[int, int],
                           blind_shape: str, blind_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of the blind region around one selected center pixel."""
    cy, cx = center
    h, w = shape
    half = blind_width // 2

    if blind_shape == "trace":       # vertical stripe -> blinds along the time/depth axis
        ys = np.arange(0, h)
        xs = np.clip(np.arange(cx - half, cx + half + 1), 0, w - 1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
    elif blind_shape == "horizontal":  # horizontal stripe -> blinds along the trace axis
        xs = np.arange(0, w)
        ys = np.clip(np.arange(cy - half, cy + half + 1), 0, h - 1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
    elif blind_shape == "diagonal":   # diagonal band -> for dipping linear noise
        length = max(h, w)
        offsets = np.arange(-length // 2, length // 2)
        ys = np.clip(cy + offsets, 0, h - 1)
        xs = np.clip(cx + offsets, 0, w - 1)
        band = []
        for dw in range(-half, half + 1):
            band.append((ys, np.clip(xs + dw, 0, w - 1)))
        yy = np.concatenate([b[0] for b in band])
        xx = np.concatenate([b[1] for b in band])
    else:
        raise ValueError(f"Unknown blind_shape: {blind_shape}")

    return yy.ravel(), xx.ravel()


def generate_mask_and_blind_regions(shape: tuple[int, int], mask_fraction: float, blind_shape: str,
                                     blind_width: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (center_mask, blind_region_mask).

    ``center_mask`` marks the pixels the loss is evaluated at (same role as
    in plain N2V). ``blind_region_mask`` marks the (larger) set of pixels
    that must be hidden from the network's input for each center pixel.
    """
    h, w = shape
    n = max(int(round(h * w * mask_fraction)), 1)
    flat_idx = rng.choice(h * w, size=n, replace=False)
    centers_y, centers_x = np.unravel_index(flat_idx, shape)

    center_mask = np.zeros(shape, dtype=bool)
    blind_region_mask = np.zeros(shape, dtype=bool)
    for cy, cx in zip(centers_y, centers_x):
        center_mask[cy, cx] = True
        yy, xx = _blind_region_indices((cy, cx), shape, blind_shape, blind_width)
        blind_region_mask[yy, xx] = True
    return center_mask, blind_region_mask


def apply_structured_blind_spot(patch: np.ndarray, blind_region_mask: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    """Scramble every pixel inside the blind region using a random shuffle of patch values.

    Unlike plain N2V's "copy a nearby pixel", here the replacement is drawn
    from a shuffled pool of the *whole patch* — nearby pixels are exactly
    what carries the coherent noise, so copying one would leak it right back
    in. Shuffling from the full patch keeps the intensity distribution
    realistic while destroying the spatial correlation the noise depends on.
    """
    h, w = patch.shape
    out = patch.copy()
    n_blind = int(blind_region_mask.sum())
    if n_blind == 0:
        return out
    pool = patch.flatten()
    replacements = rng.choice(pool, size=n_blind, replace=True)
    out[blind_region_mask] = replacements
    return out


def make_training_pair(patch: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (masked_input, center_mask, target) for one patch, StructN2V-style."""
    s_cfg = cfg["masking"]["struct_n2v"]
    center_mask, blind_region_mask = generate_mask_and_blind_regions(
        patch.shape, s_cfg["mask_fraction"], s_cfg["blind_shape"], s_cfg["blind_width"], rng
    )
    masked_input = apply_structured_blind_spot(patch, blind_region_mask, rng)
    return masked_input, center_mask, patch


def estimate_noise_type(patch: np.ndarray) -> str:
    """Cheap heuristic noise-type estimator (the "Noise type estimator" box in Sec. 4's diagram).

    Compares the 2D autocorrelation's spatial extent along the trace axis
    against the sample axis: coherent/ground-roll noise produces a much
    longer-range lateral autocorrelation than incoherent noise does.

    .. warning::
       Measured against known inputs, this does **not** separate coherent noise
       from coherent *signal*, because it reads the whole patch and a seismic
       patch is dominated by laterally continuous reflectors either way:

           synthetic + ground roll   -> 98% of patches "coherent"
           synthetic + white noise   -> 100% of patches "coherent"   <-- same verdict
           real F3                   -> 76% of patches "coherent"

       A section with no coherent noise at all scores as high as one full of it,
       so ``masking.mode: "auto"`` routes patches on geology rather than on noise.
       Estimating from a high-pass residual instead does not rescue it: a 3x3
       residual calls both synthetic cases 0% (ground roll is low-frequency and
       gets removed with the signal), a 7x7 residual calls both ~100%.

       Prefer setting ``masking.mode`` explicitly to ``"n2v"`` or ``"struct_n2v"``
       for a survey you know, rather than relying on this. It is kept because the
       spec calls for it and it is honest about incoherent-vs-coherent on bare
       noise fields; it is not reliable on real seismic.
    """
    from numpy.fft import fft2, ifft2

    p = patch - patch.mean()
    f = fft2(p)
    autocorr = np.real(ifft2(f * np.conj(f)))
    autocorr = np.fft.fftshift(autocorr)
    autocorr /= autocorr.max() + 1e-8

    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    lateral_extent = np.sum(autocorr[cy, :] > 0.3)
    vertical_extent = np.sum(autocorr[:, cx] > 0.3)

    return "coherent" if lateral_extent > 1.8 * vertical_extent else "random"
