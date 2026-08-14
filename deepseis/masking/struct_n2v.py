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
                           blind_shape: str, blind_width: int,
                           blind_length: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of the blind region around one selected center pixel.

    The region is a *bounded* segment ``blind_length`` long, not a stripe
    spanning the whole patch.

    That distinction is the difference between StructN2V working and
    destroying the data. The blind region only has to cover the noise's
    correlation length, which is a handful of samples; extending it across the
    full extent of the axis blinds far more than the noise correlates over.
    With the shipped settings -- a 64x64 patch, ``mask_fraction: 0.02``, and a
    3-wide full-height stripe per centre -- the 82 centres between them
    replaced **97.9% of every patch** with a random shuffle of its own values.
    Measured on real F3, ``masking.mode: "auto"`` routed 76% of patches down
    this path, so three-quarters of training was noise-to-noise regression with
    almost no input left to predict from, and the only function that minimises
    that is a smooth low-order guess. It is the direct cause of the previous
    denoiser behaving as a low-pass filter.

    Broaddus et al. (2020) use a short segment for exactly this reason.
    """
    cy, cx = center
    h, w = shape
    half = blind_width // 2
    reach = max(blind_length // 2, 0)

    if blind_shape == "trace":       # short vertical segment -> along time/depth
        ys = np.clip(np.arange(cy - reach, cy + reach + 1), 0, h - 1)
        xs = np.clip(np.arange(cx - half, cx + half + 1), 0, w - 1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
    elif blind_shape == "horizontal":  # short horizontal segment -> along traces
        xs = np.clip(np.arange(cx - reach, cx + reach + 1), 0, w - 1)
        ys = np.clip(np.arange(cy - half, cy + half + 1), 0, h - 1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
    elif blind_shape == "diagonal":   # short diagonal band -> for dipping linear noise
        offsets = np.arange(-reach, reach + 1)
        ys = np.clip(cy + offsets, 0, h - 1)
        xs = np.clip(cx + offsets, 0, w - 1)
        band = [(ys, np.clip(xs + dw, 0, w - 1)) for dw in range(-half, half + 1)]
        yy = np.concatenate([b[0] for b in band])
        xx = np.concatenate([b[1] for b in band])
    else:
        raise ValueError(f"Unknown blind_shape: {blind_shape}")

    return yy.ravel(), xx.ravel()


def generate_mask_and_blind_regions(shape: tuple[int, int], mask_fraction: float, blind_shape: str,
                                     blind_width: int, rng: np.random.Generator,
                                     blind_length: int = 9,
                                     max_blind_fraction: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
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
    budget = int(max_blind_fraction * h * w)

    for cy, cx in zip(centers_y, centers_x):
        yy, xx = _blind_region_indices((cy, cx), shape, blind_shape, blind_width, blind_length)
        # Hard ceiling on how much of the patch may be blinded at once. Each
        # blind region is small, but they accumulate, and once they cover most
        # of the patch the network is being asked to reconstruct an image from
        # a shuffle of itself. That is what the previous configuration did, and
        # nothing stopped it, so the limit is enforced here rather than left to
        # whoever picks mask_fraction.
        # Checked before the region is added, not after, so the ceiling is a
        # real bound rather than one that can be overshot by a whole region.
        already = blind_region_mask[yy, xx].sum()
        if blind_region_mask.sum() + (len(yy) - already) > budget:
            break
        center_mask[cy, cx] = True
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
        patch.shape, s_cfg["mask_fraction"], s_cfg["blind_shape"], s_cfg["blind_width"], rng,
        blind_length=int(s_cfg.get("blind_length", 9)),
        max_blind_fraction=float(s_cfg.get("max_blind_fraction", 0.35)),
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
