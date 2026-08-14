"""
Structure-preservation terms, aimed at the features they claim to protect.

The two terms these replace were both misdirected.

``edge_preserve.py`` selected the top 15% of *isotropic Sobel magnitude* and
defended it. On a post-stack section that set is reflector amplitude --
near-horizontal, high-contrast bedding -- and almost none of it is fault. A
fault is not a large gradient, it is a place where lateral **coherence
breaks**, and Sobel magnitude barely responds to that. The term was a
reflector-contrast penalty wearing a fault-preservation label.

``frequency.py`` took an unwindowed ``fft2`` of a 64x64 patch, so a large part
of what it measured above its cutoff was the wrap-around discontinuity at the
patch edge rather than anything in the data, and it used a *radial* cutoff,
which mixes temporal frequency with wavenumber. Physically those are not
interchangeable: seismic signal occupies a dip fan through the origin, and
coherent noise -- ground roll, linear cable noise -- is exactly what falls
*outside* that fan at the same temporal frequencies. A radial band defends a
steeply dipping reflector (high wavenumber, low frequency) and white noise
identically, and cannot express "remove the ground roll, keep the dip".

Both replacements are tested against progressively blurred inputs in
``tests/test_geology_losses.py``: a term that claims to detect lost structure
has to actually increase when structure is lost.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Local coherence
# ---------------------------------------------------------------------------

def _box(x: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    """Separable box filter with reflect padding."""
    pad = (kw // 2, kw // 2, kh // 2, kh // 2)
    xp = F.pad(x, pad, mode="reflect")
    kernel_h = torch.ones(1, 1, kh, 1, device=x.device, dtype=x.dtype) / kh
    kernel_w = torch.ones(1, 1, 1, kw, device=x.device, dtype=x.dtype) / kw
    return F.conv2d(F.conv2d(xp, kernel_h), kernel_w)


def local_semblance(x: torch.Tensor, win_t: int = 9, win_x: int = 5) -> torch.Tensor:
    """Taner-style semblance in a local window: coherent energy over total energy.

    Returns values in ``[0, 1]``. High where traces agree with one another --
    a continuous reflector -- and low where they do not, which on a migrated
    section means faults, channel edges, and unconformity truncations.
    """
    stacked = _box(x, 1, win_x) * win_x            # sum across traces
    num = _box(stacked ** 2, win_t, 1)
    den = _box(_box(x ** 2, 1, win_x) * win_x, win_t, 1) * win_x
    return num / (den + 1e-8)


def fault_contrast_loss(denoised: torch.Tensor, reference: torch.Tensor,
                        win_t: int = 9, win_x: int = 5, smooth: int = 15) -> torch.Tensor:
    """Penalise the *healing* of coherence troughs.

    Removing incoherent noise legitimately raises semblance everywhere, so a
    term that simply forbade semblance from rising would forbid denoising --
    which is why this compares local *contrast* rather than level. A fault is
    a local minimum in the coherence field; what must survive denoising is the
    depth of that minimum relative to its surroundings, not its absolute
    value.

    The penalty is one-sided and restricted to pixels where the input already
    shows a trough, so filling a fault in is punished while sharpening one, or
    raising coherence in conformable bedding, is free.
    """
    c_ref = local_semblance(reference, win_t, win_x)
    c_den = local_semblance(denoised, win_t, win_x)

    trough_ref = c_ref - _box(c_ref, smooth, smooth)
    trough_den = c_den - _box(c_den, smooth, smooth)

    is_trough = (trough_ref < 0).to(denoised.dtype)
    # trough_* are negative in troughs, so "shallower than the reference"
    # means trough_den > trough_ref.
    healed = torch.relu(trough_den - trough_ref) * is_trough
    return (healed ** 2).sum() / is_trough.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Dip-fan F-K
# ---------------------------------------------------------------------------

def _hann2d(h: int, w: int, device, dtype) -> torch.Tensor:
    """Separable Hann taper.

    Without this, ``fft2`` of a patch measures the step between its opposite
    edges, which is broadband and often larger than the signal being assessed.
    """
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return torch.outer(wy, wx).view(1, 1, h, w)


def dip_fan_mask(h: int, w: int, max_dip: float, device, dtype) -> torch.Tensor:
    """Boolean-ish mask of the F-K fan that geology can occupy.

    An event with apparent dip ``p`` samples per trace maps to the line
    ``k = p * f`` in the F-K plane, so all geological dips up to ``max_dip``
    together fill the double cone ``|k| <= max_dip * |f|``. Coherent noise
    with an apparent velocity lower than anything geological -- ground roll is
    the canonical case -- lands outside it.
    """
    f = torch.fft.fftfreq(h, device=device, dtype=dtype).view(h, 1)   # cycles/sample
    k = torch.fft.fftfreq(w, device=device, dtype=dtype).view(1, w)   # cycles/trace
    inside = (k.abs() <= max_dip * f.abs() + 1e-12)
    return inside.to(dtype).view(1, 1, h, w)


def dip_fan_loss(denoised: torch.Tensor, reference: torch.Tensor, max_dip: float = 1.0,
                 min_freq: float = 0.05, floor_ratio: float = 0.7) -> torch.Tensor:
    """Penalise collapse of the signal fan's amplitude spectrum.

    Only the inside of the fan is defended, and only against falling below
    ``floor_ratio`` of the input's magnitude there; everything outside the fan
    is left entirely to the reconstruction objective to remove. The very
    lowest temporal frequencies are excluded because the fan degenerates to a
    point at DC, where the mask is meaningless.
    """
    _, _, h, w = denoised.shape
    taper = _hann2d(h, w, denoised.device, denoised.dtype)

    spec_den = torch.fft.fft2(denoised * taper)
    spec_ref = torch.fft.fft2(reference * taper)

    f = torch.fft.fftfreq(h, device=denoised.device, dtype=denoised.dtype).view(1, 1, h, 1)
    band = dip_fan_mask(h, w, max_dip, denoised.device, denoised.dtype) * (f.abs() >= min_freq)

    mag_den = torch.abs(spec_den) * band
    mag_ref = torch.abs(spec_ref) * band

    deficit = torch.relu(floor_ratio * mag_ref - mag_den)
    scale = (mag_ref ** 2).sum().clamp_min(1e-8)
    return (deficit ** 2).sum() / scale
