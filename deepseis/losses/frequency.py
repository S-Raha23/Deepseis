"""
F-K (frequency-wavenumber) high-frequency preservation loss —
``lambda_freq * L_frequency`` in ``L_total`` (spec Sec. 3.2).

Faults, pinch-outs, and thin beds live in the high-frequency / high-
wavenumber part of the F-K spectrum — exactly the band an aggressive
denoiser tends to erase first, since a lot of *noise* energy also lives
there. This term keeps the denoiser honest about not throwing away that
whole band, complementing ``edge_preserve.py`` (which works in the pixel
domain) with a spectral-domain constraint.
"""
from __future__ import annotations

import torch


def _radial_frequency_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """Normalized radial frequency (0 at DC, 1 at Nyquist) for every F-K bin."""
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)  # cycles/sample, in [-0.5, 0.5)
    fx = torch.fft.fftfreq(w, device=device, dtype=dtype)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    radial = torch.sqrt(fy ** 2 + fx ** 2) / 0.5  # normalize by Nyquist (0.5 cycles/sample)
    return radial.clamp(max=1.0)


def fk_high_freq_loss(denoised: torch.Tensor, reference: torch.Tensor, cutoff: float = 0.55) -> torch.Tensor:
    """Match log-magnitude F-K spectra between denoised output and reference, above ``cutoff``.

    ``reference`` is the noisy input. This deliberately does *not* try to
    remove high-frequency energy the way the reconstruction loss does —
    instead it discourages the *global* high-frequency band from being
    wiped out disproportionately, which is the spectral signature of
    over-smoothing. The masked reconstruction loss is still what's
    responsible for actually removing the incoherent noise.
    """
    b, c, h, w = denoised.shape
    radial = _radial_frequency_grid(h, w, denoised.device, denoised.dtype)
    band = (radial >= cutoff).float()[None, None]

    spec_den = torch.fft.fft2(denoised)
    spec_ref = torch.fft.fft2(reference)

    mag_den = torch.log1p(torch.abs(spec_den)) * band
    mag_ref = torch.log1p(torch.abs(spec_ref)) * band

    denom = band.sum().clamp_min(1.0) * b
    return ((mag_den - mag_ref) ** 2).sum() / denom


def fk_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Return the (shifted) log-magnitude F-K spectrum for visualization. x: (H, W) numpy-friendly tensor."""
    spec = torch.fft.fftshift(torch.fft.fft2(x))
    return torch.log1p(torch.abs(spec))
