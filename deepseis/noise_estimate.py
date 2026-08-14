"""
Reference-free estimation of the incoherent noise level.

This exists because of a concrete failure in the blind-spot model. Its
objective is the likelihood of the observed sample under

    y ~ N(mu, sigma_p^2 + sigma_n^2)

which constrains only the **sum** of the two variances. The split between
"the context could not predict this" (``sigma_p``) and "this is noise"
(``sigma_n``) is supposed to be pinned down by ``sigma_n`` being shared across
pixels while ``sigma_p`` varies freely -- but early in training ``sigma_p``
barely varies, so the split is close to unidentifiable and lands wherever
initialisation left it.

Measured on F3 with a known 0.187 noise variance, a partly-trained model had
converged to a nearly correct *total* (0.256 predicted against 0.225 observed,
and calibrated to within 15% across every predicted-variance decile) while
splitting it almost exactly backwards: ``sigma_p^2 = 0.151`` against a true
model-error variance of 0.037, and ``sigma_n^2 = 0.115`` against a true noise
variance of 0.187. Since the denoised value is

    x_hat = (mu * sigma_n^2 + y * sigma_p^2) / (sigma_p^2 + sigma_n^2)

the wrong split is not a cosmetic problem: it put 57% of the weight on the raw
observation where the correct split puts 17%, so the denoiser removed roughly
a quarter of the noise it should have. Substituting the MSE-optimal
``sigma_n`` recovered 3.1 dB.

The fix is to measure ``sigma_n`` instead of hoping the optimiser resolves the
degeneracy. Two estimators are provided, because neither is valid everywhere.

**Corner** (:func:`corner_noise_variance`). Incoherent noise is white, so it
spreads its power evenly over the whole F-K plane, while seismic signal is
band-limited in temporal frequency *and* confined to a dip fan in wavenumber.
The corner of the spectrum -- high frequency and high wavenumber together --
therefore contains noise and essentially nothing else. On F3 sections with
known added noise this recovers the true variance to within 1%.

**Dip fan** (:func:`fan_noise_variance`). The corner estimator has a failure
mode that is silent and severe. Processed seismic is band-limited on purpose,
and F3's published volume has been through a filter chain that leaves the
corner five orders of magnitude below the passband. Run there, the corner
estimator returns a noise standard deviation of ~0.005 against a section
standard deviation of ~0.87 -- it is measuring the filter's stopband, not a
noise floor -- and a denoiser told the noise is zero becomes an identity. The
fan estimator instead reads the wavenumbers outside the geological dip fan
*within* the passband, which survives band-limiting; on raw F3 it gives ~0.23,
about a quarter of the section amplitude. It reads ~13% high where coherent
non-white noise is present, since ground-roll-like events also fall outside
the fan.

:func:`estimate_noise_variance` with ``method="auto"`` picks between them by
testing whether the corner holds a real noise floor or a stopband.
"""
from __future__ import annotations

import numpy as np


def _power_spectrum(section: np.ndarray):
    """Power density and the normalised |f|, |k| grids that index it."""
    sec = np.asarray(section, dtype=np.float64)
    h, w = sec.shape
    power = np.abs(np.fft.fft2(sec - sec.mean())) ** 2 / (h * w)
    fy = np.broadcast_to(np.abs(np.fft.fftfreq(h))[:, None] / 0.5, (h, w))
    fx = np.broadcast_to(np.abs(np.fft.fftfreq(w))[None, :] / 0.5, (h, w))
    return power, fy, fx


def corner_noise_variance(section: np.ndarray, f_lo: float = 0.7, k_lo: float = 0.7) -> float:
    """Noise variance from the high-frequency, high-wavenumber corner of the F-K plane.

    ``f_lo`` and ``k_lo`` are fractions of Nyquist; the default corner is the
    outer 30% of both axes at once, which no geological dip reaches at any
    temporal frequency a seismic wavelet carries. Accurate to about 1% when
    the corner really does contain the noise floor -- see
    :func:`corner_is_reliable` for when it does not.
    """
    power, fy, fx = _power_spectrum(section)
    corner = (fy >= f_lo) & (fx >= k_lo)
    if not corner.any():
        return float(power.mean())
    return float(power[corner].mean())


def fan_noise_variance(section: np.ndarray, max_dip: float = 1.5, f_lo: float = 0.05,
                       f_hi: float = 0.95, margin: float = 1.5) -> float:
    """Noise variance from outside the geological dip fan, inside the passband.

    At a given temporal frequency, signal is confined to ``|k| <= dip * f``, so
    the wavenumbers beyond that are noise even at frequencies where the signal
    is strong. Unlike the corner estimator this does not depend on the data
    retaining any energy at high temporal frequency, which is what makes it
    usable on processed data.

    It reads slightly high -- about 13% on sections with known added noise --
    because coherent noise that is not white, ground roll being the obvious
    case, also lives outside the fan and is counted here.
    """
    power, fy, fx = _power_spectrum(section)
    outside = (fx > margin * max_dip * fy) & (fy >= f_lo) & (fy <= f_hi)
    if not outside.any():
        return corner_noise_variance(section)
    return float(power[outside].mean())


def corner_flatness(section: np.ndarray, k_lo: float = 0.7) -> float:
    """How flat the corner's spectrum is in temporal frequency. White noise -> ~1.

    Compares mean power in the outer frequency band against the one just
    inside it, both restricted to high wavenumber. A white noise floor is flat
    by definition, so the ratio sits at 1 regardless of how loud the signal is
    or how much noise there is; a filter stopband is still rolling off, so the
    ratio collapses.
    """
    power, fy, fx = _power_spectrum(section)
    high_k = fx >= k_lo
    inner = high_k & (fy >= 0.6) & (fy < 0.8)
    outer = high_k & (fy >= 0.8)
    if not inner.any() or not outer.any():
        return 0.0
    return float(power[outer].mean() / (power[inner].mean() + 1e-30))


def corner_is_reliable(section: np.ndarray, min_flatness: float = 0.7,
                       k_lo: float = 0.7) -> bool:
    """Whether the F-K corner holds a noise floor rather than a filter stopband.

    Processed seismic is band-limited on purpose, and reading a noise level off
    a stopband does not measure the noise -- it measures the filter. It returns
    almost zero, which tells the denoiser there is nothing to remove and
    silently turns it into an identity, so this has to be detected rather than
    assumed.

    The test is flatness, not level. An earlier version compared corner power
    to passband power, which fails on high signal-to-noise data: raising the
    signal by 10x with the noise unchanged drove that ratio down by two orders
    of magnitude and condemned a perfectly good corner. Flatness is invariant
    to both, measuring 0.97-1.00 across a 30x range of noise amplitudes and a
    10x range of signal amplitudes, against 0.05 for band-limited data and
    0.26-0.57 for raw F3.
    """
    return corner_flatness(section, k_lo) >= min_flatness


def estimate_noise_variance(section: np.ndarray, method: str = "auto",
                            f_lo: float = 0.7, k_lo: float = 0.7) -> float:
    """Noise variance, choosing the estimator that is valid for this data.

    ``"auto"`` uses the corner when it holds a real noise floor and falls back
    to the dip-fan estimator when the corner has been filtered away.
    """
    if method == "corner":
        return corner_noise_variance(section, f_lo, k_lo)
    if method == "fan":
        return fan_noise_variance(section)
    if method != "auto":
        raise ValueError(f"unknown noise estimation method: {method}")

    if corner_is_reliable(section, k_lo=k_lo):
        return corner_noise_variance(section, f_lo, k_lo)
    return fan_noise_variance(section)


def estimate_noise_profile(section: np.ndarray, n_samples: int | None = None,
                           n_bands: int = 6, f_lo: float = 0.7, k_lo: float = 0.7,
                           min_band: int = 48, method: str = "auto") -> np.ndarray:
    """Per-depth noise standard deviation, as a length-``n_samples`` vector.

    Estimated in overlapping depth bands and interpolated, because the
    noise-to-signal ratio in a seismic section grows with two-way time --
    signal decays through spherical divergence and absorption while much of the
    noise does not -- so a single scalar systematically over-denoises the
    shallow section and under-denoises the deep part.

    Bands are kept at least ``min_band`` samples tall so each still resolves
    the corner of its own spectrum.
    """
    sec = np.asarray(section, dtype=np.float64)
    h = sec.shape[0]
    n_samples = n_samples or h

    band = max(h // max(n_bands, 1), min_band)
    centres, variances = [], []
    starts = np.linspace(0, max(h - band, 0), max(n_bands, 1)).astype(int)
    for y0 in starts:
        y1 = min(y0 + band, h)
        if y1 - y0 < 8:
            continue
        centres.append(0.5 * (y0 + y1 - 1))
        variances.append(estimate_noise_variance(sec[y0:y1], method, f_lo, k_lo))

    if not variances:
        return np.full(n_samples, np.sqrt(estimate_noise_variance(sec, method, f_lo, k_lo)))

    grid = np.arange(n_samples)
    interpolated = np.interp(grid, np.array(centres), np.array(variances))
    return np.sqrt(np.maximum(interpolated, 1e-12))


def estimate_from_sections(sections, n_samples: int, mode: str = "depth",
                           f_lo: float = 0.7, k_lo: float = 0.7,
                           method: str = "auto") -> np.ndarray:
    """Aggregate an estimate over many training sections.

    The median across sections is used rather than the mean so that a single
    anomalously noisy or anomalously quiet line cannot set the level for the
    whole survey.
    """
    if mode == "scalar":
        values = [estimate_noise_variance(s, method, f_lo, k_lo) for s in sections]
        return np.array([np.sqrt(np.median(values))])

    profiles = [estimate_noise_profile(s, n_samples, f_lo=f_lo, k_lo=k_lo, method=method)
                for s in sections]
    return np.median(np.stack(profiles), axis=0)
