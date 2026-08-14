"""
Tests for the reference-free noise-level estimator.

This estimator replaces a learned parameter that the training objective cannot
identify on its own, so its accuracy is load-bearing: it sets the blend between
the model's prediction and the observed amplitude, and therefore how much the
denoiser removes. A biased estimate here shows up directly as over- or
under-denoising, with nothing else in the pipeline to catch it.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepseis.noise_estimate import (corner_is_reliable, corner_noise_variance,
                                     estimate_from_sections, estimate_noise_profile,
                                     fan_noise_variance,
                                     estimate_noise_variance)


def layered(nt=255, nx=400, dip=0.3, period=11.0):
    """Band-limited, laterally coherent structure: signal, not noise."""
    t = np.arange(nt)[:, None]
    x = np.arange(nx)[None, :]
    sec = np.zeros((nt, nx))
    for k, amp in enumerate([1.0, 0.6, 0.35]):
        sec += amp * np.sin(2 * np.pi * (t - dip * (k + 1) * x) / (period * (k + 1)))
    return sec.astype(np.float32)


@pytest.mark.parametrize("noise_std", [0.1, 0.3, 0.5, 1.0])
def test_estimator_recovers_a_known_white_noise_level(noise_std):
    rng = np.random.default_rng(0)
    sec = layered()
    noisy = sec + noise_std * rng.standard_normal(sec.shape).astype(np.float32)

    est = np.sqrt(estimate_noise_variance(noisy))
    assert est == pytest.approx(noise_std, rel=0.12), f"estimated {est:.4f} for a true {noise_std}"


def test_estimator_returns_almost_nothing_on_a_noiseless_section():
    """It must not invent noise, or the denoiser will remove signal to match."""
    sec = layered()
    assert np.sqrt(estimate_noise_variance(sec)) < 0.05 * sec.std()


def test_estimator_is_not_fooled_by_strong_coherent_signal():
    """Signal is band-limited and confined to a dip fan; the F-K corner it is
    measured in should stay empty however loud the geology gets."""
    rng = np.random.default_rng(0)
    quiet = layered()
    loud = 10.0 * layered()
    noise = 0.3 * rng.standard_normal(quiet.shape).astype(np.float32)

    a = np.sqrt(estimate_noise_variance(quiet + noise))
    b = np.sqrt(estimate_noise_variance(loud + noise))
    assert a == pytest.approx(b, rel=0.15), f"signal amplitude changed the estimate: {a:.4f} vs {b:.4f}"
    assert a == pytest.approx(0.3, rel=0.1) and b == pytest.approx(0.3, rel=0.1)


def test_corner_reliability_is_invariant_to_signal_and_noise_amplitude():
    """The discriminator must not depend on SNR. An earlier level-based test
    condemned a valid corner as soon as the signal got loud."""
    rng = np.random.default_rng(0)
    base = layered()
    noise = rng.standard_normal(base.shape)
    for sig in (1.0, 10.0):
        for ns in (0.01, 0.3, 1.0):
            assert corner_is_reliable(sig * base + ns * noise),                 f"valid corner rejected at signal x{sig}, noise {ns}"


def test_estimator_is_not_fooled_by_steeply_dipping_coherent_noise():
    """Ground-roll-like events are coherent, so they belong to neither the
    signal fan nor the white-noise floor; they must not inflate the estimate
    much, or the denoiser will over-remove everywhere."""
    rng = np.random.default_rng(0)
    sec = layered()
    t = np.arange(255)[:, None]
    x = np.arange(400)[None, :]
    ground_roll = 1.5 * np.sin(2 * np.pi * (t - 5.0 * x) / 40.0)
    noise = 0.3 * rng.standard_normal(sec.shape)

    clean_est = np.sqrt(estimate_noise_variance(sec + noise))
    rolled_est = np.sqrt(estimate_noise_variance(sec + ground_roll + noise))
    assert rolled_est == pytest.approx(clean_est, rel=0.3)


def test_depth_profile_tracks_depth_varying_noise():
    """The reason the profile exists: noise-to-signal grows with two-way time."""
    rng = np.random.default_rng(0)
    nt, nx = 256, 400
    sec = layered(nt, nx)
    ramp = np.linspace(0.1, 0.8, nt)[:, None]
    noisy = sec + ramp * rng.standard_normal((nt, nx))

    profile = estimate_noise_profile(noisy, n_samples=nt, n_bands=6)
    assert profile.shape == (nt,)
    assert profile[-1] > 2.5 * profile[0], \
        f"profile did not follow the ramp: {profile[0]:.3f} -> {profile[-1]:.3f}"
    assert profile[10] == pytest.approx(ramp[10, 0], rel=0.5)
    assert profile[-10] == pytest.approx(ramp[-10, 0], rel=0.35)


def test_profile_is_monotone_for_monotonically_increasing_noise():
    rng = np.random.default_rng(1)
    nt, nx = 256, 400
    sec = layered(nt, nx)
    ramp = np.linspace(0.1, 0.9, nt)[:, None]
    profile = estimate_noise_profile(sec + ramp * rng.standard_normal((nt, nx)),
                                     n_samples=nt, n_bands=6)
    coarse = profile[::40]
    assert all(b > a * 0.9 for a, b in zip(coarse, coarse[1:])), f"not increasing: {coarse}"


def test_aggregation_uses_the_median_and_resists_one_bad_section():
    """One anomalous line must not set the noise level for the whole survey."""
    rng = np.random.default_rng(0)
    sections = [layered(128, 256) + 0.3 * rng.standard_normal((128, 256)) for _ in range(5)]
    sections.append(layered(128, 256) + 5.0 * rng.standard_normal((128, 256)))

    est = estimate_from_sections(sections, n_samples=128, mode="scalar")
    assert est.shape == (1,)
    assert est[0] == pytest.approx(0.3, rel=0.25), f"outlier dragged the estimate to {est[0]:.3f}"


def test_scalar_and_depth_modes_return_the_expected_shapes():
    rng = np.random.default_rng(0)
    sections = [layered(128, 256) + 0.3 * rng.standard_normal((128, 256)) for _ in range(3)]
    assert estimate_from_sections(sections, 128, mode="scalar").shape == (1,)
    assert estimate_from_sections(sections, 128, mode="depth").shape == (128,)


# ---------------------------------------------------------------------------
# Estimator selection: the corner is not always a noise floor
# ---------------------------------------------------------------------------

def band_limited(sec: np.ndarray, cutoff: float = 0.4) -> np.ndarray:
    """Apply the kind of anti-alias/bandpass chain processed seismic has been through."""
    h, w = sec.shape
    spec = np.fft.fft2(sec)
    fy = np.broadcast_to(np.abs(np.fft.fftfreq(h))[:, None] / 0.5, (h, w))
    return np.real(np.fft.ifft2(spec * (fy <= cutoff)))


def test_corner_is_reliable_on_data_with_a_real_white_noise_floor():
    rng = np.random.default_rng(0)
    noisy = layered() + 0.3 * rng.standard_normal((255, 400))
    assert corner_is_reliable(noisy) is True


def test_corner_is_judged_unreliable_once_the_data_is_band_limited():
    """The failure that would silently turn the denoiser into an identity."""
    rng = np.random.default_rng(0)
    noisy = layered() + 0.3 * rng.standard_normal((255, 400))
    filtered = band_limited(noisy)

    assert corner_is_reliable(filtered) is False
    # and the corner estimator would indeed report almost no noise
    assert np.sqrt(corner_noise_variance(filtered)) < 0.1 * np.sqrt(corner_noise_variance(noisy))


def test_auto_falls_back_to_the_fan_estimator_on_band_limited_data():
    rng = np.random.default_rng(0)
    filtered = band_limited(layered() + 0.3 * rng.standard_normal((255, 400)))
    auto = estimate_noise_variance(filtered, method="auto")
    assert auto == pytest.approx(fan_noise_variance(filtered), rel=1e-9)
    assert auto > 4 * corner_noise_variance(filtered), "auto did not escape the stopband"


def test_auto_uses_the_corner_when_it_is_valid():
    rng = np.random.default_rng(0)
    noisy = layered() + 0.3 * rng.standard_normal((255, 400))
    assert estimate_noise_variance(noisy, method="auto") == pytest.approx(
        corner_noise_variance(noisy), rel=1e-9)


def test_fan_estimator_is_usable_on_band_limited_data():
    """It should still track the true level within the passband."""
    rng = np.random.default_rng(0)
    noise = 0.3 * rng.standard_normal((255, 400))
    filtered_signal = band_limited(layered())
    filtered = filtered_signal + band_limited(noise)

    true_std = float(band_limited(noise).std())
    est = np.sqrt(fan_noise_variance(filtered))
    # Reads high: the fan region spans the full frequency axis while the noise
    # that remains after band-limiting does not, so the estimate is an upper
    # bound rather than a match. It is the usable option here regardless, since
    # the corner estimator returns essentially zero on this data.
    assert true_std < est < 2.0 * true_std, f"estimated {est:.4f} for a true {true_std:.4f}"


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        estimate_noise_variance(layered(64, 64), method="nonsense")
