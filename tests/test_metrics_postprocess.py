"""
Tests for the no-reference diagnostics and the orthogonalization post-process.

The diagnostics matter because on field data they are the only evidence
available, and the project previously leaned on one of them (local similarity)
without the companion number that makes it readable. The tests below pin down
what each metric does at its extremes, including the case that motivated the
change: a filter that removes nothing scores perfectly on leakage.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepseis import metrics as M
from deepseis.postprocess import local_orthogonalization_weight, orthogonalize


def layered_section(nt=96, nx=96, dip=0.3, period=9.0, seed=0):
    t = np.arange(nt)[:, None]
    x = np.arange(nx)[None, :]
    return np.sin(2 * np.pi * (t - dip * x) / period).astype(np.float32)


# ---------------------------------------------------------------------------
# Metric extremes
# ---------------------------------------------------------------------------

def test_a_filter_that_removes_nothing_scores_a_perfect_leakage():
    """The reason leakage must never be reported on its own."""
    sec = layered_section()
    assert M.leakage_score(sec, sec) == pytest.approx(0.0, abs=1e-9)
    assert M.energy_removed_fraction(sec, sec) == pytest.approx(0.0, abs=1e-12)


def test_spectral_retention_is_unity_for_an_untouched_section():
    sec = layered_section()
    for _, v in M.spectral_retention(sec, sec).items():
        assert v == pytest.approx(1.0, rel=1e-6)


def test_spectral_retention_shows_a_low_pass_as_a_collapse_of_the_high_bands():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    sec = layered_section() + 0.5 * rng.standard_normal((96, 96)).astype(np.float32)
    blurred = gaussian_filter(sec.astype(np.float64), 1.5)
    r = M.spectral_retention(sec, blurred)
    bands = list(r.values())
    assert all(nxt <= cur + 1e-9 for cur, nxt in zip(bands, bands[1:])), \
        f"retention should fall monotonically with frequency for a blur: {bands}"
    assert bands[-1] < 0.1, "a blur should gut the highest band"
    assert bands[0] > 10 * bands[-1], \
        f"a blur should preserve the low band far better than the high one: {bands}"


def test_energy_removed_fraction_tracks_how_much_was_taken_out():
    rng = np.random.default_rng(0)
    sec = layered_section()
    noise = rng.standard_normal(sec.shape).astype(np.float32)
    # removing exactly `noise` removes var(noise) / var(sec + noise)
    noisy = sec + noise
    expected = noise.var() / noisy.var()
    assert M.energy_removed_fraction(noisy, sec) == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Residual coherence
# ---------------------------------------------------------------------------

def test_residual_coherence_is_low_when_only_white_noise_was_removed():
    rng = np.random.default_rng(0)
    clean = layered_section()
    noisy = clean + 0.6 * rng.standard_normal(clean.shape).astype(np.float32)
    incoherent = M.residual_coherence(noisy, clean)          # removed exactly the noise
    coherent = M.residual_coherence(noisy, 0.6 * rng.standard_normal(clean.shape).astype(np.float32))
    assert incoherent < 0.5, f"white-noise residual scored {incoherent}"
    assert incoherent < coherent


def test_residual_coherence_is_high_when_a_reflector_was_removed():
    """The number that catches the failure the difference section is displayed for."""
    rng = np.random.default_rng(0)
    clean = layered_section()
    noisy = clean + 0.3 * rng.standard_normal(clean.shape).astype(np.float32)

    took_the_noise = M.residual_coherence(noisy, clean)
    took_the_geology = M.residual_coherence(noisy, noisy - clean)
    assert took_the_geology > took_the_noise, \
        f"removing the geology ({took_the_geology}) did not score worse than removing noise ({took_the_noise})"


def test_denoising_report_includes_reference_metrics_only_when_a_reference_exists():
    rng = np.random.default_rng(0)
    clean = layered_section()
    noisy = clean + 0.4 * rng.standard_normal(clean.shape).astype(np.float32)

    without = M.denoising_report(noisy, clean)
    assert "psnr" not in without and "leakage" in without

    with_ref = M.denoising_report(noisy, clean, clean)
    assert {"psnr", "ssim", "snr_db"} <= set(with_ref)


# ---------------------------------------------------------------------------
# Orthogonalization
# ---------------------------------------------------------------------------

def test_orthogonalization_recovers_signal_that_was_wrongly_removed():
    """The case it exists for: an over-attenuating denoiser pushed 40% of the
    signal into the residual, where it shows up as a scaled copy of what was
    kept and can be projected back out."""
    rng = np.random.default_rng(0)
    clean = layered_section()
    noisy = clean + 0.05 * rng.standard_normal(clean.shape).astype(np.float32)

    over_attenuated = 0.6 * clean
    corrected, _ = orthogonalize(noisy, over_attenuated, window=11, iterations=1)

    err_before = float(((over_attenuated - clean) ** 2).mean())
    err_after = float(((corrected - clean) ** 2).mean())
    assert err_after < 0.1 * err_before, \
        f"barely recovered anything: {err_before:.4f} -> {err_after:.4f}"


def test_orthogonalization_makes_the_residual_orthogonal_to_the_signal():
    """The defining property, and the one that picks out the correct update
    rule: the widely quoted ``s + w*n0`` form does not achieve it."""
    rng = np.random.default_rng(0)
    clean = layered_section()
    noisy = clean + 0.3 * rng.standard_normal(clean.shape).astype(np.float32)
    over_attenuated = 0.6 * clean

    before = abs(float(np.corrcoef((noisy - over_attenuated).ravel(), over_attenuated.ravel())[0, 1]))
    s, n = orthogonalize(noisy, over_attenuated, window=11, iterations=2)
    after = abs(float(np.corrcoef(n.ravel(), s.ravel())[0, 1]))
    assert after < 0.25 * before, f"residual still correlates with signal: {before:.3f} -> {after:.3f}"


def test_orthogonalization_is_only_worth_it_when_the_residual_is_signal_dominated():
    """Honest bound on the method, and why it is off by default.

    The correction restores leaked signal, but where the residual is mostly
    genuine noise there is little to restore and the gain correction it applies
    amplifies what is left, so the referenced error barely moves. Its value has
    to be established per survey by the referenced metrics in evaluate.py, not
    by the leakage score it is built to improve.
    """
    rng = np.random.default_rng(0)
    clean = layered_section()

    quiet = clean + 0.05 * rng.standard_normal(clean.shape).astype(np.float32)
    noisy = clean + 0.8 * rng.standard_normal(clean.shape).astype(np.float32)

    def improvement(d):
        s0 = 0.6 * clean
        s, _ = orthogonalize(d, s0, window=11, iterations=1)
        return float(((s0 - clean) ** 2).mean()) - float(((s - clean) ** 2).mean())

    assert improvement(quiet) > improvement(noisy), \
        "the signal-dominated case should benefit more than the noise-dominated one"


def test_orthogonalization_barely_moves_an_already_orthogonal_split():
    """When the residual really is just noise there is nothing to give back."""
    rng = np.random.default_rng(1)
    clean = layered_section()
    noisy = clean + 0.4 * rng.standard_normal(clean.shape).astype(np.float32)

    corrected, _ = orthogonalize(noisy, clean, window=11, iterations=1)
    shift = float(np.abs(corrected - clean).mean() / (np.abs(clean).mean() + 1e-12))
    assert shift < 0.15, f"moved an already-clean estimate by {shift:.3f}"


def test_orthogonalization_weight_is_bounded():
    """An unbounded weight can inject a spike wherever the local energy vanishes."""
    rng = np.random.default_rng(0)
    kept = np.zeros((32, 32), dtype=np.float32)     # degenerate: no energy at all
    removed = rng.standard_normal((32, 32)).astype(np.float32)
    w = local_orthogonalization_weight(removed, kept, max_weight=1.0)
    assert np.isfinite(w).all()
    assert np.abs(w).max() <= 1.0


def test_orthogonalize_preserves_the_decomposition():
    """signal + noise must still add back up to the input."""
    rng = np.random.default_rng(0)
    clean = layered_section()
    noisy = clean + 0.4 * rng.standard_normal(clean.shape).astype(np.float32)
    s, n = orthogonalize(noisy, 0.7 * clean, iterations=2)
    assert np.allclose(s + n, noisy, atol=1e-4)
