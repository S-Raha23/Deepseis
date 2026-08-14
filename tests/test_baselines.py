"""
Tests for the classical baselines.

A baseline that silently underperforms makes the learned model look good for
the wrong reason, so these check that each one actually denoises, and that the
dip estimator underneath the structure-oriented filter recovers a dip it is
given. The first version of ``local_dip`` used the standard
``0.5*arctan2(2*jxy, jyy-jxx)`` orientation shortcut, which returns the
gradient direction rather than the structure direction; it cost roughly 4 dB
and was caught here rather than in review.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepseis import baselines as B


def dipping_section(nt: int = 128, nx: int = 128, dip: float = 0.5, period: float = 10.0) -> np.ndarray:
    t = np.arange(nt)[:, None]
    x = np.arange(nx)[None, :]
    return np.sin(2 * np.pi * (t - dip * x) / period).astype(np.float32)


def snr_db(pred: np.ndarray, clean: np.ndarray) -> float:
    return float(10 * np.log10((clean ** 2).mean() / (((pred - clean) ** 2).mean() + 1e-30)))


@pytest.mark.parametrize("true_dip", [-1.5, -0.5, 0.0, 0.5, 1.5])
def test_local_dip_recovers_a_known_dip(true_dip):
    sec = dipping_section(dip=true_dip)
    dip = B.local_dip(sec)
    interior = dip[30:-30, 30:-30]
    assert np.median(interior) == pytest.approx(true_dip, abs=0.15), \
        f"estimated {np.median(interior):.3f} for a true dip of {true_dip}"


def test_structure_oriented_smoothing_preserves_a_dipping_reflector():
    """Smoothing along the dip must leave a noiseless dipping event alone."""
    sec = dipping_section(dip=0.5)
    out = B.structure_oriented_smoothing(sec)
    assert snr_db(out[20:-20, 20:-20], sec[20:-20, 20:-20]) > 20.0


def test_every_baseline_improves_snr_on_a_noisy_dipping_section():
    rng = np.random.default_rng(0)
    clean = dipping_section(dip=0.4)
    noisy = clean + 0.7 * rng.standard_normal(clean.shape).astype(np.float32)
    base = snr_db(noisy, clean)

    for name, fn in B.BASELINES.items():
        if name == "identity":
            continue
        out = fn(noisy)
        assert snr_db(out, clean) > base, f"{name} made SNR worse ({snr_db(out, clean):.2f} vs {base:.2f})"


def test_identity_changes_nothing():
    sec = dipping_section()
    assert np.array_equal(B.identity(sec), sec)


def test_fx_decon_keeps_a_purely_predictable_event():
    """A single dipping event is exactly AR-predictable along the trace axis,
    so f-x decon should pass it through nearly untouched."""
    sec = dipping_section(dip=0.3)
    out = B.fx_decon(sec)
    assert snr_db(out[10:-10, 30:-30], sec[10:-10, 30:-30]) > 15.0


def test_baselines_preserve_shape_and_finiteness():
    rng = np.random.default_rng(1)
    sec = (dipping_section(nt=255, nx=101) + 0.4 * rng.standard_normal((255, 101))).astype(np.float32)
    for name, fn in B.BASELINES.items():
        out = fn(sec)
        assert out.shape == sec.shape, f"{name} changed the shape"
        assert np.isfinite(out).all(), f"{name} produced non-finite values"
