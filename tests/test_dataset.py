"""
Tests for the data layer.

The properties asserted here are the ones whose failure would invalidate every
number the project reports, without necessarily breaking anything visibly:
that held-out data is genuinely held out, that normalisation is identical at
train and serve time, and that injected validation noise is fixed per section
rather than resampled (resampling would quietly convert the benchmark into the
easier Noise2Noise problem).
"""
from __future__ import annotations

import numpy as np
import pytest

from deepseis.io.dataset import (PatchSampler, SeismicVolume, inject_noise, make_splits,
                                 section_seed)


@pytest.fixture
def fake_volume(tmp_path):
    """A small stand-in volume with laterally correlated structure."""
    rng = np.random.default_rng(0)
    n_il, n_xl, n_t = 120, 60, 48
    t = np.arange(n_t)[None, None, :]
    il = np.arange(n_il)[:, None, None]
    vol = np.sin(2 * np.pi * (t - 0.05 * il) / 7.0) + 0.1 * rng.standard_normal((n_il, n_xl, n_t))
    path = tmp_path / "vol.npy"
    np.save(path, vol.astype(np.float32))
    return path


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_splits_are_disjoint_and_buffered():
    s = make_splits(401, buffer=30)
    assert s.train[1] <= s.val[0] and s.val[1] <= s.test[0]
    assert s.val[0] - s.train[1] == 30, "missing buffer between train and val"
    assert s.test[0] - s.val[1] == 30, "missing buffer between val and test"
    assert s.test[1] <= 401


def test_splits_reject_a_volume_too_small_for_the_buffer():
    with pytest.raises(ValueError):
        make_splits(40, buffer=30)


def test_no_inline_appears_in_two_splits(fake_volume):
    vol = SeismicVolume(fake_volume, buffer=10)
    train = set(vol.split_indices("train").tolist())
    val = set(vol.split_indices("val").tolist())
    test = set(vol.split_indices("test").tolist())
    assert not (train & val) and not (train & test) and not (val & test)


def test_training_crosslines_are_truncated_to_the_training_block(fake_volume):
    """A full-length crossline cuts through every inline, so an untruncated one
    would carry test-block samples into the training set."""
    vol = SeismicVolume(fake_volume, buffer=10)
    lo, hi = vol.splits.train
    sec = vol.crossline(0, inline_range=(lo, hi))
    assert sec.shape[1] == hi - lo, "crossline was not truncated to the training range"
    assert sec.shape[1] < vol.n_inlines


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_normalization_is_computed_from_training_data_only(fake_volume):
    vol = SeismicVolume(fake_volume, buffer=10)
    lo, hi = vol.splits.train
    idx = np.linspace(lo, hi - 1, min(40, hi - lo)).astype(int)
    expected = float(np.asarray(vol.volume[idx], dtype=np.float64).std())
    assert vol.scale == pytest.approx(expected, rel=1e-9)


def test_normalization_is_a_single_global_constant_not_per_section(fake_volume):
    """Per-section normalisation makes a line's denoised amplitude depend on
    what else is in it, which breaks amplitude comparison between lines."""
    vol = SeismicVolume(fake_volume, buffer=10)

    # std is shift invariant, so std(normalized)/std(raw) isolates the scale
    # factor actually applied to each section.
    ratios = []
    for i in (5, 50, 90):
        raw = vol.inline(i, normalized=False)
        ratios.append(float(vol.inline(i).std() / raw.std()))
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-5), "sections were scaled differently"
    assert ratios[0] == pytest.approx(ratios[2], rel=1e-5), "sections were scaled differently"
    assert ratios[0] == pytest.approx(1.0 / vol.scale, rel=1e-5)

    # and the map is exactly the stored affine transform, not a per-section fit
    raw = vol.inline(5, normalized=False)
    assert np.allclose(vol.inline(5), (raw - vol.offset) / vol.scale, atol=1e-6)


def test_normalize_denormalize_roundtrip(fake_volume):
    vol = SeismicVolume(fake_volume, buffer=10)
    raw = vol.inline(3, normalized=False)
    assert np.allclose(vol.denormalize(vol.normalize(raw)), raw, atol=1e-4)


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def test_injected_noise_is_identical_across_repeated_calls():
    """Fixed per section: resampling every epoch would show the model the same
    geology under independent noise, which is Noise2Noise and strictly easier
    than the field problem this stands in for."""
    rng = np.random.default_rng(0)
    sec = rng.standard_normal((32, 32)).astype(np.float32)
    cfg = {"random_std": 0.5, "coherent_amp": 0.3, "n_coherent_events": 2}
    a = inject_noise(sec, cfg, seed=7)
    b = inject_noise(sec, cfg, seed=7)
    assert np.array_equal(a, b), "same seed produced different noise"


def test_different_sections_get_different_noise():
    rng = np.random.default_rng(0)
    sec = rng.standard_normal((32, 32)).astype(np.float32)
    cfg = {"random_std": 0.5}
    a = inject_noise(sec, cfg, section_seed(1234, "inline", 10))
    b = inject_noise(sec, cfg, section_seed(1234, "inline", 11))
    assert not np.array_equal(a, b)


def test_section_seed_separates_orientations():
    assert section_seed(1, "inline", 5) != section_seed(1, "crossline", 5)


def test_injected_noise_actually_lowers_snr():
    rng = np.random.default_rng(0)
    sec = rng.standard_normal((64, 64)).astype(np.float32)
    noisy = inject_noise(sec, {"random_std": 0.5, "coherent_amp": 0.3}, seed=1)
    assert np.abs(noisy - sec).mean() > 0.1


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sampler_returns_correct_shapes_and_depth_offsets(fake_volume):
    vol = SeismicVolume(fake_volume, buffer=10)
    s = PatchSampler(vol, n_inline_sections=6, n_crossline_sections=3,
                     rng=np.random.default_rng(0))
    patches, offsets = s.sample_batch(batch_size=4, patch_size=16)
    assert patches.shape == (4, 1, 16, 16)
    assert offsets.shape == (4,)
    assert (offsets >= 0).all()
    assert np.isfinite(patches).all()


def test_neighbor_stacking_produces_the_requested_channel_count(fake_volume):
    vol = SeismicVolume(fake_volume, buffer=10)
    s = PatchSampler(vol, n_inline_sections=6, n_crossline_sections=0, n_neighbors=2,
                     rng=np.random.default_rng(0))
    patches, _ = s.sample_batch(batch_size=2, patch_size=16)
    assert patches.shape == (2, 5, 16, 16)


def test_neighbor_channels_get_independent_noise_realisations(fake_volume):
    """If every channel shared one realisation, the neighbours would be a free
    look at the centre line's noise and the blind spot would be pointless."""
    vol = SeismicVolume(fake_volume, buffer=10)
    s = PatchSampler(vol, n_inline_sections=4, n_crossline_sections=0, n_neighbors=1,
                     rng=np.random.default_rng(0),
                     noise_cfg={"random_std": 0.8}, noise_seed=99)
    sec = s.sections[0]
    assert not np.allclose(sec[0], sec[1]), "neighbouring channels carry identical noise"


def test_sampler_only_draws_from_the_training_block(fake_volume):
    """Every preloaded section must be reproducible from training-block data."""
    vol = SeismicVolume(fake_volume, buffer=10)
    s = PatchSampler(vol, n_inline_sections=5, n_crossline_sections=0,
                     rng=np.random.default_rng(0))
    lo, hi = vol.splits.train
    train_sections = [vol.inline(int(i)) for i in range(lo, hi)]
    for sec in s.sections:
        assert any(np.allclose(sec[0], t, atol=1e-5) for t in train_sections), \
            "a preloaded section does not come from the training block"
