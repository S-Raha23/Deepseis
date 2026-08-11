import numpy as np
import pytest

from deepseis.io import noise as noise_mod
from deepseis.io import patches as patch_mod
from deepseis.io import synthetic as synth_mod
from deepseis import metrics as metrics_mod
from deepseis.masking import noise2void as n2v_mod
from deepseis.masking import struct_n2v as struct_mod


def test_synthetic_volume_shapes():
    vol = synth_mod.generate_synthetic_volume(n_traces=64, n_samples=64, n_layers=6, n_faults=2, random_seed=1)
    assert vol.clean.shape == (64, 64)
    assert vol.fault_mask.shape == (64, 64)
    assert vol.facies.shape == (64, 64)
    assert vol.fault_mask.sum() > 0, "at least some fault pixels should be painted"
    assert set(np.unique(vol.fault_mask)).issubset({0.0, 1.0})


def test_synthetic_volume_deterministic():
    v1 = synth_mod.generate_synthetic_volume(n_traces=32, n_samples=32, random_seed=42)
    v2 = synth_mod.generate_synthetic_volume(n_traces=32, n_samples=32, random_seed=42)
    np.testing.assert_array_equal(v1.clean, v2.clean)


def test_noise_increases_distance_from_clean():
    vol = synth_mod.generate_synthetic_volume(n_traces=64, n_samples=64, random_seed=3)
    rng = np.random.default_rng(0)
    cfg = {"data": {"synthetic": {"dt_ms": 4.0}, "noise": {"random_std": 0.2, "coherent_amp": 0.3,
                                                             "coherent_type": "ground_roll"}}}
    noisy = noise_mod.make_noisy(vol.clean, cfg, rng=rng)
    assert metrics_mod.psnr(noisy, vol.clean) < 60  # noisy should clearly differ from clean
    assert not np.allclose(noisy, vol.clean)


def test_patch_extract_stitch_roundtrip():
    img = np.random.default_rng(0).normal(size=(96, 96)).astype(np.float32)
    patches = patch_mod.extract_patches(img, size=32, stride=16)
    assert patches.shape[1:] == (32, 32)
    stitched = patch_mod.stitch_patches(patches, 96, 96, 32, 16)
    # averaging overlaps with a window should stay close to the original signal
    assert np.corrcoef(img.ravel(), stitched.ravel())[0, 1] > 0.95


def test_n2v_mask_hides_true_value():
    rng = np.random.default_rng(0)
    patch = rng.normal(size=(32, 32)).astype(np.float32)
    cfg = {"masking": {"n2v": {"mask_fraction": 0.1, "neighborhood_radius": 4}}}
    masked_input, mask, target = n2v_mod.make_training_pair(patch, cfg, rng)
    assert masked_input.shape == patch.shape
    assert mask.sum() > 0
    # at masked locations, input should (almost always) differ from the true value
    diffs = masked_input[mask] != target[mask]
    assert diffs.mean() > 0.5


def test_struct_n2v_blinds_region_not_just_center():
    rng = np.random.default_rng(0)
    patch = rng.normal(size=(32, 32)).astype(np.float32)
    cfg = {"masking": {"struct_n2v": {"mask_fraction": 0.02, "blind_shape": "trace", "blind_width": 3}}}
    masked_input, center_mask, target = struct_mod.make_training_pair(patch, cfg, rng)
    assert center_mask.sum() >= 1
    n_changed = (masked_input != patch).sum()
    assert n_changed >= center_mask.sum()  # blind region is >= the center-pixel-only mask


def test_psnr_perfect_match_is_inf():
    x = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    assert metrics_mod.psnr(x, x) == float("inf")


def test_dice_perfect_match():
    x = np.zeros((10, 10), dtype=bool)
    x[2:5, 2:5] = True
    assert metrics_mod.dice_score(x, x) == pytest.approx(1.0)


def test_roc_auc_perfect_separation():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 500 + [1] * 500)
    scores = np.concatenate([rng.uniform(0, 0.4, 500), rng.uniform(0.6, 1.0, 500)])
    auc = metrics_mod.roc_auc(scores, labels)
    assert auc > 0.99


def test_distance_metric_zero_for_identical_masks():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:8, 5:8] = True
    d = metrics_mod.distance_based_fault_metric(mask, mask)
    assert d == pytest.approx(0.0, abs=1e-6)
