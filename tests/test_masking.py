"""
Regression tests for the legacy masking denoiser's blind spots.

Both of this module's blind spots were broken in ways that ran cleanly and
were only visible if you measured them:

* ``apply_blind_spot`` could replace a masked pixel with itself, and the loss
  is evaluated at exactly those pixels, so the network was rewarded for
  copying its input.
* ``struct_n2v`` blinded a stripe spanning the full extent of an axis, so with
  the shipped settings 97.9% of every patch was replaced by a shuffle of
  itself and training became noise-to-noise regression.

The path is retained only so the previous model can be scored against the
current one, but it is kept honest here so nobody rebuilds on top of it
unaware.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepseis.masking import noise2void as n2v
from deepseis.masking import struct_n2v as sn2v


CFG = {
    "masking": {
        "n2v": {"mask_fraction": 0.02, "neighborhood_radius": 5},
        "struct_n2v": {"mask_fraction": 0.02, "blind_shape": "trace",
                       "blind_width": 3, "blind_length": 9, "max_blind_fraction": 0.35},
    }
}


# ---------------------------------------------------------------------------
# N2V: the blind spot must actually be blind
# ---------------------------------------------------------------------------

def test_no_masked_pixel_is_ever_replaced_by_itself():
    """A self-donation makes the reconstruction target visible in the input."""
    rng = np.random.default_rng(0)
    self_donations = 0
    total = 0
    for _ in range(30):
        patch = rng.standard_normal((64, 64)).astype(np.float32)
        mask = n2v.generate_mask(patch.shape, 0.05, rng)
        out = n2v.apply_blind_spot(patch, mask, 5, rng)
        ys, xs = np.nonzero(mask)
        self_donations += int(np.sum(out[ys, xs] == patch[ys, xs]))
        total += ys.size
    assert self_donations == 0, f"{self_donations} of {total} masked pixels saw themselves"


def test_blind_spot_leaves_unmasked_pixels_untouched():
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((32, 32)).astype(np.float32)
    mask = n2v.generate_mask(patch.shape, 0.1, rng)
    out = n2v.apply_blind_spot(patch, mask, 5, rng)
    assert np.array_equal(out[~mask], patch[~mask])


def test_reflect_never_returns_an_out_of_range_index():
    for n in (1, 2, 5, 64):
        idx = np.arange(-3 * n, 3 * n)
        out = n2v._reflect(idx, n)
        assert out.min() >= 0 and out.max() <= n - 1


# ---------------------------------------------------------------------------
# StructN2V: the blind region must not swallow the patch
# ---------------------------------------------------------------------------

def test_blind_region_is_a_bounded_segment_not_a_full_axis_stripe():
    """The defect that made the previous denoiser a low-pass filter."""
    rng = np.random.default_rng(0)
    centre, blind = sn2v.generate_mask_and_blind_regions(
        (64, 64), 0.02, "trace", 3, rng, blind_length=9, max_blind_fraction=0.35)

    assert blind.mean() <= 0.35 + 1e-9, f"blinded {blind.mean():.1%} of the patch"
    assert centre.sum() >= 1
    # a single centre must not reach the top and bottom of the patch
    ys, xs = sn2v._blind_region_indices((32, 32), (64, 64), "trace", 3, 9)
    assert ys.max() - ys.min() <= 9, "blind region still spans a full column"


@pytest.mark.parametrize("shape", ["trace", "horizontal", "diagonal"])
def test_every_blind_shape_stays_within_the_budget(shape):
    rng = np.random.default_rng(0)
    _, blind = sn2v.generate_mask_and_blind_regions(
        (64, 64), 0.05, shape, 3, rng, blind_length=9, max_blind_fraction=0.35)
    assert blind.mean() <= 0.4, f"{shape} blinded {blind.mean():.1%}"


def test_most_of_the_patch_survives_so_there_is_something_to_predict_from():
    rng = np.random.default_rng(0)
    patch = rng.standard_normal((64, 64)).astype(np.float32)
    masked, centre, target = sn2v.make_training_pair(patch, CFG, rng)

    altered = float((masked != patch).mean())
    assert altered < 0.4, f"{altered:.1%} of the input was replaced"
    assert centre.sum() >= 1
    assert np.array_equal(target, patch)


def test_blind_region_covers_every_centre_it_kept():
    """Each retained centre must itself be hidden, or its target is visible."""
    rng = np.random.default_rng(0)
    centre, blind = sn2v.generate_mask_and_blind_regions(
        (48, 48), 0.02, "trace", 3, rng, blind_length=9, max_blind_fraction=0.35)
    assert blind[centre].all(), "a scored centre pixel was left visible in the input"


def test_noise_type_estimator_is_documented_as_unreliable_and_not_the_default():
    """`auto` used to be the shipped default and sent 76% of F3 down the
    struct_n2v path; the docstring warning is load-bearing."""
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["masking"]["mode"] != "auto", "the unreliable estimator is the default again"
    assert "warning" in sn2v.estimate_noise_type.__doc__.lower()
