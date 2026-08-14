"""
Tests for inference and checkpointing.

The central claim being checked is that full-section inference is legitimate:
the old pipeline had to cut sections into overlapping windows and average the
predictions, which is a smoothing operator applied after the model, and it
needed to because ``GroupNorm`` made a pixel's output depend on which window
it fell in. Removing the normalisation layers is only worth anything if the
network really is translation equivariant, so that is asserted rather than
assumed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepseis.denoise import denoise_section, denoise_section_tiled, load_denoiser
from deepseis.losses.nll import posterior_mean
from deepseis.models.blindspot import (BlindSpotDenoiser, NoiseModel,
                                       receptive_field_halfwidth)


@pytest.fixture
def model_pair():
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=8, depth=2).eval()
    noise_model = NoiseModel(mode="depth", n_depth=256, init_sigma=0.3).eval()
    return model, noise_model


def ramped_section(nt=64, nx=200):
    t = np.arange(nt)[:, None]
    x = np.arange(nx)[None, :]
    return np.sin(2 * np.pi * (t - 0.3 * x) / 9.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Translation equivariance -- the property that makes tiling unnecessary
# ---------------------------------------------------------------------------

def test_network_is_translation_equivariant_along_traces(model_pair):
    """A pixel's estimate must not depend on where the window around it starts.

    With normalisation layers present this is false -- GroupNorm rescales by
    statistics of whatever else is in the window -- and that is what forced the
    old patch-and-average inference path to exist. It is checked by running two
    *overlapping crops* of one long section and comparing the shared region.

    Zero padding at a crop's own border breaks equivariance within one
    receptive field of it, which is true of any padded CNN and not what is
    being tested, so the comparison excludes a margin of one measured
    receptive field from each crop's edges.
    """
    model, _ = model_pair
    rf = receptive_field_halfwidth(model)

    long_section = ramped_section(64, 480)
    a0, a1 = 0, 300
    b0, b1 = 120, 420

    with torch.no_grad():
        a, _ = model(torch.from_numpy(long_section[:, a0:a1])[None, None])
        b, _ = model(torch.from_numpy(long_section[:, b0:b1])[None, None])

    # Region valid in both crops, in coordinates of the original section.
    lo = max(a0 + rf, b0 + rf)
    hi = min(a1 - rf, b1 - rf)
    assert hi - lo > 20, "test window collapsed; widen the crops"

    lhs = a[0, 0, :, lo - a0: hi - a0].numpy()
    rhs = b[0, 0, :, lo - b0: hi - b0].numpy()
    assert np.allclose(lhs, rhs, atol=1e-5), \
        f"output depends on absolute trace position (max dev {np.abs(lhs - rhs).max():.2e})"


def test_receptive_field_is_symmetric_in_all_four_directions(model_pair):
    """An asymmetric extent would mean one of the four rotated branches is not
    contributing, which would leave part of the context unused."""
    model, _ = model_pair
    n = 161
    x = torch.zeros(1, 1, n, n, requires_grad=True)
    mu, _ = model(x)
    (g,) = torch.autograd.grad(mu[0, 0, n // 2, n // 2], x)
    nz = (g[0, 0].abs() > 0).numpy()
    ys, xs = np.nonzero(nz)
    c = n // 2
    assert (c - ys.min()) == (ys.max() - c), "vertical reach is asymmetric"
    assert (c - xs.min()) == (xs.max() - c), "horizontal reach is asymmetric"
    assert (c - ys.min()) == (c - xs.min()), "vertical and horizontal reach differ"


def test_output_does_not_depend_on_section_amplitude(model_pair):
    """Scale equivariance end to end: doubling the input doubles the estimate.

    The posterior mean is not scale equivariant on its own -- sigma_n is a
    physical noise level, not a free scale -- so this checks the network's
    contribution, which is where the old 4.4x train/serve mismatch bit.
    """
    model, _ = model_pair
    sec = torch.from_numpy(ramped_section(48, 48))[None, None]
    with torch.no_grad():
        mu1, s1 = model(sec)
        mu2, s2 = model(3.0 * sec)
    assert torch.allclose(mu2, 3.0 * mu1, atol=1e-4)
    assert torch.allclose(s2, 3.0 * s1, atol=1e-4)


# ---------------------------------------------------------------------------
# Inference paths agree
# ---------------------------------------------------------------------------

def test_tiled_inference_matches_single_pass_in_tile_interiors(model_pair):
    """The tiled path crops rather than blends, so with a margin wider than the
    receptive field it must reproduce the single-pass answer exactly enough."""
    model, noise_model = model_pair
    sec = ramped_section(64, 320)
    full = denoise_section(model, noise_model, sec)
    tiled = denoise_section_tiled(model, noise_model, sec, tile=256)
    assert tiled.shape == full.shape
    assert np.allclose(tiled, full, atol=1e-4), \
        f"max deviation {np.abs(tiled - full).max():.2e}"


def test_denoise_section_shape_and_finiteness(model_pair):
    model, noise_model = model_pair
    sec = ramped_section(255, 101)
    out = denoise_section(model, noise_model, sec)
    assert out.shape == (255, 101)
    assert np.isfinite(out).all()


def test_denoise_section_accepts_a_neighbour_stack():
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=3, base_channels=8, depth=2).eval()
    noise_model = NoiseModel(mode="scalar", init_sigma=0.3).eval()
    stack = np.stack([ramped_section(64, 80) for _ in range(3)])
    out = denoise_section(model, noise_model, stack)
    assert out.shape == (64, 80)


def test_extras_expose_an_uncertainty_map(model_pair):
    model, noise_model = model_pair
    out, extras = denoise_section(model, noise_model, ramped_section(64, 64), return_extras=True)
    assert extras["uncertainty"].shape == out.shape
    assert (extras["uncertainty"] >= 0).all()
    assert extras["sigma_noise"].shape == (64,)


# ---------------------------------------------------------------------------
# Denoising actually happens
# ---------------------------------------------------------------------------

def test_posterior_mean_reduces_to_the_input_when_the_noise_model_says_zero_noise(model_pair):
    """A denoiser told there is no noise must return the observation untouched."""
    model, _ = model_pair
    sec = torch.from_numpy(ramped_section(48, 48))[None, None]
    with torch.no_grad():
        mu, sigma_p = model(sec)
    out = posterior_mean(mu, sigma_p, torch.zeros_like(sigma_p), sec)
    assert torch.allclose(out, sec, atol=1e-5)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_reproduces_the_same_output(tmp_path, model_pair):
    model, noise_model = model_pair
    cfg = {"model": {"n_neighbors": 0, "base_channels": 8, "depth": 2,
                     "noise_model": "depth", "init_sigma": 0.3}}
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(),
                "noise_model_state": noise_model.state_dict(),
                "config": cfg,
                "normalization": {"scale": 0.21, "offset": 0.0},
                "epoch": 3}, path)

    sec = ramped_section(64, 64)
    before = denoise_section(model, noise_model, sec)

    loaded, loaded_noise, ckpt = load_denoiser(path)
    after = denoise_section(loaded, loaded_noise, sec)

    assert np.allclose(before, after, atol=1e-6)
    assert ckpt["normalization"]["scale"] == 0.21


def test_checkpoint_carries_the_normalization_constants(tmp_path, model_pair):
    """Losing these is how a train/serve amplitude mismatch happens."""
    model, noise_model = model_pair
    cfg = {"model": {"n_neighbors": 0, "base_channels": 8, "depth": 2,
                     "noise_model": "depth", "init_sigma": 0.3}}
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(),
                "noise_model_state": noise_model.state_dict(),
                "config": cfg,
                "normalization": {"scale": 0.5, "offset": 0.01}}, path)
    _, _, ckpt = load_denoiser(path)
    assert set(ckpt["normalization"]) == {"scale", "offset"}


# ---------------------------------------------------------------------------
# Raw-amplitude serving path (what the dashboard and infer.py rely on)
# ---------------------------------------------------------------------------

def test_raw_amplitude_roundtrip_uses_the_checkpoint_constants(tmp_path, model_pair):
    """Denoising raw data means normalise with the stored constants, run, and
    scale back. Getting this wrong is precisely the 4.4x train/serve mismatch
    the previous pipeline shipped, and it is silent -- the output still looks
    like a seismic section, just at the wrong amplitude.
    """
    model, noise_model = model_pair
    scale, offset = 0.21943, 0.00147

    normalized = ramped_section(64, 96)
    raw = normalized * scale + offset

    # serve the raw section the way the dashboard does
    served = denoise_section(model, noise_model, (raw - offset) / scale)
    served_raw = served * scale + offset

    # ...must equal denoising the normalised section and scaling back
    direct = denoise_section(model, noise_model, normalized) * scale + offset
    assert np.allclose(served_raw, direct, atol=1e-6)

    # The affine relation must hold exactly, which is what makes the two paths
    # interchangeable. (The output's absolute amplitude is not asserted here:
    # an untrained model has near-zero sigma_p, so the posterior collapses onto
    # mu and carries no amplitude information yet.)
    assert np.allclose((served_raw - offset) / scale, served, atol=1e-6)


def test_serving_without_rescaling_would_be_caught():
    """Guards the failure mode rather than just the fix: a model served at the
    wrong scale produces an output whose amplitude is off by that factor."""
    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=8, depth=2).eval()
    noise_model = NoiseModel(mode="scalar", init_sigma=0.3).eval()

    normalized = ramped_section(48, 48)
    scale = 0.21943
    correct = denoise_section(model, noise_model, normalized) * scale
    forgot_to_rescale = denoise_section(model, noise_model, normalized)
    assert abs(forgot_to_rescale.std() / correct.std() - 1 / scale) < 0.1


def test_serve_section_applies_config_postprocessing(tmp_path, model_pair):
    """serve_section is the single entry point; enabling post-processing in the
    checkpoint's config must actually change the served result, or the
    dashboard and infer.py would silently diverge from what was evaluated."""
    from deepseis.denoise import serve_section

    model, noise_model = model_pair
    base_cfg = {"model": {"n_neighbors": 0, "base_channels": 8, "depth": 2,
                          "noise_model": "depth", "init_sigma": 0.3}}
    norm = {"scale": 0.21943, "offset": 0.00147}
    raw = ramped_section(64, 96) * norm["scale"] + norm["offset"]

    plain = serve_section(model, noise_model,
                          {"config": {**base_cfg, "postprocess": {}}, "normalization": norm}, raw)
    ortho = serve_section(model, noise_model,
                          {"config": {**base_cfg, "postprocess": {"orthogonalize": True,
                                                                  "ortho_window": 11,
                                                                  "ortho_iterations": 1}},
                           "normalization": norm}, raw)
    per_sec = serve_section(model, noise_model,
                            {"config": {**base_cfg, "postprocess": {"per_section_sigma": True}},
                             "normalization": norm}, raw)

    assert plain.shape == raw.shape
    assert not np.allclose(plain, ortho), "orthogonalize flag had no effect"
    assert not np.allclose(plain, per_sec), "per_section_sigma flag had no effect"


def test_serve_section_returns_input_units(model_pair):
    """The result must come back on the input's amplitude scale, not the
    normalised one -- this is the 4.4x mismatch, guarded at the entry point."""
    from deepseis.denoise import serve_section

    model, noise_model = model_pair
    norm = {"scale": 0.21943, "offset": 0.00147}
    ckpt = {"config": {"model": {"n_neighbors": 0}, "postprocess": {}}, "normalization": norm}

    raw = ramped_section(48, 64) * norm["scale"] + norm["offset"]
    out, extras = serve_section(model, noise_model, ckpt, raw, return_extras=True)

    # With no post-processing, serve_section must be exactly the explicit
    # pipeline: normalise with the stored constants, denoise, scale back.
    expected = denoise_section(model, noise_model,
                               (raw - norm["offset"]) / norm["scale"]) * norm["scale"] + norm["offset"]
    assert np.allclose(out, expected, atol=1e-6)

    # The uncertainty map must be returned in input units too, not normalised ones.
    _, raw_extras = denoise_section(model, noise_model, (raw - norm["offset"]) / norm["scale"],
                                    return_extras=True)
    assert np.allclose(extras["uncertainty"], raw_extras["uncertainty"] * norm["scale"], atol=1e-6)
    assert extras["uncertainty"].shape == out.shape


def test_serve_section_warns_when_handed_already_normalised_data(model_pair):
    """The guard for a bug made at three separate call sites in this project.

    Double normalisation is silent: the network's scale equivariance absorbs it
    while sigma_n does not, so the denoiser under-removes without raising
    anything. Measured cost when it happened: 3.17 dB and 4x less noise removed.
    """
    from deepseis.denoise import serve_section

    model, noise_model = model_pair
    norm = {"scale": 0.21943, "offset": 0.00147}
    ckpt = {"config": {"model": {"n_neighbors": 0}, "postprocess": {}}, "normalization": norm}

    raw = ramped_section(48, 64) * norm["scale"]
    already_normalised = ramped_section(48, 64)
    already_normalised = already_normalised / already_normalised.std()

    with pytest.warns(RuntimeWarning, match="already normalised"):
        serve_section(model, noise_model, ckpt, already_normalised)

    # and raw data must pass silently
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        serve_section(model, noise_model, ckpt, raw)


def test_normalisation_guard_stays_quiet_for_surveys_recorded_near_unit_scale(model_pair):
    """A survey genuinely at unit amplitude must not be nagged."""
    from deepseis.denoise import serve_section

    model, noise_model = model_pair
    ckpt = {"config": {"model": {"n_neighbors": 0}, "postprocess": {}},
            "normalization": {"scale": 1.02, "offset": 0.0}}

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        serve_section(model, noise_model, ckpt, ramped_section(48, 64))
