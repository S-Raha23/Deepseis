"""
End-to-end test of the evaluation harness.

The harness is what every claim about the denoiser rests on, so the things
checked here are the ones whose silent failure would produce a plausible but
wrong table: that scoring happens on the split it says it does, that the
reference used for PSNR/SSIM is the clean section rather than the contaminated
one, and that the checkpoint's stored normalisation is used instead of being
recomputed from whatever data happens to be at hand.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepseis.evaluate import evaluate_methods, format_table, gather_sections
from deepseis.io.dataset import SeismicVolume
from deepseis.models.blindspot import BlindSpotDenoiser, NoiseModel


@pytest.fixture
def tiny_setup(tmp_path):
    """A small volume plus a matching (untrained) checkpoint."""
    rng = np.random.default_rng(0)
    n_il, n_xl, n_t = 120, 48, 40
    t = np.arange(n_t)[None, None, :]
    il = np.arange(n_il)[:, None, None]
    vol = np.sin(2 * np.pi * (t - 0.05 * il) / 7.0) + 0.1 * rng.standard_normal((n_il, n_xl, n_t))
    vol_path = tmp_path / "vol.npy"
    np.save(vol_path, vol.astype(np.float32))

    torch.manual_seed(0)
    model = BlindSpotDenoiser(in_channels=1, base_channels=4, depth=1)
    noise_model = NoiseModel(mode="depth", n_depth=n_t, init_sigma=0.3)

    cfg = {
        "seed": 0,
        "data": {"volume_path": str(vol_path), "split_buffer": 10, "noise_seed": 5,
                 "inject_noise": {"random_std": 0.4, "coherent_amp": 0.0}},
        "model": {"n_neighbors": 0, "base_channels": 4, "depth": 1,
                  "noise_model": "depth", "init_sigma": 0.3},
        "postprocess": {"ortho_window": 9, "ortho_iterations": 1},
    }

    ckpt_path = tmp_path / "ckpt.pt"
    volume = SeismicVolume(vol_path, buffer=10)
    torch.save({"model_state": model.state_dict(),
                "noise_model_state": noise_model.state_dict(),
                "config": cfg,
                "normalization": {"scale": volume.scale, "offset": volume.offset},
                "splits": volume.splits.as_dict(),
                "epoch": 0}, ckpt_path)
    return ckpt_path, cfg, volume


def test_evaluation_runs_and_scores_every_method(tiny_setup):
    ckpt, _, _ = tiny_setup
    report = evaluate_methods(ckpt, split="test", n_sections=3)

    assert report["has_reference"] is True
    assert report["n_sections"] == 3
    assert "blindspot" in report["methods"]
    assert "blindspot+ortho" in report["methods"]
    assert "fx_decon" in report["methods"]
    assert "identity" in report["methods"]

    for name, r in report["methods"].items():
        assert {"snr_db", "psnr", "ssim", "leakage", "energy_removed_fraction"} <= set(r), name
        assert np.isfinite(r["snr_db"]), name
        assert r["snr_db_std"] >= 0, f"{name} is missing a spread across sections"


def test_evaluation_only_touches_the_requested_split(tiny_setup):
    ckpt, cfg, volume = tiny_setup
    for split in ("train", "val", "test"):
        lo, hi = getattr(volume.splits, split)
        indices = [i for i, _, _, _ in gather_sections(volume, cfg, split, 3)]
        assert all(lo <= i < hi for i in indices), f"{split} scored out-of-split lines {indices}"


def test_identity_control_reproduces_the_input_snr(tiny_setup):
    """Anchors the table: 'identity' must score exactly the SNR of the noisy
    input, so every other row can be read as a gain over doing nothing."""
    ckpt, cfg, volume = tiny_setup
    report = evaluate_methods(ckpt, split="test", n_sections=3)

    expected = []
    for _, _, centre, clean in gather_sections(volume, cfg, "test", 3):
        expected.append(10 * np.log10((clean.astype(np.float64) ** 2).mean() /
                                      (((centre - clean).astype(np.float64) ** 2).mean())))
    assert report["methods"]["identity"]["snr_db"] == pytest.approx(float(np.mean(expected)), rel=1e-4)
    assert report["methods"]["identity"]["energy_removed_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert report["methods"]["identity"]["leakage"] == pytest.approx(0.0, abs=1e-9)


def test_reference_is_the_clean_section_not_the_contaminated_one(tiny_setup):
    """If the 'clean' reference were the noisy input, identity would score
    infinite SNR and the whole table would be meaningless."""
    _, cfg, volume = tiny_setup
    for _, _, centre, clean in gather_sections(volume, cfg, "test", 2):
        assert not np.allclose(centre, clean), "reference equals the contaminated input"
        assert centre.shape == clean.shape


def test_checkpoint_normalization_is_used_rather_than_recomputed(tiny_setup, tmp_path):
    """Recomputing normalisation at evaluation time would reintroduce exactly
    the train/serve mismatch the rebuild removed, and would do it silently."""
    ckpt, _, _ = tiny_setup
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state["normalization"] = {"scale": 999.0, "offset": 0.0}
    doctored = tmp_path / "doctored.pt"
    torch.save(state, doctored)

    normal = evaluate_methods(ckpt, split="test", n_sections=2)
    scaled = evaluate_methods(doctored, split="test", n_sections=2)
    assert normal["methods"]["blindspot"]["snr_db"] != pytest.approx(
        scaled["methods"]["blindspot"]["snr_db"], rel=1e-6), \
        "the stored normalisation had no effect, so it is not being used"


def test_format_table_renders_without_error(tiny_setup):
    ckpt, _, _ = tiny_setup
    text = format_table(evaluate_methods(ckpt, split="test", n_sections=2))
    assert "identity" in text and "fx_decon" in text
    assert "spectral energy retained" in text


def test_field_mode_reports_no_reference_metrics(tiny_setup, tmp_path):
    ckpt, _, _ = tiny_setup
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state["config"]["data"]["inject_noise"] = None
    field_ckpt = tmp_path / "field.pt"
    torch.save(state, field_ckpt)

    report = evaluate_methods(field_ckpt, split="test", n_sections=2)
    assert report["has_reference"] is False
    assert "snr_db" not in report["methods"]["blindspot"]
    assert "leakage" in report["methods"]["blindspot"]
