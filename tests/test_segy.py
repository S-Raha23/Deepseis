import numpy as np
import pytest

from deepseis.io import segy as segy_mod
from pathlib import Path


def test_npy_roundtrip(tmp_path):
    vol = np.random.default_rng(0).normal(size=(48, 48)).astype(np.float32)
    path = tmp_path / "section.npy"
    segy_mod.save_npy(path, vol)
    loaded = segy_mod.load_volume(path)
    np.testing.assert_allclose(loaded, vol, atol=1e-6)


def test_faultseg3d_dat_sample_loads():
    """Real-data smoke test: a genuine FaultSeg3D synthetic seis/fault validation pair,
    fetched via `data/download.py` (git clone of xinwucwp/faultSeg) and bundled here."""
    samples_dir = Path(__file__).parent.parent / "data" / "samples" / "faultseg3d"
    seis_path = samples_dir / "sample_seis_0.dat"
    fault_path = samples_dir / "sample_fault_0.dat"
    if not seis_path.exists():
        pytest.skip("bundled FaultSeg3D sample not present")

    seis = segy_mod.load_volume(seis_path)
    fault = segy_mod.load_volume(fault_path)
    assert seis.shape == (128, 128, 128)
    assert fault.shape == (128, 128, 128)
    assert set(np.unique(fault)).issubset({0.0, 1.0})
    assert seis.std() > 0


def test_segy_write_and_read_roundtrip(tmp_path):
    section = np.random.default_rng(1).normal(size=(64, 20)).astype(np.float32)  # (n_samples, n_traces)
    out_path = tmp_path / "synthetic.sgy"
    segy_mod.write_segy_like(out_path, section, template_path=None, dt_ms=4.0)

    assert out_path.exists()
    loaded = segy_mod.load_volume(out_path)  # (n_inlines, n_crosslines, n_samples) for a 3D cube
    assert loaded.ndim == 3
    assert loaded.shape == (1, section.shape[1], section.shape[0])
    # transpose the single inline back to our (n_samples, n_traces) convention and compare
    recovered = loaded[0].T
    np.testing.assert_allclose(recovered, section, atol=1e-4)
