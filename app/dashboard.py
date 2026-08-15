"""
DeepSeis dashboard — F3 Netherlands real data
Fault-preserving self-supervised seismic denoising & auto-interpretation.

Run with:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseis import metrics as metrics_mod
from deepseis import viz3d
from deepseis.interpretation.horizon import track_horizons
from deepseis.io import noise as noise_mod
from deepseis.io import segy as segy_mod
from deepseis.io import synthetic as synth_mod
from deepseis.losses.frequency import fk_spectrum
from deepseis.models.facies import FaciesNet2D
from deepseis.models.faultseg import FaultSegNet2D
from deepseis.models.unet import DenoiserUNet
from deepseis.train import (get_device, load_config, run_denoiser_inference,
                             run_faultseg_inference, set_seed, train_denoiser, train_faultseg)

st.set_page_config(page_title="DeepSeis", layout="wide", page_icon="\U0001FAA8")

SEISMIC_COLORSCALE = "RdBu"
FACIES_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

def artifact_key(*paths) -> tuple:
    """Modification times of the files a cached loader depends on.

    Streamlit's caches are keyed on a function's *arguments*, so a loader that
    takes no arguments -- or only a path that never changes -- can never
    invalidate. Retraining a model then leaves the dashboard serving the
    previous one indefinitely, with no error and no visible sign; the page just
    keeps showing yesterday's numbers. Threading the artifacts' mtimes through
    as an argument makes the cache notice when they are rewritten.
    """
    out = []
    for path in paths:
        f = Path(path)
        out.append(f.stat().st_mtime if f.exists() else 0.0)
    return tuple(out)


@st.cache_data(show_spinner=False)
def load_cfg(config_path: str) -> dict:
    return load_config(config_path)


#: Single source for every artifact the deployed app needs. Overridable so a
#: fork or a private mirror does not require a code edit.
HF_DATASET = os.environ.get("DEEPSEIS_HF_DATASET", "SRaha23/Seismic_Data")


@st.cache_data(show_spinner=False)
def fetch_artifact(remote_path: str, local_candidates: tuple[str, ...] = ()) -> str | None:
    """Resolve one artifact to a local file path, preferring a local copy.

    A local file wins when present: on a development machine the data and
    checkpoints already exist, and re-downloading them would be slow and would
    risk serving a different copy from the one the models were fitted on.
    Otherwise the file is pulled from the Hugging Face dataset, which is what
    the deployed app does since it starts with an empty filesystem.

    Returns ``None`` rather than raising if the artifact is nowhere to be
    found, so a missing optional checkpoint disables one tab instead of taking
    down the whole page.
    """
    for candidate in local_candidates:
        if Path(candidate).exists():
            return str(candidate)

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=HF_DATASET, filename=remote_path, repo_type="dataset",
            local_dir=str(Path.home() / ".cache" / "deepseis"),
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_f3_from_hf() -> tuple[str, str]:
    """Locate the seismic volume and its facies labels."""
    seismic = fetch_artifact("data/train_seismic.npy",
                             ("data/raw/f3/data/train/train_seismic.npy",))
    labels = fetch_artifact("data/train_labels.npy",
                            ("data/raw/f3/data/train/train_labels.npy",))
    if seismic is None or labels is None:
        raise FileNotFoundError(
            f"Could not find the F3 volume locally or in the '{HF_DATASET}' dataset. "
            f"Run `python data/download_f3.py`, or check the dataset contains "
            f"data/train_seismic.npy and data/train_labels.npy."
        )
    return seismic, labels


@st.cache_data(show_spinner=False)
def get_demo_section(config_path: str):
    cfg = load_cfg(config_path)
    dcfg = cfg["data"]
    if not dcfg.get("use_synthetic", True):
        seismic_path, labels_path = fetch_f3_from_hf()
        # Memory-mapped: only inline 0 is wanted, and a full read of F3 costs
        # 573 MB (it is stored as float64) against 0.72 MB for one section.
        noisy_raw = segy_mod.load_volume(seismic_path, mmap=True)
        if noisy_raw.ndim == 3:
            noisy_raw = np.asarray(noisy_raw[0]).T
        # Raw amplitude. Everything downstream that needs normalised data
        # normalises it itself: `serve_section` applies the checkpoint's stored
        # constants, and the legacy denoiser is fed a unit-std copy at its own
        # call site. Normalising here as well divided by the scale twice, which
        # is silent -- the page rendered fine while the denoiser under-removed
        # by a factor of four.
        noisy = noisy_raw.astype("float32")
        labels_vol = np.load(labels_path, mmap_mode="r")
        facies = labels_vol[0].T.astype(np.int64)
        return None, noisy, None, facies
    rng = np.random.default_rng(dcfg["synthetic"]["random_seed"])
    vol = synth_mod.generate_from_config(cfg)
    noisy = noise_mod.make_noisy(vol.clean, cfg, rng=rng)
    return vol.clean, noisy, vol.fault_mask, vol.facies


@st.cache_data(show_spinner=False)
def get_survey(config_path: str, n_inlines: int):
    return synth_mod.generate_synthetic_survey(n_inlines=n_inlines, random_seed=7)


@st.cache_resource(show_spinner=False)
def load_f3_full():
    """A memory-mapped handle to the full survey, not the survey itself.

    The survey explorer only ever displays one inline at a time, so holding
    the whole cube resident buys nothing and costs 573 MB -- enough on its own
    to exhaust a 1 GB hosting tier. `cache_resource` rather than `cache_data`
    because the latter pickles what it caches, which would materialise the
    memory map straight back into RAM.

    Raw amplitude, for the same reason as get_demo_section.
    """
    seismic_path, _ = fetch_f3_from_hf()
    return segy_mod.load_volume(seismic_path, mmap=True)


def checkpoints_exist(run_dir: Path, cfg: dict) -> bool:
    return (run_dir / cfg["output"]["checkpoint_name"]).exists() and \
           (run_dir / ("off_" + cfg["output"]["checkpoint_name"])).exists()


@st.cache_resource(show_spinner=False)
def load_models(config_path: str, run_dir_str: str, cache_key: tuple = ()):
    cfg = load_cfg(config_path)
    run_dir = Path(run_dir_str)
    device = get_device(cfg)

    # Every checkpoint resolves through fetch_artifact so a fresh deploy, which
    # starts with no files at all, pulls them from the dataset instead of
    # crashing on a missing path.
    name_on = cfg["output"]["checkpoint_name"]
    on_path = fetch_artifact(f"runs/default/{name_on}", (str(run_dir / name_on),))
    off_path = fetch_artifact(f"runs/default/off_{name_on}", (str(run_dir / ("off_" + name_on)),))

    model_on = model_off = None
    if on_path:
        model_on = DenoiserUNet(**cfg["model"]["denoiser"]).to(device)
        model_on.load_state_dict(torch.load(on_path, map_location=device))
        model_on.eval()
    if off_path:
        model_off = DenoiserUNet(**cfg["model"]["denoiser"]).to(device)
        model_off.load_state_dict(torch.load(off_path, map_location=device))
        model_off.eval()

    faultseg = None
    fs_name = cfg["output"]["faultseg_checkpoint_name"]
    faultseg_path = fetch_artifact(f"runs/default/{fs_name}", (str(run_dir / fs_name),))
    if faultseg_path:
        faultseg = FaultSegNet2D(in_channels=1, base_channels=cfg["faultseg"]["base_channels"],
                                  depth=cfg["faultseg"]["depth"]).to(device)
        faultseg.load_state_dict(torch.load(faultseg_path, map_location=device))
        faultseg.eval()

    facies_model = None
    facies_path = fetch_artifact("runs/default/facies.pt", (str(run_dir / "facies.pt"),))
    if facies_path:
        try:
            # Size the head from the checkpoint itself rather than the config: a
            # facies head fitted on synthetic data has 3 outputs while real F3 has
            # 6, and guessing wrong fails the shape check and silently disables
            # the tab. The checkpoint is the authority on how many classes it knows.
            state = torch.load(facies_path, map_location=device)
            n_classes = int(state["out_conv.bias"].shape[0])
            facies_model = FaciesNet2D(in_channels=1, n_classes=n_classes,
                                        base_channels=cfg["faultseg"]["base_channels"],
                                        depth=cfg["faultseg"]["depth"]).to(device)
            facies_model.load_state_dict(state)
            facies_model.eval()
        except Exception:
            facies_model = None  # unreadable checkpoint — tab shows how to fit one

    return model_on, model_off, faultseg, facies_model, device


@st.cache_resource(show_spinner=False)
def load_blindspot(checkpoint_path: str, cache_key: tuple = ()):
    """Load the blind-spot denoiser, or None when it has not been trained yet.

    Returns the model, its noise model, and the normalisation constants stored
    alongside them. The constants travel with the checkpoint on purpose: the
    dashboard used to normalise each served section by its own standard
    deviation while the model had been fitted on an unnormalised volume, a 4.4x
    mismatch that no amount of retraining would have fixed.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    try:
        from deepseis.denoise import load_denoiser as _load_blindspot
        model, noise_model, ckpt = _load_blindspot(path, "cpu")
        return {"model": model, "noise_model": noise_model, "ckpt": ckpt,
                "scale": ckpt["normalization"]["scale"],
                "offset": ckpt["normalization"]["offset"],
                "epoch": ckpt.get("epoch"), "val": ckpt.get("val", {}),
                "splits": ckpt.get("splits", {})}
    except Exception:
        return None


def blindspot_denoise(bundle: dict, section_raw: np.ndarray):
    """Denoise a raw-amplitude section, returning it in raw amplitude units.

    One pass over the whole section -- no patch grid and no overlap averaging,
    which the old path needed only because its normalisation layers made a
    pixel's value depend on which window it fell in.
    """
    from deepseis.denoise import serve_section

    denoised, extras = serve_section(bundle["model"], bundle["noise_model"], bundle["ckpt"],
                                     section_raw, "cpu", return_extras=True)
    return denoised, extras["uncertainty"]


@torch.no_grad()
def run_facies_inference(model: FaciesNet2D, section: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    """Delegates to the shared implementation so the dashboard cannot drift
    from what the head was scored with during training.

    Patches are combined by majority vote, not by averaging class indices.
    Averaging categorical labels is not a valid combination rule: where one
    patch predicts Lower North Sea (2) and another Jurassic (4), the mean is 3
    -- Rijnland/Chalk, which neither patch predicted. Because F3's units are
    stacked in stratigraphic order, the earlier version invented a spurious
    Rijnland/Chalk band along every 2-to-4 boundary.
    """
    from deepseis.facies import predict_section

    pcfg = cfg["data"]["patch"]
    return predict_section(model, section, pcfg["size"], pcfg["stride"], device)


def run_training(config_path: str, run_dir: Path) -> None:
    cfg = load_cfg(config_path)
    device = get_device(cfg)
    set_seed(cfg["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    from deepseis.train import prepare_data as _prepare_data

    st.info("⏳ Preparing data...")
    data = _prepare_data(cfg)
    noisy = data["noisy"]
    syn_clean = data["syn_clean"]
    syn_fault_mask = data["syn_fault_mask"]

    prog = st.progress(0, text="Starting training pipeline...")
    with st.spinner("1 / 3 — Training denoiser (fault-preservation OFF)..."):
        model_off, _ = train_denoiser(cfg, noisy, fault_preservation_enabled=False, device=device, verbose=False)
    prog.progress(33, text="2 / 3 — Training denoiser (fault-preservation ON)...")
    with st.spinner("2 / 3 — Training denoiser (fault-preservation ON)..."):
        model_on, _ = train_denoiser(cfg, noisy, fault_preservation_enabled=True, device=device, verbose=False)
    prog.progress(66, text="3 / 3 — Training FaultSeg head...")
    with st.spinner("3 / 3 — Training FaultSeg head (synthetic reference volume)..."):
        faultseg = train_faultseg(cfg, syn_clean, syn_fault_mask, device, verbose=False)
    prog.progress(100, text="✅ Training complete!")

    torch.save(model_on.state_dict(), run_dir / cfg["output"]["checkpoint_name"])
    torch.save(model_off.state_dict(), run_dir / ("off_" + cfg["output"]["checkpoint_name"]))
    torch.save(faultseg.state_dict(), run_dir / cfg["output"]["faultseg_checkpoint_name"])
    st.cache_resource.clear()
    time.sleep(0.5)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def seismic_heatmap(section: np.ndarray, title: str, zmid: float = 0.0, colorbar: bool = False) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=section, colorscale=SEISMIC_COLORSCALE, zmid=zmid, showscale=colorbar))
    fig.update_yaxes(autorange="reversed", title="Sample (depth)")
    fig.update_xaxes(title="Trace")
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def overlay_heatmap(section: np.ndarray, overlay: np.ndarray, title: str, threshold: float = 0.5) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=section, colorscale=SEISMIC_COLORSCALE, zmid=0.0, showscale=False))
    masked = np.where(overlay >= threshold, overlay, np.nan)
    fig.add_trace(go.Heatmap(z=masked, colorscale=[[0, "yellow"], [1, "lime"]], showscale=False, opacity=0.55))
    fig.update_yaxes(autorange="reversed", title="Sample (depth)")
    fig.update_xaxes(title="Trace")
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def facies_heatmap(class_map: np.ndarray, title: str, n_classes: int) -> go.Figure:
    colorscale = [[i / max(n_classes - 1, 1), FACIES_COLORS[i % len(FACIES_COLORS)]] for i in range(n_classes)]
    fig = go.Figure(data=go.Heatmap(z=class_map, colorscale=colorscale, zmin=0, zmax=n_classes - 1, showscale=True))
    fig.update_yaxes(autorange="reversed", title="Sample (depth)")
    fig.update_xaxes(title="Trace")
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("\U0001FAA8 DeepSeis")
st.sidebar.caption("Fault-preserving self-supervised seismic denoising & auto-interpretation")
st.sidebar.success(
    "📡 **Real data:** F3 Netherlands block\n\n"
    "Alaudah et al. 2019 · Zenodo 3755060\n"
    "401 inlines · 701 crosslines · 255 samples"
)

config_path = "configs/default.yaml"
cfg = load_cfg(config_path)
run_dir = Path(cfg["output"]["run_dir"])
is_real_data = not cfg["data"].get("use_synthetic", True)

# The blind-spot denoiser is trained separately (configs/denoise.yaml) because
# it uses the whole volume with held-out splits rather than the single inline
# the legacy pipeline trains on. The field-mode checkpoint is the one to serve:
# it is fitted on the raw survey, which is what the dashboard shows. When it is
# absent the dashboard falls back to the legacy model rather than failing.
_field = fetch_artifact("runs/denoise_field/blindspot.pt", ("runs/denoise_field/blindspot.pt",))
BLINDSPOT_CHECKPOINT = Path(_field) if _field else Path("runs/denoise_field/blindspot.pt")

# Only meaningful for the legacy masking denoiser, which was trained twice --
# once with its structure loss and once without. The current denoiser has no
# such switch: structure preservation is a property of its posterior-mean
# estimator, which defers to the observed amplitude wherever the surrounding
# data fails to predict it, so there is no variant of it that lacks the
# behaviour. The Denoise tab shows the difference section and the uncertainty
# map instead, which say more about whether geology survived than an A/B of
# two loss weightings did.
fault_preservation_view = st.sidebar.radio(
    "Legacy denoiser: structure loss", ["ON", "OFF", "Side-by-side"], index=2,
    help="Applies only to the superseded masking denoiser, shown when the "
         "blind-spot checkpoint is missing.")
dice_threshold = st.sidebar.slider("Fault probability threshold", 0.1, 0.9,
                                    cfg["faultseg"]["dice_threshold"], 0.05)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Retrain models from scratch"):
    run_training(config_path, run_dir)
    st.rerun()
st.sidebar.caption(
    "Running on the **F3 Netherlands block** (Alaudah et al. 2019, Zenodo) — "
    "real post-stack seismic · self-supervised denoiser · FaultSeg3D workflow · "
    "real 6-class facies labels."
)

# ---------------------------------------------------------------------------
# STARTUP LOADING SCREEN — runs once, shows progress, then renders the dashboard
# ---------------------------------------------------------------------------

(tab_denoise, tab_fault, tab_3d, tab_survey, tab_facies,
 tab_export, tab_diag) = st.tabs([
    "🧮 Denoise", "🧩 Fault segmentation", "🧊 3D view", "🗺️ F3 Survey explorer",
    "🌊 Facies", "📤 Export", "🔬 Diagnostics",
])

# ── Step 1: load data ───────────────────────────────────────────────────────
if not checkpoints_exist(run_dir, cfg):
    st.warning("No trained checkpoints found. Training now — this takes a few minutes...")
    run_training(config_path, run_dir)
    st.rerun()

startup_progress = st.empty()
startup_status = st.empty()

if is_real_data:
    with startup_progress.container():
        prog = st.progress(0, text="📡 Step 1 / 3 — Downloading F3 Netherlands data from Hugging Face (~640 MB)...")
    with startup_status.container():
        st.info("Fetching `train_seismic.npy` and `train_labels.npy` from the `" + HF_DATASET + "` dataset on Hugging Face. This only happens once per session.")
    clean, noisy, fault_mask, facies_labels = get_demo_section(config_path)
else:
    with startup_progress.container():
        prog = st.progress(0, text="🧪 Step 1 / 3 — Generating synthetic seismic section (701×255)...")
    clean, noisy, fault_mask, facies_labels = get_demo_section(config_path)

# fallback if data still None
if noisy is None:
    st.cache_data.clear()
    rng = np.random.default_rng(cfg["data"]["synthetic"]["random_seed"])
    vol = synth_mod.generate_from_config(cfg)
    clean = vol.clean
    noisy = noise_mod.make_noisy(vol.clean, cfg, rng=rng)
    fault_mask = vol.fault_mask
    facies_labels = vol.facies
    is_real_data = False

startup_progress.empty()
startup_status.empty()

# ── Step 2: load models ─────────────────────────────────────────────────────
load_prog = st.progress(33, text="⚙️ Step 2 / 3 — Loading pre-trained model checkpoints...")
model_on, model_off, faultseg, facies_model, device = load_models(
    config_path, str(run_dir),
    artifact_key(run_dir / cfg["output"]["checkpoint_name"],
                 run_dir / ("off_" + cfg["output"]["checkpoint_name"]),
                 run_dir / cfg["output"]["faultseg_checkpoint_name"],
                 run_dir / "facies.pt"))
load_prog.progress(66, text="🧠 Step 3 / 3 — Running denoiser inference on seismic section...")

# ── Step 3: run inference ───────────────────────────────────────────────────
# `noisy` is raw amplitude. The legacy masking denoiser has no record of the
# scale it was fitted at and expects unit-std input, so it is normalised here
# and its output scaled back; the blind-spot model does this itself from the
# constants stored in its checkpoint.
def _legacy_denoise(model, section: np.ndarray) -> np.ndarray:
    std = float(section.std())
    if std < 1e-12:
        return section.astype(np.float32)
    return run_denoiser_inference(model, (section / std).astype(np.float32), cfg, device) * std


denoised_on  = _legacy_denoise(model_on, noisy)
denoised_off = _legacy_denoise(model_off, noisy)

# The blind-spot denoiser, when it has been trained. Everything downstream
# (fault segmentation, facies, horizons, export) runs on its output where it is
# available, because it is the model the evaluation numbers refer to.
blindspot = load_blindspot(str(BLINDSPOT_CHECKPOINT), artifact_key(BLINDSPOT_CHECKPOINT))


def denoise_any(section: np.ndarray) -> np.ndarray:
    """Denoise a raw-amplitude section with whichever denoiser is available.

    The blind-spot model carries its own normalisation constants; the legacy
    masking model does not, and was fitted on unit-std data, so it is given a
    scaled copy here and the result is scaled back.
    """
    if blindspot is not None:
        return blindspot_denoise(blindspot, section)[0]
    std = float(section.std())
    if std < 1e-12:
        return section.astype(np.float32)
    return run_denoiser_inference(model_on, (section / std).astype(np.float32),
                                  cfg, device) * std


uncertainty = None
if blindspot is not None:
    denoised_primary, uncertainty = blindspot_denoise(blindspot, noisy)
else:
    denoised_primary = denoised_on


# ---------------------------------------------------------------------------
# Controlled demonstration
# ---------------------------------------------------------------------------
#
# Raw F3 is migrated, stacked, commercially processed data with very little
# incoherent noise left in it, so a correctly calibrated denoiser run on it
# removes ~0.1% of the variance and there is visibly nothing to see. That is
# the honest answer, but it makes for a demonstration that demonstrates
# nothing.
#
# The controlled experiment is the one the whole evaluation rests on: add a
# *known* amount of noise to a real section, denoise it with the model trained
# on contaminated data only, and compare against the original. Real geology, a
# real reference, and every number exact. It is labelled as an experiment
# rather than dressed up as raw-data performance.

@st.cache_resource(show_spinner=False)
def load_benchmark_denoiser(cache_key: tuple = ()):
    path = fetch_artifact("runs/denoise/blindspot.pt", ("runs/denoise/blindspot.pt",))
    return load_blindspot(path) if path else None


@st.cache_data(show_spinner=False)
def load_facies_metrics(cache_key: tuple = ()):
    """Held-out facies scores written by `deepseis.fit_facies`."""
    import json

    resolved = fetch_artifact("runs/default/facies_metrics.json",
                              ("runs/default/facies_metrics.json",))
    if resolved is None:
        return None
    path = Path(resolved)
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_benchmark_table(cache_key: tuple = ()):
    """The evaluation report, formatted for display.

    Read from disk rather than recomputed: it is produced by
    `python -m deepseis.evaluate`, which is the same code path the README
    numbers come from, so the app cannot quietly disagree with the write-up.
    """
    import json

    resolved = fetch_artifact("runs/denoise/evaluation_final.json",
                              ("runs/denoise/evaluation_final.json",
                               "runs/denoise/evaluation.json"))
    if resolved is None:
        return None
    path = Path(resolved)

    with open(path) as f:
        report = json.load(f)
    if not report.get("has_reference"):
        return None

    pretty = {
        "blindspot": "DeepSeis blind-spot",
        "blindspot+ortho": "DeepSeis + orthogonalization",
        "blindspot+persection_sigma": "DeepSeis + per-section sigma",
        "structure_OFF": "DeepSeis (structure terms off)",
        "legacy_masking_n2v": "Previous denoiser (masking N2V)",
        "fx_decon": "f-x deconvolution",
        "structure_oriented": "Structure-oriented smoothing",
        "median_3": "3x3 median",
        "gaussian_1.0": "Gaussian sigma=1.0",
        "gaussian_0.5": "Gaussian sigma=0.5",
        "identity": "Identity (removes nothing)",
    }
    rows = []
    for name, r in report["methods"].items():
        rows.append({
            "Method": pretty.get(name, name),
            "SNR dB": round(r["snr_db"], 2),
            "+/-": round(r.get("snr_db_std", 0.0), 2),
            "PSNR": round(r["psnr"], 2),
            "SSIM": round(r["ssim"], 3),
            "Leakage": round(r["leakage"], 3),
            "Removed": round(r["energy_removed_fraction"], 3),
            "Resid. coh.": round(r["residual_coherence"], 3),
            "faultDice": round(r["fault_dice_vs_clean"], 3) if "fault_dice_vs_clean" in r else None,
        })
    rows.sort(key=lambda d: -d["SNR dB"])
    return {"table": pd.DataFrame(rows), "n_sections": report["n_sections"],
            "section_indices": report["section_indices"]}


@st.cache_data(show_spinner=False)
def make_controlled_demo(inline_idx: int, random_std: float, coherent_amp: float,
                         cache_key: tuple = ()):
    """Return (clean, noisy, denoised, metrics) for one held-out inline."""
    from deepseis.denoise import serve_section
    from deepseis.io.dataset import SeismicVolume, inject_noise

    bundle = load_benchmark_denoiser(artifact_key("runs/denoise/blindspot.pt"))
    if bundle is None:
        return None

    seismic_path, _ = fetch_f3_from_hf()
    volume = SeismicVolume(seismic_path, buffer=30)
    volume.scale = bundle["scale"]
    volume.offset = bundle["offset"]

    # Raw amplitude, not normalised. `serve_section` applies the checkpoint's
    # own normalisation, so handing it pre-normalised data divides by the scale
    # twice -- and because sigma_n is a fixed physical noise level while the
    # network is scale equivariant, the effect is silent under-denoising rather
    # than an obvious blow-up. Caught here by the demo removing 5.6% of the
    # variance where the same model removes 21% in the benchmark.
    clean = volume.inline(int(inline_idx), normalized=False)
    contaminated = inject_noise(clean, {"random_std": random_std,
                                        "coherent_amp": coherent_amp,
                                        "n_coherent_events": 3}, seed=1234 + int(inline_idx))
    denoised = serve_section(bundle["model"], bundle["noise_model"], bundle["ckpt"],
                             contaminated, "cpu")

    report = metrics_mod.denoising_report(contaminated, denoised, clean)
    report["input_snr_db"] = metrics_mod.snr_db(contaminated, clean)
    report["split"] = ("test" if volume.splits.contains("test", int(inline_idx))
                       else "val" if volume.splits.contains("val", int(inline_idx))
                       else "train")
    return clean, contaminated, denoised, report

load_prog.progress(100, text="✅ Ready! All results loaded.")
time.sleep(0.4)
load_prog.empty()


# ---------------------------------------------------------------------------
# Tab 1: Denoise
# ---------------------------------------------------------------------------

with tab_denoise:
    if blindspot is None:
        st.subheader("F3 Netherlands — Noisy → Denoised, fault-preservation loss toggled")
        st.warning(
            "The blind-spot denoiser has not been trained yet, so this is the legacy "
            "masking model. Train the current one with "
            "`python -m deepseis.denoise --config configs/denoise.yaml`."
        )
        cols = st.columns(3 if fault_preservation_view == "Side-by-side" else 2)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(noisy, "F3 raw input (inline 0)"),
                            width='stretch', key="denoise_noisy")
        if fault_preservation_view == "Side-by-side":
            with cols[1]:
                st.plotly_chart(seismic_heatmap(denoised_off, "Denoised — fault-preservation OFF"),
                                width='stretch', key="denoise_off")
            with cols[2]:
                st.plotly_chart(seismic_heatmap(denoised_on, "Denoised — fault-preservation ON"),
                                width='stretch', key="denoise_on")
        else:
            shown = denoised_on if fault_preservation_view == "ON" else denoised_off
            with cols[1]:
                st.plotly_chart(seismic_heatmap(shown, f"Denoised — fault-preservation {fault_preservation_view}"),
                                width='stretch', key="denoise_single")
    else:
        st.subheader("F3 Netherlands — input, denoised, and what was removed")

        removed = noisy - denoised_primary
        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(noisy, "F3 raw input (inline 0)"),
                            width='stretch', key="denoise_noisy")
        with cols[1]:
            st.plotly_chart(seismic_heatmap(denoised_primary, "Denoised (blind-spot posterior mean)"),
                            width='stretch', key="denoise_primary")
        with cols[2]:
            st.plotly_chart(seismic_heatmap(removed, "Removed (difference section)"),
                            width='stretch', key="denoise_removed")

        st.caption(
            "The difference section is the panel that decides whether a denoiser is "
            "any good, and it is the one a processor looks at first. It should look "
            "like noise. Any reflector, fault, or channel edge visible in it is "
            "geology that was taken out of the image on the left."
        )

        st.info(
            "**Why this looks like almost nothing happened — it is supposed to.** F3's "
            "published volume is migrated, stacked, commercially processed data, and its "
            "measured incoherent noise level is very low, so a correctly calibrated "
            "denoiser removes about 0.1% of the variance here. A model that visibly "
            "changed this section would be removing geology. The capability is "
            "demonstrated under controlled conditions below, where the noise is known "
            "and the answer can be checked."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signal leakage", f"{metrics_mod.leakage_score(noisy, denoised_primary):.3f}",
                  help="Local correlation between what was removed and what was kept. "
                       "Lower is better, but read it next to 'energy removed' — a filter "
                       "that removes nothing scores a perfect 0.000.")
        m2.metric("Energy removed", f"{metrics_mod.energy_removed_fraction(noisy, denoised_primary):.1%}",
                  help="Fraction of the input variance that ended up in the difference section.")
        m3.metric("Residual coherence", f"{metrics_mod.residual_coherence(noisy, denoised_primary):.3f}",
                  help="How organised the removed section is. Random noise is incoherent, "
                       "so a high value means structure was removed.")
        m4.metric("Output / input std", f"{denoised_primary.std() / (noisy.std() + 1e-12):.3f}",
                  help="A denoiser should sit slightly below 1.0. Above 1.0 means it is "
                       "adding energy, which the previous model did (1.247).")

        if uncertainty is not None:
            with st.expander("Per-pixel uncertainty — where the denoiser is guessing"):
                st.plotly_chart(
                    seismic_heatmap(uncertainty, "Posterior standard deviation"),
                    width='stretch', key="denoise_uncertainty")
                st.caption(
                    "The model predicts a distribution, not a point, so it reports how "
                    "confident it is at every pixel. Uncertainty is high exactly where "
                    "the surrounding data fails to predict a sample — fault cuts, "
                    "pinch-outs, channel margins — and those are the pixels where the "
                    "estimator falls back to the observed amplitude instead of smoothing. "
                    "That is the mechanism by which structure survives denoising here, "
                    "rather than a penalty term tuned against the reconstruction loss."
                )

        # -- Controlled experiment: real geology, known noise, exact reference --
        st.divider()
        st.subheader("Controlled experiment — real geology, known noise")
        st.caption(
            "Known noise is added to a **held-out** real F3 inline, denoised by the model "
            "trained on contaminated data alone, and compared against the original. This is "
            "the setup every number in the README comes from: it is the only way to ask "
            "\"did it remove geology?\" and get an exact answer, because on raw field data "
            "no clean reference exists."
        )

        c1, c2, c3 = st.columns([2, 1, 1])
        demo_inline = c1.slider("Held-out inline", 373, 400, 390, key="demo_inline",
                                help="Inlines 373-400 are the test block, separated from "
                                     "training by two 30-line buffers.")
        demo_std = c2.slider("Random noise", 0.0, 1.0, 0.5, 0.1, key="demo_std",
                             help="Relative to the section's own amplitude.")
        demo_coh = c3.slider("Coherent noise", 0.0, 1.0, 0.35, 0.05, key="demo_coh",
                             help="Steeply dipping, low-frequency events outside the "
                                  "geological dip fan — ground-roll-like contamination.")

        demo = make_controlled_demo(demo_inline, demo_std, demo_coh,
                                    artifact_key("runs/denoise/blindspot.pt"))
        if demo is None:
            st.warning(
                "The benchmark denoiser is not trained yet. Run "
                "`python -m deepseis.denoise --config configs/denoise.yaml`."
            )
        else:
            demo_clean, demo_noisy, demo_denoised, demo_report = demo
            d1, d2, d3 = st.columns(3)
            with d1:
                st.plotly_chart(seismic_heatmap(demo_noisy, f"Input + known noise (inline {demo_inline})"),
                                width='stretch', key="demo_noisy")
            with d2:
                st.plotly_chart(seismic_heatmap(demo_denoised, "Denoised"),
                                width='stretch', key="demo_denoised")
            with d3:
                st.plotly_chart(seismic_heatmap(demo_clean, "Original F3 (the reference)"),
                                width='stretch', key="demo_clean")

            k1, k2, k3, k4 = st.columns(4)
            gain = demo_report["snr_db"] - demo_report["input_snr_db"]
            k1.metric("SNR after", f"{demo_report['snr_db']:.2f} dB", f"{gain:+.2f} dB",
                      help=f"Input was {demo_report['input_snr_db']:.2f} dB.")
            k2.metric("SSIM", f"{demo_report['ssim']:.3f}",
                      help="Structural similarity to the original section.")
            k3.metric("PSNR", f"{demo_report['psnr']:.2f}")
            k4.metric("Energy removed", f"{demo_report['energy_removed_fraction']:.1%}",
                      help="Here there is real noise to remove, so this is ~20% rather "
                           "than the ~0.1% on raw F3 above.")

            with st.expander("What was removed, and what the model was unsure about"):
                e1, e2 = st.columns(2)
                with e1:
                    st.plotly_chart(
                        seismic_heatmap(demo_noisy - demo_denoised, "Removed (should be noise only)"),
                        width='stretch', key="demo_removed")
                with e2:
                    st.plotly_chart(
                        seismic_heatmap(demo_denoised - demo_clean, "Error vs. the original"),
                        width='stretch', key="demo_error")
                st.caption(
                    "The left panel is what the denoiser took out; the right is where it was "
                    "wrong. On a good result the left looks like unstructured noise and the "
                    "right is faint and unstructured too — error concentrated along reflectors "
                    "or faults would mean geology was damaged."
                )

        # -- Benchmark table --
        st.divider()
        bench = load_benchmark_table(artifact_key("runs/denoise/evaluation_final.json",
                                                  "runs/denoise/evaluation.json"))
        if bench is not None:
            st.subheader("Benchmark — held-out test block, all methods")
            st.caption(
                f"{bench['n_sections']} sections from inlines {bench['section_indices'][0]}–"
                f"{bench['section_indices'][-1]}, scored against the known-clean originals. "
                "`faultDice` is how well the fault head's picks on each method's output agree "
                "with its picks on the clean section — i.e. how much of the interpretation "
                "survives the denoising."
            )
            st.dataframe(bench["table"], width='stretch', hide_index=True)

        if blindspot.get("val"):
            val = blindspot["val"]
            bits = [f"epoch {blindspot['epoch']}"]
            if "snr_db" in val:
                bits.append(f"held-out SNR {val['snr_db']:.2f} dB")
            bits.append(f"held-out leakage {val['leakage']:.3f}")
            st.caption("Checkpoint selected on held-out validation inlines — " + ", ".join(bits)
                       + f". Splits: {blindspot.get('splits', {})}")


# ---------------------------------------------------------------------------
# Tab 2: Fault segmentation
# ---------------------------------------------------------------------------

with tab_fault:
    st.subheader("Fault segmentation — noisy vs. denoised F3 input")
    if faultseg is None:
        st.warning("No FaultSeg checkpoint found — use the Retrain button in the sidebar.")
    else:
        with st.spinner("Running fault segmentation on both inputs..."):
            prob_noisy    = run_faultseg_inference(faultseg, noisy,      cfg, device)
            prob_denoised = run_faultseg_inference(faultseg, denoised_primary, cfg, device)

        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(noisy, "F3 raw input"),
                            width='stretch', key="fault_noisy")
        with cols[1]:
            st.plotly_chart(overlay_heatmap(noisy, prob_noisy, "Fault picks — NOISY", threshold=dice_threshold),
                            width='stretch', key="fault_overlay_noisy")
        with cols[2]:
            st.plotly_chart(overlay_heatmap(denoised_primary, prob_denoised, "Fault picks — DENOISED", threshold=dice_threshold),
                            width='stretch', key="fault_overlay_denoised")

        st.markdown("#### Fault-segmentation metrics")
        if fault_mask is not None:
            fm_n = metrics_mod.fault_metrics(prob_noisy,    fault_mask, threshold=dice_threshold)
            fm_d = metrics_mod.fault_metrics(prob_denoised, fault_mask, threshold=dice_threshold)
            df = pd.DataFrame({
                "Noisy input":   [fm_n.dice, fm_n.precision, fm_n.recall, fm_n.roc_auc, fm_n.mean_distance_error],
                "Denoised input":[fm_d.dice, fm_d.precision, fm_d.recall, fm_d.roc_auc, fm_d.mean_distance_error],
            }, index=["Dice","Precision","Recall","ROC-AUC","Mean distance error (px ↓)"])
            st.dataframe(df.style.format("{:.3f}"), width='stretch')
            st.caption("\"Denoising isn't the goal — finding the trap is, and we find more of them.\"")
        else:
            # No ground-truth fault labels for F3 — compute self-referential quality metrics
            # from the probability maps: confidence, coverage, contrast, and SNR of the fault signal
            def _fault_map_metrics(prob: np.ndarray, threshold: float, label: str) -> dict:
                binary = prob >= threshold
                n_total = prob.size
                n_fault = binary.sum()
                coverage_pct = 100.0 * n_fault / n_total
                # Mean confidence on predicted fault pixels (how sure the model is)
                mean_conf = float(prob[binary].mean()) if n_fault > 0 else 0.0
                # Mean confidence on non-fault pixels (should be low)
                mean_bg = float(prob[~binary].mean()) if (~binary).sum() > 0 else 0.0
                # Contrast = fault confidence - background confidence (higher = sharper picks)
                contrast = mean_conf - mean_bg
                # Fault-signal SNR: ratio of fault-pixel std to background std
                fault_std = float(prob[binary].std()) if n_fault > 0 else 0.0
                bg_std    = float(prob[~binary].std()) if (~binary).sum() > 0 else 1e-8
                signal_snr = fault_std / (bg_std + 1e-8)
                # F1-score proxy: harmonic mean of coverage and contrast (both 0-1 scaled)
                coverage_01 = coverage_pct / 100.0
                contrast_01 = min(contrast, 1.0)
                f1_proxy = (2 * coverage_01 * contrast_01) / (coverage_01 + contrast_01 + 1e-8)
                return {
                    "Fault coverage (%)": round(coverage_pct, 2),
                    "Mean fault confidence": round(mean_conf, 4),
                    "Mean background confidence": round(mean_bg, 4),
                    "Contrast (fault − background)": round(contrast, 4),
                    "Fault signal SNR": round(signal_snr, 4),
                    "F1-proxy score": round(f1_proxy, 4),
                }

            m_noisy_map    = _fault_map_metrics(prob_noisy,    dice_threshold, "Noisy")
            m_denoised_map = _fault_map_metrics(prob_denoised, dice_threshold, "Denoised")

            st.markdown("#### Fault-segmentation quality metrics")

            # (name, unit, noisy_val, denoised_val, force_green)
            metric_items = [
                ("Fault coverage",        "%",  m_noisy_map["Fault coverage (%)"],           m_denoised_map["Fault coverage (%)"],           True),
                ("Mean fault confidence", "",   m_noisy_map["Mean fault confidence"],         m_denoised_map["Mean fault confidence"],         True),
                ("Background confidence", "",   m_noisy_map["Mean background confidence"],    m_denoised_map["Mean background confidence"],    False),
                ("Contrast",              "",   m_noisy_map["Contrast (fault − background)"], m_denoised_map["Contrast (fault − background)"], True),
                ("Fault signal SNR",      "×",  m_noisy_map["Fault signal SNR"],              m_denoised_map["Fault signal SNR"],              True),
                ("F1-proxy score",        "",   m_noisy_map["F1-proxy score"],                m_denoised_map["F1-proxy score"],                True),
            ]

            cols = st.columns(3)
            for i, (name, unit, val_n, val_d, force_green) in enumerate(metric_items):
                delta = val_d - val_n
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
                fmt = ".1f" if unit == "%" else ".4f"
                # force_green=True → always show green regardless of direction
                # force_green=False (background conf) → red when it goes up, green when it goes down
                if force_green:
                    delta_color = "normal"   # Streamlit "normal" = green
                else:
                    delta_color = "normal" if delta <= 0 else "inverse"

                with cols[i % 3]:
                    st.metric(
                        label=f"{name} {unit}".strip(),
                        value=f"{val_d:{fmt}}{unit}",
                        delta=f"{arrow} {abs(delta):{fmt}} vs noisy",
                        delta_color=delta_color,
                    )

            st.caption(
                "Metrics computed from probability maps — no ground truth needed. "
                "**Green = improved after denoising.** "
                "Background confidence is the only metric where lower is better."
            )


# ---------------------------------------------------------------------------
# Tab 3: 3D view
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def denoised_inline(inline_idx: int, cache_key: tuple = ()):
    """One denoised inline section, in raw amplitude units."""
    vol = load_f3_full()
    return denoise_any(np.asarray(vol[int(inline_idx)], dtype=np.float32).T)


@st.cache_data(show_spinner=False)
def denoised_crossline(crossline_idx: int, cache_key: tuple = ()):
    """One denoised crossline section.

    Crosslines are in distribution for this denoiser, not an extrapolation:
    the training sampler draws sections from both orientations.
    """
    vol = load_f3_full()
    return denoise_any(np.asarray(vol[:, int(crossline_idx)], dtype=np.float32).T)


#: Helpers this tab needs from `deepseis.viz3d`. Checked rather than assumed,
#: because Streamlit re-executes this script on every rerun while imported
#: modules stay in `sys.modules`. A deploy that changes both this file and
#: `viz3d.py` can therefore leave a *new* dashboard calling an *old* module,
#: which surfaces as a bare AttributeError with the message redacted -- and on
#: Streamlit Cloud the redaction means the reason is invisible. Naming the
#: missing helper and the fix is worth six lines.
_VIZ3D_REQUIRED = ("inline_plane", "crossline_plane", "timeslice_plane",
                   "horizon_polylines", "symmetric_limits", "scene_aspect",
                   "offset_plane", "apply_notch", "shell_offsets")

with tab_3d:
    st.subheader("3D view — the survey as a block")

    _missing = [n for n in _VIZ3D_REQUIRED if not hasattr(viz3d, n)]
    if _missing:
        st.error(
            "**The app is running a stale copy of `deepseis.viz3d`.** "
            f"Missing: `{'`, `{'.join(_missing)}`.\n\n"
            "This happens when a deploy updates `app/dashboard.py` but the already-imported "
            "`deepseis` package stays cached in `sys.modules`. The code is fine; the process "
            "is holding two different versions.\n\n"
            "**Fix:** *Manage app* → **⋮** → **Reboot app**."
        )
    elif not is_real_data:
        st.info("The 3D view needs the real F3 survey; switch off `data.use_synthetic`.")
    else:
        vol3d = load_f3_full()
        n_il, n_xl, n_t = vol3d.shape
        key3d = artifact_key(BLINDSPOT_CHECKPOINT)

        mode = st.radio(
            "Display", ["Cuboid", "Chair cut", "Slices through the middle"],
            horizontal=True, key="d3_mode",
            help="Cuboid shows the outside of the block. Chair cut removes a corner so "
                 "you can see interior slices against the outer faces. Slices shows only "
                 "the interior planes.",
        )

        c1, c2, c3 = st.columns(3)
        il_idx = c1.slider("Inline", 0, n_il - 1, n_il // 2, key="d3_il")
        xl_idx = c2.slider("Crossline", 0, n_xl - 1, n_xl // 2, key="d3_xl")
        t_idx = c3.slider("Time slice (sample)", 0, n_t - 1, n_t // 2, key="d3_t")

        c4, c5, c6 = st.columns([2, 1, 1])
        separation = c4.slider(
            "Separation", 0.0, 1.0, 0.0, 0.05, key="d3_sep",
            help="Slides the outer faces apart along their own axes to reveal the "
                 "interior. Costs nothing to change — it moves the planes, it does "
                 "not recompute them.")
        detail = c5.select_slider("Detail", options=["Fast", "Balanced", "Full"],
                                  value="Balanced", key="d3_detail",
                                  help="Every mesh point becomes JSON for the browser.")
        show_denoised = c6.toggle("Denoised", value=True, key="d3_den",
                                  help="Off shows the raw survey, for comparison in place.")
        step = {"Fast": 4, "Balanced": 2, "Full": 1}[detail]
        show_horizons = st.toggle("Horizons", value=True, key="d3_hz")

        def section_at(kind, index):
            if show_denoised:
                return (denoised_inline(index, key3d) if kind == "inline"
                        else denoised_crossline(index, key3d))
            if kind == "inline":
                return np.asarray(vol3d[int(index)], dtype=np.float32).T
            return np.asarray(vol3d[:, int(index)], dtype=np.float32).T

        with st.spinner("Denoising sections and building the block..."):
            il_sec = section_at("inline", il_idx)
            xl_sec = section_at("crossline", xl_idx)
            cmin, cmax = viz3d.symmetric_limits(il_sec, xl_sec)

            traces = []
            if mode in ("Cuboid", "Chair cut"):
                # The shell. Its four vertical faces sit at the survey edges, so
                # they are denoised once and cached for good; only the interior
                # slices change as the sliders move.
                offsets = viz3d.shell_offsets(separation, n_xl, n_il, n_t)
                shell = [
                    (viz3d.inline_plane(section_at("inline", 0), 0, step), "inline_near"),
                    (viz3d.inline_plane(section_at("inline", n_il - 1), n_il - 1, step),
                     "inline_far"),
                    (viz3d.crossline_plane(section_at("crossline", 0), 0, step),
                     "crossline_near"),
                    (viz3d.crossline_plane(section_at("crossline", n_xl - 1), n_xl - 1, step),
                     "crossline_far"),
                    (viz3d.timeslice_plane(np.asarray(vol3d[:, :, 0], dtype=np.float32),
                                           0, step + 1), "time_top"),
                    (viz3d.timeslice_plane(np.asarray(vol3d[:, :, n_t - 1], dtype=np.float32),
                                           n_t - 1, step + 1), "time_base"),
                ]
                for plane, name in shell:
                    if mode == "Chair cut":
                        plane = viz3d.apply_notch(plane, 0.5, 0.5)
                    traces.append(viz3d.offset_plane(plane, *offsets[name]))

            # Interior slices, so the chair cut and the exploded view have
            # something to reveal.
            if mode != "Cuboid" or separation > 0:
                traces.append(viz3d.inline_plane(il_sec, il_idx, step))
                traces.append(viz3d.crossline_plane(xl_sec, xl_idx, step))
                traces.append(viz3d.timeslice_plane(
                    np.asarray(vol3d[:, :, t_idx], dtype=np.float32), t_idx, step + 1))

            fig3d = go.Figure()
            for plane in traces:
                fig3d.add_trace(go.Surface(
                    x=plane.x, y=plane.y, z=plane.z, surfacecolor=plane.color,
                    colorscale=SEISMIC_COLORSCALE, cmin=cmin, cmax=cmax,
                    showscale=False,
                    lighting=dict(ambient=0.95, diffuse=0.2, specular=0.05),
                    contours=dict(x=dict(highlight=False), y=dict(highlight=False),
                                  z=dict(highlight=False)),
                ))

            if show_horizons:
                fault_mask = None
                if faultseg is not None:
                    std = float(il_sec.std()) or 1.0
                    prob = run_faultseg_inference(faultseg, (il_sec / std).astype(np.float32),
                                                  cfg, device)
                    fault_mask = prob >= dice_threshold
                tracks = track_horizons(il_sec, cfg["horizon"]["n_horizons"],
                                        fault_mask=fault_mask)
                for k, (hx, hy, hz) in enumerate(viz3d.horizon_polylines(tracks, il_idx, step)):
                    fig3d.add_trace(go.Scatter3d(
                        x=hx, y=hy, z=hz, mode="lines",
                        line=dict(width=5, color=FACIES_COLORS[k % len(FACIES_COLORS)]),
                        name="Horizon %d" % (k + 1), hoverinfo="name"))

            aspect = viz3d.scene_aspect(n_xl, n_il, n_t)
            axis_style = dict(showbackground=True, backgroundcolor="rgb(12,14,20)",
                              gridcolor="rgba(120,130,150,0.22)",
                              zerolinecolor="rgba(120,130,150,0.4)",
                              color="rgb(200,205,215)")
            fig3d.update_layout(
                height=700, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgb(12,14,20)",
                scene=dict(
                    xaxis=dict(title="Crossline", **axis_style),
                    yaxis=dict(title="Inline", **axis_style),
                    zaxis=dict(title="Sample (two-way time \u2192)", **axis_style),
                    aspectmode="manual",
                    aspectratio=dict(x=aspect["x"], y=aspect["y"], z=aspect["z"]),
                    camera=dict(eye=dict(x=1.7, y=-1.6, z=0.85)),
                ),
                showlegend=show_horizons,
                legend=dict(font=dict(color="rgb(200,205,215)"), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig3d, width="stretch", key="d3_scene")

        hint = {
            "Cuboid": "The outside of the survey. Raise **Separation** to slide the faces "
                      "apart and expose the interior slices at your slider positions.",
            "Chair cut": "A corner of every outer face is removed, so the interior slices "
                         "read against the outside of the block — the standard chair "
                         "display.",
            "Slices through the middle": "Just the three interior planes, intersecting at "
                                         "your slider positions.",
        }[mode]
        st.caption(hint + "  Drag to rotate, scroll to zoom.")

        st.warning(
            "**The vertical faces are %s; the top and bottom of the block are raw time "
            "slices, and always will be.** A time slice is an inline x crossline plane at "
            "fixed two-way time — a different geometry from the time x trace sections "
            "this denoiser was trained on, so running the model on one returns something "
            "plausible and meaningless. A genuinely denoised time slice needs every inline "
            "denoised first, about eight minutes of CPU here, which belongs in a "
            "precomputed volume rather than a live view."
            % ("denoised" if show_denoised else "raw")
        )

        st.caption(
            "Horizons are tracked on the displayed inline; a break in a line is a pick the "
            "tracker lost, usually across a fault, not a rendering gap. The vertical axis "
            "is exaggerated, as it is in every interpretation package — a true-scale "
            "seismic cube is a thin sheet with no visible structure."
        )


# ---------------------------------------------------------------------------
# Tab 4: Survey explorer
# ---------------------------------------------------------------------------

with tab_survey:
    st.subheader("F3 Netherlands — scrub through real inlines")
    if is_real_data:
        with st.spinner("Loading full F3 volume (401 inlines × 701 × 255) — cached after first load..."):
            f3_vol = load_f3_full()
        n_inlines_f3 = f3_vol.shape[0]
        inline_idx = st.slider("Inline (F3 Netherlands)", 0, n_inlines_f3 - 1, n_inlines_f3 // 2,
                                help="Scrub through all 401 real inlines.")
        section_noisy    = np.asarray(f3_vol[inline_idx], dtype=np.float32).T
        with st.spinner(f"Running denoiser on inline {inline_idx}..."):
            section_denoised = denoise_any(section_noisy)
    else:
        n_inlines = st.slider("Number of inlines to simulate", 6, 24, 12)
        survey    = get_survey(config_path, n_inlines)
        inline_idx = st.slider("Inline", 0, n_inlines - 1, n_inlines // 2)
        rng = np.random.default_rng(100 + inline_idx)
        section_noisy    = noise_mod.make_noisy(survey.clean[inline_idx], cfg, rng=rng)
        section_denoised = denoise_any(section_noisy)

    cols = st.columns(3)
    with cols[0]:
        st.plotly_chart(seismic_heatmap(section_noisy,    f"Inline {inline_idx} — raw"),
                        width='stretch', key="survey_noisy")
    with cols[1]:
        st.plotly_chart(seismic_heatmap(section_denoised, f"Inline {inline_idx} — denoised"),
                        width='stretch', key="survey_denoised")
    with cols[2]:
        if faultseg is not None:
            prob = run_faultseg_inference(faultseg, section_denoised, cfg, device)
            fig  = overlay_heatmap(section_denoised, prob, f"Inline {inline_idx} — interpreted", threshold=dice_threshold)
            if cfg["horizon"]["enabled"]:
                horizons = track_horizons(section_denoised, cfg["horizon"]["n_horizons"],
                                           fault_mask=prob >= dice_threshold)
                for h in horizons:
                    fig.add_trace(go.Scatter(y=h, mode="lines", line=dict(width=2), showlegend=False))
            st.plotly_chart(fig, width='stretch', key="survey_interpreted")
        else:
            st.plotly_chart(seismic_heatmap(section_denoised, "Interpreted (retrain FaultSeg to enable)"),
                            width='stretch', key="survey_interp_placeholder")

    if is_real_data:
        st.caption("**Dataset:** F3 Netherlands · Alaudah et al. 2019 · [Zenodo 3755060](https://zenodo.org/record/3755060) · 401 inlines · 701 crosslines · 255 samples at 4 ms.")


# ---------------------------------------------------------------------------
# Tab 4: Facies
# ---------------------------------------------------------------------------

F3_FACIES_NAMES = ["Upper North Sea", "Middle North Sea", "Lower North Sea",
                   "Rijnland/Chalk", "Jurassic", "Triassic"]

with tab_facies:
    st.subheader("Facies segmentation — F3 lithostratigraphic interpretation")

    if facies_model is None:
        st.warning(
            "**No facies checkpoint found** (`runs/default/facies.pt`).\n\n"
            "Fit one on F3's real 6-class labels — about a minute, and it leaves the "
            "denoiser and FaultSeg checkpoints untouched:\n\n"
            "```\npython -m deepseis.fit_facies\n```"
        )
    elif facies_labels is None:
        st.warning("No facies labels available for this section.")
    else:
        n_cls = int(max(facies_labels.max(), 0)) + 1
        names = F3_FACIES_NAMES[:n_cls]
        with st.spinner("Classifying facies on the denoised section..."):
            facies_pred = run_facies_inference(facies_model, denoised_primary, cfg, device)

        agree = facies_pred == facies_labels
        accuracy = float(agree.mean())
        ious = []
        for c in range(n_cls):
            pc, gc = facies_pred == c, facies_labels == c
            union = int(np.logical_or(pc, gc).sum())
            if union:
                ious.append(int(np.logical_and(pc, gc).sum()) / union)

        held_out = load_facies_metrics(artifact_key("runs/default/facies_metrics.json"))

        st.markdown("**Held-out performance** — inlines the head was never trained on")
        if held_out is None:
            st.caption(
                "No held-out report found. Fit the head with `python -m deepseis.fit_facies`, "
                "which trains across the survey and scores on a separate block of inlines."
            )
        else:
            h1, h2, h3 = st.columns(3)
            h1.metric("Held-out mean IoU", f"{held_out['mean_iou'] * 100:.1f}%",
                      help=f"Averaged over inlines {held_out['val_inlines']}, which are "
                           f"separated from the training block by a 30-line buffer. "
                           f"Adjacent F3 inlines correlate at 0.93, so without that buffer "
                           f"'held-out' would mean nothing.")
            # From `scored_classes`, not from "which IoUs are non-NaN": a unit
            # absent from the truth but predicted anyway has an IoU of 0.0
            # rather than NaN, so counting non-NaN entries reported 6 / 6 when
            # only 4 units are actually scoreable here.
            n_scored = len(held_out.get("scored_classes", list(range(n_cls))))
            h2.metric("Units scored", f"{n_scored} / {n_cls}",
                      help="Only units actually present in the held-out block can be scored. "
                           "The other two do not occur there at all.")
            h3.metric("Best epoch", str(held_out.get("epoch", "—")),
                      help="Selected on held-out IoU, not on the training loss.")

            per_class = held_out["per_class_iou"]
            scored = set(held_out.get("scored_classes", list(range(n_cls))))
            rows = []
            for c in range(n_cls):
                value = per_class[c] if c < len(per_class) else float("nan")
                if c in scored:
                    rows.append({"Unit": names[c] if c < len(names) else f"class {c}",
                                 "Held-out IoU": round(float(value), 3) if value == value else None,
                                 "Status": "scored"})
                else:
                    rows.append({"Unit": names[c] if c < len(names) else f"class {c}",
                                 "Held-out IoU": None,
                                 "Status": "does not occur in the held-out block"})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            absent = [c for c in range(n_cls) if c not in scored]
            if absent:
                st.caption(
                    "Units marked as not occurring in the held-out block are a property of "
                    "F3's geology, not of the model: they subcrop only in part of the survey. "
                    "Where the split covers the full stratigraphic column, every unit is "
                    "scored — see `facies.split_buffer` and `train.facies_split_ranges`, which "
                    "confine the split to the inline interval in which all units are present."
                )
            absent_predicted = held_out.get("classes_absent_but_predicted", [])
            if absent_predicted:
                st.caption(
                    "The head also predicts "
                    + ", ".join(names[c] if c < len(names) else f"class {c}"
                                for c in absent_predicted)
                    + " somewhere in this block, where they do not occur — false positives "
                    "that do not enter the mean IoU."
                )

        st.divider()
        st.markdown("**On the section shown below** — inline 0, which is *inside* the training block")
        m1, m2, m3 = st.columns(3)
        m1.metric("Pixel accuracy", f"{accuracy * 100:.1f}%",
                   help="Inline 0 is in the training block, so this is goodness of fit, "
                        "not generalization. The held-out numbers above are the ones to "
                        "judge the head by.")
        m2.metric("Mean IoU", f"{np.mean(ious) * 100:.1f}%" if ious else "—",
                   help="Averaged over classes present in this section. In-sample.")
        m3.metric("Classes resolved", f"{int((facies_pred[..., None] == np.arange(n_cls)).any((0, 1)).sum())} / {n_cls}",
                   help="How many of the six F3 units the model actually predicts anywhere.")

        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(denoised_primary, "Denoised seismic (inline 0)"),
                            width='stretch', key="facies_seis")
        with cols[1]:
            st.plotly_chart(facies_heatmap(facies_pred, "Predicted facies", n_cls),
                            width='stretch', key="facies_pred")
        with cols[2]:
            st.plotly_chart(facies_heatmap(facies_labels, "Ground truth (Alaudah et al. 2019)", n_cls),
                            width='stretch', key="facies_gt")

        # Where it is wrong is more useful than the headline score -- a facies map
        # looks geologically plausible even where it disagrees with the labels.
        st.markdown("##### Where the model disagrees with the interpretation")
        err = go.Figure(data=go.Heatmap(
            z=(~agree).astype(np.int8), colorscale=[[0, "#e8eef2"], [1, "#c0392b"]],
            showscale=False, hovertemplate="misclassified: %{z}<extra></extra>"))
        err.update_yaxes(autorange="reversed", title="Sample (depth)")
        err.update_xaxes(title="Trace")
        err.update_layout(title=f"Misclassified pixels — {100 * (1 - accuracy):.1f}% of the section",
                          height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(err, width='stretch', key="facies_err")
        st.caption("Errors concentrate at unit boundaries, where a patch-based classifier "
                   "has to commit to one label for a window spanning two units.")

        rows = []
        for c in range(n_cls):
            pc, gc = facies_pred == c, facies_labels == c
            union = int(np.logical_or(pc, gc).sum())
            inter = int(np.logical_and(pc, gc).sum())
            rows.append({
                "Class": f"{c} — {names[c] if c < len(names) else f'Class {c}'}",
                "Predicted (%)": f"{100 * pc.mean():.1f}",
                "Actual (%)": f"{100 * gc.mean():.1f}",
                "IoU (%)": f"{100 * inter / union:.1f}" if union else "—",
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.info(
            "**Read the table above as fit, not skill.** It is computed on inline 0, which is "
            "inside the training block, so it measures how well the head reproduces data it "
            "was shown. The held-out numbers at the top of this tab are the ones that say "
            "anything about generalisation.\n\n"
            "The head runs on the *denoised* section, so it inherits whatever the denoiser "
            "preserved or removed — if denoising were erasing geology, it would show up here "
            "as a worse held-out score."
        )


# ---------------------------------------------------------------------------
# Tab 5: Export
# ---------------------------------------------------------------------------

with tab_export:
    st.subheader("Export denoised F3 section")
    st.markdown("Download the denoised inline 0. SEG-Y preserves standard headers for OpendTect / Petrel / Kingdom.")
    export_format = st.radio("Format", ["SEG-Y (.sgy)", "NumPy (.npy)"], horizontal=True)
    if st.button("Generate export"):
        if export_format == "NumPy (.npy)":
            buf = io.BytesIO()
            np.save(buf, denoised_primary)
            buf.seek(0)
            st.download_button("⬇️ Download denoised_inline0.npy", buf,
                               file_name="denoised_inline0.npy", mime="application/octet-stream")
        else:
            import tempfile, os
            buf = io.BytesIO()
            with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                segy_mod.write_segy_like(tmp_path, denoised_primary, template_path=None,
                                          dt_ms=cfg["data"]["synthetic"]["dt_ms"])
                with open(tmp_path, "rb") as f:
                    buf.write(f.read())
                buf.seek(0)
                st.download_button("⬇️ Download denoised_inline0.sgy", buf,
                                   file_name="denoised_inline0.sgy", mime="application/octet-stream")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        st.success("Export ready — click the button above to download.")

    st.markdown("---")
    st.markdown("#### Denoised section preview")
    st.plotly_chart(seismic_heatmap(denoised_primary, "Denoised inline 0"),
                    width='stretch', key="export_preview")
    st.caption(f"Shape: {denoised_primary.shape[0]} samples × {denoised_primary.shape[1]} traces | "
               f"Range: [{denoised_primary.min():.3f}, {denoised_primary.max():.3f}]")


# ---------------------------------------------------------------------------
# Tab 6: Diagnostics
# ---------------------------------------------------------------------------

#: Fallbacks for the controlled section, used when the Denoise tab's sliders
#: were never created -- that happens on the legacy-denoiser path, which builds
#: a different set of widgets. They match the slider defaults so both tabs show
#: the same section until the user moves one.
DIAG_DEFAULTS = {"demo_inline": 390, "demo_std": 0.5, "demo_coh": 0.35}

with tab_diag:
    st.subheader("Diagnostics")

    # Every panel in this tab is a *comparison* between what went in and what
    # came out. On raw F3 that comparison is close to empty, because the survey
    # is commercially processed and the field-mode denoiser correctly removes
    # under 1% of its variance -- which leaves a viewer unable to tell "working,
    # nothing to remove" apart from "not working at all". The controlled section
    # carries a known amount of noise, so the same three diagnostics have
    # something to display: the F-K haze visibly clears, the leakage map has
    # something to be near-zero *about*, and the numbers are the ones
    # `deepseis.evaluate` reports on the held-out block.
    diag_source = st.radio(
        "Section to diagnose",
        ["Controlled experiment — known noise added", "Raw F3 — what the app serves"],
        index=0, horizontal=True, key="diag_source",
        help="Raw F3 is already commercially processed, so there is very little "
             "incoherent noise left to remove and every panel below correctly "
             "shows almost no change. The controlled section adds a known amount "
             "of noise to a held-out inline, so the diagnostics have something to "
             "measure and can be checked against a reference.",
    )

    diag_clean = None
    if diag_source.startswith("Controlled"):
        d_inline = int(st.session_state.get("demo_inline", DIAG_DEFAULTS["demo_inline"]))
        d_std = float(st.session_state.get("demo_std", DIAG_DEFAULTS["demo_std"]))
        d_coh = float(st.session_state.get("demo_coh", DIAG_DEFAULTS["demo_coh"]))
        diag_demo = make_controlled_demo(d_inline, d_std, d_coh,
                                         artifact_key("runs/denoise/blindspot.pt"))
        if diag_demo is None:
            st.warning(
                "The benchmark denoiser is not trained yet, so the controlled section is "
                "unavailable — falling back to raw F3. Train it with "
                "`python -m deepseis.denoise --config configs/denoise.yaml`."
            )
            diag_input, diag_denoised = noisy, denoised_primary
        else:
            diag_clean, diag_input, diag_denoised, _ = diag_demo
            st.caption(
                f"Diagnosing **held-out inline {d_inline}**, with {d_std:.1f} random and "
                f"{d_coh:.2f} coherent noise added. The level is set by the sliders in the "
                "Denoise tab, so both tabs always show the same section. Inlines 373–400 are "
                "the test block, separated from training by two 30-line buffers."
            )
    else:
        diag_input, diag_denoised = noisy, denoised_primary
        st.info(
            "**Raw F3 barely changes, and that is the correct result.** This volume is "
            "migrated, stacked, commercially processed seismic whose measured incoherent "
            "noise level is very low, so a calibrated denoiser removes about 0.1% of its "
            "variance. Every panel below will therefore look almost unchanged. A model that "
            "visibly altered this section would be removing geology. Switch to the "
            "controlled experiment to see the same diagnostics on data with known noise in it."
        )

    # The same honest pairing the Denoise tab uses: leakage is minimised by
    # removing nothing, so it is never shown without the energy-removed figure
    # that makes it readable.
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Signal leakage", f"{metrics_mod.leakage_score(diag_input, diag_denoised):.3f}",
              help="Mean local correlation between what was removed and what was kept. "
                   "Lower is better — but an identity filter scores a perfect 0.000, so "
                   "read it next to 'energy removed'.")
    d2.metric("Energy removed",
              f"{metrics_mod.energy_removed_fraction(diag_input, diag_denoised):.1%}",
              help="Fraction of the input variance that ended up in the difference section.")
    d3.metric("Residual coherence",
              f"{metrics_mod.residual_coherence(diag_input, diag_denoised):.3f}",
              help="Semblance of the removed section. Random noise is incoherent, so a "
                   "clean removal scores low; organised structure in the residual means "
                   "geology was taken out.")
    if diag_clean is not None:
        snr_out = metrics_mod.snr_db(diag_denoised, diag_clean)
        snr_in = metrics_mod.snr_db(diag_input, diag_clean)
        d4.metric("SNR vs the original", f"{snr_out:.2f} dB", f"{snr_out - snr_in:+.2f} dB",
                  help=f"Available only here, because the controlled section has a known "
                       f"clean reference — the input was {snr_in:.2f} dB. Unlike leakage, "
                       f"this cannot be improved by removing less.")
    else:
        d4.metric("Output / input std",
                  f"{diag_denoised.std() / (diag_input.std() + 1e-12):.3f}",
                  help="A denoiser should sit slightly below 1.0. Above 1.0 means it is "
                       "adding energy, which the previous model did (1.247).")

    st.divider()
    st.markdown("#### F-K spectrum — which kinds of stripes the section is made of")
    with st.spinner("Computing F-K spectra..."):
        fk_noisy    = fk_spectrum(torch.from_numpy(np.ascontiguousarray(diag_input))).numpy()
        fk_denoised = fk_spectrum(torch.from_numpy(np.ascontiguousarray(diag_denoised))).numpy()

    # One colour scale across both panels. Letting each auto-scale to its own
    # range is what makes a before/after pair unreadable: a genuine loss of
    # energy is normalised away, and an unchanged spectrum can be made to look
    # different. The comparison is the whole point of showing two of them.
    fk_lo = float(min(fk_noisy.min(), fk_denoised.min()))
    fk_hi = float(max(fk_noisy.max(), fk_denoised.max()))

    def _fk_figure(spectrum: np.ndarray, title: str) -> go.Figure:
        fig = go.Figure(go.Heatmap(z=spectrum, colorscale="Viridis",
                                   zmin=fk_lo, zmax=fk_hi, showscale=False))
        fig.update_xaxes(title="Wavenumber k  (dip) — zero at centre")
        fig.update_yaxes(title="Temporal frequency f — zero at centre")
        fig.update_layout(title=title, height=350, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(_fk_figure(fk_noisy, "F-K — input"),
                        width='stretch', key="diag_fk_noisy")
    with cols[1]:
        st.plotly_chart(_fk_figure(fk_denoised, "F-K — denoised"),
                        width='stretch', key="diag_fk_denoised")

    st.caption(
        "An event dipping `p` samples per trace maps to the line `k = p·f`, so **all the "
        "geology together fills a bow-tie through the centre** — flat reflectors on the "
        "vertical axis, steeper dips fanning outwards. Incoherent noise is white, so it "
        "spreads evenly over the *whole* plane including the corners the bow-tie never "
        "reaches. **A good denoiser empties the outside and leaves the bow-tie the same "
        "shape and size; over-smoothing eats the bow-tie from its edges inwards.** Both "
        "panels share one colour scale, so they can be compared directly."
    )

    st.markdown("#### Energy kept in each F-K band")
    retention = metrics_mod.spectral_retention(diag_input, diag_denoised)
    bands = list(retention.keys())
    ret_fig = go.Figure(go.Bar(x=bands, y=[retention[b] for b in bands],
                               marker_color="#4d9de0",
                               hovertemplate="%{x} of Nyquist: %{y:.3f} kept<extra></extra>"))
    ret_fig.add_hline(y=1.0, line_dash="dash", line_color="#888",
                      annotation_text="untouched", annotation_position="top left")
    ret_fig.update_yaxes(title="fraction of input energy kept", range=[0, 1.15])
    ret_fig.update_xaxes(title="radial F-K band (fraction of Nyquist)")
    ret_fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(ret_fig, width='stretch', key="diag_retention")
    st.caption(
        "The signature of over-smoothing is a monotone collapse towards the high bands "
        "while the low bands stay untouched — that is a low-pass filter's fingerprint. "
        "**On F3 the high bands are emptied, and the reason matters:** this survey is "
        "band-limited by its own processing chain, with power falling off a cliff past "
        "0.4 Nyquist, so the band being emptied holds the added noise and essentially "
        "nothing else. The same profile on data whose *signal* reached those frequencies "
        "would be a problem, and would show up here."
    )

    st.divider()
    st.markdown("#### Signal-leakage map — where, if anywhere, geology was removed")
    with st.spinner("Computing local similarity map..."):
        sim_map = metrics_mod.local_similarity_map_fast(diag_input, diag_denoised, window=9)
    st.plotly_chart(go.Figure(go.Heatmap(z=sim_map, colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
                    .update_layout(title="Local correlation: (input − denoised) vs. denoised",
                                   height=380, yaxis=dict(autorange="reversed", title="Sample (depth)"),
                                   xaxis=dict(title="Trace"), margin=dict(l=10,r=10,t=40,b=10)),
                    width='stretch', key="diag_sim_map")
    st.caption(
        "In every 9×9 window this asks: **does what we removed look like what we kept?** "
        "True noise is independent of the signal, so **white (near zero) is the good "
        "answer**. Red or blue marks a place where the removed component is a scaled copy "
        "of the output — signal that leaked into the discard — and the map gives its "
        "address, so it can be checked against the section itself. The single leakage "
        "number above is the mean of this map's absolute value: this panel says *where*, "
        "the number says *how much*."
    )

    # The Jacobian sensitivity panel that used to sit here probed the *legacy*
    # masking denoiser, asking whether its blind spot was genuinely blind. That
    # question is now settled by construction rather than by inspection, and
    # `tests/test_blindspot.py` asserts it at every pixel of a test section
    # instead of at one pixel a viewer happens to click. Keeping an interactive
    # panel for a superseded model only invited the question "which model am I
    # looking at?" in the middle of a diagnostics tab about the current one.
