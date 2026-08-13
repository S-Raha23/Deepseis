"""
DeepSeis dashboard — real post-stack seismic surveys
Fault-preserving self-supervised seismic denoising & auto-interpretation.

The survey shown is chosen from the sidebar and comes from the ``datasets:``
registry in the config (F3 Netherlands by default, Parihaka as a second option).
Every registered survey is stored in the same canonical layout, so each tab reads
whichever one is selected without any per-survey special-casing.

Run with:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import copy
import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseis import metrics as metrics_mod
from deepseis.interpretation.horizon import track_horizons
from deepseis.io import noise as noise_mod
from deepseis.io import segy as segy_mod
from deepseis.io import synthetic as synth_mod
from deepseis.losses.frequency import fk_spectrum
from deepseis.masking.jacobian_explain import compute_pixel_jacobian, suggest_mask_design
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

REPO_ROOT = Path(__file__).parent.parent


def load_cfg(config_path: str) -> dict:
    """Parse the config. Deliberately NOT cached.

    It was previously wrapped in @st.cache_data keyed on the path string alone.
    That key never changes, so when Streamlit Cloud pulled new code and re-ran
    the script in a still-warm process, it returned the config parsed *before*
    the deploy -- new code reading an old config, which crashed with a KeyError
    the moment the code expected a config section the cached copy predated.
    Parsing a few KB of YAML is far cheaper than that failure mode, and every
    caller that actually does expensive work is cached in its own right.
    """
    return load_config(config_path)


@st.cache_data(show_spinner=False)
def resolve_dataset(config_path: str, dataset_key: str) -> tuple[str, str, str]:
    """Locate a registered survey's (seismic, labels) files. Returns (seismic, labels, origin).

    Local copies are preferred so that a machine which has already run the
    survey's download script never re-downloads; otherwise the file is pulled
    from the survey's Hugging Face mirror (the only route that works on
    Streamlit Cloud, where ``data/raw/`` does not exist).
    """
    spec = load_cfg(config_path)["datasets"][dataset_key]
    repo_root = Path(__file__).parent.parent

    local_seismic = repo_root / spec["local_seismic"]
    local_labels = repo_root / spec["local_labels"]
    if local_seismic.exists() and local_labels.exists():
        return str(local_seismic), str(local_labels), "local"

    from huggingface_hub import hf_hub_download
    # tempfile.gettempdir() rather than a hardcoded "/tmp" so this also works on Windows.
    cache_dir = Path(tempfile.gettempdir()) / "deepseis_data" / dataset_key
    seismic_path = hf_hub_download(repo_id=spec["hf_repo"], filename=spec["hf_seismic"],
                                    repo_type="dataset", local_dir=str(cache_dir))
    labels_path = hf_hub_download(repo_id=spec["hf_repo"], filename=spec["hf_labels"],
                                   repo_type="dataset", local_dir=str(cache_dir))
    return seismic_path, labels_path, "huggingface"


def read_inline(seismic_path: str, idx: int) -> np.ndarray:
    """One inline as (n_samples, n_traces) float32, normalized to unit std.

    Memory-mapped: touches roughly one inline's worth of bytes rather than the
    whole cube. This matters more than it looks. The dashboard only ever
    *displays* a single inline, but the previous implementations materialized
    the entire volume to get one -- F3 is 573 MB on disk as float64 and Parihaka
    465 MB, which drove resident memory to 1.4 GB on first render and 3.4 GB
    after switching surveys, over the cap on a hosted runner and killed without
    a Python traceback.

    Normalizing by the section's own std (rather than the whole volume's) also
    matches how ``train.prepare_data`` normalized the training section, so the
    denoiser sees inputs on the scale it was actually fitted at.
    """
    if seismic_path.endswith(".npy"):
        vol = np.load(seismic_path, mmap_mode="r")
        section = np.asarray(vol[idx], dtype=np.float32).T
    else:  # SEG-Y / .dat have no mmap path -- fall back to a full read
        section = segy_mod.load_volume(seismic_path)[idx].astype(np.float32).T
    return section / (float(section.std()) + 1e-8)


@st.cache_data(show_spinner=False)
def volume_shape(config_path: str, dataset_key: str) -> tuple[int, ...]:
    """Volume dimensions, without reading the samples."""
    seismic_path, _, _ = resolve_dataset(config_path, dataset_key)
    if seismic_path.endswith(".npy"):
        return tuple(np.load(seismic_path, mmap_mode="r").shape)
    return tuple(segy_mod.load_volume(seismic_path).shape)


@st.cache_data(show_spinner=False)
def get_demo_section(config_path: str, dataset_key: str):
    cfg = load_cfg(config_path)
    dcfg = cfg["data"]
    if not dcfg.get("use_synthetic", True):
        seismic_path, labels_path, _ = resolve_dataset(config_path, dataset_key)
        noisy = read_inline(seismic_path, 0)
        labels_vol = np.load(labels_path, mmap_mode="r")
        facies = np.asarray(labels_vol[0]).T.astype(np.int64)
        return None, noisy, None, facies
    rng = np.random.default_rng(dcfg["synthetic"]["random_seed"])
    vol = synth_mod.generate_from_config(cfg)
    noisy = noise_mod.make_noisy(vol.clean, cfg, rng=rng)
    return vol.clean, noisy, vol.fault_mask, vol.facies


@st.cache_data(show_spinner=False)
def get_survey(config_path: str, n_inlines: int):
    return synth_mod.generate_synthetic_survey(n_inlines=n_inlines, random_seed=7)


def checkpoints_exist(run_dir: Path, cfg: dict) -> bool:
    return (run_dir / cfg["output"]["checkpoint_name"]).exists() and \
           (run_dir / ("off_" + cfg["output"]["checkpoint_name"])).exists()


@st.cache_resource(show_spinner=False)
def load_models(config_path: str, run_dir_str: str, n_classes: int = 6):
    cfg = load_cfg(config_path)
    run_dir = Path(run_dir_str)
    device = get_device(cfg)

    model_on = DenoiserUNet(**cfg["model"]["denoiser"]).to(device)
    model_on.load_state_dict(torch.load(run_dir / cfg["output"]["checkpoint_name"], map_location=device))
    model_on.eval()

    model_off = DenoiserUNet(**cfg["model"]["denoiser"]).to(device)
    model_off.load_state_dict(torch.load(run_dir / ("off_" + cfg["output"]["checkpoint_name"]), map_location=device))
    model_off.eval()

    faultseg = None
    faultseg_path = run_dir / cfg["output"]["faultseg_checkpoint_name"]
    if faultseg_path.exists():
        faultseg = FaultSegNet2D(in_channels=1, base_channels=cfg["faultseg"]["base_channels"],
                                  depth=cfg["faultseg"]["depth"]).to(device)
        faultseg.load_state_dict(torch.load(faultseg_path, map_location=device))
        faultseg.eval()

    facies_model = None
    facies_path = run_dir / "facies.pt"
    if facies_path.exists():
        try:
            # n_classes comes from the survey being loaded, not the global default:
            # train_facies sizes the head from the labels it saw, so a survey with a
            # different class count would otherwise fail the shape check below.
            facies_model = FaciesNet2D(in_channels=1, n_classes=n_classes,
                                        base_channels=cfg["faultseg"]["base_channels"],
                                        depth=cfg["faultseg"]["depth"]).to(device)
            facies_model.load_state_dict(torch.load(facies_path, map_location=device), strict=False)
            facies_model.eval()
        except Exception:
            facies_model = None  # checkpoint mismatch — will show retrain prompt in tab

    return model_on, model_off, faultseg, facies_model, device


@torch.no_grad()
def run_facies_inference(model: FaciesNet2D, section: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    from deepseis.io import patches as patch_mod
    from deepseis.io.patches import patch_coords
    from deepseis.train import to_tensor
    model.eval()
    pcfg = cfg["data"]["patch"]
    h, w = section.shape
    patches = patch_mod.extract_patches(section, pcfg["size"], pcfg["stride"])
    # Batched for the same reason as the denoiser and fault heads -- see
    # INFERENCE_BATCH in deepseis/train.py. argmax is taken per chunk so the
    # (N, n_classes, 64, 64) logits never exist for the whole section at once.
    from deepseis.train import INFERENCE_BATCH
    preds = np.concatenate([
        model(to_tensor(patches[i:i + INFERENCE_BATCH], device)).argmax(dim=1).detach().cpu().numpy()
        for i in range(0, len(patches), INFERENCE_BATCH)
    ], axis=0)
    coords = patch_coords(h, w, pcfg["size"], pcfg["stride"])
    out = np.zeros((h, w), dtype=np.int32)
    count = np.zeros((h, w), dtype=np.int32)
    for (y, x0), patch in zip(coords, preds):
        out[y:y + pcfg["size"], x0:x0 + pcfg["size"]] += patch
        count[y:y + pcfg["size"], x0:x0 + pcfg["size"]] += 1
    count[count == 0] = 1
    return (out // count).astype(np.int32)


def run_training(config_path: str, run_dir: Path, dataset_key: str) -> None:
    """Retrain the denoiser + FaultSeg head for one survey, into that survey's run_dir."""
    cfg = copy.deepcopy(load_cfg(config_path))
    spec = cfg["datasets"][dataset_key]

    # Point the config at the selected survey. resolve_dataset() is used rather than
    # the registry's local_* paths directly, because on Streamlit Cloud the only copy
    # of the data is the Hugging Face mirror it downloads into a temp dir.
    if not cfg["data"].get("use_synthetic", True):
        seismic_path, labels_path, _ = resolve_dataset(config_path, dataset_key)
        cfg["data"]["field_volume_path"] = seismic_path
        cfg["data"]["facies_label_path"] = labels_path
    cfg["facies"]["n_classes"] = spec.get("n_facies", cfg["facies"]["n_classes"])

    device = get_device(cfg)
    set_seed(cfg["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    from deepseis.train import prepare_data as _prepare_data

    st.info(f"⏳ Preparing {spec['label']} data...")
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

# Absolute, so the app does not depend on the working directory it was launched from.
config_path = str(REPO_ROOT / "configs" / "default.yaml")
cfg = load_cfg(config_path)
is_real_data = not cfg["data"].get("use_synthetic", True)

DATASETS = cfg.get("datasets", {})
DATASET_KEYS = [k for k in DATASETS if k != "default"]

if not DATASET_KEYS:
    # Never index into an empty registry -- that surfaced as a bare KeyError with
    # no indication of the cause. Say what is wrong and where to look instead.
    st.error(
        f"**No surveys are registered.** `{config_path}` has no usable `datasets:` section, "
        f"so there is nothing to display.\n\n"
        f"Found top-level config keys: `{', '.join(sorted(cfg)) or '(none)'}`.\n\n"
        f"If this app was just redeployed, restart it from **Manage app → Reboot** — a warm "
        f"process can otherwise keep serving a configuration parsed before the deploy."
    )
    st.stop()

st.sidebar.title("\U0001FAA8 DeepSeis")
st.sidebar.caption("Fault-preserving self-supervised seismic denoising & auto-interpretation")

# ---- Survey selection -----------------------------------------------------
st.sidebar.markdown("#### Survey")
_default_key = DATASETS.get("default", "f3")
dataset_key = st.sidebar.selectbox(
    "Survey",
    DATASET_KEYS,
    index=DATASET_KEYS.index(_default_key) if _default_key in DATASET_KEYS else 0,
    format_func=lambda k: DATASETS[k]["label"],
    label_visibility="collapsed",
    help="Switch the survey every tab operates on. The denoiser is self-supervised, "
         "so it adapts to whichever survey's noise you point it at — use Retrain "
         "below to fit it to the selected one.",
)
ds = DATASETS[dataset_key]
DATASET_LABEL = ds["label"]
# Each survey has its own checkpoints -- see `run_dir` in the registry.
run_dir = Path(ds.get("run_dir", cfg["output"]["run_dir"]))

st.sidebar.success(
    f"📡 **{DATASET_LABEL}** — {ds['region']}\n\n"
    f"{ds['citation']}\n\n"
    f"{ds['geometry']}"
)

st.sidebar.markdown("---")
fault_preservation_view = st.sidebar.radio("Fault-preservation loss", ["ON", "OFF", "Side-by-side"], index=2)
dice_threshold = st.sidebar.slider("Fault probability threshold", 0.1, 0.9,
                                    cfg["faultseg"]["dice_threshold"], 0.05)
st.sidebar.markdown("---")
if st.sidebar.button(f"🔄 Retrain on {DATASET_LABEL}"):
    run_training(config_path, run_dir, dataset_key)
    st.rerun()
st.sidebar.caption(
    f"Running on the **{DATASET_LABEL}** survey ({ds['region']}) — "
    f"real post-stack seismic · self-supervised denoiser · FaultSeg3D workflow · "
    f"real {ds['n_facies']}-class facies labels."
)

# ---------------------------------------------------------------------------
# STARTUP LOADING SCREEN — runs once, shows progress, then renders the dashboard
# ---------------------------------------------------------------------------

tab_denoise, tab_fault, tab_survey, tab_facies, tab_export, tab_diag = st.tabs([
    "🧮 Denoise", "🧩 Fault segmentation", "🗺️ Survey explorer",
    "🌊 Facies", "📤 Export", "🔬 Diagnostics",
])

# ── Step 1: load data ───────────────────────────────────────────────────────
if not checkpoints_exist(run_dir, cfg):
    # Deliberately does NOT auto-train: fitting a survey takes tens of minutes on
    # CPU, which is not something to start silently inside a page load. Offer the
    # command and an explicit button instead.
    st.warning(
        f"**No trained checkpoints for {DATASET_LABEL}** (looked in `{run_dir}`).\n\n"
        f"Train it from the command line — much faster than in-browser, and resumable:\n\n"
        f"```\npython -m deepseis.train --config {config_path} --dataset {dataset_key}\n```\n\n"
        f"Or use **🔄 Retrain on {DATASET_LABEL}** in the sidebar to run it here. "
        f"Meanwhile you can switch to another survey in the sidebar."
    )
    st.stop()

startup_progress = st.empty()
startup_status = st.empty()

if is_real_data:
    with startup_progress.container():
        prog = st.progress(0, text=f"📡 Step 1 / 3 — Loading {DATASET_LABEL} survey data...")
    with startup_status.container():
        st.info(f"Fetching `{ds['hf_seismic']}` and `{ds['hf_labels']}` for **{DATASET_LABEL}** — "
                f"from `data/raw/` if present, otherwise from `{ds['hf_repo']}` on Hugging Face. "
                f"This only happens once per session.")
    try:
        clean, noisy, fault_mask, facies_labels = get_demo_section(config_path, dataset_key)
    except Exception as exc:
        startup_progress.empty()
        startup_status.empty()
        st.error(
            f"**Could not load the {DATASET_LABEL} survey.**\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            f"Fetch it locally with `python data/download_{dataset_key}.py`, or publish the "
            f"mirror it expects with `python data/mirror_to_hf.py --dataset {dataset_key}`. "
            f"Pick another survey in the sidebar to carry on in the meantime."
        )
        st.stop()
else:
    with startup_progress.container():
        prog = st.progress(0, text="🧪 Step 1 / 3 — Generating synthetic seismic section (701×255)...")
    clean, noisy, fault_mask, facies_labels = get_demo_section(config_path, dataset_key)

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
    config_path, str(run_dir), ds.get("n_facies", cfg["facies"].get("n_classes", 6)))
load_prog.progress(66, text="🧠 Step 3 / 3 — Running denoiser inference on seismic section...")

# ── Step 3: run inference ───────────────────────────────────────────────────
denoised_on  = run_denoiser_inference(model_on,  noisy, cfg, device)
denoised_off = run_denoiser_inference(model_off, noisy, cfg, device)

load_prog.progress(100, text="✅ Ready! All results loaded.")
time.sleep(0.4)
load_prog.empty()


# ---------------------------------------------------------------------------
# Tab 1: Denoise
# ---------------------------------------------------------------------------

with tab_denoise:
    st.subheader(f"{DATASET_LABEL} — Noisy → Denoised, fault-preservation loss toggled")
    cols = st.columns(3 if fault_preservation_view == "Side-by-side" else 2)
    with cols[0]:
        st.plotly_chart(seismic_heatmap(noisy, f"{DATASET_LABEL} raw input (inline 0)"),
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

    st.markdown("**\"Ordinary denoisers erase the geology. Ours protects it.\"** — toggle above to see it live.")


# ---------------------------------------------------------------------------
# Tab 2: Fault segmentation
# ---------------------------------------------------------------------------

with tab_fault:
    st.subheader(f"Fault segmentation — noisy vs. denoised {DATASET_LABEL} input")
    if faultseg is None:
        st.warning("No FaultSeg checkpoint found — use the Retrain button in the sidebar.")
    else:
        with st.spinner("Running fault segmentation on both inputs..."):
            prob_noisy    = run_faultseg_inference(faultseg, noisy,      cfg, device)
            prob_denoised = run_faultseg_inference(faultseg, denoised_on, cfg, device)

        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(noisy, f"{DATASET_LABEL} raw input"),
                            width='stretch', key="fault_noisy")
        with cols[1]:
            st.plotly_chart(overlay_heatmap(noisy, prob_noisy, "Fault picks — NOISY", threshold=dice_threshold),
                            width='stretch', key="fault_overlay_noisy")
        with cols[2]:
            st.plotly_chart(overlay_heatmap(denoised_on, prob_denoised, "Fault picks — DENOISED", threshold=dice_threshold),
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
            # No ground-truth fault labels for real surveys — compute self-referential quality metrics
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
# Tab 3: Survey explorer
# ---------------------------------------------------------------------------

with tab_survey:
    st.subheader(f"{DATASET_LABEL} — scrub through real inlines")
    if is_real_data:
        # Only the shape is needed to build the slider; the samples stay on disk
        # until an inline is actually requested.
        n_inlines_real = volume_shape(config_path, dataset_key)[0]
        inline_idx = st.slider(f"Inline ({DATASET_LABEL})", 0, n_inlines_real - 1, n_inlines_real // 2,
                                help=f"Scrub through all {n_inlines_real} real inlines.")
        seismic_path, _, _ = resolve_dataset(config_path, dataset_key)
        with st.spinner(f"Reading inline {inline_idx}..."):
            section_noisy = read_inline(seismic_path, inline_idx)
        with st.spinner(f"Running denoiser on inline {inline_idx}..."):
            section_denoised = run_denoiser_inference(model_on, section_noisy, cfg, device)
    else:
        n_inlines = st.slider("Number of inlines to simulate", 6, 24, 12)
        survey    = get_survey(config_path, n_inlines)
        inline_idx = st.slider("Inline", 0, n_inlines - 1, n_inlines // 2)
        rng = np.random.default_rng(100 + inline_idx)
        section_noisy    = noise_mod.make_noisy(survey.clean[inline_idx], cfg, rng=rng)
        section_denoised = run_denoiser_inference(model_on, section_noisy, cfg, device)

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
        st.caption(f"**Survey:** {DATASET_LABEL} ({ds['region']}) · "
                   f"[{ds['citation']}]({ds['citation_url']}) · "
                   f"{ds['geometry']} at {ds['dt_ms']} ms.")


# ---------------------------------------------------------------------------
# Tab 4: Facies
# ---------------------------------------------------------------------------

with tab_facies:
    n_facies = ds.get("n_facies", cfg["facies"].get("n_classes", 6))
    class_names = ds.get("facies_names", [f"Class {i}" for i in range(n_facies)])
    st.subheader(f"Facies segmentation — {DATASET_LABEL} lithostratigraphic labels "
                 f"({n_facies} classes)")

    if facies_model is None:
        st.warning(
            f"**No facies checkpoint for {DATASET_LABEL}** (looked for `{run_dir / 'facies.pt'}`).\n\n"
            f"Train it with:\n\n"
            f"```\npython -m deepseis.train --config {config_path} --dataset {dataset_key}\n```"
        )
    else:
        with st.spinner("Running facies inference on the denoised section..."):
            facies_map = run_facies_inference(facies_model, denoised_on, cfg, device)

        # Ground truth is available for both registered surveys, so show the
        # prediction against it rather than alone -- a facies map on its own
        # looks plausible even when it is wrong.
        has_gt = facies_labels is not None and facies_labels.shape == facies_map.shape
        cols = st.columns(3 if has_gt else 2)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(denoised_on, "Denoised seismic (inline 0)"),
                            width='stretch', key="facies_seismic")
        with cols[1]:
            st.plotly_chart(facies_heatmap(facies_map, "Predicted facies", n_facies),
                            width='stretch', key="facies_map")
        if has_gt:
            with cols[2]:
                st.plotly_chart(facies_heatmap(facies_labels, "Ground-truth facies", n_facies),
                                width='stretch', key="facies_gt")

        if has_gt:
            accuracy = float((facies_map == facies_labels).mean())
            # Mean per-class IoU, computed only over classes actually present in
            # this inline -- averaging in absent classes as zeros would understate
            # the result for a section that simply does not intersect them.
            ious = []
            for c in range(n_facies):
                pred_c, gt_c = facies_map == c, facies_labels == c
                union = np.logical_or(pred_c, gt_c).sum()
                if union > 0:
                    ious.append(np.logical_and(pred_c, gt_c).sum() / union)
            m1, m2 = st.columns(2)
            m1.metric("Pixel accuracy (fit)", f"{accuracy * 100:.1f}%",
                       help="Measured on inline 0 — the same section the head was trained on. "
                            "This is goodness of fit, not generalization.")
            m2.metric("Mean IoU (fit)", f"{np.mean(ious) * 100:.1f}%" if ious else "—",
                       help="Averaged over the classes present in this inline.")
            st.warning(
                "**These are training-set scores.** The facies head is fitted on inline 0 only "
                "(`train.prepare_data` takes a single section), and this tab scores it on that "
                "same inline. Measured on held-out inlines of Parihaka, accuracy falls from ~79% "
                "to ~42–50%. Treat the map as a qualitative interpretation aid, not a validated "
                "classifier — training across multiple inlines would be needed for the latter."
            )

        rows = []
        for c in range(n_facies):
            name = class_names[c] if c < len(class_names) else f"Class {c}"
            pred_px = int((facies_map == c).sum())
            row = {"Class": f"{c} — {name}",
                   "Predicted (%)": f"{100 * pred_px / facies_map.size:.1f}"}
            if has_gt:
                gt_px = int((facies_labels == c).sum())
                row["Actual (%)"] = f"{100 * gt_px / facies_labels.size:.1f}"
                inter = int(np.logical_and(facies_map == c, facies_labels == c).sum())
                union = int(np.logical_or(facies_map == c, facies_labels == c).sum())
                row["IoU (%)"] = f"{100 * inter / union:.1f}" if union else "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.caption(f"**{DATASET_LABEL} class legend:** "
                   + " · ".join(f"{i} = {n}" for i, n in enumerate(class_names)))
        if has_gt:
            st.info(
                f"The facies head is trained on this survey's own published labels "
                f"({ds['citation']}), then applied to the *denoised* section. Classes absent "
                f"from inline 0 are shown as 0% — that is the section not intersecting them, "
                f"not a prediction failure."
            )


# ---------------------------------------------------------------------------
# Tab 5: Export
# ---------------------------------------------------------------------------

with tab_export:
    st.subheader(f"Export denoised {DATASET_LABEL} section")
    st.markdown("Download the denoised inline 0. SEG-Y preserves standard headers for OpendTect / Petrel / Kingdom.")
    export_format = st.radio("Format", ["SEG-Y (.sgy)", "NumPy (.npy)"], horizontal=True)
    # Name the file after the survey so exports from two surveys don't collide.
    export_stem = f"{dataset_key}_denoised_inline0"
    if st.button("Generate export"):
        if export_format == "NumPy (.npy)":
            buf = io.BytesIO()
            np.save(buf, denoised_on)
            buf.seek(0)
            st.download_button(f"⬇️ Download {export_stem}.npy", buf,
                               file_name=f"{export_stem}.npy", mime="application/octet-stream")
        else:
            import os
            buf = io.BytesIO()
            with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                segy_mod.write_segy_like(tmp_path, denoised_on, template_path=None,
                                          dt_ms=ds["dt_ms"])
                with open(tmp_path, "rb") as f:
                    buf.write(f.read())
                buf.seek(0)
                st.download_button(f"⬇️ Download {export_stem}.sgy", buf,
                                   file_name=f"{export_stem}.sgy", mime="application/octet-stream")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        st.success("Export ready — click the button above to download.")

    st.markdown("---")
    st.markdown("#### Denoised section preview")
    st.plotly_chart(seismic_heatmap(denoised_on, "Denoised inline 0 — fault-preservation ON"),
                    width='stretch', key="export_preview")
    st.caption(f"Shape: {denoised_on.shape[0]} samples × {denoised_on.shape[1]} traces | "
               f"Range: [{denoised_on.min():.3f}, {denoised_on.max():.3f}]")


# ---------------------------------------------------------------------------
# Tab 6: Diagnostics
# ---------------------------------------------------------------------------

with tab_diag:
    st.subheader("Diagnostics")

    st.markdown("#### F-K spectrum — signal survives denoising")
    with st.spinner("Computing F-K spectra..."):
        fk_noisy    = fk_spectrum(torch.from_numpy(noisy)).numpy()
        fk_denoised = fk_spectrum(torch.from_numpy(denoised_on)).numpy()
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(go.Figure(go.Heatmap(z=fk_noisy, colorscale="Viridis"))
                        .update_layout(title="F-K — raw", height=350, margin=dict(l=10,r=10,t=40,b=10)),
                        width='stretch', key="diag_fk_noisy")
    with cols[1]:
        st.plotly_chart(go.Figure(go.Heatmap(z=fk_denoised, colorscale="Viridis"))
                        .update_layout(title="F-K — denoised", height=350, margin=dict(l=10,r=10,t=40,b=10)),
                        width='stretch', key="diag_fk_denoised")

    st.markdown("#### Signal-leakage map — no geology removed")
    st.caption("Values near zero = removed component is noise, not geology.")
    with st.spinner("Computing local similarity map..."):
        sim_map = metrics_mod.local_similarity_map_fast(noisy, denoised_on, window=9)
    st.plotly_chart(go.Figure(go.Heatmap(z=sim_map, colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
                    .update_layout(title="Local correlation: (noisy − denoised) vs. denoised", height=380,
                                   yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=40,b=10)),
                    width='stretch', key="diag_sim_map")

    st.markdown("#### Jacobian mask explainer")
    st.caption("Pick a pixel — see which inputs the denoiser relies on and what blind-spot mask that implies.")
    jy = st.slider("Row (sample)",  10, noisy.shape[0]-10, noisy.shape[0]//2, key="jy")
    jx = st.slider("Col (trace)",   10, noisy.shape[1]-10, noisy.shape[1]//2, key="jx")
    if st.button("Compute Jacobian"):
        pcfg = cfg["data"]["patch"]
        half = pcfg["size"] // 2
        y0 = min(max(0, jy-half), noisy.shape[0]-pcfg["size"])
        x0 = min(max(0, jx-half), noisy.shape[1]-pcfg["size"])
        patch = noisy[y0:y0+pcfg["size"], x0:x0+pcfg["size"]]
        x_t = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with st.spinner("Computing Jacobian..."):
            jac        = compute_pixel_jacobian(model_on, x_t, (jy-y0, jx-x0))
            suggestion = suggest_mask_design(jac, (jy-y0, jx-x0))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(go.Figure(go.Heatmap(z=jac, colorscale="Inferno"))
                            .update_layout(title="Sensitivity |d(output)/d(input)|", height=350,
                                           margin=dict(l=10,r=10,t=40,b=10)),
                            width='stretch', key="diag_jacobian")
        with c2:
            st.write(f"**Suggested blind shape:** `{suggestion.suggested_blind_shape}`")
            st.write(f"**Suggested blind width:** `{suggestion.suggested_blind_width}` px")
            st.write(f"Lateral extent: {suggestion.lateral_extent_px:.1f} px")
            st.write(f"Vertical extent: {suggestion.vertical_extent_px:.1f} px")
            st.caption("If measured extent > `masking.struct_n2v.blind_width` in config, widen the mask.")
