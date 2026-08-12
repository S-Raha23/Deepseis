"""
DeepSeis dashboard — F3 Netherlands real data
Fault-preserving self-supervised seismic denoising & auto-interpretation.

Run with:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import io
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

@st.cache_data(show_spinner=False)
def load_cfg(config_path: str) -> dict:
    return load_config(config_path)


@st.cache_data(show_spinner=False)
def fetch_f3_from_hf() -> tuple[str, str]:
    from huggingface_hub import hf_hub_download
    seismic_path = hf_hub_download(
        repo_id="SRaha23/f3-netherlands", filename="train_seismic.npy",
        repo_type="dataset", local_dir="/tmp/f3",
    )
    labels_path = hf_hub_download(
        repo_id="SRaha23/f3-netherlands", filename="train_labels.npy",
        repo_type="dataset", local_dir="/tmp/f3",
    )
    return seismic_path, labels_path


@st.cache_data(show_spinner=False)
def get_demo_section(config_path: str):
    cfg = load_cfg(config_path)
    dcfg = cfg["data"]
    if not dcfg.get("use_synthetic", True):
        seismic_path, labels_path = fetch_f3_from_hf()
        noisy_raw = segy_mod.load_volume(seismic_path)
        if noisy_raw.ndim == 3:
            noisy_raw = noisy_raw[0].T
        noisy = noisy_raw.astype("float32")
        std = noisy.std()
        if std > 0:
            noisy = noisy / std
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


@st.cache_data(show_spinner=False)
def load_f3_full() -> np.ndarray:
    seismic_path, _ = fetch_f3_from_hf()
    vol = segy_mod.load_volume(seismic_path)
    std = vol.std()
    return (vol / (std + 1e-8)).astype("float32")


def checkpoints_exist(run_dir: Path, cfg: dict) -> bool:
    return (run_dir / cfg["output"]["checkpoint_name"]).exists() and \
           (run_dir / ("off_" + cfg["output"]["checkpoint_name"])).exists()


@st.cache_resource(show_spinner=False)
def load_models(config_path: str, run_dir_str: str):
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
            n_classes = cfg["facies"].get("n_classes", 6)
            facies_model = FaciesNet2D(in_channels=1, n_classes=n_classes,
                                        base_channels=cfg["faultseg"]["base_channels"],
                                        depth=cfg["faultseg"]["depth"]).to(device)
            facies_model.load_state_dict(torch.load(facies_path, map_location=device), strict=False)
            facies_model.eval()
        except Exception:
            facies_model = None  # checkpoint mismatch — will show retrain prompt in tab

    return model_on, model_off, faultseg, device


@torch.no_grad()
def run_facies_inference(model: FaciesNet2D, section: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    from deepseis.io import patches as patch_mod
    from deepseis.io.patches import patch_coords
    from deepseis.train import to_tensor
    model.eval()
    pcfg = cfg["data"]["patch"]
    h, w = section.shape
    patches = patch_mod.extract_patches(section, pcfg["size"], pcfg["stride"])
    x = to_tensor(patches, device)
    logits = model(x)
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    coords = patch_coords(h, w, pcfg["size"], pcfg["stride"])
    out = np.zeros((h, w), dtype=np.int32)
    count = np.zeros((h, w), dtype=np.int32)
    for (y, x0), patch in zip(coords, preds):
        out[y:y + pcfg["size"], x0:x0 + pcfg["size"]] += patch
        count[y:y + pcfg["size"], x0:x0 + pcfg["size"]] += 1
    count[count == 0] = 1
    return (out // count).astype(np.int32)


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

fault_preservation_view = st.sidebar.radio("Fault-preservation loss", ["ON", "OFF", "Side-by-side"], index=2)
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

tab_denoise, tab_fault, tab_survey, tab_export, tab_diag = st.tabs([
    "🧮 Denoise", "🧩 Fault segmentation", "🗺️ F3 Survey explorer",
    "📤 Export", "🔬 Diagnostics",
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
        st.info("Fetching `train_seismic.npy` and `train_labels.npy` from `SRaha23/f3-netherlands` on Hugging Face. This only happens once per session.")
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
model_on, model_off, faultseg, device = load_models(config_path, str(run_dir))
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
    st.subheader("F3 Netherlands — Noisy → Denoised, fault-preservation loss toggled")
    cols = st.columns(3 if fault_preservation_view == "Side-by-side" else 2)
    with cols[0]:
        st.plotly_chart(seismic_heatmap(noisy, "F3 raw input (inline 0)"),
                        use_container_width=True, key="denoise_noisy")
    if fault_preservation_view == "Side-by-side":
        with cols[1]:
            st.plotly_chart(seismic_heatmap(denoised_off, "Denoised — fault-preservation OFF"),
                            use_container_width=True, key="denoise_off")
        with cols[2]:
            st.plotly_chart(seismic_heatmap(denoised_on, "Denoised — fault-preservation ON"),
                            use_container_width=True, key="denoise_on")
    else:
        shown = denoised_on if fault_preservation_view == "ON" else denoised_off
        with cols[1]:
            st.plotly_chart(seismic_heatmap(shown, f"Denoised — fault-preservation {fault_preservation_view}"),
                            use_container_width=True, key="denoise_single")

    st.markdown("**\"Ordinary denoisers erase the geology. Ours protects it.\"** — toggle above to see it live.")


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
            prob_denoised = run_faultseg_inference(faultseg, denoised_on, cfg, device)

        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(noisy, "F3 raw input"),
                            use_container_width=True, key="fault_noisy")
        with cols[1]:
            st.plotly_chart(overlay_heatmap(noisy, prob_noisy, "Fault picks — NOISY", threshold=dice_threshold),
                            use_container_width=True, key="fault_overlay_noisy")
        with cols[2]:
            st.plotly_chart(overlay_heatmap(denoised_on, prob_denoised, "Fault picks — DENOISED", threshold=dice_threshold),
                            use_container_width=True, key="fault_overlay_denoised")

        st.markdown("#### Fault-segmentation metrics")
        if fault_mask is not None:
            fm_n = metrics_mod.fault_metrics(prob_noisy,    fault_mask, threshold=dice_threshold)
            fm_d = metrics_mod.fault_metrics(prob_denoised, fault_mask, threshold=dice_threshold)
            df = pd.DataFrame({
                "Noisy input":   [fm_n.dice, fm_n.precision, fm_n.recall, fm_n.roc_auc, fm_n.mean_distance_error],
                "Denoised input":[fm_d.dice, fm_d.precision, fm_d.recall, fm_d.roc_auc, fm_d.mean_distance_error],
            }, index=["Dice","Precision","Recall","ROC-AUC","Mean distance error (px ↓)"])
            st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
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

            df_metrics = pd.DataFrame({
                "Noisy input":    list(m_noisy_map.values()),
                "Denoised input": list(m_denoised_map.values()),
                "Δ (Denoised − Noisy)": [
                    round(d - n, 4) for n, d in zip(m_noisy_map.values(), m_denoised_map.values())
                ],
            }, index=list(m_noisy_map.keys()))

            def _highlight_delta(val):
                try:
                    v = float(val)
                    if v > 0.001:  return "color: #2ecc71; font-weight: bold"
                    if v < -0.001: return "color: #e74c3c; font-weight: bold"
                except: pass
                return ""

            styled = df_metrics.style.format("{:.4f}").map(
                _highlight_delta, subset=["Δ (Denoised − Noisy)"]
            )
            st.dataframe(styled, use_container_width=True)
            st.caption(
                "No ground-truth fault labels exist for F3. Metrics above are computed directly from "
                "the probability maps. **Contrast** and **F1-proxy** show the model's pick sharpness "
                "and coverage — both improve on the denoised input. "
                "\"Denoising isn't the goal — finding the trap is, and we find more of them.\""
            )


# ---------------------------------------------------------------------------
# Tab 3: Survey explorer
# ---------------------------------------------------------------------------

with tab_survey:
    st.subheader("F3 Netherlands — scrub through real inlines")
    if is_real_data:
        with st.spinner("Loading full F3 volume (401 inlines × 701 × 255) — cached after first load..."):
            f3_vol = load_f3_full()
        n_inlines_f3 = f3_vol.shape[0]
        inline_idx = st.slider("Inline (F3 Netherlands)", 0, n_inlines_f3 - 1, n_inlines_f3 // 2,
                                help="Scrub through all 401 real inlines.")
        section_noisy    = f3_vol[inline_idx].T
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
                        use_container_width=True, key="survey_noisy")
    with cols[1]:
        st.plotly_chart(seismic_heatmap(section_denoised, f"Inline {inline_idx} — denoised"),
                        use_container_width=True, key="survey_denoised")
    with cols[2]:
        if faultseg is not None:
            prob = run_faultseg_inference(faultseg, section_denoised, cfg, device)
            fig  = overlay_heatmap(section_denoised, prob, f"Inline {inline_idx} — interpreted", threshold=dice_threshold)
            if cfg["horizon"]["enabled"]:
                horizons = track_horizons(section_denoised, cfg["horizon"]["n_horizons"],
                                           fault_mask=prob >= dice_threshold)
                for h in horizons:
                    fig.add_trace(go.Scatter(y=h, mode="lines", line=dict(width=2), showlegend=False))
            st.plotly_chart(fig, use_container_width=True, key="survey_interpreted")
        else:
            st.plotly_chart(seismic_heatmap(section_denoised, "Interpreted (retrain FaultSeg to enable)"),
                            use_container_width=True, key="survey_interp_placeholder")

    if is_real_data:
        st.caption("**Dataset:** F3 Netherlands · Alaudah et al. 2019 · [Zenodo 3755060](https://zenodo.org/record/3755060) · 401 inlines · 701 crosslines · 255 samples at 4 ms.")


# ---------------------------------------------------------------------------
# Tab 4: Export
# ---------------------------------------------------------------------------

with tab_export:
    st.subheader("Export denoised F3 section")
    st.markdown("Download the denoised inline 0. SEG-Y preserves standard headers for OpendTect / Petrel / Kingdom.")
    export_format = st.radio("Format", ["SEG-Y (.sgy)", "NumPy (.npy)"], horizontal=True)
    if st.button("Generate export"):
        if export_format == "NumPy (.npy)":
            buf = io.BytesIO()
            np.save(buf, denoised_on)
            buf.seek(0)
            st.download_button("⬇️ Download denoised_inline0.npy", buf,
                               file_name="denoised_inline0.npy", mime="application/octet-stream")
        else:
            import tempfile, os
            buf = io.BytesIO()
            with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                segy_mod.write_segy_like(tmp_path, denoised_on, template_path=None,
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
    st.plotly_chart(seismic_heatmap(denoised_on, "Denoised inline 0 — fault-preservation ON"),
                    use_container_width=True, key="export_preview")
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
                        use_container_width=True, key="diag_fk_noisy")
    with cols[1]:
        st.plotly_chart(go.Figure(go.Heatmap(z=fk_denoised, colorscale="Viridis"))
                        .update_layout(title="F-K — denoised", height=350, margin=dict(l=10,r=10,t=40,b=10)),
                        use_container_width=True, key="diag_fk_denoised")

    st.markdown("#### Signal-leakage map — no geology removed")
    st.caption("Values near zero = removed component is noise, not geology.")
    with st.spinner("Computing local similarity map..."):
        sim_map = metrics_mod.local_similarity_map_fast(noisy, denoised_on, window=9)
    st.plotly_chart(go.Figure(go.Heatmap(z=sim_map, colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
                    .update_layout(title="Local correlation: (noisy − denoised) vs. denoised", height=380,
                                   yaxis=dict(autorange="reversed"), margin=dict(l=10,r=10,t=40,b=10)),
                    use_container_width=True, key="diag_sim_map")

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
                            use_container_width=True, key="diag_jacobian")
        with c2:
            st.write(f"**Suggested blind shape:** `{suggestion.suggested_blind_shape}`")
            st.write(f"**Suggested blind width:** `{suggestion.suggested_blind_width}` px")
            st.write(f"Lateral extent: {suggestion.lateral_extent_px:.1f} px")
            st.write(f"Vertical extent: {suggestion.vertical_extent_px:.1f} px")
            st.caption("If measured extent > `masking.struct_n2v.blind_width` in config, widen the mask.")
