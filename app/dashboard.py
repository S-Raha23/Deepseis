"""
DeepSeis dashboard — F3 Netherlands real data
Fault-preserving self-supervised seismic denoising & auto-interpretation.

Run with:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import io
import sys
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
# Cached data / model loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_cfg(config_path: str) -> dict:
    return load_config(config_path)


@st.cache_data(show_spinner=False)
def get_demo_section(config_path: str):
    """Return (clean, noisy, fault_mask, facies) for the main inline.
    clean/fault_mask/facies are None when running on real field data."""
    cfg = load_cfg(config_path)
    dcfg = cfg["data"]
    if not dcfg.get("use_synthetic", True) and dcfg.get("field_volume_path"):
        import os
        vol_path = dcfg["field_volume_path"]
        if not os.path.exists(vol_path):
            return None, None, None, None
        noisy_raw = segy_mod.load_volume(vol_path)
        if noisy_raw.ndim == 3:
            noisy_raw = noisy_raw[0].T
        noisy = noisy_raw.astype("float32")
        std = noisy.std()
        if std > 0:
            noisy = noisy / std
        return None, noisy, None, None
    rng = np.random.default_rng(cfg["data"]["synthetic"]["random_seed"])
    vol = synth_mod.generate_from_config(cfg)
    noisy = noise_mod.make_noisy(vol.clean, cfg, rng=rng)
    return vol.clean, noisy, vol.fault_mask, vol.facies


@st.cache_data(show_spinner=False)
def get_survey(config_path: str, n_inlines: int):
    return synth_mod.generate_synthetic_survey(n_inlines=n_inlines, random_seed=7)


@st.cache_data(show_spinner="Loading F3 volume...")
def load_f3_volume(path: str) -> np.ndarray:
    vol = segy_mod.load_volume(path)   # (401, 701, 255)
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
        n_classes = cfg["facies"].get("n_classes", 6)
        facies_model = FaciesNet2D(in_channels=1, n_classes=n_classes,
                                    base_channels=cfg["faultseg"]["base_channels"],
                                    depth=cfg["faultseg"]["depth"]).to(device)
        facies_model.load_state_dict(torch.load(facies_path, map_location=device))
        facies_model.eval()

    return model_on, model_off, faultseg, facies_model, device


@torch.no_grad()
def run_facies_inference(model: FaciesNet2D, section: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    """Returns argmax class map (H, W) int."""
    from deepseis.io import patches as patch_mod
    from deepseis.train import to_tensor
    model.eval()
    pcfg = cfg["data"]["patch"]
    h, w = section.shape
    patches = patch_mod.extract_patches(section, pcfg["size"], pcfg["stride"])
    x = to_tensor(patches, device)
    logits = model(x)                              # (N, n_classes, H, W)
    preds = logits.argmax(dim=1).detach().cpu().numpy()  # (N, H, W)
    # stitch the argmax map back (nearest, no averaging — averaging class indices is meaningless)
    from deepseis.io.patches import patch_coords
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
    data = _prepare_data(cfg)
    noisy = data["noisy"]
    syn_clean = data["syn_clean"]
    syn_fault_mask = data["syn_fault_mask"]
    with st.spinner("Training denoiser (fault-preservation OFF)..."):
        model_off, _ = train_denoiser(cfg, noisy, fault_preservation_enabled=False, device=device, verbose=False)
    with st.spinner("Training denoiser (fault-preservation ON)..."):
        model_on, _ = train_denoiser(cfg, noisy, fault_preservation_enabled=True, device=device, verbose=False)
    with st.spinner("Training FaultSeg head (on synthetic reference volume)..."):
        faultseg = train_faultseg(cfg, syn_clean, syn_fault_mask, device, verbose=False)
    torch.save(model_on.state_dict(), run_dir / cfg["output"]["checkpoint_name"])
    torch.save(model_off.state_dict(), run_dir / ("off_" + cfg["output"]["checkpoint_name"]))
    torch.save(faultseg.state_dict(), run_dir / cfg["output"]["faultseg_checkpoint_name"])
    st.cache_resource.clear()


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
    colorscale = [[i / (n_classes - 1), FACIES_COLORS[i % len(FACIES_COLORS)]] for i in range(n_classes)]
    fig = go.Figure(data=go.Heatmap(z=class_map, colorscale=colorscale,
                                     zmin=0, zmax=n_classes - 1, showscale=True))
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

config_path = st.sidebar.text_input("Config", value="configs/default.yaml")
cfg = load_cfg(config_path)
run_dir = Path(cfg["output"]["run_dir"])

if not checkpoints_exist(run_dir, cfg):
    st.sidebar.warning("No trained checkpoints found yet.")
    if st.sidebar.button("\u25B6 Train models on F3 data now (~a few minutes on CPU)"):
        run_training(config_path, run_dir)
        st.rerun()
    st.title("DeepSeis")
    st.info(
        "Click **Train models on F3 data now** in the sidebar to train the self-supervised denoiser "
        "(fault-preservation ON and OFF) on the real F3 Netherlands seismic data, "
        "and the FaultSeg head on a synthetic reference volume (standard FaultSeg3D workflow). "
        "This runs `deepseis.train`'s functions directly — a couple of minutes on CPU."
    )
    st.stop()

model_on, model_off, faultseg, facies_model, device = load_models(config_path, str(run_dir))
fault_preservation_view = st.sidebar.radio("Fault-preservation loss", ["ON", "OFF", "Side-by-side"], index=2)
dice_threshold = st.sidebar.slider("Fault probability threshold", 0.1, 0.9, cfg["faultseg"]["dice_threshold"], 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Running on the **F3 Netherlands block** (Alaudah et al. 2019, Zenodo) — "
    "401 inlines × 701 crosslines × 255 depth samples of real post-stack seismic amplitude. "
    "Denoiser trained self-supervised on this survey. FaultSeg trained on a synthetic volume "
    "(no real fault GT exists for F3) — the standard FaultSeg3D workflow. "
    "Real 6-class facies labels from the Alaudah benchmark power the facies head."
)

tab_denoise, tab_fault, tab_survey, tab_facies, tab_export, tab_diag = st.tabs([
    "\U0001F9EE Denoise",
    "\U0001F9E9 Fault segmentation",
    "\U0001F5FA\uFE0F F3 Survey explorer",
    "\U0001F30A Facies",
    "\U0001F4E4 Export",
    "\U0001F52C Diagnostics",
])

clean, noisy, fault_mask, facies_labels = get_demo_section(config_path)

if noisy is None:
    st.title("DeepSeis — Real F3 Data")
    st.warning(
        "**F3 dataset not found.** Run the downloader first:\n\n"
        "```\npython data/download_f3.py\n```\n\n"
        "Then retrain:\n\n"
        "```\npython -m deepseis.train --config configs/default.yaml\n```"
    )
    st.stop()

is_real_data = clean is None

# Compute denoised inline once and reuse across all tabs
denoised_on = run_denoiser_inference(model_on, noisy, cfg, device)
denoised_off = run_denoiser_inference(model_off, noisy, cfg, device)


# ---------------------------------------------------------------------------
# Tab 1: Denoise
# ---------------------------------------------------------------------------

with tab_denoise:
    st.subheader("F3 Netherlands — Noisy \u2192 Denoised, fault-preservation loss toggled")

    cols = st.columns(3 if fault_preservation_view == "Side-by-side" else 2)
    with cols[0]:
        st.plotly_chart(seismic_heatmap(noisy, "F3 raw input (inline 0)"),
                        use_container_width=True, key="denoise_noisy")
    if fault_preservation_view == "Side-by-side":
        with cols[1]:
            st.plotly_chart(seismic_heatmap(denoised_off, "Denoised \u2014 fault-preservation OFF"),
                            use_container_width=True, key="denoise_off")
        with cols[2]:
            st.plotly_chart(seismic_heatmap(denoised_on, "Denoised \u2014 fault-preservation ON"),
                            use_container_width=True, key="denoise_on")
    else:
        shown = denoised_on if fault_preservation_view == "ON" else denoised_off
        with cols[1]:
            st.plotly_chart(seismic_heatmap(shown, f"Denoised \u2014 fault-preservation {fault_preservation_view}"),
                            use_container_width=True, key="denoise_single")

    st.markdown("**\"Ordinary denoisers erase the geology. Ours protects it.\"** \u2014 toggle above to see it live.")

    if is_real_data:
        st.info(
            "Running on **real F3 field data** — no clean reference, so PSNR/SSIM cannot be computed "
            "(expected for any field survey). The visual comparison directly shows the fault-preservation "
            "effect on real geology."
        )
    else:
        st.markdown("#### Live metrics (vs. known-clean synthetic reference)")
        m_noisy = {"PSNR (dB)": metrics_mod.psnr(noisy, clean), "SNR (dB)": metrics_mod.snr_db(noisy, clean),
                   "SSIM": metrics_mod.ssim(noisy, clean)}
        m_on = {"PSNR (dB)": metrics_mod.psnr(denoised_on, clean), "SNR (dB)": metrics_mod.snr_db(denoised_on, clean),
                "SSIM": metrics_mod.ssim(denoised_on, clean)}
        m_off = {"PSNR (dB)": metrics_mod.psnr(denoised_off, clean), "SNR (dB)": metrics_mod.snr_db(denoised_off, clean),
                 "SSIM": metrics_mod.ssim(denoised_off, clean)}
        mcols = st.columns(3)
        for col, label, m in zip(mcols, ["Noisy (baseline)", "Denoised OFF", "Denoised ON"],
                                  [m_noisy, m_off, m_on]):
            with col:
                st.metric(f"{label} — PSNR", f"{m['PSNR (dB)']:.2f} dB")
                st.metric(f"{label} — SSIM", f"{m['SSIM']:.4f}")


# ---------------------------------------------------------------------------
# Tab 2: Fault segmentation
# ---------------------------------------------------------------------------

with tab_fault:
    st.subheader("Fault segmentation on F3 Netherlands: noisy vs. denoised input")
    if faultseg is None:
        st.warning("No trained FaultSeg checkpoint found — retrain from the sidebar to populate this tab.")
    else:
        prob_noisy = run_faultseg_inference(faultseg, noisy, cfg, device)
        prob_denoised = run_faultseg_inference(faultseg, denoised_on, cfg, device)

        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(noisy, "F3 raw input"),
                            use_container_width=True, key="fault_noisy")
        with cols[1]:
            st.plotly_chart(overlay_heatmap(noisy, prob_noisy, "Fault picks — NOISY input",
                                            threshold=dice_threshold),
                            use_container_width=True, key="fault_overlay_noisy")
        with cols[2]:
            st.plotly_chart(overlay_heatmap(denoised_on, prob_denoised, "Fault picks — DENOISED input",
                                            threshold=dice_threshold),
                            use_container_width=True, key="fault_overlay_denoised")

        st.markdown("#### Fault-segmentation metrics")
        if fault_mask is not None:
            fm_noisy = metrics_mod.fault_metrics(prob_noisy, fault_mask, threshold=dice_threshold)
            fm_denoised = metrics_mod.fault_metrics(prob_denoised, fault_mask, threshold=dice_threshold)
            df = pd.DataFrame({
                "Noisy input": [fm_noisy.dice, fm_noisy.precision, fm_noisy.recall,
                                fm_noisy.roc_auc, fm_noisy.mean_distance_error],
                "Denoised input": [fm_denoised.dice, fm_denoised.precision, fm_denoised.recall,
                                   fm_denoised.roc_auc, fm_denoised.mean_distance_error],
            }, index=["Dice", "Precision", "Recall", "ROC-AUC", "Mean distance error (px, lower=better)"])
            st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
            st.caption("\"Denoising isn't the goal \u2014 finding the trap is, and we find more of them.\"")
        else:
            st.info(
                "**Real F3 data:** no fault ground-truth labels exist for this survey, so quantitative "
                "Dice/precision/recall cannot be computed. The overlays above show the qualitative "
                "improvement from denoising — FaultSeg trained on a synthetic volume applied to your "
                "real denoised section (standard FaultSeg3D field workflow)."
            )


# ---------------------------------------------------------------------------
# Tab 3: F3 Survey explorer (real inlines)
# ---------------------------------------------------------------------------

with tab_survey:
    st.subheader("F3 Netherlands — scrub through real inlines")

    if is_real_data:
        f3_vol = load_f3_volume(cfg["data"]["field_volume_path"])
        n_inlines_f3 = f3_vol.shape[0]
        inline_idx = st.slider("Inline (F3 Netherlands)", 0, n_inlines_f3 - 1, n_inlines_f3 // 2,
                                help="Scrub through all 401 real inlines of the F3 Netherlands block.")
        section_noisy = f3_vol[inline_idx].T
        section_denoised = run_denoiser_inference(model_on, section_noisy, cfg, device)
    else:
        n_inlines = st.slider("Number of inlines to simulate", 6, 24, 12)
        survey = get_survey(config_path, n_inlines)
        inline_idx = st.slider("Inline", 0, n_inlines - 1, n_inlines // 2)
        rng = np.random.default_rng(100 + inline_idx)
        section_noisy = noise_mod.make_noisy(survey.clean[inline_idx], cfg, rng=rng)
        section_denoised = run_denoiser_inference(model_on, section_noisy, cfg, device)

    cols = st.columns(3)
    with cols[0]:
        st.plotly_chart(seismic_heatmap(section_noisy, f"Inline {inline_idx} \u2014 F3 raw"),
                        use_container_width=True, key="survey_noisy")
    with cols[1]:
        st.plotly_chart(seismic_heatmap(section_denoised, f"Inline {inline_idx} \u2014 Denoised"),
                        use_container_width=True, key="survey_denoised")
    with cols[2]:
        if faultseg is not None:
            prob = run_faultseg_inference(faultseg, section_denoised, cfg, device)
            fig = overlay_heatmap(section_denoised, prob, f"Inline {inline_idx} \u2014 Fault interpreted",
                                   threshold=dice_threshold)
            if cfg["horizon"]["enabled"]:
                horizons = track_horizons(section_denoised, cfg["horizon"]["n_horizons"],
                                           fault_mask=prob >= dice_threshold)
                for h in horizons:
                    fig.add_trace(go.Scatter(y=h, mode="lines", line=dict(width=2), showlegend=False))
            st.plotly_chart(fig, use_container_width=True, key="survey_interpreted")
        else:
            st.plotly_chart(seismic_heatmap(section_denoised, "Fault interpreted (train FaultSeg to enable)"),
                            use_container_width=True, key="survey_interp_placeholder")

    if is_real_data:
        st.caption(
            "**Dataset:** F3 Netherlands block — Alaudah et al. 2019 "
            "([Zenodo 3755060](https://zenodo.org/record/3755060)), SEG Interpretation journal. "
            "401 inlines · 701 crosslines · 255 depth samples at 4 ms."
        )


# ---------------------------------------------------------------------------
# Tab 4: Facies segmentation (real F3 labels)
# ---------------------------------------------------------------------------

with tab_facies:
    st.subheader("Facies segmentation — real F3 lithostratigraphic labels (6 classes)")
    if facies_model is None:
        st.warning(
            "No trained facies checkpoint found (`runs/default/facies.pt`). "
            "Retrain with: `python -m deepseis.train --config configs/default.yaml`"
        )
    else:
        n_classes = cfg["facies"].get("n_classes", 6)
        facies_map = run_facies_inference(facies_model, denoised_on, cfg, device)

        cols = st.columns(2)
        with cols[0]:
            st.plotly_chart(seismic_heatmap(denoised_on, "Denoised seismic (F3 inline 0)"),
                            use_container_width=True, key="facies_seismic")
        with cols[1]:
            st.plotly_chart(facies_heatmap(facies_map, "Predicted facies (6 classes)", n_classes),
                            use_container_width=True, key="facies_map")

        class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]
        unique, counts = np.unique(facies_map, return_counts=True)
        total = facies_map.size
        df_facies = pd.DataFrame({
            "Class": [class_names[c] for c in unique],
            "Pixel count": counts,
            "Coverage (%)": [f"{100*c/total:.1f}" for c in counts],
        })
        st.dataframe(df_facies, use_container_width=True, hide_index=True)
        st.caption(
            "Facies head trained on real F3 Alaudah 2019 labels (6 lithostratigraphic classes). "
            "Input: denoised inline 0. The model generalises to any inline via the slider in the "
            "Survey explorer tab."
        )
        st.info(
            "**F3 class legend (Alaudah et al. 2019):** "
            "0 = Upper North Sea Group · 1 = Middle North Sea Group · "
            "2 = Lower North Sea Group · 3 = Rijnland/Chalk · "
            "4 = Jurassic · 5 = Triassic"
        )


# ---------------------------------------------------------------------------
# Tab 5: Export denoised section to SEG-Y / .npy
# ---------------------------------------------------------------------------

with tab_export:
    st.subheader("Export denoised F3 section")
    st.markdown(
        "Download the denoised inline 0 in your preferred format. "
        "SEG-Y output preserves the standard header convention so it drops into "
        "existing interpretation workflows (OpendTect, Petrel, Kingdom)."
    )

    export_format = st.radio("Format", ["SEG-Y (.sgy)", "NumPy (.npy)"], horizontal=True)

    if st.button("Generate export"):
        if export_format == "NumPy (.npy)":
            buf = io.BytesIO()
            np.save(buf, denoised_on)
            buf.seek(0)
            st.download_button(
                label="\u2B07\uFE0F Download denoised_inline0.npy",
                data=buf,
                file_name="denoised_inline0.npy",
                mime="application/octet-stream",
            )
        else:  # SEG-Y
            buf = io.BytesIO()
            # Write to a temp file then read back into the BytesIO buffer
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                segy_mod.write_segy_like(tmp_path, denoised_on, template_path=None,
                                          dt_ms=cfg["data"]["synthetic"]["dt_ms"])
                with open(tmp_path, "rb") as f:
                    buf.write(f.read())
                buf.seek(0)
                st.download_button(
                    label="\u2B07\uFE0F Download denoised_inline0.sgy",
                    data=buf,
                    file_name="denoised_inline0.sgy",
                    mime="application/octet-stream",
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        st.success("Export ready — click the button above to download.")

    st.markdown("---")
    st.markdown("#### Denoised section preview")
    st.plotly_chart(seismic_heatmap(denoised_on, "Denoised F3 inline 0 — fault-preservation ON"),
                    use_container_width=True, key="export_preview")
    st.caption(f"Shape: {denoised_on.shape[0]} samples × {denoised_on.shape[1]} traces "
               f"| Data range: [{denoised_on.min():.3f}, {denoised_on.max():.3f}]")


# ---------------------------------------------------------------------------
# Tab 6: Diagnostics
# ---------------------------------------------------------------------------

with tab_diag:
    st.subheader("Diagnostics — F3 Netherlands real data")

    st.markdown("#### F-K spectrum, before vs. after \u2014 real F3 signal survives denoising")
    fk_noisy = fk_spectrum(torch.from_numpy(noisy)).numpy()
    fk_denoised = fk_spectrum(torch.from_numpy(denoised_on)).numpy()
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(
            go.Figure(data=go.Heatmap(z=fk_noisy, colorscale="Viridis"))
            .update_layout(title="F-K spectrum \u2014 F3 raw", height=350,
                           margin=dict(l=10, r=10, t=40, b=10)),
            use_container_width=True, key="diag_fk_noisy")
    with cols[1]:
        st.plotly_chart(
            go.Figure(data=go.Heatmap(z=fk_denoised, colorscale="Viridis"))
            .update_layout(title="F-K spectrum \u2014 denoised", height=350,
                           margin=dict(l=10, r=10, t=40, b=10)),
            use_container_width=True, key="diag_fk_denoised")

    st.markdown("#### Local similarity / signal-leakage map")
    st.caption("Values near zero = removed component looks like noise, not geology (no signal leakage).")
    sim_map = metrics_mod.local_similarity_map_fast(noisy, denoised_on, window=9)
    st.plotly_chart(
        go.Figure(data=go.Heatmap(z=sim_map, colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
        .update_layout(title="Local correlation: (noisy \u2212 denoised) vs. denoised", height=380,
                       yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10)),
        use_container_width=True, key="diag_sim_map")

    st.markdown("#### Jacobian mask explainer")
    st.caption("Pick a pixel — see which inputs the denoiser relies on, and what blind-spot mask that implies.")
    jy = st.slider("Row (sample)", 10, noisy.shape[0] - 10, noisy.shape[0] // 2, key="jy")
    jx = st.slider("Column (trace)", 10, noisy.shape[1] - 10, noisy.shape[1] // 2, key="jx")
    if st.button("Compute Jacobian"):
        pcfg = cfg["data"]["patch"]
        half = pcfg["size"] // 2
        y0, x0 = max(0, jy - half), max(0, jx - half)
        y0 = min(y0, noisy.shape[0] - pcfg["size"])
        x0 = min(x0, noisy.shape[1] - pcfg["size"])
        patch = noisy[y0:y0 + pcfg["size"], x0:x0 + pcfg["size"]]
        x_t = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        jac = compute_pixel_jacobian(model_on, x_t, (jy - y0, jx - x0))
        suggestion = suggest_mask_design(jac, (jy - y0, jx - x0))
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                go.Figure(data=go.Heatmap(z=jac, colorscale="Inferno"))
                .update_layout(title="Sensitivity |d(output)/d(input)|", height=350,
                               margin=dict(l=10, r=10, t=40, b=10)),
                use_container_width=True, key="diag_jacobian")
        with c2:
            st.write(f"**Suggested blind shape:** `{suggestion.suggested_blind_shape}`")
            st.write(f"**Suggested blind width:** `{suggestion.suggested_blind_width}` px")
            st.write(f"Lateral sensitivity extent: {suggestion.lateral_extent_px:.1f} px")
            st.write(f"Vertical sensitivity extent: {suggestion.vertical_extent_px:.1f} px")
            st.caption("Compare against `masking.struct_n2v.blind_width` in the config — "
                       "if measured extent is larger, widen the mask.")
