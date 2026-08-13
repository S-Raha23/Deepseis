# DeepSeis — Fault-Preserving Self-Supervised Seismic Denoising & Auto-Interpretation

**MC²Plus – Oil India Ltd. – IIT Kharagpur Energy Innovation Challenge 2026**  
**Track 1: AI-Driven Subsurface Intelligence**

> Recovers a clean, interpretable subsurface image from noisy real seismic data **without ever needing clean training data**, while explicitly preserving faults — the structural features where hydrocarbons are trapped — then runs automated fault, horizon, and facies interpretation on top.

Trained and evaluated on the **F3 Netherlands block** (Alaudah et al. 2019, [Zenodo 3755060](https://zenodo.org/record/3755060)) — 401 inlines × 701 crosslines × 255 depth samples of real post-stack seismic amplitude with ground-truth lithostratigraphic facies labels.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the real F3 Netherlands dataset (~1 GB from Zenodo)
python data/download_f3.py

# 3. Train on real F3 data  (~72 s on CPU, resumable)
python -m deepseis.train --config configs/default.yaml
# Add --resume to continue from the last checkpoint

# 4. Launch the dashboard
streamlit run app/dashboard.py
# Open http://localhost:8501
```

Run the test suite: `pytest tests/`

---

## Dataset — F3 Netherlands block

| File | Shape | Contents |
|---|---|---|
| `data/raw/f3/data/train/train_seismic.npy` | (401, 701, 255) | Real post-stack seismic amplitude |
| `data/raw/f3/data/train/train_labels.npy` | (401, 701, 255) | 6-class facies labels (0–5) |

The denoiser is **self-supervised** — it trains directly on the noisy F3 amplitude, learning this survey's noise characteristics with no clean reference needed.

FaultSeg trains on a freshly generated synthetic volume and is applied to the real denoised data — exactly the workflow from the FaultSeg3D paper (Wu et al. 2019). No real fault ground-truth exists for F3, so quantitative Dice is not reported; qualitative fault-probability overlays are shown in the dashboard.

**F3 facies classes (Alaudah et al. 2019):**  
`0` Upper North Sea · `1` Middle North Sea · `2` Lower North Sea · `3` Rijnland/Chalk · `4` Jurassic · `5` Triassic

> To use synthetic data instead, set `data.use_synthetic: true` in `configs/default.yaml`. This enables exact PSNR/SSIM/Dice scoring against a known-clean reference.

---

## Dashboard — 6 tabs

| Tab | What you see |
|---|---|
| 🧮 **Denoise** | F3 noisy → denoised, fault-preservation ON/OFF toggle side-by-side |
| 🧩 **Fault segmentation** | Fault-probability overlays on noisy vs. denoised F3 input |
| 🗺️ **F3 Survey explorer** | Inline slider — scrub through all 401 real F3 inlines live |
| 🌊 **Facies** | Predicted 6-class lithostratigraphic map on the denoised inline |
| 📤 **Export** | Download denoised section as SEG-Y or .npy |
| 🔬 **Diagnostics** | F-K spectrum, signal-leakage map, Jacobian mask explainer |

---

## What's implemented

### Core pipeline
- [x] Load real F3 `.npy` volume — SEG-Y (`.sgy`) and FaultSeg3D `.dat` also supported
- [x] Self-supervised blind-spot denoiser — Noise2Void and Structured N2V — trained on the real survey
- [x] Fault-preservation loss (edge-preservation + F-K high-frequency terms), ON/OFF comparison
- [x] FaultSeg-style fault segmentation — synthetic-trained, applied to real denoised F3 data
- [x] 6-class facies segmentation — trained on real F3 Alaudah 2019 labels
- [x] SEG-Y export with in-dashboard download button
- [x] 6-tab Streamlit dashboard, inline slider across all 401 real F3 inlines

### Stretch goals
- [x] Structured N2V for coherent noise with automatic noise-type estimator
- [x] Physics-guided diffusion refiner (DDPM, Laplacian-smoothness prior)
- [x] Horizon tracking (amplitude-peak auto-tracking, fault-aware)
- [x] F-K spectrum viewer (before/after denoising)
- [x] Jacobian mask explainer — sensitivity map + blind-mask-shape recommendation

---

## Repository layout

```
deepseis/
├── configs/default.yaml          # all hyperparameters and F3 data paths
├── data/
│   ├── download_f3.py            # downloads F3 from Zenodo, auto-patches config
│   └── samples/faultseg3d/       # bundled FaultSeg3D validation pair (.dat)
├── deepseis/
│   ├── io/                       # segy.py · synthetic.py · noise.py · patches.py
│   ├── masking/                  # noise2void.py · struct_n2v.py · jacobian_explain.py
│   ├── models/                   # unet.py · faultseg.py · facies.py · diffusion.py
│   ├── losses/                   # reconstruction.py · edge_preserve.py · frequency.py
│   ├── interpretation/           # horizon.py
│   └── train.py · infer.py · metrics.py
├── app/dashboard.py              # Streamlit 6-tab dashboard
├── Dockerfile                    # builds and launches the full pipeline in one command
└── tests/                        # pytest test suite
```

---

## Docker deployment

```bash
docker build -t deepseis .
docker run -p 8501:8501 deepseis
# Open http://localhost:8501
```

The container automatically downloads F3 data, trains all models, then launches the dashboard.

---

## How the denoiser convergence problem was solved

Standard N2V masks ~0.2% of pixels per step. On a single real inline with a modest CPU budget, that gives too little gradient signal to converge properly.

Fixes:
1. **Higher mask fraction** — 0.02 → 0.12 (more supervised pixels per step)
2. **Flip augmentation** — h/v flips give ~4× effective patches from one inline
3. **Smaller network** — `base_channels: 16, depth: 2` converges faster on limited data
4. **Tuned loss weights** — `lambda_edge: 0.04, lambda_freq: 0.02`; higher values (0.15/0.10) dominated the reconstruction loss and stalled convergence
5. **Checkpoint/resume** — `--resume` flag for long CPU runs (`training.checkpoint_every` epochs)

---

## References

Krull et al. 2019 (Noise2Void) · Batson & Royer 2019 (Noise2Self) · Wu et al. 2019 (FaultSeg3D)  
Alaudah et al. 2019 (F3 benchmark) · Birnie et al. 2021/2022 · Li et al. 2024/2025  
See `DeepSeis_Project_Spec.md` §15 for the full bibliography.
