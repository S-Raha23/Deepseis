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

# 3. Train on real F3 data (resumable)
python -m deepseis.train --config configs/default.yaml
# Add --resume to continue from the last checkpoint

# 4. Launch the dashboard
streamlit run app/dashboard.py
# Open http://localhost:8501
```

Optionally add the second survey (Parihaka, New Zealand) to the sidebar selector:

```bash
python data/download_parihaka.py                             # fetch + normalize
python -m deepseis.train --dataset parihaka                  # train it (own checkpoints)
python data/mirror_to_hf.py --dataset parihaka               # publish to your HF account, once
```

Run the test suite: `pytest tests/`

---

## Datasets — two real surveys

The dashboard's **Survey** selector (top of the sidebar) switches every tab between two
independent public surveys. **F3 Netherlands is the default.**

| | F3 Netherlands | Parihaka |
|---|---|---|
| Basin | Dutch North Sea | Taranaki Basin, New Zealand |
| Geometry | 401 × 701 × 255 | 782 × 590 × 1006 |
| Facies classes | 6 | 6 |
| Source | [Zenodo 3755060](https://zenodo.org/record/3755060) (Alaudah et al. 2019) | [Mendeley 10.17632/gnvyh3msrj.1](https://data.mendeley.com/datasets/gnvyh3msrj/1), CC BY 4.0 (SEAM/SEG 2020 facies challenge) |
| Fetch with | `python data/download_f3.py` | `python data/download_parihaka.py` |

Both are stored in one canonical layout — `(n_inlines, n_crosslines, n_samples)` float32
amplitude plus `int8` facies labels numbered `0..5` — so no tab needs per-survey
special-casing. The download scripts do the normalization on the way in; Parihaka ships as
`(sample, inline, crossline)` with labels numbered `1..6`, and both deviations are declared
in the `datasets:` block of `configs/default.yaml` rather than hardcoded.

Amplitude scale is normalized at load time rather than on disk. Surveys are published at
very different scales — F3 arrives pre-scaled to `[-1, 1]` (std ≈ 0.23), Parihaka arrives as
raw amplitude (std ≈ 366, a factor of ~1600) — so `train.prepare_data` divides the field
volume by its standard deviation before patching. The learning rate and the
`lambda_edge`/`lambda_freq` balance are tuned for unit-scale data, and `infer.py` and the
dashboard both normalize the same way, so this is what keeps training and inference on the
same footing.

**Adding Parihaka to the dashboard** takes two commands. The dashboard cannot read
`data/raw/` (gitignored, and absent on Streamlit Cloud), so each survey needs a network
mirror — the same way F3 already works:

```bash
python data/download_parihaka.py            # fetch + normalize (~465 MB after decimation)
python data/mirror_to_hf.py --dataset parihaka   # publish to your HF account, once
```

Then pick **Parihaka** in the sidebar. The dashboard resolves each survey **local first,
Hugging Face second**, so a machine that has already run the download script never
re-downloads.

> `download_parihaka.py` keeps every 4th inline by default (`--inline-stride`). This is
> lossless for what DeepSeis actually does: training reads inline 0 only, and inline 0 is
> preserved at every stride, so a model trained on the decimated volume is identical to one
> trained on the full survey. Use `--inline-stride 1` for the complete ~1.9 GB volume.

**Why a second survey matters:** the denoiser is self-supervised, so it adapts to *each
survey's own* noise rather than to a fixed training distribution. Two basins with different
acquisition and different noise character is the cross-survey generalization argument in
§12 of the spec, demonstrated rather than asserted.

### Per-survey checkpoints

Because the denoiser is fitted to one survey's noise, **each survey has its own trained
models** rather than sharing a single set:

| Survey | `run_dir` | Train with |
|---|---|---|
| F3 Netherlands | `runs/default/` | `python -m deepseis.train --dataset f3` |
| Parihaka | `runs/parihaka/` | `python -m deepseis.train --dataset parihaka` |

`--dataset` reads the `datasets:` registry and sets the volume path, the label path, the
facies class count and the output directory for you — no config editing. Selecting a survey
in the sidebar loads that survey's checkpoints automatically. If a survey has no checkpoints
yet, the dashboard says so and offers the command rather than silently starting a
tens-of-minutes CPU training run inside a page load.

### Running the dashboard on a memory-capped host

The hosted app is memory-limited, and two things dominated its footprint. Both are
fixed; the numbers are measured RSS for a full 6-tab render:

| | Before | After |
|---|---|---|
| First render (F3) | 1413 MB | 506 MB |
| After switching to Parihaka | 3378 MB | 616 MB |

- **Inference is batched** (`INFERENCE_BATCH` in `deepseis/train.py`). Running a whole
  section's patches through the network in one pass cost 1.29 GB for Parihaka, because a
  U-Net skip connection keeps an `(N, C, 64, 64)` tensor alive from encoder to decoder, so
  peak scales with patch count. Batching caps it at a fixed ~75 MB. The network is
  per-patch, so output is bit-identical.
- **Sections are read through a memory map** (`read_inline` in `app/dashboard.py`). The
  dashboard displays one inline at a time, but previously loaded the whole cube to get it —
  573 MB read to use 0.7 MB.

### Facies results, and what they do and don't mean

| | F3 Netherlands | Parihaka |
|---|---|---|
| Pixel accuracy on the **training** inline | 83.5% | 78.9% |
| Mean IoU | 55.4% | 49.6% |
| Best class | Upper North Sea, 90.6% IoU | Basement/Other, 90.3% IoU |
| Failing class | Triassic — never predicted (2.3% of section) | Slope Valley — predicted over 11.4%, true 0% |

**The headline accuracies are goodness of fit, not generalization.** `prepare_data` fits the
facies head on inline 0 alone, and the dashboard scores it on that same inline. Measured on
held-out inlines of Parihaka (40, 98, 150, 195), accuracy falls to **41.9–50.1%** — above
chance for six classes, but not a validated classifier.

Both failures are class imbalance from single-inline training, in opposite directions. F3's
Triassic occupies 2.3% of inline 0 and the head never predicts it at all. Parihaka's Slope
Valley has essentially zero pixels in inline 0, so its output unit is never trained and wins
in ambiguous regions — visible as a large false region in the upper section. Training the
facies head across multiple inlines is the fix for both; the tab states this caveat inline so
the number is never read as more than it is.

**Facies classes**
`F3` — 0 Upper North Sea · 1 Middle North Sea · 2 Lower North Sea · 3 Rijnland/Chalk · 4 Jurassic · 5 Triassic
`Parihaka` — 0 Basement/Other · 1 Slope Mudstone A · 2 Mass Transport Deposit · 3 Slope Mudstone B · 4 Slope Valley · 5 Submarine Canyon System

The denoiser is **self-supervised** — it trains directly on the noisy amplitude of whichever
survey you point it at, learning that survey's noise characteristics with no clean reference
needed.

FaultSeg trains on a freshly generated synthetic volume and is applied to the real denoised data — exactly the workflow from the FaultSeg3D paper (Wu et al. 2019). No real fault ground-truth exists for either survey, so quantitative Dice is not reported; qualitative fault-probability overlays are shown in the dashboard.

> To use synthetic data instead, set `data.use_synthetic: true` in `configs/default.yaml`. This enables exact PSNR/SSIM/Dice scoring against a known-clean reference.

---

## Dashboard — 6 tabs

Every tab operates on whichever survey is selected in the sidebar.

| Tab | What you see |
|---|---|
| 🧮 **Denoise** | Noisy → denoised, fault-preservation ON/OFF toggle side-by-side |
| 🧩 **Fault segmentation** | Fault-probability overlays on noisy vs. denoised input |
| 🗺️ **Survey explorer** | Inline slider — scrub through the selected survey's inlines live |
| 🌊 **Facies** | Predicted vs. ground-truth facies on the denoised inline, with per-class IoU |
| 📤 **Export** | Download the denoised section as SEG-Y or .npy |
| 🔬 **Diagnostics** | F-K spectrum, signal-leakage map, Jacobian mask explainer |

---

## What's implemented

### Core pipeline
- [x] Load real `.npy` volumes — SEG-Y (`.sgy`) and FaultSeg3D `.dat` also supported
- [x] Two real surveys — F3 Netherlands and Parihaka — switchable from the sidebar
- [x] Self-supervised blind-spot denoiser — Noise2Void and Structured N2V — trained on the real survey
- [x] Fault-preservation loss (edge-preservation + F-K high-frequency terms), ON/OFF comparison
- [x] FaultSeg-style fault segmentation — synthetic-trained, applied to real denoised data
- [x] 6-class facies segmentation — trained on real facies labels
- [x] SEG-Y export with in-dashboard download button
- [x] 6-tab Streamlit dashboard, inline slider across the selected survey

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
├── configs/default.yaml          # hyperparameters + the `datasets:` survey registry
├── data/
│   ├── download_f3.py            # downloads F3 from Zenodo, auto-patches config
│   ├── download_parihaka.py      # downloads Parihaka, normalizes axes + label base
│   ├── mirror_to_hf.py           # publishes a prepared survey to Hugging Face
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
