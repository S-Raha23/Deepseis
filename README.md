# DeepSeis — Fault-Preserving Self-Supervised Seismic Denoising & Auto-Interpretation

**MC²Plus – Oil India Ltd. – IIT Kharagpur Energy Innovation Challenge 2026**  
**Track 1: AI-Driven Subsurface Intelligence**

> Recovers a clean, interpretable subsurface image from noisy real seismic data **without ever needing clean training data**, while preserving faults — the structural features where hydrocarbons are trapped — then runs automated fault, horizon, and facies interpretation on top.

Trained and evaluated on the **F3 Netherlands block** (Alaudah et al. 2019, [Zenodo 3755060](https://zenodo.org/record/3755060)) — 401 inlines × 701 crosslines × 255 depth samples of real post-stack seismic amplitude with ground-truth lithostratigraphic facies labels.

On held-out test inlines with known noise added to real F3 geology, the denoiser reaches
**13.66 dB SNR / 0.900 SSIM**, against 10.98 dB for f-x deconvolution, 10.32 dB for this
project's previous denoiser, and 6.02 dB for the untouched input. Fault picks agree with
clean-section picks at **0.643 Dice**, up from 0.345 on the noisy input — so the denoising
is measurably helping the interpretation rather than erasing it. Fault preservation comes
from the estimator itself (a posterior mean that defers to the observed amplitude wherever
context fails to predict it) rather than from a tuned penalty term.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the real F3 Netherlands dataset (~1 GB from Zenodo)
python data/download_f3.py

# 3a. Train the denoiser. Validation mode: known noise is added to real F3
#     sections, the model sees only the contaminated data, and it is scored
#     against the originals — real geology with a reference.
python -m deepseis.denoise --config configs/denoise.yaml

# 3b. Field mode: train on the raw survey, which is the model that ships.
python -m deepseis.denoise --config configs/denoise.yaml --field \
       --run-dir runs/denoise_field

# 3c. Train the interpretation heads on the denoised survey. The facies head
#     trains across many inlines and is scored on held-out ones.
python -m deepseis.train --config configs/default.yaml \
       --blindspot runs/denoise_field/blindspot.pt

# 4. Score the denoiser on the held-out test block against classical baselines
python -m deepseis.evaluate --checkpoint runs/denoise/blindspot.pt --split test

#    ...and against the previous model, and a structure-term ablation:
python -m deepseis.train --config configs/default.yaml --train-legacy-denoiser
python -m deepseis.denoise --config configs/denoise.yaml --no-structure \
       --run-dir runs/denoise_nostruct
python -m deepseis.evaluate --checkpoint runs/denoise/blindspot.pt --split test \
       --legacy runs/default/denoiser.pt \
       --compare no_structure=runs/denoise_nostruct/blindspot.pt

# 5. Launch the dashboard
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

The facies head trains on denoised sections drawn from across the training block and is
scored on **held-out inlines**, using the same buffered split layout as the denoiser
(`deepseis/facies.py`). It previously reported 94.4% pixel accuracy on inline 0 — the one
inline it had been fitted on — which could not distinguish fitting from generalising in
either direction.

Held-out mean IoU is **0.892**, over **all six units**, on inlines 101–130 — a block the head
never saw, behind a 15-line buffer:

| Unit | Held-out IoU |
|---|---|
| Upper North Sea | 0.979 |
| Middle North Sea | 0.868 |
| Lower North Sea | 0.962 |
| Rijnland/Chalk | 0.877 |
| Jurassic | 0.826 |
| Triassic | 0.839 |

Every unit clears 0.82, including the two rarest. On the training block it reaches 97.8%
pixel accuracy and 0.940 mean IoU — the gap between that and 0.892 is what generalisation
costs, and it is small.

Getting that number to mean anything required fixing the split, and the story is worth
recording because two earlier versions of it were wrong in opposite directions.

F3's facies are **spatially segregated along the inline axis**: all six units occur only in
inlines 0–168, Triassic disappearing around 168 and Jurassic thinning to nothing by ~235.
The denoiser's survey-wide split (validation at 307–342) therefore lands entirely in a
region where two of the six units do not exist. Under that split they could not be scored at
all — so a first draft of this README reported them at **0.0 IoU**, an artefact of counting
absent-from-truth classes as zeroes, and a second draft reported their generalisation as
*unmeasured*, which was true but only because the split was wrong.

The facies split is now confined to the interval where the full stratigraphic column is
present (`train.facies_split_ranges`): train 0–85, validation 101–130, test 146–169, with
15-line buffers, and every partition verified to contain all six units. It is still a spatial
hold-out — just a smaller one that asks a question the data can answer. The units that
"couldn't generalise" score 0.826 and 0.839.

The same bug had a second effect worth recording, because it is the kind that flatters a
model rather than embarrassing it. Mean IoU was averaged over whichever classes were
non-NaN *that epoch*, so a model that gave up on a hard class saw it drop out of the
denominator instead of contributing a zero — the score went **up** for getting worse, jumping
from 0.573 to 0.739 between two epochs. The denominator is now pinned to the classes present
in the held-out truth (`facies.mean_iou_over`, tested in `tests/test_facies.py`).

One thing the fixed metric does *not* penalise, and which the report records separately so it
cannot hide: the head still predicts Jurassic and Triassic somewhere in a block where neither
occurs. Those are false positives, and excluding absent classes from mean IoU is the
conventional treatment, but `classes_absent_but_predicted` in `facies_metrics.json` names them
and the Facies tab flags them in place.

## The denoiser

### Why it was rebuilt

The previous denoiser was a masking-based Noise2Void model. Measured on F3 it removed
**1.8% of the input variance** and left `corr(input, output) = 0.992`, while deleting 70–75%
of the energy above 0.6 Nyquist and almost nothing below 0.4. That is a mild low-pass
filter, not a denoiser, and its headline leakage score of 0.334 was mostly earned by barely
touching the data — a filter that removes nothing scores a perfect 0.000 on that metric.

Three things caused it, and none were fixable by tuning:

- **The masking destroyed the training data.** `masking.mode` defaulted to `auto`, whose
  own estimator routed 76% of F3 patches to StructN2V; StructN2V then blinded a *full-height*
  column per masked pixel, so with 82 masked pixels in a 64×64 patch, **97.9% of the input
  was replaced by a random shuffle**. Three-quarters of training was noise-to-noise
  regression, and the only function that minimises that is a smooth low-order guess.
- **Supervision reached 2% of pixels.** Masked N2V computes its loss only at masked
  locations, and feeds a corrupted input at training time and a clean one at inference.
- **Everything was fitted on one inline.** `volume[0].T`, one of 401, then served across the
  whole survey — with no held-out data anywhere in the project.

### What replaced it

**A blind-spot architecture, not a blind-spot mask** (`models/blindspot.py`). The receptive
field is restricted so the prediction at a pixel provably cannot see that pixel, using
half-plane convolutions over the four rotations of the input (Laine et al. 2019). Nothing is
masked, the loss is evaluated at every pixel, and train and inference see identical inputs.
The blindness is verified exhaustively by autograd in `tests/test_blindspot.py` rather than
assumed — the previous blind spot was silently not blind, and nothing caught it.

**It predicts a distribution, then folds the observation back in** (`losses/nll.py`). The
network outputs a per-pixel Gaussian `(mu, sigma_p)` over the clean value given the
surroundings, and is trained by maximum likelihood on the noisy data. Crucially the output
is *not* `mu`: a blind-spot network can only ever produce `E[x | context excluding the pixel]`,
which is intrinsically over-smoothed because it throws away the most informative measurement
of the pixel — the pixel. Combining prior and observation through the noise model recovers
`E[x | context AND pixel]`:

```
x_hat = (mu · sigma_n² + y · sigma_p²) / (sigma_p² + sigma_n²)
```

This is what makes it fault-preserving, and it is a property of the estimator rather than a
penalty term. Where context predicts a sample well — conformable bedding — `sigma_p` is small
and the noise is removed. Where context fails — fault cuts, pinch-outs, channel margins,
exactly the features being interpreted — `sigma_p` is large and the observed amplitude is kept.

**Bias-free and normalisation-free.** Every convolution has `bias=False` and every activation
is positively homogeneous, so `f(a·x) = a·f(x)` exactly (Mohan et al. 2020). Amplitude-scale
train/serve mismatch becomes structurally impossible rather than a bug to be reintroduced,
and removing `GroupNorm` (which rescaled by whatever else was in the patch, destroying
relative amplitude) makes the network exactly translation equivariant. That in turn removes
the need for patch-and-average inference — itself a post-hoc smoothing operator — so a
section is denoised in a single pass.

**The noise level is measured, not learned** (`noise_estimate.py`). The training likelihood
constrains only `sigma_p² + sigma_n²`, so the split between "unpredictable" and "noise" is
close to unidentifiable — and that split is exactly what sets how much gets removed. A
partly-trained model here had the total variance right to within 15% while splitting it
almost backwards (`sigma_p² = 0.151` against a true 0.037; `sigma_n² = 0.115` against a true
0.187), which put 57% of the weight on the raw observation where the correct split puts 17%.
Measuring `sigma_n` from the F-K spectrum instead was worth **+3.1 dB**.

Two estimators are needed, because the obvious one fails silently on real data. White noise
is flat across the F-K plane, so the high-frequency/high-wavenumber corner holds a noise
floor and nothing else — accurate to 1% on sections with known added noise. But processed
seismic is band-limited on purpose: F3's corner sits five orders of magnitude below its
passband, so reading a noise level there measures the *anti-alias filter*, returns ~0.005
against a section amplitude of 0.87, and would turn the denoiser into an identity. The
fallback reads outside the geological dip fan within the passband, which survives
band-limiting. The choice between them is made by testing whether the corner's spectrum is
*flat* (a real noise floor, ratio 0.97–1.00 across a 30× range of noise levels) or still
rolling off (a stopband, 0.05).

### Evaluation

Three weaknesses in the old evaluation are fixed. Sections come from a held-out **test
block**, separated from training by a 30-line buffer — adjacent F3 inlines correlate at
**0.93**, so a random split would have made "held-out" meaningless. Scoring uses a
**reference on real geology**: known noise is added to real F3 sections, the model is trained
self-supervised on the contaminated data alone, and scored against the originals, so
PSNR/SSIM/SNR are exact on real structure rather than on a synthetic model of it. And the
controls are **baselines worth losing to** — f-x deconvolution and structure-oriented
smoothing, not just an identity and two blurs.

> Field mode (`--field`) trains on the raw survey with no injected noise. It is the model the
> dashboard serves, and it has no reference, so it is selected on held-out predictive
> likelihood. Not on leakage: that is minimised by removing nothing, and a field run selecting
> on it kept its epoch-0 checkpoint over every later one.

### Results

Held-out **test** block (inlines 373–400, behind two 30-line buffers), 8 sections, known
noise added to real F3 geology. `+/-` is the spread across sections. `faultDice` is the
agreement of the fault head's picks with its picks on the *clean* section — i.e. how much of
the interpretation survives.

| method | SNR dB | +/- | PSNR | SSIM | leakage | removed | resid. coh | faultDice |
|---|---|---|---|---|---|---|---|---|
| **blind-spot + per-section σₙ** | **13.79** | 0.96 | 34.27 | 0.901 | 0.114 | 0.199 | 0.162 | **0.654** |
| blind-spot + orthogonalization | 13.72 | 0.94 | 34.21 | 0.900 | **0.059** | 0.207 | 0.157 | 0.647 |
| blind-spot | 13.66 | 0.94 | 34.15 | 0.900 | 0.100 | 0.209 | 0.162 | 0.643 |
| f-x deconvolution | 10.98 | 0.75 | 31.47 | 0.848 | 0.370 | 0.134 | 0.178 | 0.597 |
| structure-oriented smoothing | 10.45 | 0.50 | 30.94 | 0.829 | 0.422 | 0.103 | 0.136 | 0.520 |
| 3×3 median | 10.39 | 0.60 | 30.87 | 0.831 | 0.179 | 0.212 | 0.194 | 0.565 |
| **previous denoiser (masking N2V)** | 10.32 | 0.43 | 30.81 | 0.807 | 0.148 | 0.260 | 0.186 | 0.531 |
| gaussian σ=1.0 | 10.25 | 0.52 | 30.74 | 0.851 | 0.436 | 0.212 | 0.236 | 0.515 |
| gaussian σ=0.5 | 9.61 | 0.06 | 30.09 | 0.803 | 0.604 | 0.040 | 0.138 | 0.496 |
| identity (removes nothing) | 6.02 | 0.02 | 26.51 | 0.690 | 0.000 | 0.000 | 0.000 | 0.345 |

**+3.34 dB over the previous denoiser** and **+2.68 dB over f-x deconvolution**, with SSIM
0.900 against 0.807 and 0.848. It removes 21% of the input variance at a leakage of 0.100,
where the previous model removed 26% at 0.148 — more removed *and* less of it was signal.

The fault-Dice column is the one that answers the project's actual claim. Fault picks on the
noisy section agree with the clean-section picks at 0.345; denoising raises that to **0.643**,
against 0.597 for f-x decon and 0.531 for the previous model. Denoising is helping the
interpretation rather than quietly erasing what is being interpreted, and that is now a
measured number rather than a qualitative overlay.

Two optional refinements were measured rather than assumed, and both earn a small keep:
re-estimating σₙ per section (+0.13 dB, since the true noise level varies 0.40–0.54 between
lines) and Chen–Fomel orthogonalization (leakage 0.100 → 0.059 with SNR still improving
slightly, so it is recovering signal rather than gaming its own metric).

### The structure-preservation terms do not earn their place

The project's signature feature used to be a "fault-preservation loss" with an ON/OFF toggle.
Rebuilt versions of both terms are in `losses/geology.py` — a coherence-trough penalty that
punishes *healing* a fault, and a dip-fan spectral term with a proper taper — and each is
tested to rise monotonically as a fault is blurred out and to charge nothing for removing
incoherent noise. They are nonetheless **off by default**, because an ablation on a matched
schedule says they do nothing:

| | SNR dB | SSIM | leakage | faultDice |
|---|---|---|---|---|
| structure terms **off** | **13.82** | 0.903 | 0.111 | 0.657 |
| structure terms **on** | 13.66 | 0.900 | 0.100 | 0.656 |

Identical seed, steps, and learning-rate curve, stopped at the same epoch; OFF was equal or
better at all six matched epochs. The gap is well inside the ±0.97 dB spread across sections,
so the honest reading is "no measurable benefit" rather than "OFF is better".

This is not really a failure of the terms — it is a sign they are redundant *here*. The
posterior-mean estimator already keeps the observed amplitude wherever context cannot predict
it, which is what a fault is, so a penalty for the same behaviour has little left to correct.
They may still pay on data with strong coherent noise, which is what the dip-fan term is
shaped for, so the code and its tests stay.

Reporting this cost the project its headline demo. It is the right answer anyway: the
previous ON/OFF comparison was 0.334 vs 0.360 leakage on a single seed with no error bars,
and the direction of that difference was not established either.

On the spectral profile: the blind-spot model keeps 0.94 / 0.84 of the two lowest F-K bands
and almost nothing above 0.6 Nyquist. That looks like a low-pass until you check where F3's
signal is — the survey is band-limited by its processing chain, with power falling off a
cliff past 0.4 Nyquist, so the band being emptied contains the added noise and nothing else.
The same profile on data whose signal reached those frequencies would be a problem.

### What field mode says about F3, which is worth stating plainly

Trained on the **raw** survey with nothing added, the same architecture removes about
**0.6% of the input variance**. That is not a failure of the model; it is what a calibrated
denoiser should do to this data. F3's published volume is migrated, stacked, commercially
processed seismic, and the measured incoherent noise level is correspondingly low. The
demonstration of denoising capability therefore has to come from the semi-synthetic control
above — real geology, known noise — and any figure claiming a dramatic visual improvement on
raw F3 should be treated with suspicion.

Field-mode training also stops early on its own: held-out predictive likelihood peaked at
epoch 2 and got worse for three consecutive epochs while the training likelihood kept
improving. The held-out selection correctly kept epoch 2. With very little noise to model,
there is not much for the network to learn beyond the geology it is already being shown.

> **Training budgets used here.** The reported semi-synthetic model is the best of 6 epochs
> (360 steps) on a CPU, selected on held-out SNR; it had plateaued (14.16 → 14.22 → 14.23 dB
> over its last three epochs). The ablation below is run on the identical schedule and
> stopped at the same epoch so the comparison is matched. These are not converged-to-death
> numbers — a longer run on a GPU would likely add a few tenths of a dB — but the margins
> over every baseline are large enough that more training does not change any conclusion.

---

## Dashboard — 6 tabs

| Tab | What you see |
|---|---|
| 🧮 **Denoise** | F3 input → denoised → **the difference section**, with the per-pixel uncertainty map and the leakage / energy-removed / residual-coherence numbers. Below it: a **controlled experiment** (add known noise to a held-out inline, denoise, compare against the original, with live sliders for the noise level) and the **full benchmark table** read straight from the evaluation report |
| 🧩 **Fault segmentation** | Fault-probability overlays on noisy vs. denoised F3 input |
| 🗺️ **F3 Survey explorer** | Inline slider — scrub through all 401 real F3 inlines live |
| 🌊 **Facies** | **Held-out** IoU per unit up front, then the in-sample map vs. published labels and error map — labelled as in-sample |
| 📤 **Export** | Download denoised section as SEG-Y or .npy |
| 🔬 **Diagnostics** | F-K spectrum, signal-leakage map, Jacobian sensitivity (legacy model) |

The Denoise tab is deliberately blunt about the fact that raw F3 barely changes: the app says
so in place, and puts the controlled experiment directly underneath so the capability can be
seen rather than asserted. The benchmark table is loaded from `runs/denoise/evaluation_final.json`
rather than typed in, so the app cannot quietly disagree with this README.

---

## What's implemented

### Core pipeline
- [x] Load real F3 `.npy` volume — SEG-Y (`.sgy`) and FaultSeg3D `.dat` also supported
- [x] Self-supervised **blind-spot denoiser** — architectural blind spot, verified by autograd
- [x] Per-pixel uncertainty and posterior-mean output — structure preservation from the estimator
- [x] Reference-free noise-level estimation, with detection of band-limited data
- [x] Bias-free, normalisation-free network; single-pass full-section inference
- [x] Multi-section training with buffered train/val/test splits along the inline axis
- [x] Structure-preservation terms — coherence-trough and dip-fan, each behaviourally tested
      (available, but **off by default**: ablated and found to give no measurable benefit)
- [x] Evaluation against f-x deconvolution, structure-oriented smoothing, and the previous model
- [x] FaultSeg-style fault segmentation — synthetic-trained, applied to real denoised F3 data
- [x] 6-class facies segmentation trained across the survey, scored on held-out inlines
- [x] SEG-Y export with in-dashboard download button
- [x] 6-tab Streamlit dashboard with a controlled-noise demo and the live benchmark table,
      covered by headless smoke tests (`tests/test_dashboard.py`)

### Stretch goals
- [x] Local signal-and-noise orthogonalization post-process (Chen & Fomel 2015)
- [x] Fault-pick agreement scoring — does denoising preserve *interpretability*, quantitatively
- [x] Physics-guided diffusion refiner (DDPM, Laplacian-smoothness prior)
- [x] Horizon tracking (amplitude-peak auto-tracking, fault-aware)
- [x] F-K spectrum viewer (before/after denoising)
- [x] Jacobian sensitivity map — applies to the legacy masking denoiser only; the current
      model's sensitivity to its own pixel is exactly zero by construction

The masking denoiser (`models/unet.py`, `masking/`) is retained so the previous model can be
loaded and scored beside the current one, which `deepseis.evaluate --legacy` does.

---

## Repository layout

```
deepseis/
├── configs/
│   ├── denoise.yaml              # blind-spot denoiser: data splits, model, losses, eval
│   └── default.yaml              # interpretation heads + legacy denoiser
├── data/
│   ├── download_f3.py            # downloads F3 from Zenodo, auto-patches config
│   └── samples/faultseg3d/       # bundled FaultSeg3D validation pair (.dat)
├── deepseis/
│   ├── io/                       # dataset.py (splits, sampling) · segy.py · synthetic.py
│   │                             #   noise.py · patches.py
│   ├── models/                   # blindspot.py · unet.py (legacy) · faultseg.py
│   │                             #   facies.py · diffusion.py
│   ├── losses/                   # nll.py · geology.py · reconstruction.py (legacy)
│   │                             #   edge_preserve.py (legacy) · frequency.py (legacy)
│   ├── masking/                  # noise2void.py · struct_n2v.py · jacobian_explain.py
│   ├── interpretation/           # horizon.py
│   ├── denoise.py                # train / infer the blind-spot denoiser
│   ├── noise_estimate.py         # reference-free sigma_n from the F-K spectrum
│   ├── baselines.py              # f-x decon · structure-oriented smoothing · blurs
│   ├── postprocess.py            # local signal-and-noise orthogonalization
│   ├── evaluate.py               # held-out scoring against every baseline
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

## What the tests actually check

The failures this project has had were not crashes — they were components that ran cleanly
while doing the opposite of what they claimed. A blind spot that was not blind. An edge loss
that rewarded *raising* gradient magnitude, so the model it supervised emitted more energy
than it was given. A masking mode that destroyed 98% of its own training data. Each survived
because nothing tested the property, only that the code ran and the loss fell.

So the suite asserts properties, not smoke:

| Test | Property |
|---|---|
| `test_blindspot.py` | Every pixel's prediction has **exactly zero** gradient w.r.t. that pixel — checked exhaustively, for both heads, all input channels, and odd section sizes. Plus: the network does use its context (a constant output would also be perfectly blind), and all four half-planes are reached. |
| `test_geology_losses.py` | Each structure term is zero on an untouched section, rises monotonically as a fault is blurred out, and does *not* charge for removing incoherent noise. |
| `test_baselines.py` | The dip estimator recovers dips it is given, and every baseline actually improves SNR — a weak baseline would flatter the model. |
| `test_noise_estimate.py` | The estimator recovers known noise levels, is invariant to signal amplitude, and detects band-limited data instead of reporting a filter stopband as silence. |
| `test_dataset.py` | Splits are disjoint and buffered, crosslines are truncated to the training block, normalisation is one global constant, and injected noise is fixed per section (resampling it would quietly turn the benchmark into the easier Noise2Noise problem). |
| `test_denoise.py` | Translation and scale equivariance, tiled inference matching single-pass, and checkpoints carrying their normalisation constants. |
| `test_evaluate.py` | Scoring touches only the requested split, the reference is the clean section, and the stored normalisation is used rather than recomputed. |
| `test_facies.py` | Held-out scoring uses unseen sections, patches combine by vote rather than by averaging class indices, and **mean IoU has a fixed denominator** so a model cannot raise its score by giving up on a hard class. |
| `test_masking.py` | The legacy blind spot never donates a pixel to itself, and StructN2V's blind region stays a bounded segment inside a hard budget. |
| `test_dashboard.py` | The page executes end to end with no exception, every tab renders, the export flow actually produces a file, and the controlled demo hits the benchmark result — which catches serving-path normalisation bugs that leave the page looking perfectly fine. |

Four of these caught real defects during the rebuild:

- the **dip estimator** used the standard `0.5·arctan2(2·jxy, jyy−jxx)` orientation shortcut,
  which returns the gradient direction rather than the structure direction — worth ~4 dB, and
  it was making the structure-oriented *baseline* look weak;
- the **orthogonalization update** was the widely quoted `s + w·n0` form, which does not
  actually orthogonalize (the projection form recovers leaked signal exactly where that one
  is break-even);
- **mean IoU had a moving denominator**, so the facies head's score rose when it gave up on a
  class;
- **double normalisation at three separate call sites** — the dashboard's controlled demo,
  the dashboard's main denoise path, and the facies head's training data. `serve_section`
  applies the checkpoint's stored scale, and each caller had already divided by roughly the
  same number. It is silent by construction: the network is scale equivariant so its own
  output absorbs the extra factor, but σₙ is a fixed physical noise level, so only the
  *ratio* shifts and the denoiser quietly under-removes. Measured cost: **3.17 dB** and 4×
  less noise removed, with the page rendering perfectly and no error raised anywhere. The
  facies head had been fitted on the doubly-scaled sections and scored 0.93 against its own
  broken input — and 0.13 once served correctly.

That last one is the archetype for this whole project, which is why `serve_section` now
carries a runtime guard that warns when it is handed a section whose amplitude is suspiciously
close to unity for the checkpoint it is being served by. The failures here do not crash. They
return a worse number that still looks entirely plausible.

---

## References

Krull et al. 2019 (Noise2Void) · Laine et al. 2019 (blind-spot architecture, Bayesian
combination) · Batson & Royer 2019 (Noise2Self) · Mohan et al. 2020 (bias-free denoising
networks) · Canales 1984 (f-x deconvolution) · Chen & Fomel 2015 (local signal-and-noise
orthogonalization) · Wu et al. 2019 (FaultSeg3D) · Alaudah et al. 2019 (F3 benchmark) ·
Birnie et al. 2021/2022 · Li et al. 2024/2025  
See `DeepSeis_Project_Spec.md` §15 for the full bibliography.
