# DeepSeis: Fault-Preserving Self-Supervised Seismic Denoising and Automated Interpretation

**Project report** · MC²Plus × Oil India Ltd. × IIT Kharagpur Energy Innovation Challenge 2026,
Track 1: AI-Driven Subsurface Intelligence · Soubhagya Raha, IIT Kanpur

[README](README.md) · [Live demo](https://sraha23-deepseis.static.hf.space) ·
[Data and trained models](https://huggingface.co/datasets/SRaha23/Seismic_Data)

---

## Summary

Seismic images guide decisions worth hundreds of millions of dollars, and every one of them
carries noise. Removing that noise is hard for two reasons. There is no clean version of real
seismic data to learn from. And the features that matter most, faults, look like noise to any
filter that smooths.

DeepSeis is a self-supervised denoiser that learns from the noisy survey alone. It is built
around a single idea: predict each sample from its surroundings, report how confident that
prediction is, and blend it with the original recording according to that confidence. Where
the geology is smooth and predictable, the noise is removed. Where it is not, at faults,
pinch-outs and channel edges, the recording is kept. No tuned penalty term is involved.

On held-out lines of the real F3 Netherlands survey with known noise added, DeepSeis reaches
**13.66 dB SNR and 0.900 SSIM**, against 10.98 dB and 0.848 for f-x deconvolution and 6.02 dB
for the noisy input. Fault picks on its output agree with picks on the clean section at a
Dice score of **0.649**, up from **0.310** on the noisy input. A facies classifier trained on
the denoised survey reaches **0.892 mean IoU on unseen lines across all six
lithostratigraphic units**. The system runs offline on a CPU, reads and writes SEG-Y, and is
covered by 179 tests that check properties rather than just whether the code runs.

---

## 1. The problem

### 1.1 Why seismic noise matters

Seismic reflection surveying sends sound into the ground and records the echoes from each
rock boundary. Processed and stacked, the recordings form an image of the layers below, and
that image is the most expensive input to an exploration decision. Hydrocarbons collect
against structural traps, most often **faults**, where rock layers have broken and slid past
each other. A misplaced fault means a misplaced well, and a dry exploration well costs
anywhere from crores onshore to more than $100 million offshore.

Seismic data carries two kinds of noise:

- **Random (incoherent) noise:** ambient, electronic and environmental noise, uncorrelated
  from trace to trace.
- **Coherent noise:** structured artefacts such as ground roll, linear noise, multiples and
  acquisition footprint. It is harder to remove because it looks like signal.

### 1.2 Three obstacles

1. **No clean labels exist.** Supervised denoisers learn from matched noisy and clean pairs.
   A real survey cannot be re-recorded with the noise switched off, so supervised models are
   trained on synthetic data and generalise poorly to real surveys.
2. **Denoising destroys the geology.** Smoothing removes high-frequency, discontinuous
   content, which is exactly where faults, pinch-outs and thin beds live. Processing teams
   therefore filter gently and accept the residual noise.
3. **Interpretation is slow.** Even on a clean image, a geoscientist picks faults and horizons
   by hand across hundreds of lines, which takes weeks per survey.

### 1.3 What DeepSeis contributes

- A **blind-spot denoiser** that trains on the noisy survey alone and provably cannot see the
  sample it predicts.
- A **posterior-mean output** that blends prediction and recording by confidence. This is what
  preserves faults, and it needs no penalty term.
- **Reference-free measurement of the noise level**, with detection of band-limited processed
  data, where the standard estimator fails silently.
- An **evaluation design** that gets real geology and an exact reference at the same time,
  with spatially buffered splits and strong classical baselines.
- **Automated interpretation** on the denoised output (faults, facies and horizons), with a
  quantitative measure of whether denoising helped it.
- A **property-based test suite** aimed at the failures that matter here: plausible wrong
  numbers, not crashes.

---

## 2. Data

### 2.1 The F3 Netherlands survey

F3 is a public 3D marine survey from the Dutch sector of the North Sea, released by dGB Earth
Sciences. DeepSeis uses the benchmark training volume of Alaudah et al. (2019): **401 inlines ×
701 crosslines × 255 time samples** of post-stack, migrated seismic amplitude, with a
lithostratigraphic interpretation in six units:

| Class | Unit |
|---|---|
| 0 | Upper North Sea group |
| 1 | Middle North Sea group |
| 2 | Lower North Sea group |
| 3 | Rijnland / Chalk |
| 4 | Jurassic |
| 5 | Triassic |

The volume and labels are published on Zenodo (record 3755060). Copies, together with every
trained checkpoint, are mirrored on Hugging Face so the dashboard can start from an empty
machine.

### 2.2 Splits for the denoiser

Inlines are split along the survey into **train (0–276), validation (307–342) and test
(373–400)**, separated by **30-line buffers**. Adjacent F3 inlines correlate at about 0.93, so
a random split would put near-copies of the test data into training and make "held out"
meaningless. Crossline sections used in training are truncated to the training block for the
same reason. Amplitudes are normalised by one global constant, fixed at training time and
stored inside the checkpoint, so training and serving can never disagree about scale.

### 2.3 Splits for the facies classifier

The facies units are not spread evenly across the survey. All six occur only in inlines
0–168: the Triassic disappears around inline 168 and the Jurassic thins out by about 235. A
survey-wide split would put validation where two units do not exist, so they could not be
scored at all. The facies split is therefore confined to the interval where the full
stratigraphic column is present: **train 0–85, validation 101–130, test 146–169, with 15-line
buffers**. Every partition is checked to contain all six units.

---

## 3. The denoiser

### 3.1 Self-supervision through an architectural blind spot

Self-supervised denoising rests on one observation. If noise is independent from sample to
sample, a network that predicts each sample from its *neighbours only* cannot learn to
reproduce the noise. It can only learn the part of the signal that its surroundings predict,
which is the geology (Krull et al. 2019; Batson & Royer 2019).

The common way to build this is masking (Noise2Void): hide random pixels in the input and
train the network to fill them in. That supervises only the masked pixels, about 2% of each
patch, and shows the network a corrupted input during training but a clean one at inference.
DeepSeis instead builds the blind spot into the architecture (Laine et al. 2019). The input is
processed in four rotated copies, each by a network of half-plane convolutions whose receptive
field covers one half-plane and strictly excludes the centre. Combined, the four branches see
every neighbour of a sample but never the sample itself. Nothing is masked, the loss is
computed at every pixel, and training and inference see identical inputs. The blindness is
verified exhaustively by automatic differentiation in the test suite (§7), not assumed.

### 3.2 Predicting a distribution, then folding the observation back in

The network does not output a single value. For every sample it outputs a Gaussian over the
clean value given the surroundings, with mean μ and standard deviation σp. The recording is
modelled as the clean value plus Gaussian noise of standard deviation σn, and the network is
trained by maximising the likelihood of the noisy recording y:

```
y ~ N(μ, σp² + σn²)
```

A blind-spot network on its own can only produce E[x | surroundings]. That estimate is
inherently over-smooth, because it ignores the most informative measurement of a sample: the
sample itself. DeepSeis therefore outputs the **posterior mean**, which combines the
prediction with the recording:

```
x̂ = (μ · σn² + y · σp²) / (σp² + σn²)
```

The weight on the recording is σp² / (σp² + σn²). In conformable bedding, the surroundings
predict a sample well, σp is small, and the prediction dominates, so the noise is removed. At
a fault, a pinch-out or a channel edge, the surroundings fail to predict the sample, σp is
large, and the output falls back to the recording. **Fault preservation is a property of the
estimator, not of a penalty term.** The same formula runs at every sample, and there is no
weight to tune. Because the output is always a blend of a data-driven prediction and a real
measurement, it cannot synthesise structure that is absent from both.

The posterior standard deviation, √(σp²σn² / (σp² + σn²)), is reported as a per-sample
uncertainty map. It highlights where the denoiser is guessing, which on seismic is where the
structure is complex.

### 3.3 Measuring the noise level instead of learning it

The likelihood above constrains only the *total* variance σp² + σn². How that total splits
between "the surroundings could not predict this" and "this is noise" is close to
unidentifiable from the training objective, yet that split sets how much the denoiser
removes. In one partly trained model, the total variance was right to within 15% but the split
was almost backwards (σp² = 0.151 against a true 0.037; σn² = 0.115 against a true 0.187).
That put 57% of the weight on the raw recording where the correct split puts 17%.

DeepSeis therefore fixes σn before training, by measuring it from the data. σn is shared
across traces but allowed to vary with two-way time, because signal amplitude and noise
character change with depth. Measuring σn instead of learning it improved the result by
**3.1 dB**.

The measurement needs two estimators, because the obvious one fails silently on real data:

- **Spectral corner.** White noise is flat across the frequency–wavenumber (F-K) plane, so the
  high-frequency, high-wavenumber corner contains only a noise floor. On sections with known
  added noise, this estimate is accurate to within 1%.
- **Outside the dip fan.** Processed seismic is band-limited on purpose. On F3, the spectral
  corner sits five orders of magnitude below the passband, so the corner estimator measures
  the anti-alias filter instead of the noise. It returns about 0.005 against a section
  amplitude of 0.87, which would turn the denoiser into an identity. The fallback reads the
  spectrum within the passband but outside the range of dips that geology occupies.

The choice between them is automatic: DeepSeis tests whether the corner of the spectrum is
**flat**, which indicates a real noise floor (ratio 0.97–1.00 across a 30-fold range of noise
levels), or **still rolling off**, which indicates a filter stopband (ratio about 0.05).

### 3.4 A network that cannot be fooled by loudness

Every convolution in the denoiser has no bias term, and every activation is positively
homogeneous, so the network satisfies f(a·x) = a·f(x) exactly (Mohan et al. 2020). It has no
normalisation layers. Two consequences follow:

- **Scale equivariance.** A survey recorded at a different gain needs no recalibration, and an
  amplitude mismatch between training and serving becomes structurally impossible.
- **Translation equivariance.** Normalisation layers rescale each patch by its contents, so a
  sample's output depends on which window it falls in. Without them, a whole section is
  denoised in a single pass, with no patch-and-average step, which would itself act as a
  smoothing filter.

The denoiser has **1.23 million parameters** and trains on a laptop CPU.

### 3.5 Training and model selection

The same architecture is trained in two modes:

- **Validation mode.** Known noise is added to real sections (§5.1). The model trains on the
  noisy versions only and is selected on held-out SNR against the originals. Every
  benchmark number in this report comes from this mode.
- **Field mode.** The model trains on the raw survey with nothing added. This is the model the
  dashboard serves. There is no reference in this mode, so it is selected on the predictive
  likelihood of held-out lines. It is not selected on signal leakage, because a model that
  removes nothing scores a perfect zero on leakage.

The reported validation-mode model is the best of 6 epochs (360 steps) on a CPU. It had
plateaued, with validation SNR at 14.16, 14.22 and 14.23 dB over its last three epochs. Longer
training on a GPU would likely add a few tenths of a decibel. That would not change any
conclusion, because the margins over every baseline are much larger.

### 3.6 Optional refinements

Two refinements were measured rather than assumed, and both earn a small place:

- **Per-section noise level.** The true noise level varies between 0.40 and 0.54 across test
  lines. Re-estimating σn for each section gains 0.13 dB.
- **Local signal-and-noise orthogonalization** (Chen & Fomel 2015) recovers signal that
  leaked into the removed component. Leakage drops from 0.100 to 0.059 while SNR still rises
  slightly, so it recovers signal rather than gaming its own metric.

### 3.7 Structure-preservation terms

Two explicit structure terms are also implemented. One is a coherence-trough penalty that
charges for healing a fault. The other is a dip-fan spectral term that penalises energy
outside the dips geology occupies. Each is tested to be zero on an untouched section, to rise
monotonically as a fault is blurred out, and to charge nothing for removing incoherent noise.
A matched ablation (§6.4) found no measurable benefit, so they are **off by default**. The
published checkpoints were trained with them on, which makes no measurable difference.

---

## 4. Automated interpretation

### 4.1 Faults

F3 has no published fault interpretation, so the fault network follows the FaultSeg3D
workflow (Wu et al. 2019), in 2D. It trains on synthetic sections with folded horizons and
normal faults, where the fault labels are exact, and is then applied to real data. The
network is a U-Net with about 0.48 million parameters. Faults are picked where the predicted
probability is at least 0.5.

### 4.2 Facies

A second U-Net of the same size classifies each sample of a denoised section into the six F3
units. It is trained on denoised sections from the facies training block (§2.3). Overlapping
patch predictions are combined by **majority vote**, not by averaging class indices. Because
F3's units are stacked in stratigraphic order, averaging a 2 and a 4 would invent a 3, a
spurious band of Rijnland/Chalk along every boundary between them.

Mean IoU is computed over a **fixed set of classes**: those present in the held-out truth.
Averaging only over the classes a model still predicts lets the score rise when the model
gives up on a hard class. Classes that are absent from the held-out block but predicted
anyway are reported separately as false positives.

### 4.3 Horizons

Horizons are tracked by amplitude-peak auto-tracking: a horizon is seeded on a strong peak and
followed to the nearest peak of the same polarity on each next trace. Where the path crosses a
fault, the search window widens, so the horizon can step across the fault with a throw rather
than being forced through it. Tracking works on the denoised section. On raw noisy data it
often locks onto noise peaks.

---

## 5. Evaluation protocol

### 5.1 Real geology with an exact reference

Real data has real geology but no answer key; synthetic data has an answer key but no real
geology. DeepSeis gets both by adding **known noise to real held-out F3 sections**:

- random noise with a standard deviation of 0.5 times the section's own;
- three coherent events with a peak amplitude of 0.35 times the section's standard deviation:
  steeply dipping, low-frequency arrivals outside the geological dip fan, resembling ground
  roll.

The model trains on contaminated data only and is scored against the originals. Each
section's noise is fixed by a seed derived from its identity. Drawing new noise every epoch
would show the network the same geology under independent noise realisations, quietly turning
the benchmark into the easier Noise2Noise problem.

### 5.2 Metrics

| Metric | What it measures | Read it with |
|---|---|---|
| SNR, PSNR | Error against the original section | — |
| SSIM | Structural similarity to the original | — |
| Signal leakage | Local correlation between what was removed and what was kept: how much geology was thrown away | Energy removed. A filter that removes nothing scores a perfect 0. |
| Energy removed | Fraction of the input variance taken out | Leakage |
| Residual coherence | How organised the removed component is. Random noise is incoherent. | — |
| Output / input std | Whether the output carries less energy than the input, as a denoiser's should | — |
| Spectral retention | Energy kept in each radial F-K band | Where the survey's signal actually lies |
| Fault Dice | Agreement of fault picks on the output with picks on the clean section | It measures what noise and denoising do to the fault network's picks, not agreement with an interpreter |

### 5.3 Baselines

The baselines are meant to be hard to beat: **f-x deconvolution** (Canales 1984), the
standard production method; **structure-oriented smoothing** along the local dip; a 3×3
median filter; Gaussian blurs; the project's earlier **masking-based Noise2Void** denoiser; and
the identity, which removes nothing. Each baseline is tested to improve SNR on its own, since
a weak baseline would flatter the model.

---

## 6. Results

### 6.1 Denoising benchmark

Held-out test block (inlines 373, 376, 380, 384, 388, 392, 396 and 400). The ± value is the
standard deviation across sections. Every row is produced by `deepseis.evaluate`.

| Method | SNR (dB) | ± | PSNR | SSIM | Leakage ↓ | Removed | Resid. coherence | Fault Dice |
|---|---|---|---|---|---|---|---|---|
| **DeepSeis** | **13.66** | 0.94 | 34.15 | **0.900** | 0.100 | 0.209 | 0.162 | 0.649 |
| DeepSeis + per-section σn | 13.79 | 0.96 | 34.27 | 0.901 | 0.114 | 0.199 | 0.162 | 0.655 |
| DeepSeis + orthogonalization | 13.72 | 0.94 | 34.21 | 0.900 | **0.059** | 0.207 | 0.157 | **0.661** |
| f-x deconvolution | 10.98 | 0.75 | 31.47 | 0.848 | 0.370 | 0.134 | 0.178 | 0.620 |
| Structure-oriented smoothing | 10.45 | 0.50 | 30.94 | 0.829 | 0.422 | 0.103 | 0.136 | 0.545 |
| 3×3 median | 10.39 | 0.60 | 30.87 | 0.831 | 0.179 | 0.212 | 0.194 | 0.566 |
| Earlier masking denoiser (Noise2Void) | 10.32 | 0.43 | 30.81 | 0.807 | 0.148 | 0.260 | 0.186 | 0.573 |
| Gaussian σ = 1.0 | 10.25 | 0.52 | 30.74 | 0.851 | 0.436 | 0.212 | 0.236 | 0.484 |
| Gaussian σ = 0.5 | 9.61 | 0.06 | 30.09 | 0.803 | 0.604 | 0.040 | 0.138 | 0.517 |
| Noisy input (identity) | 6.02 | 0.02 | 26.51 | 0.690 | 0.000 | 0.000 | 0.000 | 0.310 |

DeepSeis improves on **f-x deconvolution by 2.68 dB** and on the earlier masking denoiser by
3.34 dB, with SSIM 0.900 against 0.848 and 0.807. It removes 20.9% of the input variance at a
leakage of 0.100. f-x deconvolution removes 13.4% at 0.370, and the earlier model 26.0% at
0.148. DeepSeis removes more than f-x deconvolution, and less of what it removes is signal.
Its output carries 85% of the input's standard deviation, as a denoiser's should.

### 6.2 Fault preservation

Fault picks on the noisy input match the clean-section picks at a Dice score of **0.310**.
On the DeepSeis output, the score more than doubles to **0.649**, and 0.661 with
orthogonalization. That is ahead of f-x deconvolution (0.620), the earlier model (0.573) and
every smoothing filter (0.484–0.566). Denoising measurably helps interpretation, rather than
erasing what is being interpreted.

### 6.3 Frequency content

Energy retained in each radial F-K band (1.0 = untouched):

| Method | 0.0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---|---|---|---|---|
| DeepSeis | 0.940 | 0.841 | 0.291 | 0.037 | 0.015 |
| f-x deconvolution | 0.874 | 0.722 | 0.316 | 0.123 | 0.113 |
| Structure-oriented smoothing | 0.963 | 0.793 | 0.368 | 0.242 | 0.207 |
| Gaussian σ = 1.0 | 0.871 | 0.461 | 0.119 | 0.017 | 0.005 |

DeepSeis keeps most of the two lowest bands and almost nothing above 0.6 Nyquist. That looks
like a low-pass filter until one checks where F3's signal lies. The survey is band-limited by
its processing chain, with power falling off sharply beyond 0.4 Nyquist, so the bands being
emptied contain the added noise and nothing else. The same profile on data whose signal
reached those frequencies would be a problem. Unlike a low-pass filter, DeepSeis keeps the
low and middle bands nearly intact, where a Gaussian blur has already lost half of the
0.2–0.4 band.

### 6.4 Ablation: explicit structure terms

The two structure terms of §3.7 were tested on a matched schedule: identical seed, steps and
learning-rate curve, stopped at the same epoch.

| | SNR (dB) | SSIM | Leakage | Fault Dice |
|---|---|---|---|---|
| Structure terms off | 13.82 | 0.903 | 0.111 | 0.653 |
| Structure terms on | 13.66 | 0.900 | 0.100 | 0.649 |

The "off" run was equal or better at all six matched epochs. The gap is well inside the
±0.97 dB spread across sections, so the honest reading is "no measurable benefit", not "off is
better". The terms are redundant here rather than wrong: the posterior-mean estimator already
keeps the recording wherever the surroundings fail to predict it, which is what a fault is, so
a penalty for the same behaviour has little left to correct. They may still help on data with
strong coherent noise, which the dip-fan term is shaped for, so the code and its tests remain.

### 6.5 Facies on unseen lines

Held-out validation inlines 101–130, behind 15-line buffers:

| Unit | Held-out IoU |
|---|---|
| Upper North Sea | 0.979 |
| Middle North Sea | 0.868 |
| Lower North Sea | 0.962 |
| Rijnland/Chalk | 0.877 |
| Jurassic | 0.826 |
| Triassic | 0.839 |
| **Mean** | **0.892** |

Every unit clears 0.82, including the two rarest. On its training block, the classifier reaches
97.8% pixel accuracy and 0.940 mean IoU. The difference between that and 0.892 is the cost of
generalising to unseen lines, and it is small.

### 6.6 On the raw survey

Trained on the raw survey with nothing added (field mode), the same architecture removes
**less than 1% of the input variance**. This is not a failure. F3's published volume is
migrated, stacked and commercially processed, and its measured incoherent noise level is
correspondingly low. A calibrated denoiser should remove very little, and any figure showing
a dramatic improvement on raw F3 should be treated with suspicion. This is why the capability
is demonstrated on the controlled benchmark, where it can be checked.

Field-mode training also stops itself early. Held-out predictive likelihood peaks at epoch 2
and worsens for three epochs while the training likelihood keeps improving, and model
selection correctly keeps epoch 2. With little noise to model, there is little for the
network to learn beyond the geology it is shown.

---

## 7. Verification

In a project like this, the dangerous failures are not crashes. They are components that run
cleanly while doing something different from what they claim, and that return a plausible but
wrong number. The test suite therefore asserts **properties**. All 179 tests pass.

| Test module | Property checked |
|---|---|
| `test_blindspot.py` | Every pixel's prediction has exactly zero gradient with respect to that pixel, checked exhaustively for both output heads, all input channels and odd section sizes. The network does use its context (a constant output would also be blind), and all four half-planes are reached. |
| `test_noise_estimate.py` | The estimator recovers known noise levels, is invariant to signal amplitude, and detects band-limited data instead of reading a filter stopband as silence. |
| `test_denoise.py` | Translation and scale equivariance, tiled inference matching single-pass inference, and checkpoints carrying their normalisation constants. |
| `test_dataset.py` | Splits are disjoint and buffered, crosslines are truncated to the training block, normalisation is one global constant, and injected noise is fixed per section. |
| `test_evaluate.py` | Scoring touches only the requested split, the reference is the clean section, and stored normalisation is used rather than recomputed. |
| `test_baselines.py` | The dip estimator recovers the dips it is given, and every baseline improves SNR. |
| `test_geology_losses.py` | Each structure term is zero on an untouched section, rises monotonically as a fault is blurred out, and does not charge for removing incoherent noise. |
| `test_facies.py` | Held-out scoring uses unseen sections, patches combine by vote, and mean IoU has a fixed denominator. |
| `test_masking.py` | The earlier masking denoiser never donates a pixel to itself, and its blind region stays within budget. |
| `test_dashboard.py` | The dashboard executes end to end, every tab renders, export produces a file, and the controlled demo reproduces the benchmark result. |

These tests caught real defects that would otherwise have shipped silently:

- **Dip estimation.** The widely used orientation shortcut 0.5·arctan2(2·Jxy, Jyy − Jxx)
  returns the gradient direction rather than the structure direction. It cost about 4 dB and
  made the structure-oriented baseline look weaker than it is.
- **Orthogonalization.** The commonly quoted update form does not actually orthogonalize. The
  projection form recovers leaked signal exactly where the other breaks even.
- **Mean IoU.** With a denominator that changed between epochs, the facies score jumped from
  0.573 to 0.739 when the model gave up on a class.
- **Double normalisation.** Three call sites normalised a section that the serving function
  normalised again. The network is scale-equivariant but σn is a fixed physical level, so the
  result was quiet under-denoising: 3.17 dB lost and four times less noise removed, with the
  page rendering perfectly. A facies classifier fitted on those sections scored 0.93 on its own
  broken input and 0.13 on correctly served data. The serving function now warns when it
  receives a section that looks already normalised.

---

## 8. Using DeepSeis

- **Interactive dashboard** (`streamlit run app/dashboard.py`), with seven tabs: denoising
  with the difference section, uncertainty map and a controlled experiment with live noise
  sliders; fault segmentation on noisy and denoised input; a 3D block view of the survey; an
  inline explorer covering all 401 inlines; facies against the published interpretation;
  SEG-Y and NumPy export; and spectral and leakage diagnostics. The benchmark table in the
  app is read from the evaluation report, so the app cannot disagree with it.
- **Live demo** at [sraha23-deepseis.static.hf.space](https://sraha23-deepseis.static.hf.space):
  a static page of precomputed results for every held-out line, which opens instantly.
- **Operationally:** SEG-Y in, SEG-Y out, so it fits into an existing Petrel or OpendTect
  workflow as one more processing step. It runs on a CPU with no GPU, and runs fully offline
  once the data is local. No outside AI service or cloud is involved, which matters because
  seismic data is among the most confidential assets an operator owns. Adapting it to a new
  survey means retraining on that survey's own data, inside the operator's network.

---

## 9. Impact

- **Fewer dry wells.** Better fault delineation improves well placement, and a single avoided
  dry hole outweighs the cost of the software many times over.
- **Reviving legacy data.** Decades of older, noisier surveys sit in national archives.
  Re-shooting a survey costs crores; re-processing costs compute. Because DeepSeis learns
  each survey's own noise, it should help most on exactly that data.
- **No labelling cost.** Clean labels cannot exist for seismic denoising, and DeepSeis needs
  none.
- **Faster interpretation.** Faults, horizons and facies are produced automatically on the
  cleaned image.
- **Trust.** It shows what it removed and where it was unsure, which makes its output possible
  to review and sign off.

---

## 10. Limitations

- **2D, not full 3D.** Sections are processed one at a time. Neighbouring lines are not used as
  context during denoising, although the code supports it.
- **One survey.** All results are on F3, which is well-processed North Sea data. Performance on
  noisier surveys has not been measured yet.
- **No interpreted fault labels.** Fault preservation is measured against the fault network's
  picks on the clean section. This isolates the effect of noise and denoising, but it is not
  agreement with an interpreter.
- **Coherent noise.** The controlled test includes ground-roll-like events, but strong field
  coherent noise and multiples are untested.
- **Training budget.** The models were trained for a few epochs on a CPU and are not fully
  converged (§3.5).

## 11. Future work

1. **Full 3D volumes**, with neighbouring lines as context, on a GPU machine.
2. **One archived Indian survey** (for example Cambay, Krishna-Godavari or Assam), processed
   entirely inside the operator's own network.
3. **Scoring against interpreted faults** once a survey that has them is available.
4. **A hardened SEG-Y round trip**, so DeepSeis sits inside a live processing chain.
5. **Field-grade coherent noise**, where the dip-fan structure term and structured blind spots
   (Liu et al. 2022) are expected to matter.

All five are engineering, not research: the method is settled and measured, and what remains
is scale, integration, and a real survey to prove it on.

---

## References

- Alaudah, Y., Michałowicz, P., Alfarraj, M., AlRegib, G. (2019). A machine-learning benchmark
  for facies classification. *Interpretation*, 7(3), SE175–SE187.
- Batson, J., Royer, L. (2019). Noise2Self: Blind denoising by self-supervision. *ICML*.
- Birnie, C., Ravasi, M., Liu, S., Alkhalifah, T. (2021). The potential of self-supervised
  networks for random noise suppression in seismic data. *Artificial Intelligence in
  Geosciences*, 2, 47–59.
- Birnie, C., Alkhalifah, T. (2022). Transfer learning for self-supervised, blind-spot seismic
  denoising. *Frontiers in Earth Science*; arXiv:2209.12210.
- Canales, L. L. (1984). Random noise reduction. *SEG Technical Program Expanded Abstracts*.
- Chen, Y., Fomel, S. (2015). Random noise attenuation using local signal-and-noise
  orthogonalization. *Geophysics*, 80(6), WD1–WD9.
- Durall, R., et al. (2023). Deep diffusion models for seismic processing. *Computers &
  Geosciences*.
- Krull, A., Buchholz, T.-O., Jug, F. (2019). Noise2Void: Learning denoising from single noisy
  images. *CVPR*.
- Laine, S., Karras, T., Lehtinen, J., Aila, T. (2019). High-quality self-supervised deep image
  denoising. *NeurIPS*.
- Lehtinen, J., et al. (2018). Noise2Noise: Learning image restoration without clean data.
  *ICML*.
- Li, Y., Wu, X., Zhu, Z., Ding, J., Wang, Q. (2024). FaultSeg3D plus: Evaluating and improving
  CNN-based fault segmentation. *Geophysics*, 89.
- Li, et al. (2025). DPN2N: Self-supervised seismic denoising with high-frequency (fault)
  preservation. *Journal of Geophysics and Engineering*.
- Liu, S., Birnie, C., Alkhalifah, T. (2022). Coherent noise suppression via a self-supervised
  (Structured Noise2Void) scheme. *EAGE Annual Conference*.
- Mohan, S., Kadkhodaie, Z., Simoncelli, E. P., Fernandez-Granda, C. (2020). Robust and
  interpretable blind image denoising via bias-free convolutional neural networks. *ICLR*.
- Silva, R. M., et al. (2019). Netherlands F3 dataset: A public dataset for machine learning in
  seismic interpretation. arXiv:1904.00770.
- Wu, X., Liang, L., Shi, Y., Fomel, S. (2019). FaultSeg3D: Using synthetic data sets to train
  an end-to-end convolutional neural network for 3D seismic fault segmentation. *Geophysics*,
  84(3), IM35–IM45.
