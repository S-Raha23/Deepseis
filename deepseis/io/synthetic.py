"""
Synthetic seismic model generator.

Real F3 Netherlands / FaultSeg3D data has to be supplied by the user (see
``data/download.py`` — this sandbox cannot reach Zenodo). To still be able to
build and exercise the *entire* pipeline end-to-end, and to get the
known-clean reference the spec asks for (Sec. 9: "use the synthetic volumes
... where ground truth exists"), this module builds a layered-earth model
with folded horizons, normal faults, and per-layer facies, then convolves it
with a Ricker wavelet to produce a synthetic post-stack seismic section.

This is intentionally a 2D (inline) generator — the whole pipeline
(denoiser, losses, FaultSeg-style head, dashboard) works slice-by-slice,
which mirrors how the spec's own MVP operates on "a 2D seismic line / 3D
slice".

Everything here is pure NumPy so it has no torch/GPU dependency.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy.signal import convolve as _scipy_convolve


@dataclasses.dataclass
class SyntheticVolume:
    clean: np.ndarray          # (n_samples, n_traces) float32, the clean seismic section
    fault_mask: np.ndarray     # (n_samples, n_traces) {0,1} binary fault label
    horizon_depths: np.ndarray  # (n_horizons, n_traces) int, sample index of each horizon
    facies: np.ndarray         # (n_samples, n_traces) int class label per layer


def ricker_wavelet(freq_hz: float, dt_ms: float, length_samples: int = 41) -> np.ndarray:
    """Classic zero-phase Ricker (Mexican hat) wavelet used in exploration seismology."""
    dt = dt_ms / 1000.0
    t = (np.arange(length_samples) - length_samples // 2) * dt
    a = (np.pi * freq_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w.astype(np.float32)


def _smooth_random_curve(n_traces: int, n_harmonics: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """A smooth 1D curve built from a handful of random sinusoids (horizon topography/folding)."""
    x = np.linspace(0, 2 * np.pi, n_traces)
    curve = np.zeros(n_traces, dtype=np.float64)
    for _ in range(n_harmonics):
        freq = rng.uniform(0.5, 3.5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = amplitude * rng.uniform(0.3, 1.0) / n_harmonics
        curve += amp * np.sin(freq * x + phase)
    return curve


def generate_synthetic_volume(
    n_traces: int = 256,
    n_samples: int = 256,
    n_layers: int = 14,
    n_faults: int = 4,
    dt_ms: float = 4.0,
    wavelet_freq_hz: float = 30.0,
    random_seed: int = 7,
) -> SyntheticVolume:
    """Build a synthetic seismic section with folded horizons, normal faults, and facies.

    Returns a :class:`SyntheticVolume` with the clean section, a binary fault
    mask, horizon-depth curves, and a facies label map — enough ground truth
    to train and *evaluate* every stage of the pipeline quantitatively.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Base (unfaulted) horizon depths: roughly evenly spaced, then folded.
    base_depths = np.linspace(n_samples * 0.08, n_samples * 0.92, n_layers)
    horizon_depths = np.zeros((n_layers, n_traces), dtype=np.float64)
    for k in range(n_layers):
        fold = _smooth_random_curve(n_traces, n_harmonics=3, amplitude=n_samples * 0.05, rng=rng)
        horizon_depths[k] = base_depths[k] + fold

    # 2. Reflection coefficients: one per horizon, randomly signed (impedance contrast).
    reflectivity_strength = rng.uniform(0.4, 1.0, size=n_layers) * rng.choice([-1.0, 1.0], size=n_layers)

    # 3. Faults: pick trace locations + dips, and shift horizons for traces on the
    #    hanging-wall side by a throw that grows with depth (typical normal-fault geometry).
    fault_trace_positions = np.sort(rng.choice(np.arange(int(n_traces * 0.15), int(n_traces * 0.85)),
                                                size=min(n_faults, n_traces // 8), replace=False))
    fault_mask = np.zeros((n_samples, n_traces), dtype=np.float32)
    fault_dips = rng.uniform(0.15, 0.45, size=len(fault_trace_positions))   # how much throw shifts laterally with depth
    fault_max_throw = rng.uniform(n_samples * 0.03, n_samples * 0.09, size=len(fault_trace_positions))
    fault_widths = rng.integers(1, 3, size=len(fault_trace_positions))  # half-width of the fault trace band, in pixels

    for f_idx, x0 in enumerate(fault_trace_positions):
        dip = fault_dips[f_idx]
        max_throw = fault_max_throw[f_idx]
        width = fault_widths[f_idx]
        for k in range(n_layers):
            mean_depth = horizon_depths[k].mean()
            throw = max_throw * (mean_depth / n_samples)  # deeper horizons see more accumulated throw
            # apply throw to the hanging wall (traces beyond the fault trace, offset by dip*depth)
            fault_x_at_depth = x0 + dip * (np.arange(n_traces) * 0.0)  # placeholder, refined below
            shift_mask = np.arange(n_traces) >= x0
            horizon_depths[k, shift_mask] += throw

        # trace out the fault plane itself into the fault_mask image (roughly vertical, slight dip)
        for t in range(n_samples):
            x_at_t = int(round(x0 + dip * (t - n_samples / 2) * 0.15))
            lo, hi = max(0, x_at_t - width), min(n_traces, x_at_t + width + 1)
            if lo < hi:
                fault_mask[t, lo:hi] = 1.0

    horizon_depths = np.clip(horizon_depths, 2, n_samples - 3)

    # 4. Build the reflectivity series (a spike train at each horizon depth, per trace).
    reflectivity = np.zeros((n_samples, n_traces), dtype=np.float64)
    facies = np.zeros((n_samples, n_traces), dtype=np.int64)
    n_facies_classes = 3
    layer_facies = rng.integers(0, n_facies_classes, size=n_layers + 1)

    for x in range(n_traces):
        depths_sorted = np.sort(horizon_depths[:, x])
        prev = 0
        for k, d in enumerate(depths_sorted):
            d_int = int(round(d))
            reflectivity[d_int, x] += reflectivity_strength[k % n_layers]
            facies[prev:d_int, x] = layer_facies[k]
            prev = d_int
        facies[prev:, x] = layer_facies[-1]

    # small amount of random reflectivity jitter (thin-bed / stratigraphic texture)
    reflectivity += rng.normal(0, 0.02, size=reflectivity.shape)

    # 5. Convolve every trace with a Ricker wavelet -> synthetic seismic section.
    wavelet = ricker_wavelet(wavelet_freq_hz, dt_ms, length_samples=41)
    clean = np.zeros_like(reflectivity)
    for x in range(n_traces):
        clean[:, x] = _scipy_convolve(reflectivity[:, x], wavelet, mode="same")

    # normalize to unit RMS amplitude (standard seismic display convention)
    clean = clean / (clean.std() + 1e-8)

    return SyntheticVolume(
        clean=clean.astype(np.float32),
        fault_mask=fault_mask.astype(np.float32),
        horizon_depths=horizon_depths.astype(np.float32),
        facies=facies.astype(np.int64),
    )


@dataclasses.dataclass
class SyntheticSurvey:
    """A small pseudo-3D survey: a stack of structurally-coherent inlines.

    Used by the dashboard's inline slider (spec Sec. 5: "a slider to scrub
    inlines"). Horizons and faults drift smoothly from inline to inline
    (shared low-frequency random fields + a slow lateral fault drift) so
    scrubbing through it looks like moving through a real survey rather
    than jumping between unrelated 2D sections.
    """
    clean: np.ndarray          # (n_inlines, n_samples, n_traces)
    fault_mask: np.ndarray     # (n_inlines, n_samples, n_traces)
    facies: np.ndarray         # (n_inlines, n_samples, n_traces)


def _smooth_random_field(n_i: int, n_x: int, n_harmonics: int, amplitude: float,
                          rng: np.random.Generator) -> np.ndarray:
    """A smooth 2D field (inline x crossline) built from a few low-frequency 2D sinusoids."""
    ii = np.linspace(0, 2 * np.pi, n_i)
    xx = np.linspace(0, 2 * np.pi, n_x)
    I, X = np.meshgrid(ii, xx, indexing="ij")
    field = np.zeros((n_i, n_x), dtype=np.float64)
    for _ in range(n_harmonics):
        f_i = rng.uniform(0.3, 1.5)
        f_x = rng.uniform(0.5, 3.5)
        phase = rng.uniform(0, 2 * np.pi)
        amp = amplitude * rng.uniform(0.3, 1.0) / n_harmonics
        field += amp * np.sin(f_x * X + f_i * I + phase)
    return field


def generate_synthetic_survey(
    n_inlines: int = 16,
    n_traces: int = 160,
    n_samples: int = 160,
    n_layers: int = 12,
    n_faults: int = 3,
    dt_ms: float = 4.0,
    wavelet_freq_hz: float = 30.0,
    random_seed: int = 7,
) -> SyntheticSurvey:
    """Build a small pseudo-3D synthetic survey (structurally-coherent stack of inlines)."""
    rng = np.random.default_rng(random_seed)

    base_depths = np.linspace(n_samples * 0.08, n_samples * 0.92, n_layers)
    reflectivity_strength = rng.uniform(0.4, 1.0, size=n_layers) * rng.choice([-1.0, 1.0], size=n_layers)

    # one smooth (inline, crossline) fold field per layer -> horizons drift coherently across inlines
    fold_fields = [_smooth_random_field(n_inlines, n_traces, 3, n_samples * 0.05, rng) for _ in range(n_layers)]

    fault_x0 = np.sort(rng.choice(np.arange(int(n_traces * 0.15), int(n_traces * 0.85)),
                                   size=min(n_faults, n_traces // 8), replace=False)).astype(np.float64)
    fault_dip = rng.uniform(0.15, 0.45, size=len(fault_x0))
    fault_max_throw = rng.uniform(n_samples * 0.03, n_samples * 0.09, size=len(fault_x0))
    fault_width = rng.integers(1, 3, size=len(fault_x0))
    fault_strike_drift = rng.uniform(-0.6, 0.6, size=len(fault_x0))  # lateral shift per inline step

    wavelet = ricker_wavelet(wavelet_freq_hz, dt_ms, length_samples=41)

    clean = np.zeros((n_inlines, n_samples, n_traces), dtype=np.float32)
    fault_mask = np.zeros((n_inlines, n_samples, n_traces), dtype=np.float32)
    facies = np.zeros((n_inlines, n_samples, n_traces), dtype=np.int64)
    layer_facies = rng.integers(0, 3, size=n_layers + 1)

    for i in range(n_inlines):
        horizon_depths = np.stack([base_depths[k] + fold_fields[k][i] for k in range(n_layers)])
        for f_idx in range(len(fault_x0)):
            x0_i = fault_x0[f_idx] + fault_strike_drift[f_idx] * i
            for k in range(n_layers):
                mean_depth = horizon_depths[k].mean()
                throw = fault_max_throw[f_idx] * (mean_depth / n_samples)
                shift_mask = np.arange(n_traces) >= x0_i
                horizon_depths[k, shift_mask] += throw
            for t in range(n_samples):
                x_at_t = int(round(x0_i + fault_dip[f_idx] * (t - n_samples / 2) * 0.15))
                lo, hi = max(0, x_at_t - int(fault_width[f_idx])), min(n_traces, x_at_t + int(fault_width[f_idx]) + 1)
                if lo < hi:
                    fault_mask[i, t, lo:hi] = 1.0
        horizon_depths = np.clip(horizon_depths, 2, n_samples - 3)

        reflectivity = np.zeros((n_samples, n_traces), dtype=np.float64)
        for x in range(n_traces):
            depths_sorted = np.sort(horizon_depths[:, x])
            prev = 0
            for k, d in enumerate(depths_sorted):
                d_int = int(round(d))
                reflectivity[d_int, x] += reflectivity_strength[k % n_layers]
                facies[i, prev:d_int, x] = layer_facies[k]
                prev = d_int
            facies[i, prev:, x] = layer_facies[-1]
        reflectivity += rng.normal(0, 0.02, size=reflectivity.shape)

        section = np.zeros_like(reflectivity)
        for x in range(n_traces):
            section[:, x] = _scipy_convolve(reflectivity[:, x], wavelet, mode="same")
        clean[i] = (section / (section.std() + 1e-8)).astype(np.float32)

    return SyntheticSurvey(clean=clean, fault_mask=fault_mask, facies=facies)


def generate_from_config(cfg: dict) -> SyntheticVolume:
    """Convenience wrapper that reads the ``data.synthetic`` block of a config dict."""
    s = cfg["data"]["synthetic"]
    return generate_synthetic_volume(
        n_traces=s["n_traces"],
        n_samples=s["n_samples"],
        n_layers=s["n_layers"],
        n_faults=s["n_faults"],
        dt_ms=s["dt_ms"],
        wavelet_freq_hz=s["wavelet_freq_hz"],
        random_seed=s["random_seed"],
    )
