"""
Controllable synthetic noise injection.

Implements the two noise families described in the spec (Sec. 2):

* **Random / incoherent noise** — i.i.d. Gaussian, no trace-to-trace correlation.
* **Coherent noise** — structured artefacts that *look like signal*:
  - ``ground_roll``: low-frequency, high-amplitude surface waves with linear
    moveout across traces and a slow amplitude decay with offset/time —
    the classic near-surface contaminant in land seismic data.
  - ``linear``: simple dipping linear noise events (e.g. cable/rig noise).

Having noise with a *known* ground truth is what lets the pipeline report
clean PSNR/SSIM numbers (Sec. 9) — this is not possible on real field data,
which is why the spec calls the synthetic case out specifically for that
purpose.
"""
from __future__ import annotations

import numpy as np


def add_random_noise(clean: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """I.i.d. Gaussian noise, independent at every sample -> the Noise2Void assumption holds exactly."""
    noise = rng.normal(0.0, std, size=clean.shape).astype(np.float32)
    return clean + noise


def add_ground_roll(clean: np.ndarray, amplitude: float, dt_ms: float, rng: np.random.Generator,
                     apparent_velocity_traces_per_sample: float = 0.9,
                     freq_hz: float = 8.0) -> np.ndarray:
    """Low-frequency coherent noise with linear moveout, concentrated at early times.

    This is the "hard" noise case the Structured Noise2Void branch targets:
    it is spatially/temporally correlated, so a plain random blind-spot mask
    cannot remove it (the network could just learn to reproduce it from
    neighbouring pixels along the noise's own direction).
    """
    n_samples, n_traces = clean.shape
    t = np.arange(n_samples)[:, None]
    x = np.arange(n_traces)[None, :]
    dt = dt_ms / 1000.0

    # a couple of superposed dispersive wavetrains, decaying with time (depth)
    gr = np.zeros_like(clean, dtype=np.float64)
    for k, (f, v) in enumerate([(freq_hz, apparent_velocity_traces_per_sample),
                                 (freq_hz * 1.6, apparent_velocity_traces_per_sample * 0.6)]):
        phase = 2 * np.pi * f * (t * dt - x / (v / dt))
        envelope = np.exp(-t / (n_samples * 0.35)) * (1.0 - 0.3 * k)
        gr += envelope * np.sin(phase)

    gr = gr / (np.abs(gr).max() + 1e-8) * amplitude * clean.std() * 4.0
    jitter = rng.normal(0, 0.05 * amplitude, size=clean.shape)
    return (clean + gr + jitter).astype(np.float32)


def add_linear_noise(clean: np.ndarray, amplitude: float, rng: np.random.Generator, n_events: int = 3) -> np.ndarray:
    """A handful of dipping linear coherent-noise events (e.g. rig/cable noise)."""
    n_samples, n_traces = clean.shape
    out = clean.copy().astype(np.float64)
    t = np.arange(n_samples)[:, None]
    x = np.arange(n_traces)[None, :]
    for _ in range(n_events):
        dip = rng.uniform(-1.5, 1.5)
        intercept = rng.uniform(0, n_samples)
        freq = rng.uniform(10, 25)
        width = rng.uniform(3, 7)
        dist = t - (intercept + dip * x)
        event = amplitude * clean.std() * 3.0 * np.exp(-(dist ** 2) / (2 * width ** 2)) * np.sin(2 * np.pi * freq * dist / n_samples)
        out += event
    return out.astype(np.float32)


def make_noisy(clean: np.ndarray, cfg: dict, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply the noise pipeline described by ``cfg['data']['noise']`` to a clean section."""
    if rng is None:
        rng = np.random.default_rng(0)
    ncfg = cfg["data"]["noise"]
    noisy = clean.copy()

    coherent_type = ncfg.get("coherent_type", "none")
    if coherent_type == "ground_roll":
        dt_ms = cfg["data"]["synthetic"]["dt_ms"]
        noisy = add_ground_roll(noisy, amplitude=ncfg["coherent_amp"], dt_ms=dt_ms, rng=rng)
    elif coherent_type == "linear":
        noisy = add_linear_noise(noisy, amplitude=ncfg["coherent_amp"], rng=rng)
    elif coherent_type == "none":
        pass
    else:
        raise ValueError(f"Unknown coherent_type: {coherent_type}")

    noisy = add_random_noise(noisy, std=ncfg["random_std"] * clean.std(), rng=rng)
    return noisy.astype(np.float32)
