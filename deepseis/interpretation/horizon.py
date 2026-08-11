"""
Horizon tracking (stretch goal, spec Sec. 3.4 / Sec. 5).

Classic amplitude-peak "auto-tracking": seed a horizon at a strong
amplitude peak on the first trace, then follow the nearest same-polarity
peak from trace to trace within a search window. Faults are handled by
letting the search window widen right at a fault-mask crossing (the
horizon may show a small vertical jump — a throw — as it steps across the
fault, which is exactly what should happen geologically) rather than
forcing continuity through it.

This works directly on the *denoised* section — running it on raw noisy
data would very often lock onto a noise peak instead of a real reflector,
which is the whole "denoising isn't the end goal, better interpretation is"
argument from the demo script (Sec. 11).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _trace_peaks(trace: np.ndarray, polarity: int) -> np.ndarray:
    signal = trace if polarity > 0 else -trace
    peaks, _ = find_peaks(signal, prominence=0.05 * (np.abs(trace).max() + 1e-8))
    return peaks


def track_horizons(section: np.ndarray, n_horizons: int, fault_mask: np.ndarray | None = None,
                    search_window: int = 6, fault_search_window: int = 16) -> np.ndarray:
    """Track ``n_horizons`` horizons across a denoised section.

    Returns an array of shape ``(n_horizons, n_traces)`` with the tracked
    sample index at every trace (NaN where tracking was lost).
    """
    n_samples, n_traces = section.shape

    # seed picks: the n_horizons strongest peaks on the first trace, evenly spread in depth
    seed_trace = section[:, 0]
    peaks, props = find_peaks(np.abs(seed_trace), prominence=0.05 * (np.abs(seed_trace).max() + 1e-8))
    if len(peaks) == 0:
        return np.full((n_horizons, n_traces), np.nan)

    # spread seeds across depth rather than just taking the top-N strongest (which tend to cluster)
    depth_bins = np.linspace(0, n_samples, n_horizons + 1)
    seeds = []
    for i in range(n_horizons):
        in_bin = peaks[(peaks >= depth_bins[i]) & (peaks < depth_bins[i + 1])]
        if len(in_bin) > 0:
            best = in_bin[np.argmax(np.abs(seed_trace[in_bin]))]
        elif len(peaks) > 0:
            best = peaks[np.argmin(np.abs(peaks - depth_bins[i:i + 2].mean()))]
        else:
            best = None
        seeds.append(best)

    tracks = np.full((n_horizons, n_traces), np.nan)
    for h_idx, seed in enumerate(seeds):
        if seed is None:
            continue
        polarity = 1 if seed_trace[seed] >= 0 else -1
        current = float(seed)
        tracks[h_idx, 0] = current

        for x in range(1, n_traces):
            near_fault = fault_mask is not None and fault_mask[int(round(current)), max(0, x - 1)] > 0.5
            window = fault_search_window if near_fault else search_window

            trace_peaks = _trace_peaks(section[:, x], polarity)
            if len(trace_peaks) == 0:
                tracks[h_idx, x] = current  # hold last value rather than losing the pick entirely
                continue

            dists = np.abs(trace_peaks - current)
            candidate_idx = np.argmin(dists)
            if dists[candidate_idx] <= window:
                current = float(trace_peaks[candidate_idx])
            # else: outside the search window -> hold previous depth (likely lost the horizon locally)
            tracks[h_idx, x] = current

    return tracks
