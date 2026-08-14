"""
Classical denoising baselines.

The project previously reported its denoiser against an identity control and
two blurs. That is enough to show it is not *literally* a blur, but not enough
to show it is worth running: the methods a processor would actually reach for
on post-stack data are f-x deconvolution and structure-oriented smoothing, and
a learned denoiser that cannot beat twenty lines of signal processing has not
earned its training budget.

Everything here operates on a ``(n_samples, n_traces)`` section and returns
one of the same shape, so all of it drops into the same evaluation harness as
the network.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter


# ---------------------------------------------------------------------------
# Trivial controls
# ---------------------------------------------------------------------------

def identity(section: np.ndarray) -> np.ndarray:
    """Removes nothing. Scores perfectly on any leakage metric, which is the
    point of including it: it calibrates what "no signal leaked" is worth."""
    return section.astype(np.float32)


def gaussian(section: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter(section.astype(np.float64), sigma=sigma).astype(np.float32)


def median(section: np.ndarray, size: int = 3) -> np.ndarray:
    return median_filter(section.astype(np.float64), size=size).astype(np.float32)


# ---------------------------------------------------------------------------
# f-x deconvolution (Canales 1984)
# ---------------------------------------------------------------------------

def _fx_prediction_filter(u: np.ndarray, order: int, damping: float) -> np.ndarray:
    """Forward-backward complex prediction filter applied to one frequency slice.

    At a single temporal frequency, a linear (dipping) event is a complex
    exponential along the trace axis, so a superposition of ``order`` dipping
    events is exactly predictable by an order-``order`` autoregressive filter.
    Random noise is not. Fitting the filter and keeping only what it predicts
    is therefore a signal/noise separation that assumes nothing about
    amplitude and everything about lateral coherence -- which is why it has
    outlived most of the alternatives.
    """
    n = u.shape[0]
    if n <= 2 * order + 1:
        return u

    # Forward: u[i] ~ sum_k a_k u[i-k]
    fwd = np.stack([u[i - order:i][::-1] for i in range(order, n)])
    fwd_rhs = u[order:]
    # Backward: the same filter must predict the conjugate-reversed series,
    # which doubles the equations and stabilises the fit on short lines.
    bwd = np.stack([np.conj(u[i + 1:i + order + 1]) for i in range(0, n - order)])
    bwd_rhs = np.conj(u[:n - order])

    A = np.vstack([fwd, bwd])
    b = np.concatenate([fwd_rhs, bwd_rhs])

    AhA = A.conj().T @ A
    lam = damping * np.trace(AhA).real / max(order, 1)
    try:
        a = np.linalg.solve(AhA + lam * np.eye(order), A.conj().T @ b)
    except np.linalg.LinAlgError:
        return u

    # Average the forward and backward reconstructions so the result is not
    # biased towards one end of the line.
    out = np.array(u, dtype=np.complex128)
    out[order:] = 0.5 * (out[order:] + fwd @ a)
    out[:n - order] = 0.5 * (out[:n - order] + np.conj(bwd @ a))
    return out


def fx_decon(section: np.ndarray, order: int = 4, window_traces: int = 40,
             window_overlap: int = 20, damping: float = 0.02,
             max_freq_fraction: float = 1.0) -> np.ndarray:
    """f-x deconvolution in overlapping spatial windows.

    Windowing matters: the AR model assumes a small number of dips that hold
    across the window, which is true locally and false across a whole line.
    """
    sec = np.asarray(section, dtype=np.float64)
    nt, nx = sec.shape
    step = max(window_traces - window_overlap, 1)

    out = np.zeros((nt, nx), dtype=np.float64)
    weight = np.zeros(nx, dtype=np.float64)

    starts = list(range(0, max(nx - window_traces, 0) + 1, step))
    if not starts or starts[-1] + window_traces < nx:
        starts.append(max(nx - window_traces, 0))

    for x0 in starts:
        x1 = min(x0 + window_traces, nx)
        block = sec[:, x0:x1]
        spec = np.fft.rfft(block, axis=0)

        n_freq = spec.shape[0]
        cutoff = int(n_freq * max_freq_fraction)
        for fi in range(1, min(cutoff, n_freq)):   # skip DC
            spec[fi] = _fx_prediction_filter(spec[fi], order, damping)

        rec = np.fft.irfft(spec, n=nt, axis=0)
        taper = np.hanning(x1 - x0 + 2)[1:-1]
        if x1 - x0 == 1:
            taper = np.ones(1)
        out[:, x0:x1] += rec * taper
        weight[x0:x1] += taper

    weight[weight == 0] = 1.0
    return (out / weight).astype(np.float32)


# ---------------------------------------------------------------------------
# Structure-oriented smoothing
# ---------------------------------------------------------------------------

def structure_tensor(section: np.ndarray, sigma_grad: float = 1.0, sigma_tensor: float = 4.0):
    """Smoothed gradient outer product ``(jyy, jxy, jxx)``, in (time, trace) order."""
    sec = np.asarray(section, dtype=np.float64)
    gy, gx = np.gradient(gaussian_filter(sec, sigma_grad))
    return (gaussian_filter(gy * gy, sigma_tensor),
            gaussian_filter(gy * gx, sigma_tensor),
            gaussian_filter(gx * gx, sigma_tensor))


def local_dip(section: np.ndarray, sigma_grad: float = 1.0, sigma_tensor: float = 4.0,
              max_dip: float = 4.0) -> np.ndarray:
    """Per-pixel apparent dip in samples per trace, from the structure tensor.

    A reflector with dip ``p`` samples/trace runs along ``(dy, dx) = (p, 1)``,
    so its gradient points along ``(1, -p)`` and the tensor it generates is
    proportional to ``[[1, -p], [-p, p^2]]``. Recovering ``p`` therefore means
    taking the eigenvector of the *smallest* eigenvalue -- the direction the
    image does not change along -- and reading off ``v_y / v_x``.

    Doing this via ``0.5 * arctan2(2*jxy, jyy - jxx)`` is the usual shortcut,
    but it returns the orientation of the *principal* axis, which is the
    gradient direction rather than the structure direction, and gets the
    branch wrong for steep dips. It cost about 4 dB here before being caught
    by ``tests/test_baselines.py::test_local_dip_recovers_a_known_dip``.
    """
    jyy, jxy, jxx = structure_tensor(section, sigma_grad, sigma_tensor)

    tr = jyy + jxx
    disc = np.sqrt(np.maximum((jyy - jxx) ** 2 + 4 * jxy ** 2, 0.0))
    lam_min = 0.5 * (tr - disc)

    v_y = jxy
    v_x = lam_min - jyy
    # Degenerate where the tensor is isotropic (no structure): fall back to flat.
    dip = np.divide(v_y, v_x, out=np.zeros_like(v_y), where=np.abs(v_x) > 1e-12)
    return np.clip(dip, -max_dip, max_dip)


def structure_oriented_smoothing(section: np.ndarray, length: int = 7,
                                 sigma_grad: float = 1.0, sigma_tensor: float = 4.0,
                                 coherence_guard: bool = True) -> np.ndarray:
    """Smooth *along* the local reflector dip and never across it.

    This is the classical answer to "denoise without destroying structure",
    and it is a genuinely strong baseline: averaging along a reflector
    suppresses incoherent noise at close to the sqrt(n) rate while leaving the
    reflector itself untouched. Its weakness is the one the learned model has
    to exploit -- the dip field is itself estimated from noisy data, and near a
    fault there is no single correct dip, so it smears exactly where it
    matters most unless it is guarded.
    """
    sec = np.asarray(section, dtype=np.float64)
    nt, nx = sec.shape
    dip = local_dip(sec, sigma_grad, sigma_tensor)

    half = length // 2
    acc = np.zeros_like(sec)
    cnt = np.zeros_like(sec)

    rows = np.arange(nt)[:, None]
    cols = np.arange(nx)[None, :]

    for k in range(-half, half + 1):
        # follow the dip k traces away, with linear interpolation in time
        tgt_x = np.clip(cols + k, 0, nx - 1)
        tgt_y = rows + dip * k
        y0 = np.floor(tgt_y).astype(int)
        frac = tgt_y - y0
        y0c = np.clip(y0, 0, nt - 1)
        y1c = np.clip(y0 + 1, 0, nt - 1)

        vals = (1 - frac) * sec[y0c, tgt_x] + frac * sec[y1c, tgt_x]
        inside = (tgt_y >= 0) & (tgt_y <= nt - 1)
        acc += np.where(inside, vals, 0.0)
        cnt += inside

    smoothed = acc / np.maximum(cnt, 1)

    if coherence_guard:
        # Where the structure tensor says there is no coherent orientation
        # (a fault, or noise), fall back towards the original sample rather
        # than averaging along a dip that does not exist.
        jyy, jxy, jxx = structure_tensor(sec, sigma_grad, sigma_tensor)
        tr = jyy + jxx
        disc = np.sqrt(np.maximum((jyy - jxx) ** 2 + 4 * jxy ** 2, 0.0))
        anisotropy = disc / (tr + 1e-12)      # 1 = perfectly linear, 0 = isotropic
        w = np.clip(anisotropy, 0.0, 1.0)
        smoothed = w * smoothed + (1 - w) * sec

    return smoothed.astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES = {
    "identity": identity,
    "gaussian_0.5": lambda s: gaussian(s, 0.5),
    "gaussian_1.0": lambda s: gaussian(s, 1.0),
    "median_3": lambda s: median(s, 3),
    "fx_decon": fx_decon,
    "structure_oriented": structure_oriented_smoothing,
}
