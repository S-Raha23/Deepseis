"""
Evaluation metrics (spec Sec. 9).

* Denoising, synthetic (clean available): PSNR, SNR, SSIM.
* Denoising, field data (no clean reference): local-similarity / signal-
  leakage map, F-K spectrum comparison (see ``losses/frequency.py:fk_spectrum``).
* Fault preservation: Dice / precision / recall / ROC-AUC of the fault
  segmentation head on noisy vs. denoised input, plus a distance-based
  metric (a la FaultSeg3D-plus) that scores *how far off* a predicted fault
  is rather than penalizing every off-by-one-pixel miss as a full miss.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as skimage_ssim


# ---------------------------------------------------------------------------
# Denoising metrics (synthetic / known-clean case)
# ---------------------------------------------------------------------------

def psnr(pred: np.ndarray, clean: np.ndarray, data_range: float | None = None) -> float:
    if data_range is None:
        data_range = clean.max() - clean.min()
    mse = np.mean((pred.astype(np.float64) - clean.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def snr_db(pred: np.ndarray, clean: np.ndarray) -> float:
    """Signal-to-noise ratio, treating (pred - clean) as residual noise."""
    signal_power = np.mean(clean.astype(np.float64) ** 2)
    noise_power = np.mean((pred.astype(np.float64) - clean.astype(np.float64)) ** 2)
    if noise_power <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def ssim(pred: np.ndarray, clean: np.ndarray) -> float:
    data_range = clean.max() - clean.min()
    return float(skimage_ssim(pred, clean, data_range=data_range))


# ---------------------------------------------------------------------------
# Field-data (no-reference) metrics
# ---------------------------------------------------------------------------

def local_similarity_map(noisy: np.ndarray, denoised: np.ndarray, window: int = 9) -> np.ndarray:
    """Windowed local correlation between the *removed noise* and the *denoised signal*.

    If the denoiser is behaving (no signal leakage into the "removed"
    component), the removed noise should look statistically independent of
    the denoised signal, so local correlation should sit near zero
    everywhere. Patches where this map spikes are exactly the places to
    manually sanity-check for over-aggressive removal of real structure.
    """
    removed = noisy - denoised
    h, w = noisy.shape
    half = window // 2
    sim = np.zeros_like(noisy, dtype=np.float64)

    # pad for border handling
    removed_p = np.pad(removed, half, mode="reflect")
    denoised_p = np.pad(denoised, half, mode="reflect")

    for y in range(h):
        for x in range(w):
            a = removed_p[y:y + window, x:x + window].ravel()
            b = denoised_p[y:y + window, x:x + window].ravel()
            a = a - a.mean()
            b = b - b.mean()
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            sim[y, x] = float(np.dot(a, b) / denom)
    return sim


def local_similarity_map_fast(noisy: np.ndarray, denoised: np.ndarray, window: int = 9) -> np.ndarray:
    """Vectorized (box-filter based) version of :func:`local_similarity_map` — use this one in practice."""
    from scipy.ndimage import uniform_filter

    removed = noisy - denoised
    a, b = removed.astype(np.float64), denoised.astype(np.float64)

    mean_a = uniform_filter(a, size=window)
    mean_b = uniform_filter(b, size=window)
    a0, b0 = a - mean_a, b - mean_b

    cov = uniform_filter(a0 * b0, size=window)
    var_a = uniform_filter(a0 * a0, size=window)
    var_b = uniform_filter(b0 * b0, size=window)

    return cov / (np.sqrt(var_a * var_b) + 1e-8)


# ---------------------------------------------------------------------------
# Fault segmentation metrics
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FaultMetrics:
    dice: float
    precision: float
    recall: float
    roc_auc: float
    mean_distance_error: float  # geologically-fair, distance-based metric (FaultSeg3D-plus style)


def dice_score(pred_binary: np.ndarray, target_binary: np.ndarray) -> float:
    inter = np.logical_and(pred_binary, target_binary).sum()
    denom = pred_binary.sum() + target_binary.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def precision_recall(pred_binary: np.ndarray, target_binary: np.ndarray) -> tuple[float, float]:
    tp = np.logical_and(pred_binary, target_binary).sum()
    fp = np.logical_and(pred_binary, ~target_binary.astype(bool)).sum()
    fn = np.logical_and(~pred_binary.astype(bool), target_binary).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return float(precision), float(recall)


def roc_auc(scores: np.ndarray, labels: np.ndarray, max_samples: int = 20000) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U statistic) — no sklearn dependency needed.

    Subsamples to ``max_samples`` pixels for speed on large volumes; AUC is
    stable under subsampling for this purpose.
    """
    scores = scores.ravel()
    labels = labels.ravel().astype(bool)
    if scores.size > max_samples:
        idx = np.random.default_rng(0).choice(scores.size, size=max_samples, replace=False)
        scores, labels = scores[idx], labels[idx]

    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[labels].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def distance_based_fault_metric(pred_binary: np.ndarray, target_binary: np.ndarray) -> float:
    """Mean of (distance from each predicted-fault pixel to nearest true fault pixel)
    and (distance from each true-fault pixel to nearest predicted-fault pixel).

    This is the "geologically fair" scoring the spec asks for (Sec. 9): a
    fault predicted 1-2 pixels off from the true location is a near-miss,
    not a total miss the way raw pixel accuracy would treat it.
    """
    if target_binary.sum() == 0 or pred_binary.sum() == 0:
        return float("nan")
    dist_to_target = distance_transform_edt(~target_binary.astype(bool))
    dist_to_pred = distance_transform_edt(~pred_binary.astype(bool))

    err_pred_to_target = dist_to_target[pred_binary.astype(bool)].mean()
    err_target_to_pred = dist_to_pred[target_binary.astype(bool)].mean()
    return float((err_pred_to_target + err_target_to_pred) / 2.0)


def fault_metrics(pred_prob: np.ndarray, target_binary: np.ndarray, threshold: float = 0.5) -> FaultMetrics:
    pred_binary = pred_prob >= threshold
    dice = dice_score(pred_binary, target_binary.astype(bool))
    precision, recall = precision_recall(pred_binary, target_binary.astype(bool))
    auc = roc_auc(pred_prob, target_binary)
    dist_err = distance_based_fault_metric(pred_binary, target_binary.astype(bool))
    return FaultMetrics(dice=dice, precision=precision, recall=recall, roc_auc=auc, mean_distance_error=dist_err)
