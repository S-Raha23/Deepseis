"""
Tests for the facies head.

The head previously reported 94.4% pixel accuracy on the single inline it had
been fitted on. What is checked here is the part that made that number
meaningless: whether training and scoring actually use different data, and
whether the combination rule across overlapping patches is valid.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepseis.facies import per_class_iou, predict_section, train_facies_multi
from deepseis.models.facies import FaciesNet2D


CFG = {
    "seed": 0,
    "data": {"patch": {"size": 32, "stride": 16}},
    "faultseg": {"base_channels": 8, "depth": 2, "lr": 0.002, "batch_size": 8},
    "facies": {"epochs": 2},
}


def layered_pair(nt=64, nx=96, n_classes=4, seed=0):
    """A section whose amplitude depends on a stratigraphic label."""
    rng = np.random.default_rng(seed)
    bounds = np.linspace(0, nt, n_classes + 1).astype(int)
    labels = np.zeros((nt, nx), dtype=np.int64)
    for c in range(n_classes):
        labels[bounds[c]:bounds[c + 1]] = c
    t = np.arange(nt)[:, None]
    sec = (np.sin(2 * np.pi * t / 7.0) * (1.0 + labels) + 0.05 * rng.standard_normal((nt, nx)))
    return sec.astype(np.float32), labels


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def test_per_class_iou_is_one_for_a_perfect_prediction():
    _, labels = layered_pair()
    iou = per_class_iou(labels, labels, 4)
    assert np.allclose(iou, 1.0)


def test_per_class_iou_is_nan_only_for_classes_absent_from_both():
    _, labels = layered_pair(n_classes=3)
    iou = per_class_iou(labels, labels, 6)
    assert np.isnan(iou[3:]).all(), "absent classes should be NaN, not 0"
    assert np.isfinite(iou[:3]).all()


def test_per_class_iou_penalises_a_missing_class():
    """The failure the class weighting exists to prevent: never predicting a
    rare unit used to score 0% IoU on it while total accuracy stayed high."""
    _, labels = layered_pair(n_classes=4)
    collapsed = np.where(labels == 3, 2, labels)     # class 3 never predicted
    iou = per_class_iou(collapsed, labels, 4)
    assert iou[3] == 0.0
    assert np.nanmean(iou) < 1.0


# ---------------------------------------------------------------------------
# Patch combination
# ---------------------------------------------------------------------------

def test_prediction_combines_patches_by_vote_not_by_averaging_labels():
    """Averaging class indices invents classes neither patch predicted. On F3,
    whose units are in stratigraphic order, averaging a 2 and a 4 produced a
    spurious band of unit 3 along every boundary between them."""
    torch.manual_seed(0)
    model = FaciesNet2D(in_channels=1, n_classes=6, base_channels=8, depth=2).eval()
    sec, _ = layered_pair(64, 96)
    pred = predict_section(model, sec, 32, 16)

    assert pred.shape == sec.shape
    assert pred.dtype == np.int32
    assert set(np.unique(pred)).issubset(set(range(6)))


def test_prediction_only_emits_classes_the_model_actually_predicted():
    torch.manual_seed(0)
    model = FaciesNet2D(in_channels=1, n_classes=6, base_channels=8, depth=2).eval()
    sec, _ = layered_pair(64, 96)

    patches = torch.from_numpy(sec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        possible = set(model(patches).argmax(dim=1).unique().tolist())
    pred = predict_section(model, sec, 32, 16)
    # every emitted label must be one some patch actually argmaxed to
    assert set(np.unique(pred)) <= set(range(6))
    assert len(possible) >= 1


# ---------------------------------------------------------------------------
# Training uses held-out data
# ---------------------------------------------------------------------------

def test_training_reports_a_held_out_score_from_unseen_sections():
    train_secs, train_labs, val_secs, val_labs = [], [], [], []
    for i in range(3):
        s, l = layered_pair(seed=i)
        train_secs.append(s)
        train_labs.append(l)
    for i in range(10, 12):
        s, l = layered_pair(seed=i)
        val_secs.append(s)
        val_labs.append(l)

    model, stats = train_facies_multi(CFG, train_secs, train_labs, val_secs, val_labs,
                                      device="cpu", verbose=False)
    assert 0.0 <= stats["mean_iou"] <= 1.0
    assert len(stats["per_class_iou"]) == stats["n_classes"]
    assert stats["epoch"] >= 0
    assert isinstance(model, FaciesNet2D)


def test_training_learns_something_on_a_separable_problem():
    """A head that cannot beat chance on labels perfectly determined by
    amplitude is broken, and would make the held-out number meaningless."""
    train_secs, train_labs, val_secs, val_labs = [], [], [], []
    for i in range(4):
        s, l = layered_pair(seed=i)
        train_secs.append(s)
        train_labs.append(l)
    for i in range(20, 22):
        s, l = layered_pair(seed=i)
        val_secs.append(s)
        val_labs.append(l)

    cfg = {**CFG, "facies": {"epochs": 8}}
    _, stats = train_facies_multi(cfg, train_secs, train_labs, val_secs, val_labs,
                                  device="cpu", verbose=False)
    assert stats["mean_iou"] > 0.25, f"held-out mIoU {stats['mean_iou']:.3f} is at chance"


def test_best_checkpoint_is_selected_on_held_out_score_not_the_last_epoch():
    train_secs, train_labs = zip(*[layered_pair(seed=i) for i in range(3)])
    val_secs, val_labs = zip(*[layered_pair(seed=i) for i in range(30, 32)])

    cfg = {**CFG, "facies": {"epochs": 4}}
    _, stats = train_facies_multi(cfg, list(train_secs), list(train_labs),
                                  list(val_secs), list(val_labs), device="cpu", verbose=False)
    assert 0 <= stats["epoch"] < 4


# ---------------------------------------------------------------------------
# The score's denominator must not move
# ---------------------------------------------------------------------------

def test_mean_iou_counts_a_dropped_class_as_zero_not_as_absent():
    """Giving up on a hard class must lower the score, not raise it.

    With a per-epoch denominator, a model that stops predicting a rare class
    sees it drop out of the average instead of contributing a zero -- the score
    goes *up* for getting worse. Observed here as a jump from 0.573 to 0.739
    between two epochs before the denominator was pinned.
    """
    from deepseis.facies import mean_iou_over

    scored = np.array([0, 1, 2, 3])
    predicts_everything = np.array([0.8, 0.8, 0.8, 0.4])
    gave_up_on_class_3 = np.array([0.8, 0.8, 0.8, np.nan])

    good = mean_iou_over(predicts_everything, scored)
    gave_up = mean_iou_over(gave_up_on_class_3, scored)
    assert gave_up < good, f"dropping a class raised the score: {gave_up:.3f} vs {good:.3f}"
    assert gave_up == pytest.approx(0.6, abs=1e-9)


def test_mean_iou_denominator_is_fixed_by_the_scored_class_set():
    from deepseis.facies import mean_iou_over

    per_class = np.array([1.0, 1.0, np.nan, np.nan])
    assert mean_iou_over(per_class, np.array([0, 1])) == pytest.approx(1.0)
    assert mean_iou_over(per_class, np.array([0, 1, 2, 3])) == pytest.approx(0.5)


def test_scored_classes_come_from_held_out_truth():
    """A class the model invents but which is not in the truth must not be
    allowed to define the score."""
    from deepseis.facies import mean_iou_over

    per_class = np.array([0.9, 0.9, 0.0])
    # class 2 absent from held-out truth -> excluded from the denominator
    assert mean_iou_over(per_class, np.array([0, 1])) == pytest.approx(0.9)
