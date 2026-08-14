"""
Facies head trained across the survey, scored on held-out inlines.

The previous facies head was fitted on inline 0 and reported at 94.4% pixel
accuracy on inline 0. The README was straight about what that meant --
"goodness of fit, not generalization" -- but a number that cannot distinguish
the two is not worth reporting at all, and there was no way to produce a
better one because nothing in the project held any data back.

That is fixed here by reusing the denoiser's split layout: the head trains on
patches drawn from many inlines in the training block and is scored on inlines
from the validation block, separated by a buffer wide enough for the
inter-line correlation to decay. The checkpoint kept is the best on held-out
mean IoU rather than the last one.

The head runs on *denoised* input, which is the point of the pipeline: if
denoising helps interpretation, it should show up here as a better score, and
if it removes geology it should show up as a worse one.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from deepseis.io import patches as patch_mod
from deepseis.models.facies import FaciesNet2D


def per_class_iou(pred: np.ndarray, target: np.ndarray, n_classes: int) -> np.ndarray:
    """IoU per class; NaN only for classes absent from both prediction and truth."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        p, t = pred == c, target == c
        union = np.logical_or(p, t).sum()
        if union:
            out[c] = np.logical_and(p, t).sum() / union
    return out


def mean_iou_over(per_class: np.ndarray, scored_classes: np.ndarray) -> float:
    """Mean IoU over a **fixed** set of classes.

    The class set has to be pinned to the ones present in the ground truth,
    decided once, rather than derived per epoch from which classes happen to
    be non-NaN. Averaging only over classes the model still predicts lets it
    raise its own score by giving up on a hard one: the class drops out of the
    average instead of contributing a zero. That is the same failure the
    inverse-frequency weighting exists to prevent, reappearing in the metric
    rather than the loss, and it showed up here as a jump from 0.573 to 0.739
    between two epochs.

    Classes present in the truth but never predicted score 0 and stay in.
    """
    if scored_classes.size == 0:
        return float("nan")
    vals = per_class[scored_classes]
    return float(np.where(np.isnan(vals), 0.0, vals).mean())


@torch.no_grad()
def predict_section(model: FaciesNet2D, section: np.ndarray, patch_size: int, stride: int,
                    device: str = "cpu") -> np.ndarray:
    """Per-pixel class prediction, combined across overlapping patches by majority vote.

    Votes rather than averaged labels: class indices are categorical, so
    averaging them invents classes that neither patch predicted. On F3, whose
    units are stacked in stratigraphic order, averaging a 2 and a 4 produced a
    spurious band of unit 3 along every boundary between them.
    """
    model.eval()
    h, w = section.shape
    patches = patch_mod.extract_patches(section, patch_size, stride)
    x = torch.from_numpy(patches.astype(np.float32)).unsqueeze(1).to(device)

    logits = model(x)
    preds = logits.argmax(dim=1).cpu().numpy()
    n_classes = int(logits.shape[1])

    votes = np.zeros((n_classes, h, w), dtype=np.int16)
    for (y, x0), patch in zip(patch_mod.patch_coords(h, w, patch_size, stride), preds):
        ys, xs = np.mgrid[y:y + patch_size, x0:x0 + patch_size]
        np.add.at(votes, (patch, ys, xs), 1)
    return votes.argmax(axis=0).astype(np.int32)


def train_facies_multi(cfg: dict, train_sections, train_labels, val_sections, val_labels,
                       device: str = "cpu", verbose: bool = True):
    """Train on many sections, select on held-out mean IoU.

    ``train_sections`` / ``val_sections`` are already-denoised 2D sections;
    the label arrays are the matching integer facies maps.
    """
    fcfg = cfg["facies"]
    pcfg = cfg["data"]["patch"]
    rng = np.random.default_rng(cfg["seed"] + 2)
    torch.manual_seed(cfg["seed"])

    n_classes = int(max(int(l.max()) for l in train_labels)) + 1

    model = FaciesNet2D(in_channels=1, n_classes=n_classes,
                        base_channels=cfg["faultseg"]["base_channels"],
                        depth=cfg["faultseg"]["depth"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["faultseg"]["lr"])

    xs, ys = [], []
    for sec, lab in zip(train_sections, train_labels):
        xs.append(patch_mod.extract_patches(sec, pcfg["size"], pcfg["stride"]))
        ys.append(patch_mod.extract_patches(lab.astype(np.float32),
                                            pcfg["size"], pcfg["stride"]).astype(np.int64))
    x_patches = np.concatenate(xs)
    y_patches = np.concatenate(ys)

    # Trace-order flip only. A vertical flip reverses the time axis, and F3's
    # units are stacked in stratigraphic order, so it would teach the head an
    # inverted stratigraphy that does not occur.
    x_patches = np.concatenate([x_patches, x_patches[:, :, ::-1]])
    y_patches = np.concatenate([y_patches, y_patches[:, :, ::-1]])

    # Inverse-frequency class weights: F3's units are very unevenly
    # represented, and unweighted cross-entropy minimises total error by never
    # predicting the rare ones -- which is what it did, scoring 0% IoU on
    # Triassic while every other unit scored 80-99%.
    counts = np.bincount(y_patches.reshape(-1), minlength=n_classes).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    if verbose:
        print(f"[facies] {len(x_patches)} patches from {len(train_sections)} sections, "
              f"{n_classes} classes")
        print(f"[facies] class weights: {np.round(weights, 2).tolist()}")

    # Fixed once, from the held-out ground truth, so the score's denominator
    # cannot drift between epochs.
    scored_classes = np.unique(np.concatenate([l.ravel() for l in val_labels]))
    scored_classes = scored_classes[(scored_classes >= 0) & (scored_classes < n_classes)]
    if verbose:
        print(f"[facies] scoring mIoU over classes present in held-out truth: "
              f"{scored_classes.tolist()}")

    best = {"iou": -np.inf, "epoch": -1, "state": None, "per_class": None}
    batch = int(cfg["faultseg"]["batch_size"])

    for epoch in range(int(fcfg.get("epochs", 24))):
        model.train()
        idx = np.arange(len(x_patches))
        rng.shuffle(idx)
        total, n_batches = 0.0, 0
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            x = torch.from_numpy(np.ascontiguousarray(x_patches[b])).unsqueeze(1).float().to(device)
            y = torch.from_numpy(np.ascontiguousarray(y_patches[b])).long().to(device)

            loss = nn.functional.cross_entropy(model(x), y, weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1

        ious = []
        for sec, lab in zip(val_sections, val_labels):
            pred = predict_section(model, sec, pcfg["size"], pcfg["stride"], device)
            ious.append(per_class_iou(pred, lab, n_classes))
        stacked = np.stack(ious)
        with np.errstate(invalid="ignore"):
            per_class = np.array([np.nan if np.all(np.isnan(col)) else np.nanmean(col)
                                  for col in stacked.T])
        mean_iou = mean_iou_over(per_class, scored_classes)

        if mean_iou > best["iou"]:
            best = {"iou": mean_iou, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "per_class": per_class}

        if verbose:
            print(f"[facies] epoch {epoch:02d}  loss={total / max(n_batches, 1):.4f}  "
                  f"held-out mIoU={mean_iou:.3f}" + ("  <- best" if best["epoch"] == epoch else ""),
                  flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if verbose:
        print(f"[facies] best held-out mIoU {best['iou']:.3f} at epoch {best['epoch']}")
        print(f"[facies] per-class IoU: {np.round(best['per_class'], 3).tolist()}")

    # `scored_classes` has to travel with the numbers. A per-class IoU of 0.0
    # means two completely different things depending on membership: for a
    # scored class it is a genuine failure to segment a unit that is there,
    # while for an unscored one it means the unit is absent from the held-out
    # block and the model predicted it anyway -- a false positive that does not
    # enter the mean. Without this field a reader cannot tell them apart.
    false_positive_classes = [int(c) for c in range(n_classes)
                              if c not in scored_classes.tolist()
                              and best["per_class"] is not None
                              and not np.isnan(best["per_class"][c])]

    return model, {"mean_iou": best["iou"], "per_class_iou": best["per_class"].tolist(),
                   "epoch": best["epoch"], "n_classes": n_classes,
                   "scored_classes": [int(c) for c in scored_classes],
                   "classes_absent_but_predicted": false_positive_classes}
