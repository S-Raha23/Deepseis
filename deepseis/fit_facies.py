"""
Fit the facies head on the real F3 facies labels.

The facies checkpoint shipped with the default run was trained on the synthetic
generator's facies, which has 3 classes. Real F3 (Alaudah et al. 2019) has 6 --
Upper/Middle/Lower North Sea, Rijnland/Chalk, Jurassic, Triassic -- so that
checkpoint cannot classify F3 at all; it will not even load into a 6-class head.

This trains *only* the facies head, reusing the existing denoiser checkpoint and
leaving the denoiser, FaultSeg and diffusion artifacts untouched. That keeps the
working dashboard exactly as it is and just gives the Facies tab something real
to show, in about a minute rather than a full 25-minute retrain.

The input pipeline deliberately mirrors what the dashboard feeds at inference
time -- inline 0, normalized to unit std, then passed through the denoiser -- so
the head is fitted on the same distribution it will be served.

Usage:
    python -m deepseis.fit_facies
    python -m deepseis.fit_facies --epochs 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from deepseis.models.unet import DenoiserUNet
from deepseis.train import (load_config, get_device, set_seed,
                             run_denoiser_inference, train_facies)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_SEISMIC = "data/raw/f3/data/train/train_seismic.npy"
DEFAULT_LABELS = "data/raw/f3/data/train/train_labels.npy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--seismic", default=DEFAULT_SEISMIC)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--inline", type=int, default=0,
                     help="which inline to fit on (default 0, the one the dashboard shows)")
    ap.add_argument("--epochs", type=int, default=None, help="override the training epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)
    set_seed(cfg["seed"])
    run_dir = REPO_ROOT / cfg["output"]["run_dir"]

    seismic_path = REPO_ROOT / args.seismic
    labels_path = REPO_ROOT / args.labels
    for p in (seismic_path, labels_path):
        if not p.exists():
            raise SystemExit(f"[facies] {p} not found. Run: python data/download_f3.py")

    # --- the dashboard's exact input path -----------------------------------
    vol = np.load(seismic_path, mmap_mode="r")
    section = np.asarray(vol[args.inline], dtype=np.float32).T
    section = section / (float(section.std()) + 1e-8)

    labels = np.asarray(np.load(labels_path, mmap_mode="r")[args.inline]).T.astype(np.int64)
    classes = np.unique(labels)
    print(f"[facies] inline {args.inline}: section {section.shape}, "
          f"labels {labels.shape}, classes {classes.tolist()}")

    ckpt = run_dir / cfg["output"]["checkpoint_name"]
    if not ckpt.exists():
        raise SystemExit(f"[facies] {ckpt} not found. Train the denoiser first: "
                         f"python -m deepseis.train --config {args.config}")
    model = DenoiserUNet(**cfg["model"]["denoiser"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"[facies] denoising inline {args.inline} with {ckpt.name} ...")
    denoised = run_denoiser_inference(model, section, cfg, device)

    if args.epochs is not None:
        cfg.setdefault("facies", {})["epochs"] = args.epochs

    print(f"[facies] fitting the facies head on {int(classes.max()) + 1} real classes ...")
    facies_model = train_facies(cfg, denoised, labels, device)

    out = run_dir / "facies.pt"
    torch.save(facies_model.state_dict(), out)
    print(f"[facies] wrote {out}")

    # --- report fit quality, honestly ---------------------------------------
    from deepseis.io import patches as patch_mod
    from deepseis.io.patches import patch_coords
    from deepseis.train import to_tensor
    pcfg = cfg["data"]["patch"]
    h, w = denoised.shape
    pts = patch_mod.extract_patches(denoised, pcfg["size"], pcfg["stride"])
    facies_model.eval()
    with torch.no_grad():
        preds = facies_model(to_tensor(pts, device)).argmax(dim=1).cpu().numpy()
    acc_map = np.zeros((h, w), np.int32)
    cnt = np.zeros((h, w), np.int32)
    for (y, x0), p in zip(patch_coords(h, w, pcfg["size"], pcfg["stride"]), preds):
        acc_map[y:y + pcfg["size"], x0:x0 + pcfg["size"]] += p
        cnt[y:y + pcfg["size"], x0:x0 + pcfg["size"]] += 1
    cnt[cnt == 0] = 1
    pred = (acc_map / cnt).round().astype(np.int64)

    print(f"\n[facies] pixel accuracy on inline {args.inline}: "
          f"{100 * float((pred == labels).mean()):.1f}%  "
          f"(this is FIT, not generalization -- same inline it was trained on)")
    for c in range(int(classes.max()) + 1):
        pc, gc = pred == c, labels == c
        union = int(np.logical_or(pc, gc).sum())
        iou = 100 * int(np.logical_and(pc, gc).sum()) / union if union else float("nan")
        print(f"           class {c}: predicted {100 * pc.mean():5.1f}%  "
              f"actual {100 * gc.mean():5.1f}%  IoU {iou:5.1f}%")


if __name__ == "__main__":
    main()
