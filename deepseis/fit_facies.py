"""
Fit only the facies head, leaving every other checkpoint alone.

Useful when the denoiser and FaultSeg artifacts are already good and only the
facies head needs refitting — it takes minutes rather than a full retrain.

What changed: this used to fit the head on **inline 0 alone** and report its
accuracy on inline 0, which cannot distinguish fitting from generalising. It
now delegates to :func:`deepseis.train.train_facies_head`, which trains across
the training block and scores on held-out inlines behind a buffer, exactly as
the denoiser does. The held-out numbers are lower than the old single-inline
figure and mean something.

Usage:
    python -m deepseis.fit_facies
    python -m deepseis.fit_facies --epochs 40 --train-sections 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepseis.train import get_device, load_config, resolve_denoiser, set_seed, train_facies_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--blindspot", default="runs/denoise_field/blindspot.pt",
                        help="denoiser whose output the head is fitted on")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-sections", type=int, default=None)
    parser.add_argument("--val-sections", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["facies"]["epochs"] = args.epochs
    if args.train_sections is not None:
        cfg["facies"]["n_train_sections"] = args.train_sections
    if args.val_sections is not None:
        cfg["facies"]["n_val_sections"] = args.val_sections

    device = get_device(cfg)
    set_seed(cfg["seed"])
    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    denoise_fn, label = resolve_denoiser(cfg, args, device)
    print(f"[facies] fitting on output of the {label} denoiser")

    stats = train_facies_head(cfg, denoise_fn, run_dir, device)
    if stats is None:
        return

    results_path = run_dir / "facies_metrics.json"
    with open(results_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[facies] held-out mIoU {stats['mean_iou']:.3f} on inlines {stats['val_inlines']}")
    print(f"[facies] wrote {run_dir / 'facies.pt'} and {results_path}")


if __name__ == "__main__":
    main()
