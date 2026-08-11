#!/usr/bin/env python3
"""
Controllable synthetic noise CLI (spec Sec. 5 MVP / Sec. 10 Phase 1).

Generates (or loads) a clean seismic section, injects noise per the config's
``data.noise`` block, and reports the *baseline* PSNR/SSIM of the raw noisy
section against the known-clean reference — i.e. how bad it is before
DeepSeis touches it, which is the number the denoiser's improvement gets
measured against.

The actual noise-generation logic lives in ``deepseis.io.noise`` /
``deepseis.io.synthetic`` (reusable by the rest of the library); this script
is just the command-line entry point the spec's repo layout calls for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from deepseis.io import noise as noise_mod
from deepseis.io import synthetic as synth_mod
from deepseis import metrics as metrics_mod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent.parent / "configs" / "default.yaml"))
    parser.add_argument("--clean-input", type=str, default=None,
                         help="optional .npy clean section to use instead of the synthetic generator")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "synthetic"))
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rng = np.random.default_rng(cfg["data"]["synthetic"]["random_seed"])

    if args.clean_input:
        clean = np.load(args.clean_input)
        fault_mask = facies = horizon_depths = None
    else:
        vol = synth_mod.generate_from_config(cfg)
        clean, fault_mask, facies, horizon_depths = vol.clean, vol.fault_mask, vol.facies, vol.horizon_depths

    noisy = noise_mod.make_noisy(clean, cfg, rng=rng)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "clean.npy", clean)
    np.save(out_dir / "noisy.npy", noisy)
    if fault_mask is not None:
        np.save(out_dir / "fault_mask.npy", fault_mask)
        np.save(out_dir / "facies.npy", facies)
        np.save(out_dir / "horizon_depths.npy", horizon_depths)

    print(f"[synth_noise] wrote clean/noisy to {out_dir}/")
    print("[synth_noise] baseline (pre-denoising) quality vs. known-clean reference:")
    print(f"  PSNR = {metrics_mod.psnr(noisy, clean):.2f} dB")
    print(f"  SNR  = {metrics_mod.snr_db(noisy, clean):.2f} dB")
    print(f"  SSIM = {metrics_mod.ssim(noisy, clean):.4f}")
    print(f"  noise config: {cfg['data']['noise']}")


if __name__ == "__main__":
    main()
