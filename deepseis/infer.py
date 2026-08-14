"""
Inference entry point.

Loads trained checkpoints from a run directory and applies the full
pipeline (denoise -> fault segmentation -> horizons -> [diffusion refine])
to a new input volume/section.

Usage:
    python -m deepseis.infer --input data/f3_inline_400.npy --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deepseis.io import segy as segy_mod
from deepseis.interpretation.horizon import track_horizons
from deepseis.models.faultseg import FaultSegNet2D
from deepseis.models.unet import DenoiserUNet
from deepseis.train import (get_device, load_config, run_denoiser_inference, run_faultseg_inference, set_seed)
from deepseis import metrics as metrics_mod


def load_denoiser(cfg: dict, checkpoint_path: str | Path, device: str) -> DenoiserUNet:
    model = DenoiserUNet(**cfg["model"]["denoiser"]).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_faultseg(cfg: dict, checkpoint_path: str | Path, device: str) -> FaultSegNet2D:
    fcfg = cfg["faultseg"]
    model = FaultSegNet2D(in_channels=1, base_channels=fcfg["base_channels"], depth=fcfg["depth"]).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_pipeline(cfg: dict, section: np.ndarray, run_dir: Path, device: str,
                 blindspot_checkpoint: Path | None = None) -> dict:
    """Denoise, then segment faults and track horizons on the result.

    Prefers the blind-spot denoiser when a checkpoint for it is available. It
    carries its own normalisation constants and runs on the whole section in
    one pass, so neither the caller's per-section scaling nor the old
    patch-and-average stitching applies to it.
    """
    if blindspot_checkpoint is not None and Path(blindspot_checkpoint).exists():
        from deepseis.denoise import load_denoiser as load_blindspot
        from deepseis.denoise import serve_section

        model, noise_model, ckpt = load_blindspot(blindspot_checkpoint, device)
        denoised = serve_section(model, noise_model, ckpt, section, device)
        out = {"noisy": section, "denoised": denoised}
        print(f"[deepseis] denoised with the blind-spot model ({blindspot_checkpoint})")
    else:
        denoiser = load_denoiser(cfg, run_dir / cfg["output"]["checkpoint_name"], device)
        denoised = run_denoiser_inference(denoiser, section, cfg, device)
        out = {"noisy": section, "denoised": denoised}

    faultseg_ckpt = run_dir / cfg["output"]["faultseg_checkpoint_name"]
    if faultseg_ckpt.exists():
        faultseg = load_faultseg(cfg, faultseg_ckpt, device)
        out["fault_prob_noisy"] = run_faultseg_inference(faultseg, section, cfg, device)
        out["fault_prob_denoised"] = run_faultseg_inference(faultseg, denoised, cfg, device)

        if cfg["horizon"]["enabled"]:
            fault_binary = out["fault_prob_denoised"] >= cfg["faultseg"]["dice_threshold"]
            out["horizons"] = track_horizons(denoised, n_horizons=cfg["horizon"]["n_horizons"],
                                              fault_mask=fault_binary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help=".npy or .sgy/.segy input section")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-dir", type=str, default=None, help="defaults to output.run_dir from config")
    parser.add_argument("--output", type=str, default=None, help="where to save the .npz result bundle")
    parser.add_argument("--blindspot", type=str, default="runs/denoise_field/blindspot.pt",
                        help="blind-spot denoiser checkpoint; used when present, otherwise the "
                             "legacy masking denoiser from --run-dir is used")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)
    set_seed(cfg["seed"])

    run_dir = Path(args.run_dir or cfg["output"]["run_dir"])
    section = segy_mod.load_volume(args.input)
    if section.ndim == 3:
        # SEG-Y cubes load as (n_inlines, n_crosslines, n_samples); take inline 0 and
        # transpose to our (n_samples, n_traces) convention.
        section = section[0].T

    section = section.astype(np.float32)
    blindspot = Path(args.blindspot) if args.blindspot else None
    use_blindspot = blindspot is not None and blindspot.exists()

    if not use_blindspot:
        # Legacy path only: the masking denoiser was fitted on unit-std data
        # and has no record of the scale it was trained at, so the section has
        # to be normalised here. The blind-spot checkpoint stores its own
        # global constants and applies them itself -- doing it here as well
        # would scale the input twice, which is the same class of train/serve
        # mismatch that cost the old pipeline a factor of 4.4.
        input_std = section.std()
        if input_std > 1e-8:
            section = section / input_std
        else:
            print("[deepseis] warning: input has near-zero std — check your input file.")

    result = run_pipeline(cfg, section, run_dir, device,
                          blindspot_checkpoint=blindspot if use_blindspot else None)

    out_path = Path(args.output or (run_dir / "inference_result.npz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    print(f"[deepseis] inference complete -> {out_path}")

    # quick console summary if we have fault probabilities to compare
    if "fault_prob_noisy" in result:
        n_faultish_noisy = float((result["fault_prob_noisy"] >= cfg["faultseg"]["dice_threshold"]).mean())
        n_faultish_denoised = float((result["fault_prob_denoised"] >= cfg["faultseg"]["dice_threshold"]).mean())
        print(json.dumps({"fault_pixel_fraction_noisy_input": n_faultish_noisy,
                           "fault_pixel_fraction_denoised_input": n_faultish_denoised}, indent=2))


if __name__ == "__main__":
    main()
