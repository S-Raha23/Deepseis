"""
Training and inference for the blind-spot denoiser.

Replaces the ``train_denoiser`` / ``run_denoiser_inference`` pair in
``train.py``. Beyond the model change, three things about the *procedure* are
different.

**Inference does not patch-and-average.** The old path cut the section into
64x64 windows at stride 32, ran each, and averaged the four overlapping
predictions per pixel through a Hann window. Averaging four different
estimates of the same pixel is a smoothing operator applied *after* the model,
so it removed resolution unconditionally -- and it was necessary only because
``GroupNorm`` made the output depend on which window a pixel landed in. With
the normalisation layers gone the network is fully convolutional and exactly
translation equivariant, so the whole section runs in one pass and the answer
is the same one the training objective was written for. A tiled path remains
for sections too large to fit in memory, but it crops each tile's interior
rather than blending overlaps.

**Validation is on held-out lines, against a known reference.** Every epoch
the model is scored on inlines from the validation block -- separated from the
training block by a buffer wide enough for the inter-line correlation to decay
-- and the checkpoint kept is the best one on that score, not the last one.

**The reported objective is not the training objective.** Model selection uses
SNR against the known-clean section in semi-synthetic mode; in field mode,
where no such reference exists, it uses the validation likelihood. Neither is
the leakage number the project used to headline, because leakage alone rewards
doing nothing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from deepseis import noise_estimate
from deepseis.io.dataset import PatchSampler, SeismicVolume, inject_noise, section_seed
from deepseis.losses.geology import dip_fan_loss, fault_contrast_loss
from deepseis.losses.nll import blindspot_nll, posterior_mean, posterior_std
from deepseis.postprocess import orthogonalize
from deepseis.models.blindspot import (BlindSpotDenoiser, NoiseModel,
                                       receptive_field_halfwidth)


#: Validation metrics where a smaller number is a better model.
LOWER_IS_BETTER = {"nll", "leakage", "residual_coherence"}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_section(model: BlindSpotDenoiser, noise_model: NoiseModel, section: np.ndarray,
                    device: str = "cpu", return_extras: bool = False,
                    estimate_noise_per_section: bool = False):
    """Denoise one full section in a single forward pass.

    ``section`` is ``(n_samples, n_traces)`` for a single-channel model, or
    ``(channels, n_samples, n_traces)`` when neighbouring lines are used; the
    result is always the 2D estimate for the centre line.

    ``estimate_noise_per_section`` re-measures the noise level from this
    section rather than using the survey-wide value stored with the
    checkpoint. That is not a train/serve mismatch of the kind this rewrite
    removed: ``sigma_n`` is a measured property of the input, not a fitted
    parameter, and the network's own outputs are untouched. It matters because
    the noise level genuinely varies between lines -- across F3's training
    block the injected noise standard deviation ranges from 0.40 to 0.54 --
    so a single global value over-denoises the quiet lines and under-denoises
    the loud ones.
    """
    model.eval()
    arr = np.asarray(section, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    x = torch.from_numpy(arr)[None].to(device)          # (1, C, H, W)

    mu, sigma_p = model(x)
    centre_idx = model.in_channels // 2
    centre = x[:, centre_idx: centre_idx + 1]

    if estimate_noise_per_section:
        profile = noise_estimate.estimate_noise_profile(
            arr[centre_idx], n_samples=arr.shape[-2],
            n_bands=1 if noise_model.mode == "scalar" else 6)
        sigma_n = torch.from_numpy(profile.astype(np.float32)).view(1, 1, -1, 1).to(device)
    else:
        sigma_n = noise_model.sigma(x.shape[-2], depth_offset=0, device=device).to(device)

    x_hat = posterior_mean(mu, sigma_p, sigma_n, centre)
    out = x_hat[0, 0].cpu().numpy()

    if not return_extras:
        return out
    return out, {
        "mu": mu[0, 0].cpu().numpy(),
        "sigma_prior": sigma_p[0, 0].cpu().numpy(),
        "uncertainty": posterior_std(sigma_p, sigma_n)[0, 0].cpu().numpy(),
        "sigma_noise": sigma_n[0, 0, :, 0].cpu().numpy(),
    }


@torch.no_grad()
def denoise_section_tiled(model: BlindSpotDenoiser, noise_model: NoiseModel, section: np.ndarray,
                          device: str = "cpu", tile: int = 512,
                          margin: int | None = None) -> np.ndarray:
    """Memory-bounded inference that keeps only each tile's interior.

    Overlapping tiles are *cropped*, never blended: blending is averaging, and
    averaging is exactly the post-hoc smoothing this rewrite set out to remove.
    Cropping is only equivalent to a single pass if the discarded margin is at
    least as wide as the receptive field, so the margin is measured from the
    model rather than guessed -- it is 100 pixels at the default depth of 3,
    which is more than the 64 a plausible-looking guess would have used.
    """
    arr = np.asarray(section, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    _, h, w = arr.shape
    if w <= tile:
        return denoise_section(model, noise_model, arr, device)

    if margin is None:
        margin = receptive_field_halfwidth(model)
    step = tile - 2 * margin
    if step <= 0:
        raise ValueError(f"tile={tile} is too small for a receptive field of {margin} px; "
                         f"it must exceed 2*{margin}")

    out = np.zeros((h, w), dtype=np.float32)
    x0 = 0
    while x0 < w:
        lo = max(x0 - margin, 0)
        hi = min(x0 + step + margin, w)
        piece = denoise_section(model, noise_model, arr[:, :, lo:hi], device)
        keep_hi = min(x0 + step, w)
        out[:, x0:keep_hi] = piece[:, x0 - lo: keep_hi - lo]
        x0 += step
    return out


def _warn_if_already_normalized(section: np.ndarray, scale: float, tol: float = 0.25) -> None:
    """Catch the caller having normalised the section before handing it over.

    This mistake has been made at three separate call sites in this project and
    is invisible every time. ``serve_section`` divides by the checkpoint's
    scale; if the caller has already divided by roughly the same number, the
    section is scaled twice. The network is scale equivariant so its own output
    tracks that harmlessly, but ``sigma_n`` is a fixed physical noise level, so
    the *ratio* between them shifts and the denoiser quietly under-removes --
    measured at 3.17 dB and a 4x drop in energy removed, with no error raised
    and no visible artefact.

    The signature is a section whose amplitude sits near unity when the
    training data's did not. A survey genuinely recorded at unit amplitude
    would trip this too, hence a warning rather than an exception.
    """
    if not np.isfinite(scale) or scale <= 0 or abs(scale - 1.0) < tol:
        return                      # training data was already ~unit scale: no signal here
    std = float(np.std(section))
    if std <= 0:
        return
    if abs(std - 1.0) < tol:
        import warnings

        warnings.warn(
            f"serve_section received a section with std={std:.3f} but the checkpoint was "
            f"fitted on data with scale={scale:.3f}. This usually means the section was "
            f"already normalised by the caller, which normalises it twice and makes the "
            f"denoiser silently under-remove. Pass raw amplitude.",
            RuntimeWarning, stacklevel=3,
        )


def serve_section(model: BlindSpotDenoiser, noise_model: NoiseModel, ckpt: dict,
                  section_raw: np.ndarray, device: str = "cpu", return_extras: bool = False):
    """Denoise a raw-amplitude section exactly as the checkpoint prescribes.

    The single entry point every consumer should use -- the dashboard,
    ``infer.py``, and anything downstream -- so that normalisation and
    post-processing cannot drift apart between them. It applies, in order:

    1. the checkpoint's stored global normalisation (never a per-section one,
       which is what produced the old 4.4x train/serve mismatch),
    2. the network and its posterior mean,
    3. whichever post-processing the config enables, and
    4. the inverse normalisation, so the result comes back in input units.
    """
    cfg = ckpt.get("config", {})
    post = cfg.get("postprocess", {}) or {}
    scale = ckpt["normalization"]["scale"]
    offset = ckpt["normalization"]["offset"]

    arr = np.asarray(section_raw, dtype=np.float32)
    _warn_if_already_normalized(arr, scale)

    normalized = (arr - offset) / scale
    denoised, extras = denoise_section(
        model, noise_model, normalized, device, return_extras=True,
        estimate_noise_per_section=bool(post.get("per_section_sigma", False)))

    if post.get("orthogonalize", False):
        centre = normalized if normalized.ndim == 2 else normalized[model.in_channels // 2]
        denoised, _ = orthogonalize(centre, denoised,
                                    window=int(post.get("ortho_window", 11)),
                                    iterations=int(post.get("ortho_iterations", 1)))

    out = denoised * scale + offset
    if not return_extras:
        return out
    extras["uncertainty"] = extras["uncertainty"] * scale
    return out, extras


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _val_sections(volume: SeismicVolume, cfg: dict, split: str = "val"):
    """Yield ``(noisy_input, clean_reference_or_None)`` for held-out lines."""
    idx = volume.split_indices(split)
    n = min(int(cfg["eval"]["n_sections"]), len(idx))
    chosen = idx[np.linspace(0, len(idx) - 1, n).astype(int)]

    n_neighbors = int(cfg["model"]["n_neighbors"])
    noise_cfg = cfg["data"].get("inject_noise") or None

    for i in chosen:
        stack = volume.inline_stack(int(i), n_neighbors)
        clean = stack[n_neighbors].copy() if noise_cfg else None
        if noise_cfg:
            seed = section_seed(int(cfg["data"]["noise_seed"]), "inline", int(i))
            stack = np.stack([inject_noise(stack[c], noise_cfg, seed + 104729 * c)
                              for c in range(stack.shape[0])])
        yield int(i), stack, clean


def evaluate_split(model, noise_model, volume: SeismicVolume, cfg: dict,
                   split: str = "val", device: str = "cpu") -> dict:
    """Mean metrics over held-out sections.

    Includes ``nll``, the predictive negative log likelihood of the held-out
    data under the model. That is the criterion field mode selects on, because
    it is the only one available without a reference that a do-nothing model
    cannot win: the prediction is made without ever seeing the pixel being
    scored, so refusing to denoise buys nothing. Selecting on leakage instead
    -- the obvious no-reference choice, and the one this project used to
    headline -- actively selects for the model that removes least.
    """
    from deepseis import metrics as metrics_mod

    rows = []
    for _, stack, clean in _val_sections(volume, cfg, split):
        denoised, extras = denoise_section(model, noise_model, stack, device, return_extras=True)
        centre = stack[cfg["model"]["n_neighbors"]]
        report = metrics_mod.denoising_report(centre, denoised, clean)

        var = extras["sigma_prior"] ** 2 + extras["sigma_noise"][:, None] ** 2
        report["nll"] = float((0.5 * (centre - extras["mu"]) ** 2 / var + 0.5 * np.log(var)).mean())
        rows.append(report)

    agg: dict = {}
    for key in rows[0]:
        if key == "spectral_retention":
            agg[key] = {b: float(np.mean([r[key][b] for r in rows])) for b in rows[0][key]}
        else:
            agg[key] = float(np.mean([r[key] for r in rows]))
    return agg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict, device: str = "cpu", verbose: bool = True) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    volume = SeismicVolume(cfg["data"]["volume_path"], buffer=int(cfg["data"]["split_buffer"]))
    if verbose:
        print(f"[data] {volume.path.name} {volume.volume.shape}  splits={volume.splits.as_dict()}", flush=True)
        print(f"[data] global scale={volume.scale:.5f} offset={volume.offset:.5f}", flush=True)

    n_neighbors = int(cfg["model"]["n_neighbors"])
    sampler = PatchSampler(
        volume,
        n_inline_sections=int(cfg["data"]["n_inline_sections"]),
        n_crossline_sections=int(cfg["data"]["n_crossline_sections"]),
        n_neighbors=n_neighbors,
        rng=rng,
        noise_cfg=cfg["data"].get("inject_noise") or None,
        noise_seed=int(cfg["data"]["noise_seed"]),
    )
    if verbose:
        print(f"[data] {len(sampler)} training sections, {2 * n_neighbors + 1} input channel(s)", flush=True)

    model = BlindSpotDenoiser(
        in_channels=2 * n_neighbors + 1,
        base_channels=int(cfg["model"]["base_channels"]),
        depth=int(cfg["model"]["depth"]),
    ).to(device)
    # Measure the noise level rather than leaving it to a degenerate
    # optimisation. The likelihood constrains only sigma_p^2 + sigma_n^2, so
    # the split between "unpredictable" and "noise" is barely identifiable and
    # a partly-trained model splits it close to backwards -- which directly
    # sets how much the denoiser removes. See deepseis/noise_estimate.py.
    estimate_cfg = cfg["model"].get("estimate_noise", True)
    init_profile = None
    if estimate_cfg:
        centre_sections = [s[n_neighbors] for s in sampler.sections]
        init_profile = noise_estimate.estimate_from_sections(
            centre_sections, volume.n_samples, mode=cfg["model"]["noise_model"])
        if verbose:
            print(f"[noise] estimated sigma_n from data: min {init_profile.min():.4f} "
                  f"median {np.median(init_profile):.4f} max {init_profile.max():.4f}", flush=True)

    noise_model = NoiseModel(
        mode=cfg["model"]["noise_model"],
        n_depth=volume.n_samples,
        init_sigma=float(cfg["model"]["init_sigma"]),
        init_profile=init_profile,
        freeze=bool(cfg["model"].get("freeze_noise", False)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[model] {n_params:,} parameters, noise model = {cfg['model']['noise_model']}"
              f"{' (frozen)' if noise_model.frozen else ''}", flush=True)

    groups = [{"params": model.parameters()}]
    if not noise_model.frozen:
        groups.append({"params": noise_model.parameters(),
                       "lr": float(cfg["training"]["noise_lr"])})
    opt = torch.optim.Adam(groups, lr=float(cfg["training"]["lr"]))
    epochs = int(cfg["training"]["epochs"])
    steps = int(cfg["training"]["steps_per_epoch"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)

    lam_fault = float(cfg["training"]["lambda_fault_contrast"])
    lam_dip = float(cfg["training"]["lambda_dip_fan"])
    structure_on = cfg["training"]["structure_preservation_enabled"]
    patch = int(cfg["data"]["patch_size"])
    centre_c = n_neighbors

    history: list[dict] = []
    best = {"score": -np.inf, "epoch": -1}
    select_on = cfg["eval"]["select_on"]
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        totals = {"nll": 0.0, "fault": 0.0, "dip": 0.0, "total": 0.0}

        for _ in range(steps):
            patches, offsets = sampler.sample_batch(int(cfg["training"]["batch_size"]), patch)
            x = torch.from_numpy(patches).to(device)
            y = x[:, centre_c: centre_c + 1]

            mu, sigma_p = model(x)
            # Every patch in the batch starts at a different depth, so the
            # depth-varying noise model is indexed per sample.
            sigma_n = torch.cat([noise_model.sigma(patch, int(o), device) for o in offsets], dim=0)

            nll = blindspot_nll(mu, sigma_p, sigma_n, y)
            total = nll
            fault_l = torch.zeros((), device=device)
            dip_l = torch.zeros((), device=device)

            if structure_on and (lam_fault > 0 or lam_dip > 0):
                x_hat = posterior_mean(mu, sigma_p, sigma_n, y)
                if lam_fault > 0:
                    fault_l = fault_contrast_loss(x_hat, y)
                    total = total + lam_fault * fault_l
                if lam_dip > 0:
                    dip_l = dip_fan_loss(x_hat, y, max_dip=float(cfg["training"]["max_dip"]))
                    total = total + lam_dip * dip_l

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            opt.step()
            sched.step()

            totals["nll"] += float(nll.detach())
            totals["fault"] += float(fault_l.detach())
            totals["dip"] += float(dip_l.detach())
            totals["total"] += float(total.detach())

        for k in totals:
            totals[k] /= max(steps, 1)
        totals["epoch"] = epoch

        val = evaluate_split(model, noise_model, volume, cfg, "val", device)
        totals["val"] = val
        history.append(totals)

        score = val.get(select_on)
        if score is None:
            raise KeyError(f"eval.select_on={select_on!r} is not among {sorted(val)}")
        if select_on in LOWER_IS_BETTER:
            score = -score
        if score > best["score"]:
            best = {"score": score, "epoch": epoch}
            torch.save({
                "model_state": model.state_dict(),
                "noise_model_state": noise_model.state_dict(),
                "config": cfg,
                "normalization": {"scale": volume.scale, "offset": volume.offset},
                "splits": volume.splits.as_dict(),
                "epoch": epoch,
                "val": val,
            }, run_dir / cfg["output"]["checkpoint_name"])

        if verbose:
            extra = f"snr={val['snr_db']:.2f}dB " if "snr_db" in val else ""
            print(f"[epoch {epoch:03d}] nll={totals['nll']:+.4f} "
                  f"fault={totals['fault']:.4f} dip={totals['dip']:.4f} | "
                  f"val nll={val['nll']:+.4f} {extra}leak={val['leakage']:.3f} "
                  f"removed={val['energy_removed_fraction']:.3f} "
                  f"resid_coh={val['residual_coherence']:.3f}"
                  + ("  <- best" if best["epoch"] == epoch else ""), flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"[done] {elapsed:.1f}s, best epoch {best['epoch']} on {select_on}", flush=True)

    with open(run_dir / "denoise_history.json", "w") as f:
        json.dump({"history": history, "best": best, "elapsed_seconds": elapsed,
                   "n_params": n_params, "volume": volume.metadata()}, f, indent=2)

    return {"history": history, "best": best, "volume": volume}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_denoiser(checkpoint_path: str | Path, device: str = "cpu"):
    """Return ``(model, noise_model, checkpoint)`` ready for inference."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    n_neighbors = int(cfg["model"]["n_neighbors"])

    model = BlindSpotDenoiser(
        in_channels=2 * n_neighbors + 1,
        base_channels=int(cfg["model"]["base_channels"]),
        depth=int(cfg["model"]["depth"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    noise_model = NoiseModel(
        mode=cfg["model"]["noise_model"],
        n_depth=len(ckpt["noise_model_state"]["raw"]),
        init_sigma=float(cfg["model"]["init_sigma"]),
        freeze=bool(cfg["model"].get("freeze_noise", False)),
    ).to(device)
    noise_model.load_state_dict(ckpt["noise_model_state"])
    noise_model.eval()

    return model, noise_model, ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Train the blind-spot denoiser.")
    parser.add_argument("--config", default="configs/denoise.yaml")
    parser.add_argument("--field", action="store_true",
                        help="field mode: train on the raw survey with no injected noise "
                             "(no reference available, so selection falls back to leakage)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--no-structure", action="store_true",
                        help="ablate the structure-preservation terms")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.field:
        cfg["data"]["inject_noise"] = None
        # Not leakage: it is minimised by removing nothing. The held-out
        # predictive likelihood is the criterion that still means something
        # when no clean reference exists.
        cfg["eval"]["select_on"] = "nll"
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.steps is not None:
        cfg["training"]["steps_per_epoch"] = args.steps
    if args.run_dir:
        cfg["output"]["run_dir"] = args.run_dir
    if args.no_structure:
        cfg["training"]["structure_preservation_enabled"] = False

    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[deepseis] cuda unavailable -> cpu", flush=True)
        device = "cpu"

    train(cfg, device=device)


if __name__ == "__main__":
    main()
