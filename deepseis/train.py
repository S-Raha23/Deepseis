"""
Training entry point.

Trains, in order:

1. The self-supervised blind-spot denoiser (twice: fault-preservation loss
   ON and OFF, so the demo's "money shot" on/off comparison, Sec. 10 Phase 2,
   is always available).
2. The FaultSeg-style fault segmentation head, on synthetic clean data +
   fault labels (mirroring how the real FaultSeg3D is trained on synthetic
   volumes and then applied to noisy field data — the domain gap between
   its training distribution and noisy input is exactly what the
   noisy-vs-denoised delta in Sec. 9 measures).
3. (Stretch, if enabled) the facies head and the diffusion refiner.

Usage:
    python -m deepseis.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from deepseis.io import noise as noise_mod
from deepseis.io import patches as patch_mod
from deepseis.io import segy as segy_mod
from deepseis.io import synthetic as synth_mod
from deepseis.losses.edge_preserve import edge_preservation_loss
from deepseis.losses.frequency import fk_high_freq_loss
from deepseis.losses.reconstruction import masked_mse_loss
from deepseis.masking import noise2void as n2v_mod
from deepseis.masking import struct_n2v as struct_mod
from deepseis.models.diffusion import DiffusionUNet, GaussianDiffusion
from deepseis.models.facies import FaciesNet2D
from deepseis.models.faultseg import FaultSegNet2D, load_pretrained_or_none
from deepseis.models.unet import DenoiserUNet
from deepseis import metrics as metrics_mod


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(cfg: dict) -> str:
    want = cfg.get("device", "cpu")
    if want == "cuda" and not torch.cuda.is_available():
        print("[deepseis] cuda requested but not available -> falling back to cpu")
        return "cpu"
    return want


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    """(N, H, W) numpy -> (N, 1, H, W) float32 tensor."""
    t = torch.from_numpy(x.astype(np.float32))
    if t.ndim == 2:
        t = t.unsqueeze(0)
    return t.unsqueeze(1).to(device)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(cfg: dict) -> dict:
    """Build the clean/noisy/label set the rest of training operates on.

    Real-data mode (``data.use_synthetic: false``):
      - Loads the field volume from ``data.field_volume_path`` as the noisy input.
      - ``clean`` and ``fault_mask`` are always None for real data (no clean reference exists).
      - ``facies`` is loaded from ``data.facies_label_path`` if provided; None otherwise.
      - FaultSeg is always trained on a *freshly generated* synthetic volume (exactly
        how the real FaultSeg3D paper does it) and then applied to the real denoised data.

    Synthetic mode (``data.use_synthetic: true``):
      - Generates a known-clean reflectivity model + fault labels + facies.
      - PSNR/SSIM/Dice can be computed exactly (spec Sec. 9).
    """
    dcfg = cfg["data"]
    rng = np.random.default_rng(dcfg["synthetic"]["random_seed"])

    if not dcfg.get("use_synthetic", True) and dcfg.get("field_volume_path"):
        noisy = segy_mod.load_volume(dcfg["field_volume_path"])
        if noisy.ndim == 3:
            # (n_inlines, n_crosslines, n_samples) -> take inline 0 as (n_samples, n_traces)
            noisy = noisy[0].T
        noisy = noisy.astype(np.float32)
        print(f"[deepseis] loaded field volume from {dcfg['field_volume_path']}, shape={noisy.shape}")

        # Normalize to unit std. This is REQUIRED, not cosmetic: surveys are published
        # at wildly different amplitude scales (F3 ships pre-scaled to [-1, 1] with
        # std ~0.23; Parihaka ships raw with std ~366, a factor of ~1600), and the
        # loss weights, learning rate and the lambda_edge/lambda_freq balance are all
        # tuned for unit-scale data -- raw Parihaka amplitudes diverge immediately.
        #
        # It is also what makes training agree with inference: `infer.py` and the
        # dashboard both divide the input by its std before calling the model, so
        # without this the model would be fitted at one scale and served at another.
        input_std = float(noisy.std())
        if input_std > 1e-8:
            noisy = noisy / input_std
            print(f"[deepseis] normalized field volume to unit std (was {input_std:.4f})")
        else:
            print("[deepseis] warning: field volume has near-zero std — check your input file.")

        # Real facies labels (optional — F3 has them, most surveys don't)
        facies = None
        if dcfg.get("facies_label_path"):
            facies_vol = segy_mod.load_volume(dcfg["facies_label_path"])
            if facies_vol.ndim == 3:
                facies = facies_vol[0].T.astype(np.int64)
            else:
                facies = facies_vol.astype(np.int64)
            print(f"[deepseis] loaded facies labels from {dcfg['facies_label_path']}, "
                  f"shape={facies.shape}, classes={np.unique(facies).tolist()}")

        # Generate a small synthetic volume purely to train FaultSeg on —
        # this mirrors the real FaultSeg3D paper: train on synthetic, infer on field data.
        print("[deepseis] generating synthetic volume for FaultSeg training "
              "(no real fault GT available for field data) ...")
        syn_vol = synth_mod.generate_from_config(cfg)
        syn_noisy = noise_mod.make_noisy(syn_vol.clean, cfg, rng=rng)

        return {
            "clean": None,
            "noisy": noisy,
            "fault_mask": None,
            "facies": facies,
            # FaultSeg trains on these, separately from the real denoiser data:
            "syn_clean": syn_vol.clean,
            "syn_noisy": syn_noisy,
            "syn_fault_mask": syn_vol.fault_mask,
        }

    # Synthetic mode — ground truth available for exact scoring
    vol = synth_mod.generate_from_config(cfg)
    noisy = noise_mod.make_noisy(vol.clean, cfg, rng=rng)
    print(f"[deepseis] generated synthetic volume: {vol.clean.shape}, "
          f"{len(np.unique(vol.fault_mask))-1 if vol.fault_mask.sum() else 0} fault trace(s) painted")
    return {
        "clean": vol.clean,
        "noisy": noisy,
        "fault_mask": vol.fault_mask,
        "facies": vol.facies,
        "syn_clean": vol.clean,
        "syn_noisy": noisy,
        "syn_fault_mask": vol.fault_mask,
    }


# ---------------------------------------------------------------------------
# Blind-spot mask dispatch
# ---------------------------------------------------------------------------

def make_mask_batch(patch_batch: np.ndarray, cfg: dict, rng: np.random.Generator):
    """Apply N2V / StructN2V (or auto-select per patch) -> (masked_inputs, masks, targets), all (N,H,W)."""
    mode = cfg["masking"]["mode"]
    masked_inputs, masks, targets = [], [], []
    for patch in patch_batch:
        this_mode = mode
        if mode == "auto":
            this_mode = "struct_n2v" if struct_mod.estimate_noise_type(patch) == "coherent" else "n2v"
        if this_mode == "n2v":
            mi, m, t = n2v_mod.make_training_pair(patch, cfg, rng)
        else:
            mi, m, t = struct_mod.make_training_pair(patch, cfg, rng)
        masked_inputs.append(mi)
        masks.append(m)
        targets.append(t)
    return np.stack(masked_inputs), np.stack(masks), np.stack(targets)


# ---------------------------------------------------------------------------
# Stage 1: self-supervised denoiser
# ---------------------------------------------------------------------------

def augment_with_flips(patches: np.ndarray) -> np.ndarray:
    """Horizontal + vertical flip augmentation: ~4x effective patches from one image.

    Both flips are physically defensible for a post-stack seismic section (mirroring
    trace order, or time order within a patch) and materially speed up convergence
    when training on a single small image instead of a large dataset -- see README,
    "Why the demo noise level looks aggressive", for the convergence study this came from.
    """
    return np.concatenate([patches, patches[:, :, ::-1], patches[:, ::-1, :], patches[:, ::-1, ::-1]], axis=0)


# ---------------------------------------------------------------------------
# Stage 1: self-supervised denoiser
# ---------------------------------------------------------------------------

def train_denoiser(cfg: dict, noisy: np.ndarray, fault_preservation_enabled: bool, device: str,
                    verbose: bool = True, checkpoint_path: "Path | None" = None,
                    resume: bool = False) -> tuple[DenoiserUNet, list[dict]]:
    """Train the self-supervised denoiser.

    If ``checkpoint_path`` is given, a resumable checkpoint (model + optimizer
    state + epoch + history) is written every ``training.checkpoint_every``
    epochs. Pass ``resume=True`` to continue from it if it already exists --
    useful for long CPU runs that may not finish in one sitting, and free on
    a GPU where you'd rarely need it.
    """
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    pcfg = cfg["data"]["patch"]
    all_patches = patch_mod.extract_patches(noisy, pcfg["size"], pcfg["stride"])
    if cfg["training"].get("augment_flips", False):
        all_patches = augment_with_flips(all_patches)

    model = DenoiserUNet(**cfg["model"]["denoiser"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    tcfg = cfg["training"]
    history: list[dict] = []
    start_epoch = 0

    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        if verbose:
            print(f"[denoiser] resumed from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, tcfg["epochs"]):
        epoch_losses = {"recon": 0.0, "edge": 0.0, "freq": 0.0, "total": 0.0}
        n_batches = 0
        for batch in patch_mod.batches(all_patches, tcfg["batch_size"], shuffle=True, rng=rng):
            masked_inputs, masks, targets = make_mask_batch(batch, cfg, rng)

            x = to_tensor(masked_inputs, device)
            m = to_tensor(masks.astype(np.float32), device)
            y = to_tensor(targets, device)

            pred = model(x)
            recon = masked_mse_loss(pred, y, m)

            total = recon
            edge_l = torch.tensor(0.0, device=device)
            freq_l = torch.tensor(0.0, device=device)
            if fault_preservation_enabled:
                edge_l = edge_preservation_loss(pred, y)
                freq_l = fk_high_freq_loss(pred, y, cutoff=tcfg["loss"]["freq_highpass_cutoff"])
                total = recon + tcfg["loss"]["lambda_edge"] * edge_l + tcfg["loss"]["lambda_freq"] * freq_l

            opt.zero_grad()
            total.backward()
            opt.step()

            epoch_losses["recon"] += recon.item()
            epoch_losses["edge"] += float(edge_l.detach()) if torch.is_tensor(edge_l) else float(edge_l)
            epoch_losses["freq"] += float(freq_l.detach()) if torch.is_tensor(freq_l) else float(freq_l)
            epoch_losses["total"] += total.item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        epoch_losses["epoch"] = epoch
        history.append(epoch_losses)

        if verbose and (epoch % tcfg["log_every"] == 0 or epoch == tcfg["epochs"] - 1):
            tag = "ON " if fault_preservation_enabled else "OFF"
            print(f"[denoiser fault-preservation={tag}] epoch {epoch:02d}  "
                  f"recon={epoch_losses['recon']:.4f}  edge={epoch_losses['edge']:.4f}  "
                  f"freq={epoch_losses['freq']:.4f}  total={epoch_losses['total']:.4f}")

        checkpoint_every = tcfg.get("checkpoint_every", 10)
        if checkpoint_path is not None and (epoch % checkpoint_every == 0 or epoch == tcfg["epochs"] - 1):
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                        "epoch": epoch, "history": history}, checkpoint_path)

    return model, history


@torch.no_grad()
def run_denoiser_inference(model: DenoiserUNet, noisy: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    """Full-image inference: patchify, run the (un-masked) forward pass, stitch back together."""
    model.eval()
    pcfg = cfg["data"]["patch"]
    h, w = noisy.shape
    patches = patch_mod.extract_patches(noisy, pcfg["size"], pcfg["stride"])
    x = to_tensor(patches, device)
    out = model(x)                              # (N, 1, H, W)
    out_np = out.detach().cpu().numpy()[:, 0]    # (N, H, W) -- always, regardless of N
    return patch_mod.stitch_patches(out_np, h, w, pcfg["size"], pcfg["stride"])


# ---------------------------------------------------------------------------
# Stage 2: fault segmentation head
# ---------------------------------------------------------------------------

def _bce_dice_loss(pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy(pred_prob.clamp(1e-6, 1 - 1e-6), target)
    inter = (pred_prob * target).sum()
    dice = 1 - (2 * inter + 1e-6) / (pred_prob.sum() + target.sum() + 1e-6)
    return bce + dice


def train_faultseg(cfg: dict, clean: np.ndarray, fault_mask: np.ndarray, device: str,
                    verbose: bool = True) -> FaultSegNet2D:
    """Trains on *clean* synthetic data, exactly as the real FaultSeg3D is trained on synthetic
    volumes before being applied to noisy field data — this is what creates the domain gap
    the noisy-vs-denoised delta in Sec. 9 is designed to measure."""
    fcfg = cfg["faultseg"]
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"] + 1)

    model = FaultSegNet2D(in_channels=1, base_channels=fcfg["base_channels"], depth=fcfg["depth"]).to(device)

    if fcfg.get("pretrained_weights_path") and load_pretrained_or_none(model, fcfg["pretrained_weights_path"], device):
        print("[faultseg] loaded pretrained weights")
        return model

    if not fcfg.get("train_on_synthetic_if_missing", True):
        print("[faultseg] no pretrained weights and train_on_synthetic_if_missing=False -> untrained model")
        return model

    pcfg = cfg["data"]["patch"]
    clean_patches = patch_mod.extract_patches(clean, pcfg["size"], pcfg["stride"])
    fault_patches = patch_mod.extract_patches(fault_mask, pcfg["size"], pcfg["stride"])
    if cfg["training"].get("augment_flips", False):
        clean_patches = augment_with_flips(clean_patches)
        fault_patches = augment_with_flips(fault_patches)

    opt = torch.optim.Adam(model.parameters(), lr=fcfg["lr"])
    for epoch in range(fcfg["epochs"]):
        epoch_loss, n_batches = 0.0, 0
        idx = np.arange(len(clean_patches))
        rng.shuffle(idx)
        for i in range(0, len(idx), fcfg["batch_size"]):
            b = idx[i:i + fcfg["batch_size"]]
            x = to_tensor(clean_patches[b], device)
            y = to_tensor(fault_patches[b], device)

            pred = model(x)
            loss = _bce_dice_loss(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        if verbose and (epoch % 2 == 0 or epoch == fcfg["epochs"] - 1):
            print(f"[faultseg] epoch {epoch:02d}  loss={epoch_loss / max(n_batches, 1):.4f}")

    return model


@torch.no_grad()
def run_faultseg_inference(model: FaultSegNet2D, section: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    model.eval()
    pcfg = cfg["data"]["patch"]
    h, w = section.shape
    patches = patch_mod.extract_patches(section, pcfg["size"], pcfg["stride"])
    x = to_tensor(patches, device)
    out = model(x)                              # (N, 1, H, W)
    out_np = out.detach().cpu().numpy()[:, 0]    # (N, H, W)
    return patch_mod.stitch_patches(out_np, h, w, pcfg["size"], pcfg["stride"])


# ---------------------------------------------------------------------------
# Stretch: facies head
# ---------------------------------------------------------------------------

def train_facies(cfg: dict, clean: np.ndarray, facies: np.ndarray, device: str) -> FaciesNet2D:
    fcfg = cfg["facies"]
    pcfg = cfg["data"]["patch"]
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"] + 2)

    # Auto-detect number of classes from the actual label data — don't rely
    # on the config value, which may be wrong (e.g. config says 3 but F3 has 6).
    n_classes = int(facies.max()) + 1
    if n_classes != fcfg.get("n_classes", n_classes):
        print(f"[facies] config n_classes={fcfg['n_classes']} but labels contain {n_classes} classes "
              f"(0..{n_classes-1}) — using {n_classes}.")

    model = FaciesNet2D(in_channels=1, n_classes=n_classes,
                         base_channels=cfg["faultseg"]["base_channels"], depth=cfg["faultseg"]["depth"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["faultseg"]["lr"])

    clean_patches = patch_mod.extract_patches(clean, pcfg["size"], pcfg["stride"])
    facies_patches = patch_mod.extract_patches(facies.astype(np.float32), pcfg["size"], pcfg["stride"]).astype(np.int64)
    if cfg["training"].get("augment_flips", False):
        clean_patches = augment_with_flips(clean_patches)
        facies_patches = augment_with_flips(facies_patches.astype(np.float32)).astype(np.int64)

    epochs = 6
    for epoch in range(epochs):
        idx = np.arange(len(clean_patches))
        rng.shuffle(idx)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(idx), cfg["faultseg"]["batch_size"]):
            b = idx[i:i + cfg["faultseg"]["batch_size"]]
            x = to_tensor(clean_patches[b], device)
            y = torch.from_numpy(facies_patches[b]).long().to(device)

            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"[facies] epoch {epoch:02d}  loss={epoch_loss / max(n_batches, 1):.4f}")

    return model


# ---------------------------------------------------------------------------
# Stretch: diffusion refiner
# ---------------------------------------------------------------------------

def train_diffusion(cfg: dict, denoised: np.ndarray, device: str) -> GaussianDiffusion:
    dcfg = cfg["diffusion"]
    pcfg = cfg["data"]["patch"]
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"] + 3)

    unet = DiffusionUNet(in_channels=1, base_channels=dcfg["base_channels"], depth=2).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=dcfg["timesteps"],
                                   physics_prior_weight=dcfg["physics_prior_weight"], device=device)
    opt = torch.optim.Adam(unet.parameters(), lr=1e-3)

    patches = patch_mod.extract_patches(denoised, pcfg["size"], pcfg["stride"])
    for epoch in range(dcfg["epochs"]):
        idx = np.arange(len(patches))
        rng.shuffle(idx)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(idx), 8):
            b = idx[i:i + 8]
            x0 = to_tensor(patches[b], device)
            t = torch.randint(0, dcfg["timesteps"], (x0.shape[0],), device=device)

            loss = diffusion.training_loss(x0, x0_ref=x0, t=t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"[diffusion] epoch {epoch:02d}  loss={epoch_loss / max(n_batches, 1):.4f}")

    return diffusion


@torch.no_grad()
def refine_with_diffusion(diffusion: GaussianDiffusion, denoised: np.ndarray, cfg: dict, device: str,
                           t_start_frac: float = 0.3) -> np.ndarray:
    """SDEdit-style partial refinement: forward-diffuse to an intermediate step, then reverse-sample back.

    Only partially re-noising (rather than starting from pure noise) keeps
    the output anchored to the stage-1 structure while letting the learned
    prior smooth/sharpen it — full resampling from pure noise would
    hallucinate a *different* section, not refine this one.
    """
    pcfg = cfg["data"]["patch"]
    h, w = denoised.shape
    patches = patch_mod.extract_patches(denoised, pcfg["size"], pcfg["stride"])
    x0 = to_tensor(patches, device)

    t_start = int(diffusion.timesteps * t_start_frac)
    t = torch.full((x0.shape[0],), t_start, device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise)

    refined = diffusion.sample(x0.shape, x_init=x_t, sample_steps=max(4, int(diffusion.timesteps * t_start_frac)))
    refined_np = refined.detach().cpu().numpy()[:, 0]   # (N, H, W)
    return patch_mod.stitch_patches(refined_np, h, w, pcfg["size"], pcfg["stride"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def apply_dataset(cfg: dict, dataset_key: str) -> dict:
    """Point the config at one survey from the ``datasets:`` registry.

    Rewrites the three fields that select what gets trained on -- the volume, the
    facies labels, and the output directory -- so that training a registered
    survey is ``--dataset <key>`` rather than three manual config edits. Each
    survey writes to its own ``run_dir``: the denoiser is self-supervised, so a
    model fitted to one survey's noise is not the model to run on another.
    """
    registry = cfg.get("datasets", {})
    if dataset_key not in registry or dataset_key == "default":
        known = [k for k in registry if k != "default"]
        raise SystemExit(f"[deepseis] unknown --dataset '{dataset_key}'. Registered: {known}")

    spec = registry[dataset_key]
    for field, key in (("field_volume_path", "local_seismic"), ("facies_label_path", "local_labels")):
        path = Path(spec[key])
        if not path.exists():
            raise SystemExit(
                f"[deepseis] {spec['label']}: {path} not found.\n"
                f"           Fetch it first:  python data/download_{dataset_key}.py"
            )
        cfg["data"][field] = str(path)

    cfg["data"]["use_synthetic"] = False
    cfg["output"]["run_dir"] = spec.get("run_dir", cfg["output"]["run_dir"])
    cfg["facies"]["n_classes"] = spec.get("n_facies", cfg["facies"]["n_classes"])

    print(f"[deepseis] dataset '{dataset_key}' -> {spec['label']} ({spec['region']})")
    print(f"[deepseis]   volume  {cfg['data']['field_volume_path']}")
    print(f"[deepseis]   labels  {cfg['data']['facies_label_path']}")
    print(f"[deepseis]   run_dir {cfg['output']['run_dir']}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None,
                         help="train on a survey from the config's `datasets:` registry "
                              "(e.g. f3, parihaka). Sets the volume, labels and run_dir "
                              "for you; without it the `data:` block is used as written.")
    parser.add_argument("--skip-stretch", action="store_true", help="skip diffusion + facies (faster run)")
    parser.add_argument("--resume", action="store_true",
                         help="resume denoiser training from the last periodic checkpoint, if present "
                              "(useful for long CPU runs split across multiple sessions)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg = apply_dataset(cfg, args.dataset)
    device = get_device(cfg)
    set_seed(cfg["seed"])

    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = prepare_data(cfg)
    noisy, clean, fault_mask, facies = data["noisy"], data["clean"], data["fault_mask"], data["facies"]
    syn_clean = data["syn_clean"]
    syn_fault_mask = data["syn_fault_mask"]

    results: dict = {"config_path": str(args.config), "dataset": args.dataset or "(config as written)"}

    # ---- Stage 1: denoiser, trained twice (fault-preservation ON vs OFF) ----
    print("\n=== Training denoiser: fault-preservation OFF ===")
    model_off, hist_off = train_denoiser(cfg, noisy, fault_preservation_enabled=False, device=device,
                                          checkpoint_path=run_dir / "training_ckpt_off.pt", resume=args.resume)
    denoised_off = run_denoiser_inference(model_off, noisy, cfg, device)

    print("\n=== Training denoiser: fault-preservation ON ===")
    model_on, hist_on = train_denoiser(cfg, noisy, fault_preservation_enabled=True, device=device,
                                        checkpoint_path=run_dir / "training_ckpt_on.pt", resume=args.resume)
    denoised_on = run_denoiser_inference(model_on, noisy, cfg, device)

    torch.save(model_on.state_dict(), run_dir / cfg["output"]["checkpoint_name"])
    torch.save(model_off.state_dict(), run_dir / ("off_" + cfg["output"]["checkpoint_name"]))
    np.save(run_dir / "noisy.npy", noisy)
    np.save(run_dir / "denoised_on.npy", denoised_on)
    np.save(run_dir / "denoised_off.npy", denoised_off)
    if clean is not None:
        np.save(run_dir / "clean.npy", clean)
    if fault_mask is not None:
        np.save(run_dir / "fault_mask.npy", fault_mask)
    if facies is not None:
        np.save(run_dir / "facies.npy", facies)

    if clean is not None:
        results["denoise_metrics"] = {
            "noisy_vs_clean": {"psnr": metrics_mod.psnr(noisy, clean), "snr_db": metrics_mod.snr_db(noisy, clean),
                                "ssim": metrics_mod.ssim(noisy, clean)},
            "denoised_ON_vs_clean": {"psnr": metrics_mod.psnr(denoised_on, clean),
                                      "snr_db": metrics_mod.snr_db(denoised_on, clean),
                                      "ssim": metrics_mod.ssim(denoised_on, clean)},
            "denoised_OFF_vs_clean": {"psnr": metrics_mod.psnr(denoised_off, clean),
                                       "snr_db": metrics_mod.snr_db(denoised_off, clean),
                                       "ssim": metrics_mod.ssim(denoised_off, clean)},
        }
        print("\n[metrics] denoising (synthetic, clean available):")
        print(json.dumps(results["denoise_metrics"], indent=2))
    else:
        print("\n[metrics] real data mode — no clean reference, skipping PSNR/SSIM (expected).")

    # ---- Stage 2: FaultSeg head ----
    # Always trains on the synthetic clean+fault data (syn_clean / syn_fault_mask),
    # mirroring how the real FaultSeg3D paper trains on synthetic and infers on field data.
    # For synthetic mode syn_clean == clean, so behaviour is identical to before.
    print("\n=== Training FaultSeg head (on synthetic clean data) ===")
    faultseg_model = train_faultseg(cfg, syn_clean, syn_fault_mask, device)
    torch.save(faultseg_model.state_dict(), run_dir / cfg["output"]["faultseg_checkpoint_name"])

    prob_noisy = run_faultseg_inference(faultseg_model, noisy, cfg, device)
    prob_denoised = run_faultseg_inference(faultseg_model, denoised_on, cfg, device)
    np.save(run_dir / "fault_prob_noisy.npy", prob_noisy)
    np.save(run_dir / "fault_prob_denoised.npy", prob_denoised)

    # Quantitative fault delta only available when we have ground-truth fault labels
    if fault_mask is not None:
        m_noisy = metrics_mod.fault_metrics(prob_noisy, fault_mask, threshold=cfg["faultseg"]["dice_threshold"])
        m_denoised = metrics_mod.fault_metrics(prob_denoised, fault_mask, threshold=cfg["faultseg"]["dice_threshold"])
        results["fault_metrics"] = {"noisy_input": dataclasses_asdict(m_noisy),
                                     "denoised_input": dataclasses_asdict(m_denoised)}
        print("\n[metrics] fault segmentation delta (noisy vs denoised input):")
        print(json.dumps(results["fault_metrics"], indent=2))
    else:
        print("\n[metrics] real data mode — no fault GT, skipping Dice/precision/recall (expected).")
        print("          Fault-probability overlays saved to runs/default/fault_prob_*.npy for visual QC.")

    # ---- Stretch: facies + diffusion ----
    if not args.skip_stretch and facies is not None and cfg["facies"]["enabled"]:
        # Real F3 facies labels — retrain the facies head on them.
        # facies is already the 2D slice matching the noisy section (from prepare_data).
        print("\n=== Training facies head on real F3 labels (stretch) ===")
        n_classes_actual = int(facies.max()) + 1
        print(f"[facies] using {n_classes_actual} classes detected from F3 labels (0..{n_classes_actual-1})")
        facies_input = denoised_on if clean is None else clean
        facies_model = train_facies(cfg, facies_input, facies, device)
        torch.save(facies_model.state_dict(), run_dir / "facies.pt")
    elif not args.skip_stretch and clean is not None and data.get("facies") is not None and cfg["facies"]["enabled"]:
        print("\n=== Training facies head (stretch) ===")
        facies_model = train_facies(cfg, clean, data["facies"], device)
        torch.save(facies_model.state_dict(), run_dir / "facies.pt")

    if not args.skip_stretch and cfg["diffusion"]["enabled"]:
        print("\n=== Training diffusion refiner (stretch) ===")
        diffusion = train_diffusion(cfg, denoised_on, device)
        torch.save(diffusion.model.state_dict(), run_dir / cfg["output"]["diffusion_checkpoint_name"])
        refined = refine_with_diffusion(diffusion, denoised_on, cfg, device)
        np.save(run_dir / "refined.npy", refined)
        if clean is not None:
            results["diffusion_metrics"] = {"psnr": metrics_mod.psnr(refined, clean),
                                             "ssim": metrics_mod.ssim(refined, clean)}
            print("\n[metrics] diffusion-refined vs clean:", results["diffusion_metrics"])

    results["history_off"] = hist_off
    results["history_on"] = hist_on
    results["elapsed_seconds"] = time.time() - t0

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[deepseis] done in {results['elapsed_seconds']:.1f}s. Artifacts written to {run_dir}/")


def dataclasses_asdict(obj) -> dict:
    import dataclasses
    return dataclasses.asdict(obj)


if __name__ == "__main__":
    main()
