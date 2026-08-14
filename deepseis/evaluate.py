"""
Evaluation harness: the learned denoiser against classical methods, on
held-out data, with a reference it cannot game.

The old evaluation could not support the claims made on it. It scored one
section -- the same one the model was fitted on -- with a single no-reference
metric that a do-nothing filter wins outright, against controls (identity, two
blurs) that no processor would ever run. This replaces all three weaknesses:

* **Held-out data.** Sections come from the test block, separated from
  training by a buffer wide enough for the inter-line correlation to decay.
* **A reference on real geology.** In semi-synthetic mode the noise added to
  each real section is known, so PSNR / SSIM / SNR are exact -- against real
  F3 structure rather than a synthetic model of it. Unlike leakage, these
  cannot be improved by removing less.
* **Baselines worth losing to.** f-x deconvolution and structure-oriented
  smoothing are what this problem is actually solved with in practice.

Every number is reported per method over the same sections, with the spread
across sections, so a difference can be read against its own variability.

Usage::

    python -m deepseis.evaluate --checkpoint runs/denoise/blindspot.pt
    python -m deepseis.evaluate --checkpoint ... --split test --ablations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deepseis import baselines as baselines_mod
from deepseis import metrics as metrics_mod
from deepseis.denoise import denoise_section, load_denoiser
from deepseis.io.dataset import SeismicVolume, inject_noise, section_seed
from deepseis.postprocess import orthogonalize


# ---------------------------------------------------------------------------
# Section assembly
# ---------------------------------------------------------------------------

def gather_sections(volume: SeismicVolume, cfg: dict, split: str, n_sections: int):
    """Yield ``(index, input_stack, centre_input, clean_reference_or_None)``."""
    idx = volume.split_indices(split)
    chosen = idx[np.linspace(0, len(idx) - 1, min(n_sections, len(idx))).astype(int)]

    n_neighbors = int(cfg["model"]["n_neighbors"])
    noise_cfg = cfg["data"].get("inject_noise") or None

    for i in chosen:
        stack = volume.inline_stack(int(i), n_neighbors)
        clean = stack[n_neighbors].copy() if noise_cfg else None
        if noise_cfg:
            seed = section_seed(int(cfg["data"]["noise_seed"]), "inline", int(i))
            stack = np.stack([inject_noise(stack[c], noise_cfg, seed + 104729 * c)
                              for c in range(stack.shape[0])])
        yield int(i), stack, stack[n_neighbors], clean


# ---------------------------------------------------------------------------
# The previous denoiser, wrapped so it can be scored on identical data
# ---------------------------------------------------------------------------

def legacy_denoiser(checkpoint: str | Path, legacy_config: str | Path = "configs/default.yaml",
                    device: str = "cpu"):
    """Wrap the masking denoiser from ``models/unet.py`` as a comparable method.

    It is served exactly the way the old pipeline served it -- per-section
    normalisation to unit standard deviation, then patch-and-average inference
    at the configured size and stride -- and the result is scaled back so the
    metrics are computed in the same units as every other method. Anything
    else would be measuring the rewrite of its serving code rather than the
    model.
    """
    import yaml

    from deepseis.models.unet import DenoiserUNet
    from deepseis.train import run_denoiser_inference

    with open(legacy_config) as f:
        legacy_cfg = yaml.safe_load(f)

    model = DenoiserUNet(**legacy_cfg["model"]["denoiser"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    def run(_stack: np.ndarray, centre: np.ndarray) -> np.ndarray:
        std = float(centre.std())
        if std < 1e-12:
            return centre.astype(np.float32)
        out = run_denoiser_inference(model, (centre / std).astype(np.float32), legacy_cfg, device)
        return (out * std).astype(np.float32)

    return run


# ---------------------------------------------------------------------------
# Downstream interpretability
# ---------------------------------------------------------------------------

def fault_agreement_scorer(faultseg_checkpoint: str | Path,
                           legacy_config: str | Path = "configs/default.yaml",
                           threshold: float = 0.5, device: str = "cpu"):
    """Score how well fault picks survive denoising, against picks on the clean section.

    This is the project's actual claim -- that denoising helps interpretation
    rather than quietly erasing what is being interpreted -- and until now
    nothing measured it, because F3 has no fault ground truth and the fault
    head could only be shown as a qualitative overlay.

    In semi-synthetic mode there is a way around that. The clean section is
    available, so the fault head can be run on it to produce reference picks,
    and every method's output scored on how closely its picks reproduce them.
    The reference is not ground-truth geology -- it is what this fault head
    says about uncontaminated data -- but that is precisely the right
    comparison here: it isolates the damage done by noise and by each
    denoiser, which is the question being asked.

    Returns ``None`` when the fault head is unavailable, so the rest of the
    evaluation still runs.
    """
    import yaml

    from deepseis.models.faultseg import FaultSegNet2D
    from deepseis.train import run_faultseg_inference

    path = Path(faultseg_checkpoint)
    if not path.exists():
        return None

    with open(legacy_config) as f:
        legacy_cfg = yaml.safe_load(f)

    model = FaultSegNet2D(in_channels=1, base_channels=legacy_cfg["faultseg"]["base_channels"],
                          depth=legacy_cfg["faultseg"]["depth"]).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    def picks(section: np.ndarray) -> np.ndarray:
        std = float(section.std())
        scaled = (section / std).astype(np.float32) if std > 1e-12 else section.astype(np.float32)
        return run_faultseg_inference(model, scaled, legacy_cfg, device)

    def score(section: np.ndarray, clean: np.ndarray) -> dict:
        p_ref = picks(clean)
        p_sec = picks(section)
        ref_bin = p_ref >= threshold
        sec_bin = p_sec >= threshold
        return {
            "fault_dice_vs_clean": metrics_mod.dice_score(sec_bin, ref_bin),
            "fault_prob_corr_vs_clean": float(np.corrcoef(p_sec.ravel(), p_ref.ravel())[0, 1]),
        }

    return score


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    for key in rows[0]:
        if isinstance(rows[0][key], dict):
            out[key] = {b: float(np.mean([r[key][b] for r in rows])) for b in rows[0][key]}
        else:
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            out[key] = float(vals.mean())
            out[key + "_std"] = float(vals.std())
    return out


def evaluate_methods(checkpoint: str | Path, split: str = "test", n_sections: int = 8,
                     device: str = "cpu", with_orthogonalization: bool = True,
                     extra_models: dict | None = None,
                     fault_scorer=None) -> dict:
    """Score the learned denoiser, its post-processed variant, and every baseline."""
    model, noise_model, ckpt = load_denoiser(checkpoint, device)
    cfg = ckpt["config"]

    volume = SeismicVolume(cfg["data"]["volume_path"], buffer=int(cfg["data"]["split_buffer"]))
    # The checkpoint's normalisation is authoritative -- recomputing it here
    # would silently re-introduce the train/serve mismatch this rebuild fixed.
    volume.scale = ckpt["normalization"]["scale"]
    volume.offset = ckpt["normalization"]["offset"]

    results: dict[str, list[dict]] = {}
    sections = list(gather_sections(volume, cfg, split, n_sections))
    if not sections:
        raise ValueError(f"no sections available in split {split!r}")

    def record(name: str, output: np.ndarray, centre: np.ndarray, clean) -> None:
        report = metrics_mod.denoising_report(centre, output, clean)
        if fault_scorer is not None and clean is not None:
            report.update(fault_scorer(output, clean))
        results.setdefault(name, []).append(report)

    for _, stack, centre, clean in sections:
        denoised = denoise_section(model, noise_model, stack, device)
        record("blindspot", denoised, centre, clean)

        # Same network, noise level re-measured from the section being denoised
        # rather than taken as the survey-wide constant. Reported as a separate
        # row so the choice is made on evidence rather than assumed.
        per_section = denoise_section(model, noise_model, stack, device,
                                      estimate_noise_per_section=True)
        record("blindspot+persection_sigma", per_section, centre, clean)

        if with_orthogonalization:
            ortho, _ = orthogonalize(centre, denoised,
                                     window=int(cfg["postprocess"]["ortho_window"]),
                                     iterations=int(cfg["postprocess"]["ortho_iterations"]))
            record("blindspot+ortho", ortho, centre, clean)

        for name, fn in baselines_mod.BASELINES.items():
            record(name, fn(centre), centre, clean)

        for name, fn in (extra_models or {}).items():
            record(name, fn(stack, centre), centre, clean)

    return {
        "split": split,
        "n_sections": len(sections),
        "section_indices": [s[0] for s in sections],
        "has_reference": sections[0][3] is not None,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "splits": ckpt.get("splits"),
        "methods": {name: _aggregate(rows) for name, rows in results.items()},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(report: dict) -> str:
    has_ref = report["has_reference"]
    methods = report["methods"]

    has_fault = has_ref and "fault_dice_vs_clean" in methods[next(iter(methods))]

    if has_ref:
        header = (f"{'method':<22}{'SNR dB':>10}{'+/-':>6}{'PSNR':>8}{'SSIM':>8}"
                  f"{'leak':>8}{'removed':>9}{'res.coh':>9}")
        if has_fault:
            header += f"{'faultDice':>11}"
        key = "snr_db"
    else:
        header = f"{'method':<22}{'leak':>10}{'removed':>9}{'res.coh':>9}{'out/in std':>12}"
        key = "leakage"

    order = sorted(methods, key=lambda m: -methods[m][key] if has_ref else methods[m][key])

    lines = [header, "-" * len(header)]
    for m in order:
        r = methods[m]
        if has_ref:
            row = (f"{m:<22}{r['snr_db']:>10.2f}{r['snr_db_std']:>6.2f}{r['psnr']:>8.2f}"
                   f"{r['ssim']:>8.3f}{r['leakage']:>8.3f}"
                   f"{r['energy_removed_fraction']:>9.3f}{r['residual_coherence']:>9.3f}")
            if has_fault:
                row += f"{r['fault_dice_vs_clean']:>11.3f}"
            lines.append(row)
        else:
            lines.append(f"{m:<22}{r['leakage']:>10.3f}{r['energy_removed_fraction']:>9.3f}"
                         f"{r['residual_coherence']:>9.3f}{r['output_std_ratio']:>12.3f}")

    lines.append("")
    lines.append("spectral energy retained, by radial F-K band (1.0 = untouched):")
    bands = list(methods[order[0]]["spectral_retention"].keys())
    lines.append(f"{'method':<22}" + "".join(f"{b:>10}" for b in bands))
    for m in order:
        sr = methods[m]["spectral_retention"]
        lines.append(f"{m:<22}" + "".join(f"{sr[b]:>10.3f}" for b in bands))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the denoiser against classical baselines.")
    parser.add_argument("--checkpoint", default="runs/denoise/blindspot.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-sections", type=int, default=8)
    parser.add_argument("--no-ortho", action="store_true")
    parser.add_argument("--output", default=None, help="write the full report as JSON")
    parser.add_argument("--legacy", default=None,
                        help="also score the previous masking denoiser, e.g. runs/default/denoiser.pt")
    parser.add_argument("--legacy-config", default="configs/default.yaml")
    parser.add_argument("--compare", action="append", default=[], metavar="NAME=PATH",
                        help="score another blind-spot checkpoint alongside (repeatable), "
                             "e.g. --compare no_structure=runs/denoise_nostruct/blindspot.pt")
    parser.add_argument("--faultseg", default="runs/default/faultseg.pt",
                        help="fault head used to score how well fault picks survive denoising; "
                             "skipped when the checkpoint is absent")
    args = parser.parse_args()

    extra: dict = {}
    if args.legacy:
        extra["legacy_masking_n2v"] = legacy_denoiser(args.legacy, args.legacy_config)
    for spec in args.compare:
        name, _, path = spec.partition("=")
        other_model, other_noise, _ = load_denoiser(path)
        extra[name] = (lambda m, nm: lambda stack, _centre: denoise_section(m, nm, stack))(
            other_model, other_noise)

    fault_scorer = fault_agreement_scorer(args.faultseg, args.legacy_config)
    if fault_scorer is None:
        print(f"[eval] no fault head at {args.faultseg} — skipping fault-pick agreement")

    report = evaluate_methods(args.checkpoint, split=args.split, n_sections=args.n_sections,
                              with_orthogonalization=not args.no_ortho, extra_models=extra,
                              fault_scorer=fault_scorer)

    print(f"\nsplit={report['split']}  sections={report['section_indices']}  "
          f"reference={'yes' if report['has_reference'] else 'no (field mode)'}")
    print(f"checkpoint={report['checkpoint']} (epoch {report['checkpoint_epoch']})\n")
    print(format_table(report))

    out = Path(args.output) if args.output else Path(args.checkpoint).parent / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    main()
