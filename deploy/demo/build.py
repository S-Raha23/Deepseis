"""Build the static DeepSeis demo: precomputed sections plus one HTML page.

The Streamlit app needs a Python server with torch loaded, which free hosting
either no longer offers (Hugging Face Docker Spaces) or offers with too little
memory and a cold start. This page needs neither: every model output is
computed here, once, by the same code paths the benchmark uses, and saved as
images. It is served as plain files, so it opens the same way on every visit.

    python deploy/demo/build.py        # writes deploy/demo/site/
    python deploy/demo/push.py         # publishes it

The page's sections are recomputed, but its headline numbers are read from the
evaluation reports. The build fails if the two disagree, so the page cannot
show sections that tell a different story from the published table.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
sys.path.insert(0, str(ROOT))

from deepseis import metrics as metrics_mod  # noqa: E402
from deepseis.denoise import denoise_section, load_denoiser, serve_section  # noqa: E402
from deepseis.evaluate import gather_sections  # noqa: E402
from deepseis.facies import mean_iou_over, per_class_iou, predict_section  # noqa: E402
from deepseis.io.dataset import SeismicVolume  # noqa: E402
from deepseis.models.facies import FaciesNet2D  # noqa: E402
from deepseis.models.faultseg import FaultSegNet2D  # noqa: E402
from deepseis.train import run_faultseg_inference  # noqa: E402

BENCH_CKPT = "runs/denoise/blindspot.pt"
FIELD_CKPT = "runs/denoise_field/blindspot.pt"
FAULT_CKPT = "runs/default/faultseg.pt"
FACIES_CKPT = "runs/default/facies.pt"
BENCH_REPORT = "runs/denoise/evaluation_final.json"
FACIES_REPORT = "runs/default/facies_metrics.json"
LEGACY_CONFIG = "configs/default.yaml"

# Same names and colours as the dashboard, so the two read alike.
FACIES_NAMES = ["Upper North Sea", "Middle North Sea", "Lower North Sea",
                "Rijnland/Chalk", "Jurassic", "Triassic"]
FACIES_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
METHOD_NAMES = {
    "blindspot": "DeepSeis blind-spot",
    "blindspot+ortho": "DeepSeis + orthogonalization",
    "blindspot+persection_sigma": "DeepSeis + per-section sigma",
    "structure_OFF": "DeepSeis (structure terms off)",
    "legacy_masking_n2v": "Previous denoiser (masking N2V)",
    "fx_decon": "f-x deconvolution",
    "structure_oriented": "Structure-oriented smoothing",
    "median_3": "3x3 median",
    "gaussian_1.0": "Gaussian sigma=1.0",
    "gaussian_0.5": "Gaussian sigma=0.5",
    "identity": "Noisy input (removes nothing)",
}

#: ColorBrewer RdBu, troughs red and peaks blue, white at zero amplitude.
RDBU = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in (
    "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7",
    "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061")], dtype=np.float32)
FAULT_RGB = np.array([255, 64, 32], dtype=np.float32)
MAX_IMAGE_BYTES = 98_000


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def diverging(section: np.ndarray, clip: float) -> np.ndarray:
    pos = (np.clip(section / clip, -1.0, 1.0) + 1.0) / 2.0 * (len(RDBU) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(RDBU) - 2)
    frac = (pos - lo)[..., None]
    return (RDBU[lo] * (1 - frac) + RDBU[lo + 1] * frac).astype(np.uint8)


def fault_overlay(section: np.ndarray, prob: np.ndarray, clip: float,
                  threshold: float) -> np.ndarray:
    """Picks drawn over a muted grey section, so they read in either page theme."""
    grey = 40 + 175 * (np.clip(section / clip, -1.0, 1.0) + 1.0) / 2.0
    rgb = np.repeat(grey[..., None], 3, axis=-1).astype(np.float32)
    alpha = np.where(prob >= threshold, 0.35 + 0.6 * prob, 0.0)[..., None]
    return (rgb * (1 - alpha) + FAULT_RGB * alpha).astype(np.uint8)


def categorical(class_map: np.ndarray) -> np.ndarray:
    palette = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in FACIES_COLORS],
                       dtype=np.uint8)
    return palette[np.clip(class_map, 0, len(palette) - 1)]


def save(rgb: np.ndarray, name: str, lossless: bool = False) -> str:
    """Write one image, small enough to be served from the Space itself.

    Spaces move files above ~100 KB to their CDN behind a redirect. Staying
    under that serves the whole page from one origin with no extra hop, so
    the noisiest sections, which compress worst, step down in quality until
    they fit. Names carry a content hash, so a rebuild can never pair new
    HTML with a stale cached image.
    """
    img = Image.fromarray(rgb)
    for quality in (90, 85, 80, 75, 70, 65, 60):
        buf = io.BytesIO()
        img.save(buf, "WEBP", lossless=lossless, quality=quality, method=6)
        if lossless or buf.tell() <= MAX_IMAGE_BYTES:
            break
    data = buf.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        print(f"  note: {name} is {len(data)} bytes and will be served through the CDN")
    fname = f"{name}.{hashlib.sha1(data).hexdigest()[:10]}.webp"
    (SITE / "img" / fname).write_bytes(data)
    return f"img/{fname}"


# ---------------------------------------------------------------------------
# Controlled test: known noise on the benchmark's own held-out inlines
# ---------------------------------------------------------------------------

def controlled_sections(report: dict, threshold: float) -> list[dict]:
    model, noise_model, ckpt = load_denoiser(BENCH_CKPT, "cpu")
    cfg = ckpt["config"]
    volume = SeismicVolume(cfg["data"]["volume_path"], buffer=int(cfg["data"]["split_buffer"]))
    volume.scale = ckpt["normalization"]["scale"]
    volume.offset = ckpt["normalization"]["offset"]

    with open(LEGACY_CONFIG) as f:
        legacy_cfg = yaml.safe_load(f)
    faultseg = FaultSegNet2D(in_channels=1, base_channels=legacy_cfg["faultseg"]["base_channels"],
                             depth=legacy_cfg["faultseg"]["depth"])
    faultseg.load_state_dict(torch.load(FAULT_CKPT, map_location="cpu"))
    faultseg.eval()

    def picks(section: np.ndarray) -> np.ndarray:
        # Unit-std input, exactly as evaluate.fault_agreement_scorer feeds it.
        std = float(section.std())
        scaled = (section / std).astype(np.float32) if std > 1e-12 else section.astype(np.float32)
        return run_faultseg_inference(faultseg, scaled, legacy_cfg, "cpu")

    out = []
    for idx, stack, noisy, clean in gather_sections(volume, cfg, report["split"],
                                                    report["n_sections"]):
        denoised = denoise_section(model, noise_model, stack, "cpu")
        after = metrics_mod.denoising_report(noisy, denoised, clean)
        before = metrics_mod.denoising_report(noisy, noisy, clean)

        p_clean, p_noisy, p_den = picks(clean), picks(noisy), picks(denoised)
        ref = p_clean >= threshold
        dice_noisy = metrics_mod.dice_score(p_noisy >= threshold, ref)
        dice_den = metrics_mod.dice_score(p_den >= threshold, ref)

        clip = float(np.percentile(np.abs(clean), 99))
        tag = f"test_{idx}"
        out.append({
            "inline": idx,
            "snr_in": before["snr_db"], "snr_out": after["snr_db"],
            "ssim_in": before["ssim"], "ssim_out": after["ssim"],
            "removed": after["energy_removed_fraction"],
            "dice_noisy": dice_noisy, "dice_denoised": dice_den,
            "img": {
                "clean": save(diverging(clean, clip), f"{tag}_clean"),
                "noisy": save(diverging(noisy, clip), f"{tag}_noisy"),
                "denoised": save(diverging(denoised, clip), f"{tag}_denoised"),
                "removed": save(diverging(noisy - denoised, clip), f"{tag}_removed"),
                "fault_clean": save(fault_overlay(clean, p_clean, clip, threshold), f"{tag}_fault_clean"),
                "fault_noisy": save(fault_overlay(noisy, p_noisy, clip, threshold), f"{tag}_fault_noisy"),
                "fault_denoised": save(fault_overlay(denoised, p_den, clip, threshold), f"{tag}_fault_denoised"),
            },
        })
        print(f"  inline {idx}: SNR {before['snr_db']:.2f} -> {after['snr_db']:.2f} dB, "
              f"fault Dice {dice_noisy:.3f} -> {dice_den:.3f}")
    return out


# ---------------------------------------------------------------------------
# Facies on real F3: the head's own held-out inlines
# ---------------------------------------------------------------------------

def facies_sections(facies_report: dict, legacy_cfg: dict) -> tuple[list[dict], np.ndarray]:
    model, noise_model, ckpt = load_denoiser(FIELD_CKPT, "cpu")
    dcfg = legacy_cfg["data"]
    volume = SeismicVolume(dcfg["field_volume_path"], buffer=30)
    labels = np.load(dcfg["facies_label_path"], mmap_mode="r")

    state = torch.load(FACIES_CKPT, map_location="cpu")
    n_classes = int(state["out_conv.bias"].shape[0])
    head = FaciesNet2D(in_channels=1, n_classes=n_classes,
                       base_channels=legacy_cfg["faultseg"]["base_channels"],
                       depth=legacy_cfg["faultseg"]["depth"])
    head.load_state_dict(state)
    head.eval()
    pcfg = dcfg["patch"]

    out, ious = [], []
    for idx in facies_report["val_inlines"]:
        # Raw amplitude: serve_section applies the checkpoint's own normalisation.
        raw = volume.inline(int(idx), normalized=False)
        denoised = serve_section(model, noise_model, ckpt, raw, "cpu")
        pred = predict_section(head, denoised, pcfg["size"], pcfg["stride"], "cpu")
        truth = np.asarray(labels[int(idx)]).T.astype(np.int64)
        ious.append(per_class_iou(pred, truth, n_classes))

        clip = float(np.percentile(np.abs(raw), 99))
        tag = f"facies_{idx}"
        out.append({
            "inline": int(idx),
            "accuracy": float((pred == truth).mean()),
            "img": {
                "seismic": save(diverging(denoised, clip), f"{tag}_seismic"),
                "pred": save(categorical(pred), f"{tag}_pred", lossless=True),
                "truth": save(categorical(truth), f"{tag}_truth", lossless=True),
            },
        })
        print(f"  inline {idx}: pixel accuracy {out[-1]['accuracy']:.3f}")

    stacked = np.stack(ious)
    with np.errstate(invalid="ignore"):
        per_class = np.array([np.nan if np.all(np.isnan(col)) else np.nanmean(col)
                              for col in stacked.T])
    return out, per_class


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def check(name: str, recomputed: float, reported: float, tol: float) -> None:
    status = "ok" if abs(recomputed - reported) <= tol else "MISMATCH"
    print(f"  {name}: recomputed {recomputed:.4f}, report {reported:.4f}  [{status}]")
    if status != "ok":
        raise SystemExit(f"{name} drifted from the report; rerun deepseis.evaluate before publishing.")


def main() -> None:
    os.chdir(ROOT)  # checkpoint configs hold repository-relative data paths
    torch.set_grad_enabled(False)

    with open(BENCH_REPORT) as f:
        report = json.load(f)
    with open(FACIES_REPORT) as f:
        facies_report = json.load(f)
    with open(LEGACY_CONFIG) as f:
        legacy_cfg = yaml.safe_load(f)
    threshold = float(legacy_cfg["faultseg"]["dice_threshold"])

    # Empty the folder rather than delete it: on Windows a directory that any
    # process has open (a shell, an editor) cannot be removed.
    SITE.mkdir(parents=True, exist_ok=True)
    for child in SITE.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    (SITE / "img").mkdir()

    print(f"Controlled test on {report['split']} inlines {report['section_indices']}")
    controlled = controlled_sections(report, threshold)
    print(f"Facies on held-out inlines {facies_report['val_inlines']}")
    facies, per_class = facies_sections(facies_report, legacy_cfg)

    ours, noisy = report["methods"]["blindspot"], report["methods"]["identity"]
    print("Consistency with the published reports")
    check("SNR after", float(np.mean([s["snr_out"] for s in controlled])), ours["snr_db"], 0.01)
    check("SSIM after", float(np.mean([s["ssim_out"] for s in controlled])), ours["ssim"], 0.001)
    scored = np.array(facies_report["scored_classes"])
    check("facies mIoU", mean_iou_over(per_class, scored), facies_report["mean_iou"], 0.005)

    # Fault Dice is the one number taken from this build rather than the report.
    # The report's values do not reproduce from the committed fault head and
    # scorer (deterministically 0.649 / 0.310 against the report's 0.656 /
    # 0.345), so the page shows what these images were actually scored at, and
    # the benchmark table leaves the column out rather than mix the two sources.
    dice_out = float(np.mean([s["dice_denoised"] for s in controlled]))
    dice_in = float(np.mean([s["dice_noisy"] for s in controlled]))
    print(f"  fault Dice (from this build): noisy {dice_in:.4f} -> denoised {dice_out:.4f}; "
          f"report says {noisy['fault_dice_vs_clean']:.4f} -> {ours['fault_dice_vs_clean']:.4f}")

    benchmark = sorted(
        ({"method": METHOD_NAMES.get(k, k), "ours": k == "blindspot",
          "snr": r["snr_db"], "snr_std": r.get("snr_db_std", 0.0), "ssim": r["ssim"],
          "removed": r["energy_removed_fraction"]}
         for k, r in report["methods"].items()),
        key=lambda row: -row["snr"])

    data = {
        "headline": {
            "snr_in": noisy["snr_db"], "snr_out": ours["snr_db"], "ssim": ours["ssim"],
            "dice_in": dice_in, "dice_out": dice_out,
            "miou": facies_report["mean_iou"], "n_sections": report["n_sections"],
        },
        "threshold": threshold,
        "controlled": controlled,
        # Open on a typical section, the median by output SNR, not the best one.
        "start_controlled": int(np.argsort([s["snr_out"] for s in controlled])[len(controlled) // 2]),
        "facies": facies,
        "facies_classes": [{"name": FACIES_NAMES[c], "color": FACIES_COLORS[c],
                            "iou": facies_report["per_class_iou"][c]}
                           for c in range(len(facies_report["per_class_iou"]))],
        "benchmark": benchmark,
    }

    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (SITE / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(HERE / "space_README.md", SITE / "README.md")

    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"Wrote {SITE} ({size / 2**20:.1f} MB, {len(list((SITE / 'img').iterdir()))} images)")


if __name__ == "__main__":
    main()
