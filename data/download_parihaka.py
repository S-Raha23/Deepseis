#!/usr/bin/env python3
"""
Download the Parihaka survey (Taranaki Basin, New Zealand) and normalize it into
DeepSeis' canonical layout.

Source: Mendeley Data 10.17632/gnvyh3msrj.1 (CC BY 4.0), which redistributes the
SEAM / SEG 2020 "Seismic Facies Identification Challenge" volumes. That mirror is
used rather than AIcrowd directly because it serves plain, unauthenticated HTTP
downloads -- the AIcrowd resources page requires a login and competition sign-up,
which a setup script cannot drive.

Two normalizations happen here, once, so that nothing downstream needs to know
Parihaka is different from F3:

  1. Axis order. The publisher ships (Z, X, Y) = (sample, inline, crossline).
     DeepSeis' convention -- set by ``deepseis.io.segy.load_volume`` and assumed
     by ``train.prepare_data`` and the dashboard -- is
     (n_inlines, n_crosslines, n_samples). ``datasets.parihaka.source_layout``
     ([1, 2, 0]) encodes that reorder.

  2. Label base. The publisher numbers facies 1..6; F3 (and therefore
     ``facies.n_classes`` and the facies head's ``argmax``) uses 0..5.
     ``datasets.parihaka.source_label_offset`` (-1) encodes that rebase.

Usage:
    python data/download_parihaka.py                    # inline stride 4 (~465 MB)
    python data/download_parihaka.py --inline-stride 8  # smaller, ~233 MB
    python data/download_parihaka.py --inline-stride 1  # full survey, ~1.9 GB

Decimating inlines is lossless for everything DeepSeis actually does with the
volume: training reads inline 0 only (``train.prepare_data``), and the dashboard's
survey slider just needs a structurally coherent stack to scrub through. Inline 0
is preserved at every stride, so the trained model is bit-identical either way.

Then, to make it reachable from the deployed dashboard:
    python data/mirror_to_hf.py --dataset parihaka
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "parihaka"
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


# Mendeley's CDN rejects the default "Python-urllib/x.y" user-agent with HTTP 403,
# so requests must be issued through urlopen() with an explicit browser UA rather
# than the simpler urlretrieve() (which offers no way to set headers).
_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_CHUNK = 1 << 20  # 1 MiB


def fetch(url: str, dest: Path, description: str) -> None:
    """Stream ``url`` to ``dest`` with a progress readout, skipping if already present."""
    if dest.exists():
        print(f"[parihaka] {dest.name} already present, skipping download.")
        return
    print(f"[parihaka] downloading {description} ...")
    print(f"           {url}")
    # Write to a .part file first so an interrupted download is never mistaken
    # for a complete one on the next run.
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec - URL from repo config
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = min(100, done * 100 // total)
                        print(f"\r  {pct:3d}%  {done / 1024 / 1024:.1f} / "
                              f"{total / 1024 / 1024:.1f} MB", end="", flush=True)
                    else:
                        print(f"\r  {done / 1024 / 1024:.1f} MB", end="", flush=True)
        print()
        if total and done != total:
            raise IOError(f"incomplete download: got {done} of {total} bytes")
        tmp.replace(dest)
        print(f"[parihaka] saved -> {dest}")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"\n[parihaka] download failed: {e}", file=sys.stderr)
        print(
            "\nManual download steps:\n"
            "  1. Open https://data.mendeley.com/datasets/gnvyh3msrj/1\n"
            f"  2. Download the file backing '{description}'\n"
            f"  3. Save it as: {dest}\n"
            "  4. Re-run this script to convert it.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def load_npz_array(path: Path, key: str) -> np.ndarray:
    """Pull one named array out of a published .npz, with a helpful error if the key moved."""
    with np.load(path) as bundle:
        available = list(bundle.keys())
        if key not in available:
            print(f"[parihaka] '{path.name}' has no array named '{key}'. "
                  f"Found: {available}", file=sys.stderr)
            sys.exit(1)
        return bundle[key]


def to_canonical(arr: np.ndarray, layout: list[int], label_offset: int = 0) -> np.ndarray:
    """Reorder axes to (inline, crossline, sample) and rebase label values."""
    out = np.transpose(arr, axes=tuple(layout))
    if label_offset:
        out = out + label_offset
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inline-stride", type=int, default=4,
                         help="keep every Nth inline (default 4). Inline 0 is always kept, "
                              "so training results are unaffected. Use 1 for the full survey.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--keep-archives", action="store_true",
                         help="keep the downloaded .npz files instead of deleting them after conversion")
    args = parser.parse_args()

    if args.inline_stride < 1:
        parser.error("--inline-stride must be >= 1")

    cfg = yaml.safe_load(open(args.config))
    spec = cfg["datasets"]["parihaka"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seismic_npz = DATA_DIR / "parihaka_data_train.npz"
    labels_npz = DATA_DIR / "parihaka_labels_train.npz"

    print("=" * 70)
    print("Parihaka — Taranaki Basin, New Zealand")
    print(spec["citation"])
    print("=" * 70)

    fetch(spec["source_seismic_url"], seismic_npz, "seismic volume (~1.7 GB)")
    fetch(spec["source_labels_url"], labels_npz, "facies labels (~7 MB)")

    layout = spec["source_layout"]
    offset = spec["source_label_offset"]

    # ---- seismic ----------------------------------------------------------
    print("\n[parihaka] converting seismic volume to canonical layout ...")
    seismic = load_npz_array(seismic_npz, spec["source_seismic_key"])
    print(f"           published shape {seismic.shape} (sample, inline, crossline)")
    seismic = to_canonical(seismic, layout)
    print(f"           canonical shape {seismic.shape} (inline, crossline, sample)")
    if args.inline_stride > 1:
        seismic = seismic[::args.inline_stride]
        print(f"           decimated to    {seismic.shape} (every {args.inline_stride}th inline)")
    seismic = np.ascontiguousarray(seismic, dtype=np.float32)

    out_seismic = REPO_ROOT / spec["local_seismic"]
    out_seismic.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_seismic, seismic)
    print(f"[parihaka] wrote {out_seismic}  "
          f"({out_seismic.stat().st_size / 1024 / 1024:.0f} MB)")
    del seismic

    # ---- labels -----------------------------------------------------------
    print("\n[parihaka] converting facies labels to canonical layout ...")
    labels = load_npz_array(labels_npz, spec["source_labels_key"])
    published_classes = np.unique(labels)
    print(f"           published shape {labels.shape}, classes {published_classes.tolist()}")
    labels = to_canonical(labels, layout, label_offset=offset)
    if args.inline_stride > 1:
        labels = labels[::args.inline_stride]
    labels = np.ascontiguousarray(labels, dtype=np.int8)

    final_classes = np.unique(labels)
    print(f"           canonical shape {labels.shape}, classes {final_classes.tolist()} "
          f"(rebased by {offset:+d})")

    expected = spec["n_facies"]
    if final_classes.min() != 0 or final_classes.max() != expected - 1:
        print(f"[parihaka] warning: expected classes 0..{expected - 1} after rebasing but got "
              f"{final_classes.min()}..{final_classes.max()}. Check "
              f"datasets.parihaka.source_label_offset in {args.config}.", file=sys.stderr)

    out_labels = REPO_ROOT / spec["local_labels"]
    np.save(out_labels, labels)
    print(f"[parihaka] wrote {out_labels}  "
          f"({out_labels.stat().st_size / 1024 / 1024:.0f} MB)")
    del labels

    if not args.keep_archives:
        for archive in (seismic_npz, labels_npz):
            archive.unlink(missing_ok=True)
        print("\n[parihaka] removed the downloaded .npz archives "
              "(pass --keep-archives to retain them).")

    rel_seismic = Path(spec["local_seismic"]).as_posix()
    rel_labels = Path(spec["local_labels"]).as_posix()
    print(
        f"\n[parihaka] Ready.\n"
        f"\nTo make it selectable in the deployed dashboard, mirror it once:\n"
        f"  python data/mirror_to_hf.py --dataset parihaka\n"
        f"\nTo train on it instead of F3, set in {args.config}:\n"
        f"  field_volume_path: \"{rel_seismic}\"\n"
        f"  facies_label_path: \"{rel_labels}\"\n"
        f"then run:\n"
        f"  python -m deepseis.train --config {args.config}\n"
    )


if __name__ == "__main__":
    main()
