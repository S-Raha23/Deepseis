#!/usr/bin/env python3
"""
Download the F3 Netherlands benchmark dataset (Alaudah et al. 2019).

Source: https://zenodo.org/record/3755060
  - train/train_seismic.npy   shape (401, 701, 255)
  - train/train_labels.npy    shape (401, 701, 255)  -- facies labels (6 classes)
  - test_once/test1_seismic.npy
  - test_once/test1_labels.npy

Usage:
    python data/download_f3.py

The downloaded files land in:
    data/raw/f3/train/train_seismic.npy
    data/raw/f3/train/train_labels.npy
    data/raw/f3/test_once/test1_seismic.npy
    data/raw/f3/test_once/test1_labels.npy
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/record/3755060/files/data.zip"
DATA_DIR = Path(__file__).parent / "raw" / "f3"
ZIP_PATH = DATA_DIR / "data.zip"


def reporthook(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        mb_done = count * block_size / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"\r  {pct:3d}%  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"[f3] zip already present at {ZIP_PATH}, skipping download.")
    else:
        print(f"[f3] downloading F3 dataset from Zenodo (~500 MB)...")
        print(f"     {ZENODO_URL}")
        try:
            urllib.request.urlretrieve(ZENODO_URL, ZIP_PATH, reporthook)  # nosec
            print()  # newline after progress
            print(f"[f3] download complete -> {ZIP_PATH}")
        except Exception as e:
            print(f"\n[f3] download failed: {e}", file=sys.stderr)
            print(
                "\nManual download steps:\n"
                "  1. Open https://zenodo.org/record/3755060 in your browser\n"
                "  2. Download data.zip\n"
                f"  3. Move it to: {ZIP_PATH}\n"
                "  4. Re-run this script to unzip, or unzip manually into data/raw/f3/\n",
                file=sys.stderr,
            )
            sys.exit(1)

    # Unzip
    expected = DATA_DIR / "train" / "train_seismic.npy"
    if expected.exists():
        print(f"[f3] already unzipped (found {expected}), nothing to do.")
    else:
        print(f"[f3] unzipping {ZIP_PATH} ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"[f3] unzipped to {DATA_DIR}")

    # The zip extracts into a nested `data/` subfolder — locate the actual .npy files
    # by searching for train_seismic.npy anywhere under DATA_DIR.
    import numpy as np
    matches = list(DATA_DIR.rglob("train_seismic.npy"))
    if not matches:
        print(f"[f3] shape check failed: train_seismic.npy not found under {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    seismic_path = matches[0]
    labels_path = seismic_path.parent / "train_labels.npy"
    print(f"[f3] found seismic at: {seismic_path}")

    try:
        vol = np.load(str(seismic_path), mmap_mode="r")
        print(f"[f3] train_seismic.npy shape: {vol.shape}  (expect (401, 701, 255))")
        labels = np.load(str(labels_path), mmap_mode="r")
        print(f"[f3] train_labels.npy   shape: {labels.shape}  (facies labels, 6 classes)")
    except Exception as e:
        print(f"[f3] shape check failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print the exact paths to use in configs/default.yaml
    # Convert to relative paths from repo root for portability
    repo_root = DATA_DIR.parent.parent.parent  # data/raw/f3 -> data/raw -> data -> repo root
    try:
        rel_seismic = seismic_path.relative_to(repo_root).as_posix()
        rel_labels = labels_path.relative_to(repo_root).as_posix()
    except ValueError:
        rel_seismic = str(seismic_path)
        rel_labels = str(labels_path)

    print(
        f"\n[f3] Ready. Update configs/default.yaml with:\n"
        f"  field_volume_path: \"{rel_seismic}\"\n"
        f"  facies_label_path: \"{rel_labels}\"\n"
        f"\nThen run:\n"
        f"  python -m deepseis.train --config configs/default.yaml\n"
        f"  streamlit run app/dashboard.py\n"
    )

    # Auto-patch configs/default.yaml with the correct paths
    cfg_path = repo_root / "configs" / "default.yaml"
    if cfg_path.exists():
        text = cfg_path.read_text()
        import re
        text = re.sub(r'field_volume_path:.*', f'field_volume_path: "{rel_seismic}"', text)
        text = re.sub(r'facies_label_path:.*', f'facies_label_path: "{rel_labels}"', text)
        cfg_path.write_text(text)
        print(f"[f3] auto-patched {cfg_path} with correct paths.")


if __name__ == "__main__":
    download()
