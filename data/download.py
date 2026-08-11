#!/usr/bin/env python3
"""
Fetch F3 Netherlands (.npy) + FaultSeg3D synthetic data + pretrained fault
weights (spec Sec. 6 / Sec. 10 Phase 0).

Real seismic datasets are large, slow-moving binary files hosted on Zenodo /
GitHub release assets / institutional mirrors — this script is a thin,
resumable wrapper around those downloads, *not* a data host itself, so run
it wherever you actually have outbound network access to those sites.

If a target can't be reached (no network, a mirror moved, etc.), this
script does NOT fail silently: it prints exactly what didn't work and where
to get it manually, and falls back to generating the synthetic dataset
(deepseis.io.synthetic) so the rest of the pipeline still has something to
train and evaluate against immediately.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "raw"

SOURCES = {
    "f3_netherlands": {
        "description": "F3 Netherlands Block, ready-to-use NumPy (Alaudah et al. 2019 benchmark)",
        "direct_urls": [],  # Zenodo requires browser/auth flow in practice; no stable direct-download link
        "repo_urls": ["https://github.com/microsoft/seismic-deeplearning"],
        "note": "Zenodo (zenodo.org/records/1471548) is the canonical source but doesn't offer a "
                "stable unauthenticated direct-download link — open it in a browser. Alternatively, "
                "clone microsoft/seismic-deeplearning and run its scripts/download_dutch_f3.sh.",
    },
    "faultseg3d": {
        "description": "200 synthetic 3D seismic + binary fault-label volume pairs, plus pretrained weights",
        "direct_urls": [],
        "repo_urls": ["https://github.com/xinwucwp/faultSeg"],
        "note": "Clone the repo and follow its data/ + pretrained-model instructions "
                "(large files may be served via Git LFS or an external link in its README).",
    },
    "thebe_kerry_penobscot": {
        "description": "Additional public field surveys for cross-survey generalization tests (optional)",
        "direct_urls": [],
        "repo_urls": [],
        "note": "Public seismic repositories (terranubis.com, wiki.seg.org/wiki/Open_data); "
                "availability/format varies by survey and typically requires a browser.",
    },
}


def try_download(url: str, dest: Path, timeout: int = 15) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  fetching {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)  # nosec - user-provided, trusted URL
        return True
    except Exception as e:  # noqa: BLE001 - we want to report *any* failure reason to the user
        print(f"  [failed] {url}: {e}", file=sys.stderr)
        return False


def try_git_clone(repo_url: str, dest_dir: Path, timeout: int = 60) -> bool:
    """Clone a GitHub repo (source code + small assets, e.g. pretrained-weight pointers/scripts).

    This gets the repo's *code*, not necessarily large binary datasets —
    those are frequently hosted outside the repo (Zenodo, release assets,
    Git LFS) and need a follow-up manual step; we say so either way.
    """
    import subprocess

    if dest_dir.exists():
        print(f"  [skip] {dest_dir} already exists")
        return True
    try:
        print(f"  git clone --depth 1 {repo_url} -> {dest_dir}")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest_dir)],
                        check=True, timeout=timeout, capture_output=True, text=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [failed] git clone {repo_url}: {e}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(SOURCES.keys()) + ["all"], default="all")
    parser.add_argument("--synthetic-fallback", action="store_true", default=True,
                         help="also (re)generate the synthetic dataset regardless of download outcome")
    args = parser.parse_args()

    targets = list(SOURCES.keys()) if args.dataset == "all" else [args.dataset]
    any_real_data = False

    print("=" * 70)
    print("DeepSeis data fetch")
    print("=" * 70)
    for name in targets:
        info = SOURCES[name]
        print(f"\n[{name}] {info['description']}")
        ok = False
        for url in info["direct_urls"]:
            ok = try_download(url, DATA_DIR / name / Path(url).name) or ok
        for repo_url in info["repo_urls"]:
            cloned = try_git_clone(repo_url, DATA_DIR / name / Path(repo_url).name)
            if cloned:
                print(f"  -> repo cloned; it may still point to an external host for the actual "
                      f"large data/weight files. Check its README under {DATA_DIR / name}.")
        if not ok:
            print(f"  -> no direct machine-downloadable dataset here. {info['note']}")
        any_real_data = any_real_data or ok

    if not any_real_data:
        print("\n" + "-" * 70)
        print("No directly machine-downloadable real datasets in this environment")
        print("(large binary datasets on Zenodo/etc. typically need a browser or")
        print("credentials). Generating the synthetic fallback dataset instead so")
        print("the pipeline has something to run against right now — this is")
        print("exactly the 'known-clean' benchmark case from spec Sec. 9.")
        print("-" * 70)

    # Always (re)generate the synthetic set unless explicitly disabled — cheap, deterministic,
    # and it's what configs/default.yaml points at by default (data.use_synthetic: true).
    if args.synthetic_fallback:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from deepseis.io import noise as noise_mod
        from deepseis.io import synthetic as synth_mod
        import numpy as np
        import yaml

        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "configs" / "default.yaml"))
        vol = synth_mod.generate_from_config(cfg)
        rng = np.random.default_rng(cfg["data"]["synthetic"]["random_seed"])
        noisy = noise_mod.make_noisy(vol.clean, cfg, rng=rng)

        out_dir = DATA_DIR / "synthetic"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "clean.npy", vol.clean)
        np.save(out_dir / "noisy.npy", noisy)
        np.save(out_dir / "fault_mask.npy", vol.fault_mask)
        np.save(out_dir / "facies.npy", vol.facies)
        np.save(out_dir / "horizon_depths.npy", vol.horizon_depths)
        print(f"\n[synthetic] wrote clean/noisy/fault_mask/facies/horizon_depths to {out_dir}/")

    print("\nTo use real data once you have it: set data.field_volume_path "
          "(and optionally data.fault_label_path) in configs/default.yaml, "
          "and set data.use_synthetic: false.")


if __name__ == "__main__":
    main()
