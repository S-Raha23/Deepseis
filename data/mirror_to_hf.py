#!/usr/bin/env python3
"""
Mirror a locally-prepared survey to Hugging Face so the deployed dashboard can reach it.

The dashboard cannot read ``data/raw/`` -- that directory is gitignored and does
not exist on Streamlit Cloud -- so every survey it offers has to be fetchable over
the network at runtime. F3 already works this way (``datasets.f3.hf_repo``); this
script gives any other registered survey the same treatment, uploading the
canonical .npy pair that the download script produced.

Run once per survey, after its download script:

    python data/download_parihaka.py
    python data/mirror_to_hf.py --dataset parihaka

Authentication (either one):
    huggingface-cli login          # interactive, stores a token
    set HF_TOKEN=hf_xxxxx          # Windows;  export HF_TOKEN=... on Linux/macOS

The target repo id comes from ``datasets.<name>.hf_repo`` in the config, so change
it there (not here) if you want a different account or name.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def human_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1024 / 1024:.0f} MB"


def build_dataset_card(name: str, spec: dict) -> str:
    """A minimal card so the mirror is self-describing and correctly attributed."""
    facies = "\n".join(f"| {i} | {n} |" for i, n in enumerate(spec.get("facies_names", [])))
    return f"""---
license: cc-by-4.0
tags:
  - seismic
  - geophysics
  - facies-segmentation
---

# {spec['label']} — {spec['region']}

Post-stack seismic amplitude and lithostratigraphic facies labels, redistributed
for use with [DeepSeis](https://github.com/S-Raha23/Deepseis).

**Original source:** {spec['citation']}
<{spec.get('citation_url', '')}>

Please cite the original publication, not this mirror.

## Files

| File | Contents |
|---|---|
| `{spec['hf_seismic']}` | `(n_inlines, n_crosslines, n_samples)` float32 amplitude |
| `{spec['hf_labels']}` | `(n_inlines, n_crosslines, n_samples)` int8 facies labels |

## Layout

Both arrays are stored in DeepSeis' canonical layout —
`(inline, crossline, sample)` — with facies labels rebased to `0..{spec['n_facies'] - 1}`.
This differs from the publisher's original layout; the conversion is done by
`data/download_{name}.py` in the DeepSeis repo.

Geometry: {spec['geometry']} · {spec['dt_ms']} ms sampling

## Facies classes

| Class | Name |
|---|---|
{facies}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True,
                         help="registry key from `datasets:` in the config, e.g. parihaka")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--repo", type=str, default=None,
                         help="override the target repo id (default: datasets.<name>.hf_repo)")
    parser.add_argument("--private", action="store_true",
                         help="create the dataset repo as private (the dashboard will then "
                              "need HF_TOKEN set wherever it runs)")
    parser.add_argument("--dry-run", action="store_true",
                         help="show what would be uploaded, then stop")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    registry = cfg.get("datasets", {})
    if args.dataset not in registry or args.dataset == "default":
        known = [k for k in registry if k != "default"]
        parser.error(f"unknown dataset '{args.dataset}'. Registered: {known}")

    spec = registry[args.dataset]
    repo_id = args.repo or spec["hf_repo"]

    seismic = REPO_ROOT / spec["local_seismic"]
    labels = REPO_ROOT / spec["local_labels"]

    missing = [p for p in (seismic, labels) if not p.exists()]
    if missing:
        print(f"[mirror] missing local files for '{args.dataset}':", file=sys.stderr)
        for p in missing:
            print(f"           {p}", file=sys.stderr)
        print(f"\nRun the download script first:\n"
              f"  python data/download_{args.dataset}.py\n", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"Mirroring '{spec['label']}' -> https://huggingface.co/datasets/{repo_id}")
    print("=" * 70)
    print(f"  {spec['hf_seismic']:<28s} <- {seismic}  ({human_mb(seismic)})")
    print(f"  {spec['hf_labels']:<28s} <- {labels}  ({human_mb(labels)})")

    if args.dry_run:
        print("\n[mirror] --dry-run: nothing uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\n[mirror] huggingface_hub is not installed. Run:\n"
              "  pip install -r requirements.txt\n", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        api.whoami()
    except Exception:
        print("\n[mirror] not authenticated with Hugging Face. Do one of:\n"
              "  huggingface-cli login\n"
              "  set HF_TOKEN=hf_xxxxx          (Windows)\n"
              "  export HF_TOKEN=hf_xxxxx       (Linux/macOS)\n", file=sys.stderr)
        sys.exit(1)

    print(f"\n[mirror] creating dataset repo {repo_id} (ok if it already exists) ...")
    api.create_repo(repo_id=repo_id, repo_type="dataset",
                     private=args.private, exist_ok=True)

    for local_path, remote_name in ((seismic, spec["hf_seismic"]), (labels, spec["hf_labels"])):
        print(f"[mirror] uploading {remote_name} ({human_mb(local_path)}) — this takes a while ...")
        api.upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_name,
                         repo_id=repo_id, repo_type="dataset")

    print("[mirror] uploading README.md ...")
    api.upload_file(
        path_or_fileobj=build_dataset_card(args.dataset, spec).encode("utf-8"),
        path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
    )

    print(f"\n[mirror] done -> https://huggingface.co/datasets/{repo_id}")
    print(f"[mirror] '{spec['label']}' is now selectable in the dashboard's Survey sidebar.")
    if args.private:
        print("[mirror] NOTE: the repo is private, so HF_TOKEN must be set wherever "
              "the dashboard runs (including Streamlit Cloud secrets).")


if __name__ == "__main__":
    main()
