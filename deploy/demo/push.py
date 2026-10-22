"""Publish the built demo (deploy/demo/site/) to a static Hugging Face Space.

Static Spaces are served as plain files: no container, no cold start, no
sleep, and free on every account.

    python deploy/demo/build.py
    python deploy/demo/push.py                    # SRaha23/DeepSeis
    python deploy/demo/push.py --space you/DeepSeis
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

SITE = Path(__file__).resolve().parent / "site"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default="SRaha23/DeepSeis", help="owner/name of the Space")
    args = parser.parse_args()

    if not (SITE / "index.html").exists():
        raise SystemExit(f"{SITE} has no index.html; run deploy/demo/build.py first.")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
    # One commit, and images from a previous build that no longer exist are removed.
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=SITE,
                      delete_patterns=["img/**"], commit_message="Publish demo")

    owner, name = args.space.split("/")
    print(f"Space: https://huggingface.co/spaces/{args.space}")
    print(f"Live:  https://{owner.lower()}-{name.lower().replace('_', '-')}.static.hf.space")


if __name__ == "__main__":
    main()
