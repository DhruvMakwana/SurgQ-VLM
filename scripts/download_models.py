#!/usr/bin/env python3
"""
download_models.py — Pre-fetch the Qwen2.5-VL backbone from HuggingFace.
=========================================================================

Training and evaluation download the backbone automatically on first use;
this script just lets you pre-fetch it (e.g. on a login node with internet
access before moving to an offline compute node).

Usage
-----
    python scripts/download_models.py                 # 7B (paper default)
    python scripts/download_models.py --model_size 3b # 3B variant

The model lands in the standard HuggingFace cache (~/.cache/huggingface,
or $HF_HOME if set). Size: ~16 GB for 7B, ~7 GB for 3B.
"""

import argparse
import sys

MODELS = {
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-fetch the Qwen2.5-VL backbone from HuggingFace."
    )
    parser.add_argument("--model_size", default="7b", choices=list(MODELS),
                        help="Backbone size (paper default: 7b).")
    args = parser.parse_args()

    repo_id = MODELS[args.model_size]
    print(f"Downloading {repo_id} to the HuggingFace cache ...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. "
              "Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    path = snapshot_download(repo_id)
    print(f"Done. Cached at: {path}")

    # Quick sanity check: can transformers resolve the config?
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo_id)
        print(f"Verified: {cfg.model_type} "
              f"(hidden_size={getattr(cfg, 'hidden_size', '?')})")
    except Exception as exc:
        print(f"Warning: transformers could not load the config ({exc}). "
              f"The download finished, but check your transformers version.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
