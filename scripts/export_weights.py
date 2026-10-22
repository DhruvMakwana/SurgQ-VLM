#!/usr/bin/env python3
"""
export_weights.py — Produce a lean, releasable checkpoint from a training
checkpoint.
===========================================================================

Training checkpoints (checkpoint.pt / best_checkpoint.pt) carry extra
state that a release does not need:

  * optimizer + scheduler state            (resume-only)
  * compressor.qwen_embed.weight           (~0.9 GB — Qwen's own frozen
                                            embedding table, restored from
                                            the HF download at load time)
  * LoRA adapters on the ViT               (~0.3 GB — created by PEFT's
                                            module-name matching, then
                                            re-frozen before training;
                                            their lora_B tensors are all
                                            zeros, i.e. exact identity)

This script verifies the ViT adapters really are untrained (lora_B == 0)
and writes a ~0.85 GB weights-only file containing the compressor and the
LLM LoRA adapters. evaluate.py loads either format transparently
(load_state_dict with strict=False: stripped keys keep their frozen /
identity initialisation, which is numerically identical).

Usage
-----
    python scripts/export_weights.py \\
        --checkpoint checkpoints/grpo/best_checkpoint.pt \\
        --output surgqvlm_grpo_weights.pt
"""

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a lean release checkpoint (weights only)."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a training best_checkpoint.pt")
    parser.add_argument("--output", required=True,
                        help="Output .pt path for the lean weights")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ts = ckpt["trainable_state"]
    full_gb = sum(v.numel() * v.element_size() for v in ts.values()) / 1e9
    print(f"  trainable_state: {len(ts)} tensors, {full_gb:.2f} GB")

    # Safety check: the ViT LoRA adapters must be untrained (identity).
    vit_b = {k: v for k, v in ts.items() if ".visual." in k and "lora_B" in k}
    nonzero = [k for k, v in vit_b.items() if not (v == 0).all()]
    if nonzero:
        print("ABORT: ViT lora_B tensors are non-zero (the ViT adapters "
              "were trained?) — refusing to strip them:", file=sys.stderr)
        for k in nonzero[:5]:
            print("   ", k, file=sys.stderr)
        sys.exit(1)
    print(f"  verified: {len(vit_b)} ViT lora_B tensors are all zero (identity)")

    lean = {
        k: v for k, v in ts.items()
        if k != "compressor.qwen_embed.weight"     # frozen Qwen embedding
        and ".visual." not in k                    # inert ViT adapters
    }
    lean_gb = sum(v.numel() * v.element_size() for v in lean.values()) / 1e9
    n_comp = sum(1 for k in lean if k.startswith("compressor."))
    n_lora = sum(1 for k in lean if "lora_" in k)
    print(f"  lean state: {len(lean)} tensors "
          f"(compressor={n_comp}, llm_lora={n_lora}), {lean_gb:.2f} GB")

    out = {
        "trainable_state": lean,
        "epoch":           ckpt.get("epoch"),
        "metrics":         ckpt.get("metrics", {}),
        "export_note": (
            "Lean release checkpoint: compressor + LLM LoRA adapters only. "
            "Stripped: optimizer/scheduler state, the frozen "
            "compressor.qwen_embed.weight (restored from the HF download), "
            "and the identity ViT LoRA adapters (lora_B == 0, verified at "
            "export). Load with load_state_dict(..., strict=False) — "
            "evaluate.py and the training scripts do this already."
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"Saved → {out_path}  ({size_gb:.2f} GB on disk)")


if __name__ == "__main__":
    main()
