"""
evaluate.py — Standalone Evaluation for SurgQ-VLM
==================================================

Loads a trained checkpoint, runs greedy generation over a split, and scores
the decoded answers with the answer-type-aware metric suite:

    K-Acc (keyword accuracy), Primary (structure-matched N-weighted
    average), binary accuracy, micro-F1, count exact-match / MAE /
    normalised error, ROUGE-L, BLEU-4, and T-Acc at ±1/±2/±3 s —
    per Q-type, per answer-structure group, and overall.

Usage
-----
    # Evaluate the best GRPO checkpoint on the sealed test split
    python evaluate.py --experiment_name grpo --split test

    # Evaluate the SFT checkpoint
    python evaluate.py --experiment_name sft --split test

    # Larger eval batch (no gradients → more headroom)
    python evaluate.py --experiment_name grpo --split test --eval_batch_size 16

Output
------
Two files in the experiment checkpoint directory (or --output_dir):
    eval_{split}_{timestamp}.jsonl         ← per-sample predictions + metrics
    eval_{split}_{timestamp}_summary.json  ← aggregated metrics
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Make the repository root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s %(message)s",
    datefmt = "%H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Load config from checkpoint dir ───────────────────────────────────────────

def load_train_config(checkpoint_dir: Path):
    """Read train_config.json and reconstruct the config dataclass.

    Works for both SFT and GRPO checkpoints — both save the LoRA/model
    fields that build_model() needs; extra keys are ignored.
    """
    import dataclasses
    from train_sft import SFTConfig

    config_path = checkpoint_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"train_config.json not found in {checkpoint_dir}. "
            "Was this experiment produced by train_sft.py / train_grpo.py?"
        )

    with open(config_path) as f:
        saved = json.load(f)

    valid_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    filtered = {k: v for k, v in saved.items() if k in valid_fields}
    return SFTConfig(**filtered)


# ── Main eval loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(
    experiment_name: str | None,
    checkpoint:      str,
    split:           str,
    output_dir:      Path | None,
    data_root:       str | None,
    mock:            bool,
    split_file:      str | None = None,
    eval_batch_size: int | None = None,
    gen_max_new_tokens: int | None = None,
    weights:         str | None = None,
    model_config:    str = "configs/sft.yaml",
) -> None:

    if weights is not None:
        # ── Standalone weights file (e.g. downloaded release weights) ──────
        # Model/LoRA hyperparameters come from a YAML config instead of a
        # training directory's train_config.json.
        from train_sft import SFTConfig
        from surgqvlm.utils import load_yaml_config
        ckpt_path = Path(weights)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Weights file not found: {ckpt_path}")
        ckpt_dir = ckpt_path.parent
        config = load_yaml_config(SFTConfig, model_config)
        ckpt_filename = ckpt_path.name
        experiment_name = experiment_name or ckpt_path.stem
    else:
        # ── Locate the checkpoint directory ─────────────────────────────────
        if experiment_name is None:
            raise ValueError("Provide --experiment_name or --weights.")
        ckpt_dir = Path("checkpoints") / experiment_name
        if not ckpt_dir.exists():
            ckpt_dir = Path(experiment_name)
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Cannot find a checkpoint directory for '{experiment_name}'. "
                f"Tried: checkpoints/{experiment_name} and {experiment_name}"
            )
        # Reconstruct the config saved at training time
        config = load_train_config(ckpt_dir)
        ckpt_filename = ("best_checkpoint.pt" if checkpoint == "best"
                         else "checkpoint.pt")
        ckpt_path = ckpt_dir / ckpt_filename
    if data_root:
        config.data_root = data_root
    if mock:
        config.mock = True
    if gen_max_new_tokens:
        config.gen_max_new_tokens = gen_max_new_tokens

    logger.info("=" * 60)
    logger.info("SurgQ-VLM Eval | %s | split=%s | ckpt=%s",
                experiment_name, split, checkpoint)
    logger.info("=" * 60)
    logger.info("Checkpoint dir : %s", ckpt_dir)
    logger.info("Data root      : %s", config.data_root)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s | %.1f GB VRAM", props.name, props.total_memory / 1e9)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── Build model ─────────────────────────────────────────────────────────
    from train_sft import build_model
    model, tokenizer, video_token_id = build_model(config)
    model = model.to(device) if config.mock else model.to_device(device)
    model.eval()

    # ── Load checkpoint weights ─────────────────────────────────────────────
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_path}. Available files: "
            + ", ".join(str(p.name) for p in ckpt_dir.iterdir())
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["trainable_state"], strict=False)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])
    # `missing` is expected: frozen base weights are not in trainable_state,
    # and lean release weights additionally omit the frozen Qwen embedding
    # and the identity ViT adapters (see scripts/export_weights.py).
    n_loaded = len(ckpt["trainable_state"])
    logger.info("Loaded %s (epoch %s): %d tensors in checkpoint, %d unexpected",
                ckpt_filename, ckpt.get("epoch", "?"), n_loaded, len(unexpected))

    # ── Build the dataloader ────────────────────────────────────────────────
    from surgqvlm.data import SurgQDataset, build_dataloader

    if split_file:
        pass  # explicit override wins
    elif split == "val":
        split_file = str(config.val_split)
    elif split == "train":
        split_file = str(config.train_split)
    elif split == "test":
        split_file = str(config.test_split)
    else:
        raise ValueError(f"--split must be 'val', 'train', or 'test', got '{split}'")

    bs = eval_batch_size or config.batch_size
    dataset = SurgQDataset(split_file=split_file, use_mock_encoder=config.mock)
    loader  = build_dataloader(
        dataset,
        batch_size  = bs,
        shuffle     = False,
        num_workers = config.num_workers,
    )
    logger.info("Dataset: %d samples (batch_size=%d) from %s",
                len(dataset), bs, split_file)

    # ── Prompt builder ──────────────────────────────────────────────────────
    from surgqvlm.utils import PromptBuilder

    prompt_builder = PromptBuilder(
        tokenizer      = tokenizer,
        video_token_id = video_token_id,
        max_seq_len    = config.max_seq_len,
    )
    prompt_builder._loss_weight_mode = "none"
    prompt_builder._dataset_freq     = None

    compressor_config = model.config.compressor

    # ── Generation + scoring ────────────────────────────────────────────────
    from train_sft import validate_generative

    val_result = validate_generative(
        model             = model,
        dataloader        = loader,
        prompt_builder    = prompt_builder,
        compressor_config = compressor_config,
        config            = config,
        device            = device,
        epoch             = ckpt.get("epoch") or 0,
        tag               = split,
    )

    # ── Save results ────────────────────────────────────────────────────────
    save_dir  = Path(output_dir) if output_dir else ckpt_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag       = f"eval_{split}_{timestamp}"

    # Per-sample predictions JSONL
    pred_path = save_dir / f"{tag}.jsonl"
    with open(pred_path, "w") as f:
        for p in val_result["predictions"]:
            f.write(json.dumps(p) + "\n")
    logger.info("Predictions saved → %s  (%d samples)",
                pred_path, len(val_result["predictions"]))

    # Aggregated summary JSON
    summary = {
        "experiment_name": experiment_name,
        "checkpoint":      ckpt_filename,
        "checkpoint_epoch": ckpt.get("epoch"),
        "split":           split,
        "overall_primary": round(val_result["metrics"].get("overall_primary", 0.0), 4),
        "overall_kacc":    round(val_result["metrics"].get("overall_kacc", 0.0), 4),
        "n_samples":       val_result["metrics"].get("n_samples", 0),
    }
    # Flatten per-answer-type and per-Q-type metrics
    for at, m in val_result["metrics"].get("by_answer_type", {}).items():
        for k, v in m.items():
            summary[f"{at}_{k}"] = round(v, 4)
    for qt, m in val_result["metrics"].get("by_qtype", {}).items():
        for k, v in m.items():
            summary[f"{qt}_{k}"] = round(v, 4)

    summary_path = save_dir / f"{tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved    → %s", summary_path)

    # Print key metrics to terminal
    logger.info("-" * 60)
    logger.info("overall_primary : %.4f", summary["overall_primary"])
    logger.info("overall_kacc    : %.4f", summary["overall_kacc"])
    for atype, agg in val_result["metrics"].get("by_answer_type", {}).items():
        kacc = agg.get("kacc", 0.0)
        logger.info("  %-20s : kacc=%.4f", atype, kacc)
    logger.info("-" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SurgQ-VLM standalone evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment_name", default=None,
        help="Experiment name (e.g. 'grpo'). Looks for checkpoints/<name>/. "
             "Not needed when --weights is given.",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to a standalone weights file (e.g. downloaded release "
             "weights from scripts/export_weights.py). Model hyperparameters "
             "are then read from --model_config instead of a training dir.",
    )
    parser.add_argument(
        "--model_config", default="configs/sft.yaml",
        help="YAML with the model/LoRA hyperparameters, used with --weights.",
    )
    parser.add_argument(
        "--checkpoint", default="best", choices=["best", "latest"],
        help="'best' = best_checkpoint.pt, 'latest' = checkpoint.pt.",
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "train", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save results. Defaults to the checkpoint directory.",
    )
    parser.add_argument(
        "--data_root", default=None,
        help="Override data root (default: the one stored in train_config.json).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock encoder (tests the script without GPU/weights).",
    )
    parser.add_argument(
        "--split_file", default=None,
        help="Override the split file directly.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=None,
        help="Override batch size for eval (no gradients → can be larger).",
    )
    parser.add_argument(
        "--gen_max_new_tokens", type=int, default=None,
        help="Max new tokens for generation. Use ~384 for GRPO checkpoints "
             "(reasoning trace + answer); the SFT default (64) suffices for "
             "SFT checkpoints.",
    )
    args = parser.parse_args()

    run_eval(
        experiment_name    = args.experiment_name,
        checkpoint         = args.checkpoint,
        split              = args.split,
        output_dir         = Path(args.output_dir) if args.output_dir else None,
        data_root          = args.data_root,
        mock               = args.mock,
        split_file         = args.split_file,
        eval_batch_size    = args.eval_batch_size,
        gen_max_new_tokens = args.gen_max_new_tokens,
        weights            = args.weights,
        model_config       = args.model_config,
    )


if __name__ == "__main__":
    main()
