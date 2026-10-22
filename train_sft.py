"""
train_sft.py — SurgQ-VLM Stage 1: Supervised Fine-Tuning
=========================================================

Trains the question-conditioned dual-stream compressor and the LoRA
adapters on the Qwen2.5-VL LLM jointly, with token-level cross-entropy and
Q-type inverse-frequency loss weighting. This is the paper's Stage 1;
Stage 2 (train_grpo.py) warm-starts from the best checkpoint produced here.

Usage
-----
    # Train with the paper recipe
    python train_sft.py --config configs/sft.yaml

    # Resume an interrupted run
    python train_sft.py --config configs/sft.yaml --resume

    # Override any config field on the command line
    python train_sft.py --config configs/sft.yaml --learning_rate 1e-4

    # CPU shape-verification dry run (no weights downloaded, no GPU needed)
    python train_sft.py --config configs/sft.yaml --mock --dry_run

Output structure
----------------
    checkpoints/{experiment_name}/
        checkpoint.pt           ← latest checkpoint (trainable weights only)
        best_checkpoint.pt      ← best checkpoint by generative primary metric
        train_config.json       ← full config used for this run
        metrics_log.jsonl       ← one JSON line per validation event
        train.log               ← full log

Validation strategy
-------------------
Two signals run during training:
  1. Loss-only teacher-forced validation (fast) — a divergence alarm.
     A NaN/Inf loss hard-aborts training.
  2. Generative validation on a small curated subset (real autoregressive
     decoding, ~528 samples) — its overall_primary metric drives best-
     checkpoint selection and early stopping. Built by
     data_prep/build_splits.py.
"""

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

# Make the repository root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).parent))


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SFTConfig:
    """
    All hyperparameters for one SFT training run.

    Field defaults are the exact recipe used for the paper's 7B model;
    configs/sft.yaml carries the same values explicitly.

    Attributes
    ----------
    experiment_name : str
        Checkpoint directory name under output_dir.
    model_size : str
        Qwen2.5-VL size: "7b" (paper) or "3b".
    mock : bool
        If True, use random weights (no Qwen download). For shape testing.
    lora_mode / lora_r / lora_alpha / lora_dropout
        LoRA adapter settings. Paper: attention_and_ffn, r=16, alpha=32,
        dropout=0.0 (single-epoch training cannot overfit; dropout would
        waste gradient signal).
    learning_rate / warmup_steps / lr_schedule / weight_decay / max_grad_norm
        Optimiser settings. Paper: 2e-4 peak LR, 50 warmup steps, cosine
        decay, weight decay 0.01, gradient clip 1.0.
    num_epochs / batch_size / grad_accum_steps
        Paper: 1 epoch at batch 8 × accumulation 8 = effective batch 64.
    loss_weight_mode / loss_weight_cap
        "inverse_freq" upweights rare Q-types so binary questions (the
        largest share of the training mixture) do not swamp the long tail.
        Rare-type weights are capped at loss_weight_cap × median.
    gradient_checkpointing : bool
        Saves ~50% GPU memory at ~20% compute cost. Required to fit the 7B
        model comfortably on an 80 GB A100 at batch 8.
    num_workers : int
        DataLoader workers. 8 keeps an A100 saturated; 0 for debugging.
    use_8bit_adam : bool
        bitsandbytes 8-bit AdamW — saves ~6 GB of optimizer state on 7B.
    data_root : str
        Root containing splits/ and clips/ (see data_prep/).
    val_every_epochs / early_stop_patience
        Epoch-level validation cadence and patience (0 disables early stop).
    val_every_steps / step_val_batches
        Mid-epoch validation every N optimizer steps: a loss-only
        divergence check over step_val_batches batches plus a generative
        pass over the curated subset. Writes a safety checkpoint.
    curated_val_split / curated_val_qids
        The curated generative-validation subset (produced by
        data_prep/build_splits.py). When missing, generative monitoring is
        skipped and only the loss signal is used.
    gen_max_new_tokens : int
        Max new tokens for generative validation (64 covers the longest
        templated answer).
    max_seq_len : int
        Hard cap on prompt + answer token length. Typical usage is ~810
        tokens (671 visual + ~50 prompt + ~90 answer); 1024 is a buffer.
    """

    experiment_name:         str   = "sft"
    description:             str   = ""

    # Model
    model_size:              str   = "7b"
    mock:                    bool  = False

    # LoRA
    lora_mode:               str   = "attention_and_ffn"
    lora_r:                  int   = 16
    lora_alpha:              float = 32.0
    lora_dropout:            float = 0.0

    # Optimisation
    learning_rate:           float = 2e-4
    warmup_steps:            int   = 50
    lr_schedule:             str   = "cosine"
    weight_decay:            float = 0.01
    max_grad_norm:           float = 1.0
    num_epochs:              int   = 1
    batch_size:              int   = 8
    grad_accum_steps:        int   = 8

    # Loss weighting
    loss_weight_mode:        str   = "inverse_freq"
    loss_weight_cap:         float = 5.0

    # Memory / performance
    gradient_checkpointing:  bool  = True
    num_workers:             int   = 8
    use_8bit_adam:           bool  = True

    # Data
    data_root:               str   = "datasets"

    # Logging / checkpointing
    output_dir:              str   = "checkpoints"
    val_every_epochs:        int   = 1
    early_stop_patience:     int   = 0
    log_every_steps:         int   = 5
    max_seq_len:             int   = 1024
    seed:                    int   = 42
    resume:                  bool  = False

    # Mid-epoch step-based validation
    val_every_steps:         int   = 400
    step_val_batches:        int   = 20

    # Debug / smoke test
    max_batches:             int   = 0   # 0 = unlimited; >0 = stop after N batches

    # Curated-val generative monitoring
    curated_val_split:       str   = "datasets/splits/curated_val.txt"
    curated_val_qids:        str   = "datasets/splits/curated_val_qids.json"
    gen_max_new_tokens:      int   = 64

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name

    @property
    def train_split(self) -> Path:
        return Path(self.data_root) / "splits" / "train.txt"

    @property
    def val_split(self) -> Path:
        return Path(self.data_root) / "splits" / "val.txt"

    @property
    def test_split(self) -> Path:
        return Path(self.data_root) / "splits" / "test.txt"


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(config: SFTConfig) -> None:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.checkpoint_dir / "train.log"

    # Append so a resume doesn't wipe the previous log; write a separator
    # so it is clear where a new run started.
    log_file = logging.FileHandler(log_path, mode="a")
    with open(log_path, "a") as f:
        import datetime
        f.write(
            f"\n{'='*60}\n"
            f"RUN STARTED {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f" | resume={config.resume}\n"
            f"{'='*60}\n"
        )

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s %(message)s",
        datefmt = "%H:%M:%S",
        handlers = [
            logging.StreamHandler(sys.stdout),
            log_file,
        ],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── Model setup ───────────────────────────────────────────────────────────────

def build_model(config):
    """
    Load SurgQVLM (real or mock) and apply optional memory optimisations.

    Shared by train_sft.py, train_grpo.py, and evaluate.py — `config` may be
    an SFTConfig or GRPOConfig; only the fields read here are required.

    Returns (model, tokenizer, video_token_id).
    """
    from surgqvlm.model import SurgQVLM, SurgQVLMConfig, LoRAConfig

    vlm_config = SurgQVLMConfig(
        model_size       = config.model_size,
        use_mock_encoder = config.mock,
        freeze_vit       = True,
        lora = LoRAConfig(
            lora_mode    = config.lora_mode,
            r            = config.lora_r,
            lora_alpha   = config.lora_alpha,
            lora_dropout = config.lora_dropout,
        ),
    )

    model = SurgQVLM(vlm_config)

    # ── Gradient checkpointing (memory saving) ────────────────────────────
    if config.gradient_checkpointing and not config.mock:
        try:
            # enable_input_require_grads is required for gradient
            # checkpointing to work with PEFT adapters.
            model.qwen_model.enable_input_require_grads()
            model.qwen_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        except Exception as e:
            logger.warning("Could not enable gradient checkpointing: %s", e)

    # ── Optionally freeze the compressor (GRPO stage) ─────────────────────
    if getattr(config, "freeze_compressor", False):
        frozen_n = 0
        for name, param in model.named_parameters():
            if "compressor" in name:
                param.requires_grad_(False)
                frozen_n += param.numel()
        logger.info(
            "freeze_compressor=True: froze %dM compressor params.",
            frozen_n // 1_000_000,
        )

    trainable = model.count_trainable_params()
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model ready | trainable: %dM / %dM total (%.1f%%)",
        trainable // 1_000_000,
        total     // 1_000_000,
        100 * trainable / total,
    )

    if config.mock:
        class _MockTok:
            pad_token_id = 0
            eos_token_id = 1
            def encode(self, text, **kw):
                return [1] * min(len(text.split()), 20)
            def decode(self, ids, **kw):
                return "mock"
            def batch_decode(self, ids, **kw):
                try:
                    n = ids.shape[0]
                except Exception:
                    n = len(ids)
                return ["mock"] * n

        return model, _MockTok(), 42

    return model, model.tokenizer, model.video_token_id


# ── Optimizer and scheduler ───────────────────────────────────────────────────

def build_optimizer(model, config):
    """AdamW (or 8-bit AdamW via bitsandbytes) on trainable parameters only."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Optimizing %d parameter tensors (%dM total trainable)",
        len(trainable_params),
        sum(p.numel() for p in trainable_params) // 1_000_000,
    )

    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr           = config.learning_rate,
                weight_decay = config.weight_decay,
            )
            logger.info("Using 8-bit AdamW (bitsandbytes).")
            return optimizer
        except ImportError:
            logger.warning(
                "bitsandbytes not installed — falling back to standard AdamW."
            )

    return AdamW(
        trainable_params,
        lr           = config.learning_rate,
        weight_decay = config.weight_decay,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )


def build_scheduler(optimizer, config, total_steps: int):
    """Warmup + main schedule. Warmup is always linear."""
    warmup = LinearLR(
        optimizer,
        start_factor = 1e-8 / config.learning_rate,   # near-zero start
        end_factor   = 1.0,
        total_iters  = config.warmup_steps,
    )

    if config.lr_schedule == "cosine":
        main = CosineAnnealingLR(
            optimizer,
            T_max   = max(1, total_steps - config.warmup_steps),
            eta_min = config.learning_rate * 0.01,
        )
    elif config.lr_schedule == "linear":
        main = LinearLR(
            optimizer,
            start_factor = 1.0,
            end_factor   = 0.01,
            total_iters  = max(1, total_steps - config.warmup_steps),
        )
    elif config.lr_schedule == "constant":
        main = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)
    else:
        raise ValueError(f"Unknown lr_schedule '{config.lr_schedule}'")

    return SequentialLR(
        optimizer,
        schedulers  = [warmup, main],
        milestones  = [config.warmup_steps],
    )


# ── Batch preparation ─────────────────────────────────────────────────────────

def prepare_batch(
    raw_batch:         dict,
    prompt_builder,
    compressor_config,
    device:            torch.device,
    pad_token_id:      int,
):
    """
    Convert a raw DataLoader batch into the tensors expected by
    SurgQVLM.forward().

    Builds one tokenized sample per batch item with exactly the number of
    <|video_pad|> placeholders that the compressor will produce for that
    clip's real (unpadded) frame counts, pads to a common length, and
    computes per-sample Q-type loss weights.

    Returns
    -------
    dict with all tensors moved to device, plus questions and weights.
    """
    from surgqvlm.utils import compute_qtype_weights, expected_visual_tokens

    slow_frames = raw_batch["slow_frames"]   # (B, N_slow, 3, H, W)
    fast_frames = raw_batch["fast_frames"]   # (B, N_fast, 3, H, W)
    slow_mask   = raw_batch["slow_mask"]     # (B, N_slow)
    fast_mask   = raw_batch["fast_mask"]     # (B, N_fast)
    meta        = raw_batch["meta"]          # list of dicts
    questions_in_batch = raw_batch["question"]
    answers_in_batch   = raw_batch["answer"]

    B = slow_frames.shape[0]

    samples        = []
    questions      = []
    question_types = []

    for i in range(B):
        # Real frame counts (ignoring padding)
        n_slow_real = int(slow_mask[i].sum().item())
        n_fast_real = int(fast_mask[i].sum().item())

        # Number of visual tokens this clip will produce
        n_vis = expected_visual_tokens(n_slow_real, n_fast_real, compressor_config)

        sample = prompt_builder.build_sample(
            question        = questions_in_batch[i],
            answer          = answers_in_batch[i],
            n_visual_tokens = n_vis,
        )
        samples.append(sample)
        questions.append(questions_in_batch[i])
        question_types.append(meta[i].get("question_type", "Q01"))

    # Pad all samples to the same length
    padded = prompt_builder.collate_batch(samples, pad_token_id)

    # Loss weights — dataset-level inverse-frequency (stable at any batch size)
    sample_weights = compute_qtype_weights(
        question_types,
        mode         = getattr(prompt_builder, "_loss_weight_mode", "none"),
        dataset_freq = getattr(prompt_builder, "_dataset_freq", None),
    )

    return {
        "slow_frames":    slow_frames.to(device),
        "fast_frames":    fast_frames.to(device),
        "slow_mask":      slow_mask,    # kept on CPU — used only for integer counts
        "fast_mask":      fast_mask,
        "input_ids":      padded["input_ids"].to(device),
        "attention_mask": padded["attention_mask"].to(device),
        "labels":         padded["labels"].to(device),
        "questions":      questions,
        "question_types": question_types,
        "sample_weights": sample_weights.to(device),
    }


# ── Mid-epoch resume helper ───────────────────────────────────────────────────

def _make_resume_loader(dataset, config, skip_batches: int):
    """Return a DataLoader covering only the samples not yet processed.

    Reproduces the same shuffled permutation as the start of the epoch
    (the caller must already have re-seeded torch for the epoch), drops the
    first skip_batches × batch_size indices, and wraps the rest in a
    SubsetRandomSampler. Workers never load the discarded batches.
    """
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from surgqvlm.data import collate_fn

    n          = len(dataset)
    skip_n     = skip_batches * config.batch_size
    perm       = torch.randperm(n).tolist()
    remaining  = perm[skip_n:]

    sampler = SubsetRandomSampler(remaining)
    kwargs = dict(
        batch_size  = config.batch_size,
        sampler     = sampler,
        num_workers = config.num_workers,
        pin_memory  = True,
        collate_fn  = collate_fn,
        drop_last   = False,
    )
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"]    = 4
    return DataLoader(dataset, **kwargs)


# ── Training epoch ────────────────────────────────────────────────────────────

def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    prompt_builder,
    compressor_config,
    config:       SFTConfig,
    epoch:        int,
    device:       torch.device,
    global_step:  int,
    on_step_end:  Optional[callable] = None,
) -> tuple[float, int, bool]:
    """
    Run one full training epoch.

    Returns
    -------
    (mean_train_loss, updated_global_step, stopped_early)
    """
    from surgqvlm.utils import compute_weighted_loss

    model.train()
    total_loss  = 0.0
    n_batches   = 0
    optimizer_steps = 0
    accum_loss  = 0.0
    accum_count = 0

    optimizer.zero_grad()

    for step, raw_batch in enumerate(dataloader):
        if config.max_batches > 0 and step >= config.max_batches:
            logger.info("max_batches=%d reached — stopping epoch early.",
                        config.max_batches)
            break

        batch = prepare_batch(
            raw_batch         = raw_batch,
            prompt_builder    = prompt_builder,
            compressor_config = compressor_config,
            device            = device,
            pad_token_id      = getattr(
                prompt_builder.tokenizer, "pad_token_id", 0
            ) or 0,
        )

        # Forward pass — get logits, compute the weighted loss ourselves
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            outputs = model(
                slow_frames    = batch["slow_frames"],
                fast_frames    = batch["fast_frames"],
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                questions      = batch["questions"],
                labels         = None,   # loss computed below with weighting
                slow_mask      = batch["slow_mask"],
                fast_mask      = batch["fast_mask"],
            )

        loss = compute_weighted_loss(
            logits         = outputs["logits"],
            labels         = batch["labels"],
            sample_weights = batch["sample_weights"]
            if config.loss_weight_mode != "none" else None,
        )

        # Gradient accumulation
        scaled_loss = loss / config.grad_accum_steps
        scaled_loss.backward()

        accum_loss  += loss.item()
        accum_count += 1

        if (step + 1) % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            optimizer_steps += 1
            global_step     += 1

            if optimizer_steps % config.log_every_steps == 0:
                avg = accum_loss / accum_count
                lr  = scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch %d | step %d (global %d) | loss=%.4f | lr=%.2e",
                    epoch, optimizer_steps, global_step, avg, lr,
                )
                accum_loss  = 0.0
                accum_count = 0

            # Mid-epoch step validation callback
            if (on_step_end is not None
                    and config.val_every_steps > 0
                    and global_step % config.val_every_steps == 0):
                model.eval()
                should_stop = on_step_end(global_step)
                model.train()
                if should_stop:
                    mean_loss = total_loss / max(n_batches, 1)
                    return mean_loss, global_step, True

        total_loss += loss.item()
        n_batches  += 1

    # Flush remaining gradients if dataset size is not divisible by accum
    if (len(dataloader) % config.grad_accum_steps) != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    mean_loss = total_loss / max(n_batches, 1)
    return mean_loss, global_step, False


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> str:
    """Extract the final answer from a possibly-tagged model output.

    Handles:
      * ``<think>...</think>{answer}`` reasoning traces (produced by the
        GRPO-trained model; rewarded via the format term R2).
      * Truncated output that opened ``<think>`` but never closed it
        (no answer was produced → empty string).
      * Plain text with no tags (returned as-is).
    """
    if not text:
        return ""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if text.startswith("<think>"):
        return ""          # truncated — no answer produced
    return text.strip()


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model,
    dataloader,
    prompt_builder,
    compressor_config,
    config,
    device:      torch.device,
    epoch:       int,
    max_batches: int  = 0,       # 0 = full val set; >0 = stop after N batches
    tag:         str  = "val",
) -> dict:
    """
    Loss-only validation: a single teacher-forced forward pass returning the
    mean cross-entropy loss.

    Teacher-forced argmax metrics are deliberately NOT computed here — every
    answer position would be conditioned on ground-truth previous tokens,
    systematically overestimating autoregressive performance. Real metrics
    come from validate_generative().
    """
    from surgqvlm.utils import compute_weighted_loss

    model.eval()
    total_loss = 0.0
    n_batches  = 0

    pad_id = getattr(prompt_builder.tokenizer, "pad_token_id", 0) or 0
    total_batches = len(dataloader) if max_batches == 0 else max_batches
    t_val_start = time.time()

    for val_step, raw_batch in enumerate(dataloader):
        if max_batches > 0 and val_step >= max_batches:
            break

        batch = prepare_batch(
            raw_batch         = raw_batch,
            prompt_builder    = prompt_builder,
            compressor_config = compressor_config,
            device            = device,
            pad_token_id      = pad_id,
        )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            outputs = model(
                slow_frames    = batch["slow_frames"],
                fast_frames    = batch["fast_frames"],
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                questions      = batch["questions"],
                labels         = None,
                slow_mask      = batch["slow_mask"],
                fast_mask      = batch["fast_mask"],
            )

        loss = compute_weighted_loss(
            logits = outputs["logits"],
            labels = batch["labels"],
        )
        total_loss += loss.item()
        n_batches  += 1

        if val_step % 100 == 0:
            elapsed = time.time() - t_val_start
            logger.info("  %s step %d / %d | running_loss=%.4f | %.0fs elapsed",
                        tag, val_step, total_batches,
                        total_loss / n_batches, elapsed)

    val_loss = total_loss / max(n_batches, 1)
    logger.info("Epoch %d %s (loss-only) | val_loss=%.4f | n_batches=%d",
                epoch, tag, val_loss, n_batches)
    return {"val_loss": val_loss, "metrics": {}, "predictions": []}


@torch.no_grad()
def validate_generative(
    model,
    dataloader,
    prompt_builder,
    compressor_config,
    config,
    device: torch.device,
    epoch:  int,
    tag:    str = "curated_val",
) -> dict:
    """
    Generative validation — runs ``model.generate()`` on every sample and
    scores the decoded answers with the answer-type-aware metrics.

    Unlike teacher-forced argmax, this reflects the model's true
    autoregressive output distribution and is what drives checkpoint
    selection. Cost: ~2-3 minutes for the ~528-sample curated subset on an
    A100; ~30 minutes for a full split.
    """
    from surgqvlm.utils import (
        evaluate_sample, aggregate_metrics, expected_visual_tokens,
    )

    model.eval()
    tokenizer = prompt_builder.tokenizer
    pad_id    = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    all_preds: list[dict] = []
    t0 = time.time()
    total_batches = len(dataloader)

    logger.info(
        "Starting generative eval | tag=%s | batches=%d | batch_size=%d",
        tag, total_batches, dataloader.batch_size or 1,
    )

    pbar = tqdm(dataloader, desc=f"eval[{tag}]", unit="batch",
                ncols=120, dynamic_ncols=False, leave=True)

    for val_step, raw_batch in enumerate(pbar):
        slow_frames = raw_batch["slow_frames"].to(device)
        fast_frames = raw_batch["fast_frames"].to(device)
        slow_mask   = raw_batch["slow_mask"]
        fast_mask   = raw_batch["fast_mask"]
        meta        = raw_batch["meta"]
        questions   = raw_batch["question"]
        answers     = raw_batch["answer"]

        B = slow_frames.shape[0]

        # Build prompt-only input_ids for generation.
        prompts = []
        for i in range(B):
            n_slow = int(slow_mask[i].sum().item())
            n_fast = int(fast_mask[i].sum().item())
            n_vis  = expected_visual_tokens(n_slow, n_fast, compressor_config)
            prompts.append(
                prompt_builder.build_prompt_only(questions[i], n_vis)
            )

        # Left-pad so all prompts end at the same index (required for
        # batched generation with variable prompt lengths).
        max_len = max(len(p["input_ids"]) for p in prompts)
        input_ids      = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, p in enumerate(prompts):
            L = len(p["input_ids"])
            input_ids[i, max_len - L:] = torch.tensor(
                p["input_ids"], dtype=torch.long
            )
            attention_mask[i, max_len - L:] = 1

        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            gen_ids = model.generate(
                slow_frames    = slow_frames,
                fast_frames    = fast_frames,
                input_ids      = input_ids,
                attention_mask = attention_mask,
                questions      = questions,
                max_new_tokens = config.gen_max_new_tokens,
                slow_mask      = slow_mask,
                fast_mask      = fast_mask,
                do_sample      = False,
            )

        # gen_ids are the newly generated tokens only (see SurgQVLM.generate).
        if config.mock:
            preds_text = [""] * B
        else:
            preds_text = tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True
            )
        preds_text = [extract_final_answer(t) for t in preds_text]

        for i, pred in enumerate(preds_text):
            all_preds.append({
                "question":      questions[i],
                "reference":     answers[i],
                "prediction":    pred,
                "question_type": meta[i].get("question_type", "Q01"),
                "binary_label":  meta[i].get("binary_label"),
            })

        elapsed = time.time() - t0
        n_done  = len(all_preds)
        pbar.set_postfix(samples=n_done,
                         sps=f"{n_done / max(elapsed, 1e-6):.1f}",
                         elapsed=f"{elapsed/60:.1f}m")

    # Per-sample metrics.
    per_sample = []
    for p in all_preds:
        m = evaluate_sample(
            prediction    = p["prediction"],
            reference     = p["reference"],
            question_type = p["question_type"],
            binary_label  = p["binary_label"],
        )
        p["metrics"] = m
        per_sample.append(m)
    metrics = aggregate_metrics(
        per_sample, [p["question_type"] for p in all_preds]
    )

    elapsed = time.time() - t0
    logger.info(
        "Epoch %d %s (generative) | overall_primary=%.3f "
        "| overall_kacc=%.3f | n=%d | %.1f s",
        epoch, tag,
        metrics.get("overall_primary", 0.0),
        metrics.get("overall_kacc",    0.0),
        len(all_preds), elapsed,
    )
    return {
        "val_loss":    None,    # generative path doesn't compute a loss
        "metrics":     metrics,
        "predictions": all_preds,
    }


# ── Metrics logger ────────────────────────────────────────────────────────────

class MetricsLog:
    """Appends one JSON line per validation event to a .jsonl file."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, epoch: int, train_loss: float, val_result: Optional[dict],
              step: Optional[int] = None) -> None:
        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6) if not math.isnan(train_loss) else None,
        }
        if step is not None:
            record["global_step"] = step
        if val_result:
            if val_result.get("val_loss") is not None:
                record["val_loss"] = round(val_result["val_loss"], 6)
            record["overall_kacc"] = round(
                val_result["metrics"].get("overall_kacc", 0.0), 4
            )
            record["overall_primary"] = round(
                val_result["metrics"].get("overall_primary", 0.0), 4
            )
            for at, m in val_result["metrics"].get("by_answer_type", {}).items():
                for k, v in m.items():
                    record[f"{at}_{k}"] = round(v, 4)
            for qt, m in val_result["metrics"].get("by_qtype", {}).items():
                for k, v in m.items():
                    record[f"{qt}_{k}"] = round(v, 4)

        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ── Main training loop ────────────────────────────────────────────────────────

def train(config: SFTConfig) -> None:
    """Full SFT training pipeline."""
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("SurgQ-VLM SFT | %s", config.experiment_name)
    logger.info("=" * 60)
    logger.info("Description : %s", config.description)
    logger.info("Train split : %s", config.train_split)
    logger.info("Val split   : %s", config.val_split)
    logger.info("Effective B : %d (%d × %d accum)", config.effective_batch_size,
                config.batch_size, config.grad_accum_steps)
    logger.info("Checkpoint  : %s", config.checkpoint_dir)

    # Reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s | %.1f GB VRAM", props.name,
                    props.total_memory / 1e9)
        # TF32 on Ampere+: tensor-core matmul/conv at ~3× FP32 throughput
        # with negligible accuracy loss vs bfloat16 training.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        logger.info("TF32 enabled for matmul and cuDNN.")

    # ── Build model ────────────────────────────────────────────────────────
    model, tokenizer, video_token_id = build_model(config)
    model = model.to(device) if config.mock else model.to_device(device)

    # Qwen2.5-VL's tokenizer ships without a pad_token by default. Without
    # this, batched generation silently uses token id 0 (an unrelated
    # vocabulary entry), which pollutes generations.
    if not config.mock and getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer pad_token was None — set to eos_token (id=%s).",
                        tokenizer.pad_token_id)
        else:
            raise RuntimeError(
                "Tokenizer has no pad_token AND no eos_token. "
                "Cannot proceed — set a pad token explicitly."
            )

    # ── Build dataloaders ──────────────────────────────────────────────────
    from surgqvlm.data import SurgQDataset, build_dataloader

    train_dataset = SurgQDataset(
        split_file       = str(config.train_split),
        use_mock_encoder = config.mock,
    )
    val_dataset = SurgQDataset(
        split_file       = str(config.val_split),
        use_mock_encoder = config.mock,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = config.num_workers,
    )

    # Curated-val dataset: fixed stratified subset for generative monitoring.
    curated_loader = None
    cval_path = Path(config.curated_val_split)
    qid_path  = Path(config.curated_val_qids)
    if (not config.mock) and cval_path.exists() and qid_path.exists():
        with open(qid_path) as f:
            qid_whitelist = set(json.load(f).keys())
        curated_ds = SurgQDataset(
            split_file       = str(cval_path),
            use_mock_encoder = config.mock,
            filter_qids      = qid_whitelist,
        )
        curated_loader = build_dataloader(
            curated_ds,
            batch_size  = config.batch_size,
            shuffle     = False,
            num_workers = min(config.num_workers, 4),
        )
        logger.info("Curated-val subset loaded: %d samples (from %s)",
                    len(curated_ds), cval_path)
    else:
        logger.info(
            "Curated-val subset not found at %s — generative monitoring "
            "disabled (best checkpoint falls back to val loss). "
            "Run data_prep/build_splits.py to create it.", cval_path,
        )

    logger.info("Data: %d train samples / %d val samples",
                len(train_dataset), len(val_dataset))

    # ── Prompt builder ─────────────────────────────────────────────────────
    from surgqvlm.utils import (
        PromptBuilder, build_dataset_weights, EarlyStopping,
        save_checkpoint, load_checkpoint,
    )

    prompt_builder = PromptBuilder(
        tokenizer      = tokenizer,
        video_token_id = video_token_id,
        max_seq_len    = config.max_seq_len,
    )
    prompt_builder._loss_weight_mode = config.loss_weight_mode

    # Precompute dataset-level Q-type weights (needed for inverse_freq mode).
    dataset_freq = None
    if config.loss_weight_mode == "inverse_freq":
        dataset_freq = build_dataset_weights(
            str(config.train_split),
            cap = config.loss_weight_cap,
        )
        logger.info(
            "Dataset Q-type weights (top 5 by weight): %s",
            sorted(dataset_freq.items(), key=lambda x: -x[1])[:5],
        )
    prompt_builder._dataset_freq = dataset_freq

    # Compressor config for visual token count calculation
    compressor_config = model.config.compressor

    # ── Optimizer and scheduler ────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
    total_steps     = steps_per_epoch * config.num_epochs

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps)

    logger.info("Training: %d epochs × %d steps/epoch = %d total optimizer steps",
                config.num_epochs, steps_per_epoch, total_steps)

    # ── Early stopping + metrics log ───────────────────────────────────────
    early_stop = EarlyStopping(
        patience  = config.early_stop_patience,
        min_delta = 1e-4,
    )
    metrics_log   = MetricsLog(config.checkpoint_dir / "metrics_log.jsonl")
    best_val_loss = float("inf")
    best_primary  = 0.0
    global_step   = 0
    start_epoch   = 0
    skip_batches  = 0

    # ── Resume from checkpoint ─────────────────────────────────────────────
    if config.resume:
        ckpt_path = config.checkpoint_dir / "checkpoint.pt"
        if ckpt_path.exists():
            start_epoch, best_val_loss, best_primary, ckpt_global_step = \
                load_checkpoint(
                    model     = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    path      = config.checkpoint_dir,
                    device    = device,
                )
            if ckpt_global_step is not None:
                global_step = ckpt_global_step
                start_epoch = global_step // steps_per_epoch
                # Skip already-processed batches so the optimizer and
                # scheduler continue from exactly where they left off.
                steps_into_epoch = global_step % steps_per_epoch
                skip_batches = steps_into_epoch * config.grad_accum_steps
            else:
                start_epoch = start_epoch - 1
                global_step = start_epoch * steps_per_epoch
                skip_batches = 0
            logger.info("Resuming from epoch %d (best_primary=%.4f, global_step=%d)",
                        start_epoch, best_primary, global_step)
            # Re-seed so shuffling continues deterministically, then rebuild
            # the train loader to cover only the remaining samples.
            torch.manual_seed(config.seed + start_epoch)
            if skip_batches > 0:
                train_loader = _make_resume_loader(train_dataset, config, skip_batches)
                logger.info(
                    "Resume DataLoader: skipped %d batches (%d samples).",
                    skip_batches, skip_batches * config.batch_size,
                )
        else:
            logger.warning("--resume requested but no checkpoint found at %s. "
                           "Starting from scratch.", ckpt_path)

    # ── Mid-epoch step validation callback ─────────────────────────────────
    _current_epoch = [start_epoch + 1]   # mutable ref so the closure can read it

    def on_step_end(step: int) -> bool:
        """Step-level validation callback.

        Runs:
          1) Loss-only teacher-forced validation on a slice of the val set —
             a divergence alarm. NaN/Inf loss hard-aborts training.
          2) Generative validation on the curated subset. Its
             overall_primary drives best-checkpoint selection.

        Checking at every step-val (~every 400 optimizer steps) catches
        peaks that land mid-epoch and would otherwise be overwritten.
        Step-vals are monitoring-only — they do not count toward early
        stopping patience (the ~528-sample generative primary has ~±5%
        noise; epoch-level checks are the stopping signal).
        """
        nonlocal best_val_loss, best_primary

        # 1) Divergence alarm — loss only.
        loss_result = validate(
            model             = model,
            dataloader        = val_loader,
            prompt_builder    = prompt_builder,
            compressor_config = compressor_config,
            config            = config,
            device            = device,
            epoch             = _current_epoch[0],
            max_batches       = config.step_val_batches,
            tag               = "val_step",
        )
        vloss = loss_result["val_loss"]
        if math.isnan(vloss) or math.isinf(vloss):
            logger.error(
                "NaN/Inf val_loss at step %d — aborting training. "
                "Most likely: LR too high, or a bad gradient update.", step
            )
            return True

        # 2) Real generative metric on the curated subset.
        gen_result = None
        if curated_loader is not None:
            gen_result = validate_generative(
                model             = model,
                dataloader        = curated_loader,
                prompt_builder    = prompt_builder,
                compressor_config = compressor_config,
                config            = config,
                device            = device,
                epoch             = _current_epoch[0],
                tag               = "curated_val_step",
            )

        step_primary = None
        if gen_result is not None:
            step_primary = gen_result["metrics"].get(
                "overall_primary",
                gen_result["metrics"].get("overall_kacc", 0.0),
            )

        is_best = (step_primary is not None and step_primary > best_primary)
        if is_best:
            logger.info(
                "Step %d: new best generative primary %.4f > %.4f — "
                "saving best_checkpoint.pt", step, step_primary, best_primary,
            )
            best_primary = step_primary

        save_checkpoint(
            model       = model,
            optimizer   = optimizer,
            scheduler   = scheduler,
            epoch       = _current_epoch[0],
            val_loss    = vloss,
            metrics     = (gen_result["metrics"] if gen_result
                           else loss_result["metrics"]),
            config      = config,
            path        = config.checkpoint_dir,
            global_step = step,
            is_best     = is_best,
        )

        merged = {
            "val_loss": vloss,
            "metrics":  (gen_result["metrics"] if gen_result else {}),
        }
        metrics_log.write(_current_epoch[0], float("nan"), merged, step=step)

        if vloss < best_val_loss:
            best_val_loss = vloss

        return False

    step_callback = on_step_end if config.val_every_steps > 0 else None

    # ── Training loop ──────────────────────────────────────────────────────
    stopped_early = False
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        _current_epoch[0] = epoch
        t0 = time.time()

        train_loss, global_step, stopped_early = train_epoch(
            model             = model,
            dataloader        = train_loader,
            optimizer         = optimizer,
            scheduler         = scheduler,
            prompt_builder    = prompt_builder,
            compressor_config = compressor_config,
            config            = config,
            epoch             = epoch,
            device            = device,
            global_step       = global_step,
            on_step_end       = step_callback,
        )

        elapsed = time.time() - t0
        logger.info("Epoch %d done | train_loss=%.4f | %.1f s",
                    epoch, train_loss, elapsed)

        # Epoch-end validation:
        #   * loss-only teacher-forced on the full val split → loss curve
        #   * generative on the curated subset → the real metric that
        #     drives selection and early stopping
        val_result = None
        if epoch % config.val_every_epochs == 0 or epoch == config.num_epochs:
            val_result = validate(
                model             = model,
                dataloader        = val_loader,
                prompt_builder    = prompt_builder,
                compressor_config = compressor_config,
                config            = config,
                device            = device,
                epoch             = epoch,
                max_batches       = config.max_batches if config.max_batches > 0 else 0,
                tag               = "val_epoch",
            )

            gen_result = None
            if curated_loader is not None:
                gen_result = validate_generative(
                    model             = model,
                    dataloader        = curated_loader,
                    prompt_builder    = prompt_builder,
                    compressor_config = compressor_config,
                    config            = config,
                    device            = device,
                    epoch             = epoch,
                    tag               = "curated_val_epoch",
                )
                val_result["metrics"]["generative"] = gen_result["metrics"]

            epoch_primary = None
            if gen_result is not None:
                epoch_primary = gen_result["metrics"].get(
                    "overall_primary",
                    gen_result["metrics"].get("overall_kacc", 0.0),
                )

            is_best = (epoch_primary is not None
                       and epoch_primary > best_primary)
            if is_best:
                best_primary = epoch_primary
            if val_result["val_loss"] < best_val_loss:
                best_val_loss = val_result["val_loss"]

            logger.info(
                "Epoch %d summary | val_loss=%.4f | generative primary=%.4f",
                epoch, val_result["val_loss"],
                epoch_primary if epoch_primary is not None else float("nan"),
            )

            save_checkpoint(
                model       = model,
                optimizer   = optimizer,
                scheduler   = scheduler,
                epoch       = epoch,
                val_loss    = val_result["val_loss"],
                metrics     = val_result["metrics"],
                config      = config,
                path        = config.checkpoint_dir,
                global_step = global_step,
                is_best     = is_best,
            )

            if gen_result is not None:
                gen_pred_path = (
                    config.checkpoint_dir
                    / f"curated_predictions_epoch{epoch:03d}.jsonl"
                )
                with open(gen_pred_path, "w") as f:
                    for p in gen_result["predictions"]:
                        f.write(json.dumps(p) + "\n")

            if (not stopped_early
                    and config.early_stop_patience > 0
                    and epoch_primary is not None):
                if early_stop.step(-epoch_primary):
                    logger.info("Early stopping at epoch %d "
                                "(no generative-primary improvement).", epoch)
                    stopped_early = True

        metrics_log.write(epoch, train_loss, val_result)

        if stopped_early:
            break

    logger.info("Training complete. Best primary=%.4f | Best val_loss=%.4f",
                best_primary, best_val_loss)
    logger.info("Checkpoints saved to %s", config.checkpoint_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    from surgqvlm.utils import (
        load_yaml_config, add_config_cli_overrides, apply_cli_overrides,
    )

    parser = argparse.ArgumentParser(
        description="SurgQ-VLM Stage 1: supervised fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/sft.yaml",
                        help="Path to the YAML training config.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint.pt in the experiment directory.")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock encoder (no weights downloaded, CPU OK).")
    parser.add_argument("--dry_run", action="store_true",
                        help="1 epoch, 10 batches — shape verification only.")
    # Any config field can be overridden: --learning_rate 1e-4, etc.
    add_config_cli_overrides(parser, SFTConfig,
                             skip=("mock", "resume", "description"))
    args = parser.parse_args()

    config = load_yaml_config(SFTConfig, args.config)
    apply_cli_overrides(config, args)

    if args.resume:
        config.resume = True
    if args.mock:
        config.mock = True
    if args.dry_run:
        config.num_epochs          = 1
        config.val_every_epochs    = 1
        config.early_stop_patience = 0
        config.log_every_steps     = 1
        config.val_every_steps     = 0     # skip step-vals in a dry run
        if config.max_batches == 0:        # don't override an explicit value
            config.max_batches = 10
        logger.info("DRY RUN mode: 1 epoch, max_batches=%d.", config.max_batches)

    train(config)


if __name__ == "__main__":
    main()
