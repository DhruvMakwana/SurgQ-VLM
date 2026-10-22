"""
train_grpo.py — SurgQ-VLM Stage 2: Group Relative Policy Optimization
======================================================================

Post-trains the SFT checkpoint with GRPO under an answer-type-aware reward.
The compressor is frozen at this stage; only the LoRA adapters receive
gradients.

Implements GRPO with DAPO-inspired enhancements:
- Answer-type-aware rewards:
      R = R1 + 0.3·R2 + 0.05·R3 − 0.3·R_len
  R1 (task correctness) dispatches on the answer-structure group: binary
  match, keyword-accuracy threshold for classification, micro-F1 for
  multi-label, normalised count error, and a content–timing blend for
  sentence-temporal answers.
  R2 is a soft format check on the reasoning trace with an anti-hack
  penalty for annotation-lookup phrasing.
  R3 rewards temporal vocabulary and numeric second references on the six
  temporal Q-types.
  R_len is a linear overlong penalty.
- Decoupled asymmetric clipping from DAPO (εlow=0.2, εhigh=0.28)
- Dynamic sampling (2× oversample, keep mixed-reward prompts only)
- Inverse-frequency Q-type weighting for balanced gradient signal
- Reference model via LoRA weight swapping (no extra GPU memory)
- Batched generation (all B×G rollouts in one call)

Usage
-----
    # Train with the paper recipe (warm-starts from checkpoints/sft)
    python train_grpo.py --config configs/grpo.yaml

    # Reward function unit tests (CPU, instant)
    python train_grpo.py --test_rewards

    # CPU shape-verification dry run
    python train_grpo.py --config configs/grpo.yaml --mock --dry_run
"""
import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Make the repository root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).parent))


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    """
    All hyperparameters for one GRPO training run.

    Field defaults are the exact recipe used for the paper's reported model;
    configs/grpo.yaml carries the same values explicitly.

    Key fields
    ----------
    num_rollouts : int
        G — candidate responses sampled per prompt (8 in the paper).
    reward_*_weight : float
        Reward term weights. R2 and R_len are set to the same value (0.3)
        so format reward and length penalty exert equal and opposite
        pressure; the R3 weight was reduced from 0.2 to 0.05 after a pilot
        run revealed the model citing annotation-schema field names to gain
        temporal reward (see the paper, §3.6).
    kl_coeff / clip_low / clip_high
        KL penalty to the frozen SFT reference (β=0.04) and DAPO decoupled
        clipping (asymmetric: εhigh > εlow encourages exploration).
    oversample_factor : int
        Dynamic sampling: load oversample_factor × batch_size prompts, keep
        at most batch_size whose G rollouts have non-zero reward variance
        (zero-variance prompts give zero advantage — no learning signal).
    freeze_compressor : bool
        True in the paper — the compressor is frozen after SFT; only the
        LoRA adapters train during GRPO.
    warm_start_from : str
        The SFT experiment directory to initialise from.
    data_fraction : float
        Fraction of the training split to use. 1.0 for the main run;
        0.05 was used for the reward-coefficient pilot (paper §3.6).
    val_every_steps / val_gen_samples / save_every_steps
        Mid-training generative eval cadence (drives best-checkpoint
        selection) and periodic safety checkpoints.
    """

    experiment_name:         str   = "grpo"
    description:             str   = ""

    # Model (must match the SFT warm-start architecture)
    model_size:              str   = "7b"
    mock:                    bool  = False
    lora_mode:               str   = "attention_and_ffn"
    lora_r:                  int   = 16
    lora_alpha:              float = 32.0
    lora_dropout:            float = 0.0
    freeze_compressor:       bool  = True

    # ── Rollout ──
    num_rollouts:            int   = 8       # G: candidates per prompt
    temperature:             float = 1.0
    top_p:                   float = 0.95
    max_new_tokens:          int   = 384     # reasoning trace + answer

    # ── Reward weights ──
    reward_r1_weight:        float = 1.0     # answer correctness
    reward_r2_weight:        float = 0.3     # reasoning format (+ anti-hack)
    reward_r3_weight:        float = 0.05    # temporal coherence
    reward_len_weight:       float = 0.3     # overlong penalty (max deduction)

    # ── Policy optimisation ──
    kl_coeff:                float = 0.04    # KL penalty coefficient
    clip_low:                float = 0.2     # DAPO εlow  (floor)
    clip_high:               float = 0.28    # DAPO εhigh (ceiling, > εlow)
    learning_rate:           float = 5e-7
    warmup_steps:            int   = 20
    lr_schedule:             str   = "cosine"
    weight_decay:            float = 0.01
    max_grad_norm:           float = 1.0
    num_epochs:              int   = 1
    batch_size:              int   = 4       # prompts per step (each gets G rollouts)
    grad_accum_steps:        int   = 2

    # ── Loss weighting by Q-type ──
    loss_weight_mode:        str   = "inverse_freq"
    loss_weight_cap:         float = 5.0

    # ── Dynamic sampling (DAPO-inspired) ──
    oversample_factor:       int   = 2

    # ── Data ──
    data_root:               str   = "datasets"
    data_fraction:           float = 1.0

    # ── Memory / performance ──
    gradient_checkpointing:  bool  = True
    num_workers:             int   = 4
    use_8bit_adam:           bool  = True
    val_batch_size:          int   = 8

    # ── Logging / checkpointing ──
    output_dir:              str   = "checkpoints"
    log_every_steps:         int   = 1
    log_samples_every:       int   = 10      # log full rollout text every N steps
    val_every_epochs:        int   = 1
    val_every_steps:         int   = 500     # generative eval every N opt steps
    val_gen_samples:         int   = 500     # val samples for generative eval
    save_every_steps:        int   = 200     # periodic checkpoint.pt (0 = off)
    max_seq_len:             int   = 2048
    seed:                    int   = 42
    warm_start_from:         str   = "checkpoints/sft"

    # ── Debug ──
    max_batches:             int   = 0       # 0 = unlimited; >0 = stop after N

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


# ── Logging setup ─────────────────────────────────────────────────────────────

logger = logging.getLogger("train_grpo")


def _setup_logging(config: GRPOConfig):
    log_dir = config.checkpoint_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "train_grpo.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# REWARD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _strip_think(text: str) -> str:
    """Strip <think>...</think> reasoning, keep only the answer."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if text.startswith("<think>"):
        return ""
    return text


def _extract_think_block(text: str) -> str:
    """Extract the content inside <think>...</think> tags."""
    if "<think>" in text and "</think>" in text:
        return text.split("<think>", 1)[1].split("</think>", 1)[0]
    if "<think>" in text:
        return text.split("<think>", 1)[1]
    return ""


def compute_r1(
    prediction:    str,
    reference:     str,
    question_type: str,
    binary_label:  Optional[bool] = None,
) -> float:
    """
    R1 — answer correctness, dispatched on the answer-structure group.

    Returns
    -------
    float in [0, 1]:
        - binary:            0/1 via binary_accuracy
        - classification:    0/1 via keyword_accuracy >= 0.5
        - multi_label:       continuous [0,1] via multilabel_f1
        - counting:          continuous [0,1] via normalised count error
        - sentence_temporal: content–timing blend
                             0.2·(k+τ)/2 + 0.8·(k·τ)
          where k is continuous keyword accuracy and τ is temporal accuracy
          within ±2 s. The dominant multiplicative term grants credit only
          when both the named content and the timing are right; the small
          additive floor keeps a gradient when one side is zero.
    """
    from surgqvlm.utils import (
        QTYPE_TO_ANSWER_TYPE,
        binary_accuracy,
        counting_metrics,
        keyword_accuracy,
        multilabel_f1,
        temporal_accuracy,
    )

    answer = _strip_think(prediction)
    atype = QTYPE_TO_ANSWER_TYPE.get(question_type, "classification")

    if atype == "binary":
        if binary_label is not None:
            return binary_accuracy(answer, binary_label)
        return float(keyword_accuracy(answer, reference) >= 0.5)

    elif atype == "classification":
        return float(keyword_accuracy(answer, reference) >= 0.5)

    elif atype == "multi_label":
        return multilabel_f1(answer, reference, question_type=question_type)

    elif atype == "counting":
        return counting_metrics(answer, reference)["normalized"]

    elif atype == "sentence_temporal":
        kacc_cont = keyword_accuracy(answer, reference)
        temp_acc  = temporal_accuracy(answer, reference, tolerance=2)
        return 0.2 * (kacc_cont + temp_acc) / 2 + 0.8 * (kacc_cont * temp_acc)

    return float(keyword_accuracy(answer, reference) >= 0.5)


# Phrases that impersonate annotation lookups. Without a penalty, GRPO can
# reinforce a fake database-lookup style in the reasoning trace instead of
# genuine visual description.
_R2_HACK_PATTERNS = (
    r"instrument_per_frame",
    r"phase_per_frame",
    r"triplet_per_frame",
    r"verb_per_frame",
    r"target_per_frame",
    r"has_\w+_change",
    r"the annotation",
    r"annotation at this frame",
    r"I look up",
    r"I scan",
    r"I examine",
    r"I count",
    r"I check",
    r"I trace",
    r"Tracing",
    r"frame index \d",
)


def compute_r2(prediction: str) -> float:
    """
    R2 — reasoning format reward in [0, 1].

    +0.33 if the <think> tag is present
    +0.33 if the </think> tag is present
    +0.34 if there are >= 2 numbered "Step N" markers
    −0.3 per annotation-lookup hack phrase (floored at 0).
    """
    score = 0.0
    if "<think>" in prediction:
        score += 0.33
    if "</think>" in prediction:
        score += 0.33
    steps = re.findall(r"Step\s+\d+", prediction, re.IGNORECASE)
    if len(steps) >= 2:
        score += 0.34

    hack_hits = sum(
        1 for pat in _R2_HACK_PATTERNS
        if re.search(pat, prediction, re.IGNORECASE)
    )
    score -= 0.3 * hack_hits

    return max(0.0, score)


def compute_r3(prediction: str, question_type: str) -> float:
    """
    R3 — temporal coherence reward in [0, 1].

    Scored only on the six temporal Q-types; others get full credit.
    +0.5 for at least one numeric second/frame reference in the trace,
    +0.5 for temporal vocabulary (before/after/transition/begins/...).
    """
    TEMPORAL_QTYPES = {"Q03", "Q04", "Q08", "Q09", "Q16", "Q18"}
    if question_type not in TEMPORAL_QTYPES:
        return 1.0  # full credit for non-temporal questions

    think_content = _extract_think_block(prediction)
    if not think_content:
        return 0.0

    score = 0.0
    # References frame numbers or seconds
    if re.search(r"(frame|second|sec)\s*\d+", think_content, re.IGNORECASE):
        score += 0.5
    # References temporal patterns
    temporal_words = [
        "before", "after", "during", "transition", "changes",
        "begins", "ends", "first", "last", "appears", "disappears",
        "earlier", "later", "sequence", "followed",
    ]
    if any(w in think_content.lower() for w in temporal_words):
        score += 0.5
    return score


def compute_r_length(response_len: int, max_len: int = 512, cache: int = 128) -> float:
    """
    R_len — overlong reward shaping (DAPO-inspired, simplified).

    Returns 0.0 for normal-length responses, linearly ramps to −1.0 for
    responses approaching or exceeding max_len. Prevents the policy from
    drifting into ever-longer rollouts as the context budget approaches.
    """
    threshold = max_len - cache
    if response_len <= threshold:
        return 0.0
    elif response_len <= max_len:
        return -(response_len - threshold) / cache  # -0.0 → -1.0
    else:
        return -1.0


def compute_reward(
    prediction:    str,
    reference:     str,
    question_type: str,
    binary_label:  Optional[bool],
    config:        GRPOConfig,
    gen_len:       int = 0,
) -> dict[str, float]:
    """
    Compute the total reward and per-component breakdown.

    Returns a dict with keys: total, r1, r2, r3, r_len.
    """
    r1    = compute_r1(prediction, reference, question_type, binary_label)
    r2    = compute_r2(prediction)
    r3    = compute_r3(prediction, question_type)
    r_len = compute_r_length(gen_len, config.max_new_tokens, cache=128)

    total = (
        config.reward_r1_weight * r1
        + config.reward_r2_weight * r2
        + config.reward_r3_weight * r3
        + config.reward_len_weight * r_len
    )
    return {"total": total, "r1": r1, "r2": r2, "r3": r3, "r_len": r_len}


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE MODEL — LoRA weight swapping
# ═══════════════════════════════════════════════════════════════════════════

_LORA_KEYS = ("lora_", "modules_to_save")


def _is_lora_key(key: str) -> bool:
    return any(tag in key for tag in _LORA_KEYS)


def capture_ref_state(model) -> dict[str, torch.Tensor]:
    """Snapshot the LoRA weights to CPU for the reference model."""
    return {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
        if _is_lora_key(name)
    }


def _build_param_lookup(model) -> dict[str, torch.nn.Parameter]:
    """Build the name→parameter lookup ONCE, reuse for fast swaps."""
    lookup = {}
    for name, param in model.named_parameters():
        lookup[name] = param
    return lookup


# Module-level cache — populated on first swap call
_param_lookup: dict[str, torch.nn.Parameter] = {}


def swap_to_ref(model, ref_state: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Load reference weights into the model; return policy weights for restore.

    Uses direct param.data.copy_() instead of state_dict()/load_state_dict()
    which are very slow on large PEFT models (state_dict creates a full copy
    of all parameters each call).
    """
    global _param_lookup
    if not _param_lookup:
        _param_lookup = _build_param_lookup(model)

    policy_state = {}
    for k, ref_v in ref_state.items():
        param = _param_lookup.get(k)
        if param is not None:
            policy_state[k] = param.data.cpu().clone()
            param.data.copy_(ref_v.to(device))
    return policy_state


def swap_to_policy(model, policy_state: dict, device: torch.device) -> None:
    """Restore the policy weights into the model."""
    global _param_lookup
    if not _param_lookup:
        _param_lookup = _build_param_lookup(model)

    for k, pol_v in policy_state.items():
        param = _param_lookup.get(k)
        if param is not None:
            param.data.copy_(pol_v.to(device))


# ═══════════════════════════════════════════════════════════════════════════
# VISUAL ENCODING
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_visual_once(model, slow_frames, fast_frames, questions,
                       slow_mask=None, fast_mask=None):
    """
    Run the frozen ViT + compressor ONCE per sample.
    The result is reused for all G rollouts of that sample.

    Returns a list of B tensors, each (T_vis, llm_dim).
    """
    import einops

    B      = slow_frames.shape[0]
    N_slow = slow_frames.shape[1]
    N_fast = fast_frames.shape[1]
    device = slow_frames.device
    dtype  = slow_frames.dtype

    if model.config.use_mock_encoder:
        slow_tokens = model._encode_frames_mock(B, N_slow, device, dtype)
        fast_tokens = model._encode_frames_mock(B, N_fast, device, dtype)
    else:
        slow_flat = einops.rearrange(slow_frames, "b n c h w -> (b n) c h w")
        fast_flat = einops.rearrange(fast_frames, "b n c h w -> (b n) c h w")
        slow_enc  = model._encode_frames_vit(slow_flat)
        fast_enc  = model._encode_frames_vit(fast_flat)
        slow_tokens = einops.rearrange(
            slow_enc, "(b n) h w d -> b n h w d", b=B, n=N_slow
        )
        fast_tokens = einops.rearrange(
            fast_enc, "(b n) h w d -> b n h w d", b=B, n=N_fast
        )

    slow_real_n = (
        [int(slow_mask[i].sum().item()) for i in range(B)]
        if slow_mask is not None else None
    )
    fast_real_n = (
        [int(fast_mask[i].sum().item()) for i in range(B)]
        if fast_mask is not None else None
    )

    visual_tokens = model._compress(
        slow_tokens, fast_tokens, questions,
        slow_real_n=slow_real_n, fast_real_n=fast_real_n,
    )
    return visual_tokens  # list of B tensors


# ═══════════════════════════════════════════════════════════════════════════
# LOG-PROBABILITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_batch_log_probs(
    model,
    prompt_ids:  torch.Tensor,        # (P,) — shared prompt for all G
    rollout_ids: list[torch.Tensor],  # G tensors of variable length
    vis_tokens:  torch.Tensor,        # (T_vis, llm_dim) — shared for all G
    prompt_len:  int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Compute log-probs for G rollouts in ONE batched forward pass.

    All rollouts share the same prompt and visual tokens (same sample).
    Returns a (G,) tensor of summed per-token log-probabilities over the
    generated positions.
    """
    G = len(rollout_ids)
    if G == 0:
        return torch.zeros(0, device=device)

    # Build full sequences: prompt + generated for each rollout
    gen_lens  = []
    full_seqs = []
    for g in range(G):
        full = torch.cat([prompt_ids, rollout_ids[g]]).to(device)
        full_seqs.append(full)
        gen_lens.append(len(rollout_ids[g]))

    # Pad to max length
    max_len = max(s.shape[0] for s in full_seqs)
    padded_ids = torch.zeros(
        (G, max_len), dtype=full_seqs[0].dtype, device=device
    )
    attn_mask = torch.zeros((G, max_len), dtype=torch.long, device=device)
    for g in range(G):
        L_g = full_seqs[g].shape[0]
        padded_ids[g, :L_g] = full_seqs[g]
        attn_mask[g, :L_g] = 1

    # Sanitize: the generated region must not contain video_token_id or
    # _build_inputs_embeds would count those positions as visual
    # placeholders (the mock model can generate random IDs that collide
    # with the mock video_token_id).
    vid_id = getattr(model, "video_token_id", None)
    if vid_id is not None:
        for g in range(G):
            gen_region = padded_ids[g, prompt_len : prompt_len + gen_lens[g]]
            collision = gen_region == vid_id
            if collision.any():
                padded_ids[g, prompt_len : prompt_len + gen_lens[g]][collision] = 0

    # Visual tokens: same for all G rollouts
    vis_list = [vis_tokens] * G

    # Build embeddings
    inputs_embeds = model._build_inputs_embeds(padded_ids, vis_list)

    # Forward pass
    if model.config.use_mock_encoder:
        logits = model.lm_head(inputs_embeds)  # (G, L, V)
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model._llm_inner(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                use_cache=False,
            )
            logits = model.lm_head(outputs.last_hidden_state)  # (G, L, V)

    # Extract per-sequence log-probs.
    # logits[g, t-1] predicts the token at position t, so logits at
    # positions [prompt_len-1 : -1] predict tokens [prompt_len:].
    sequence_lps = []
    for g in range(G):
        gen_len = gen_lens[g]
        if gen_len <= 0:
            sequence_lps.append(
                torch.tensor(0.0, device=device, dtype=torch.float32)
            )
            continue
        gen_logits = logits[g, prompt_len - 1 : prompt_len - 1 + gen_len, :]
        gen_tokens = padded_ids[g, prompt_len : prompt_len + gen_len]
        lps = F.log_softmax(gen_logits.float(), dim=-1)
        token_lps = lps.gather(1, gen_tokens.unsqueeze(1)).squeeze(1)
        sequence_lps.append(token_lps.sum())

    return torch.stack(sequence_lps)  # (G,)


# ═══════════════════════════════════════════════════════════════════════════
# GRPO TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_grpo(config: GRPOConfig):
    """Main GRPO training function."""
    _setup_logging(config)
    logger.info("=" * 70)
    logger.info("GRPO Training — %s", config.experiment_name)
    logger.info("=" * 70)
    logger.info("Config: %s", json.dumps(asdict(config), indent=2, default=str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    # TF32 on Ampere+ for faster matmul/cuDNN ops on fp32 paths
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── Build model (reusing SFT infrastructure: LoRA + compressor) ────────
    from train_sft import build_model, build_optimizer, build_scheduler
    from surgqvlm.utils import (
        PromptBuilder, build_dataset_weights, save_checkpoint,
        aggregate_metrics, evaluate_sample,
    )
    from surgqvlm.data import SurgQDataset, collate_fn

    model, tokenizer, video_token_id = build_model(config)
    model = model.to(device)

    # The mock tokenizer from build_model lacks decode() with the signature
    # GRPO needs — patch it for rollout decoding.
    if not hasattr(tokenizer, "decode"):
        tokenizer.decode = lambda ids, **kw: (
            f"mock_text_{len(ids) if hasattr(ids, '__len__') else 0}_tokens"
        )

    # ── Warm-start from the SFT checkpoint ─────────────────────────────────
    if config.warm_start_from and not config.mock:
        ckpt_path = Path(config.warm_start_from)
        ckpt_file = ckpt_path / "best_checkpoint.pt"
        if not ckpt_file.exists():
            ckpt_file = ckpt_path / "checkpoint.pt"
        if ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(
                ckpt["trainable_state"], strict=False
            )
            logger.info(
                "Warm-start from %s: loaded %d keys (missing=%d, unexpected=%d)",
                ckpt_file, len(ckpt["trainable_state"]), len(missing), len(unexpected),
            )
        else:
            raise FileNotFoundError(
                f"Warm-start checkpoint not found under {ckpt_path}. "
                f"Run train_sft.py first (expected best_checkpoint.pt or "
                f"checkpoint.pt)."
            )
    elif config.mock:
        logger.info("Mock mode — skipping warm-start.")

    trainable    = model.count_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model ready | trainable: %dM / %dM total (%.1f%%)",
        trainable // 1_000_000, total_params // 1_000_000,
        100 * trainable / total_params,
    )

    # ── Capture the reference LoRA state (frozen SFT policy) ───────────────
    ref_lora_state = capture_ref_state(model)
    logger.info(
        "Reference LoRA state captured: %d keys, %.1f MB on CPU.",
        len(ref_lora_state),
        sum(v.numel() * v.element_size() for v in ref_lora_state.values()) / 1e6,
    )

    # ── Prompt builder ─────────────────────────────────────────────────────
    prompt_builder = PromptBuilder(tokenizer, video_token_id, config.max_seq_len)

    # ── Dataset & dataloader ───────────────────────────────────────────────
    train_dataset = SurgQDataset(
        split_file       = str(config.train_split),
        use_mock_encoder = config.mock,
    )
    # Sub-sample for pilot runs (data_fraction < 1.0)
    if config.data_fraction < 1.0:
        n_total = len(train_dataset)
        n_use   = max(1, int(n_total * config.data_fraction))
        indices = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(config.seed)
        )[:n_use]
        train_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())
        logger.info("Using %.0f%% of training data: %d / %d samples.",
                    config.data_fraction * 100, n_use, n_total)

    _use_workers = config.num_workers > 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size         = config.batch_size * config.oversample_factor,
        shuffle            = True,
        num_workers        = config.num_workers,
        collate_fn         = collate_fn,
        pin_memory         = True,
        drop_last          = False,
        persistent_workers = _use_workers,
        prefetch_factor    = 2 if _use_workers else None,
    )

    val_dataset = SurgQDataset(
        split_file       = str(config.val_split),
        use_mock_encoder = config.mock,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size         = config.val_batch_size,
        shuffle            = False,
        num_workers        = config.num_workers,
        collate_fn         = collate_fn,
        pin_memory         = True,
        persistent_workers = _use_workers,
        prefetch_factor    = 2 if _use_workers else None,
    )

    # ── Q-type weights ─────────────────────────────────────────────────────
    dataset_freq = None
    if config.loss_weight_mode == "inverse_freq":
        dataset_freq = build_dataset_weights(
            str(config.train_split), cap=config.loss_weight_cap
        )
        logger.info("Q-type weights: %s",
                    {k: f"{v:.2f}" for k, v in dataset_freq.items()})

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = build_optimizer(model, config)

    total_samples     = len(train_dataset)
    batches_per_epoch = math.ceil(
        total_samples / (config.batch_size * config.oversample_factor)
    )
    if config.max_batches > 0:
        # Ensure at least grad_accum_steps batches so ≥1 optimizer step occurs
        effective_max = max(config.max_batches, config.grad_accum_steps)
        if effective_max != config.max_batches:
            logger.info(
                "max_batches=%d < grad_accum_steps=%d — raised to %d.",
                config.max_batches, config.grad_accum_steps, effective_max,
            )
            config.max_batches = effective_max
        batches_per_epoch = min(batches_per_epoch, config.max_batches)
    steps_per_epoch = math.ceil(batches_per_epoch / config.grad_accum_steps)
    total_steps     = steps_per_epoch * config.num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)

    logger.info(
        "Training plan: %d samples, %d batches/epoch, %d opt steps/epoch, "
        "%d total steps, effective_batch=%d",
        total_samples, batches_per_epoch, steps_per_epoch,
        total_steps, config.effective_batch_size,
    )

    metrics_log_path = config.checkpoint_dir / "metrics_log.jsonl"
    samples_log_path = config.checkpoint_dir / "generated_samples.jsonl"

    # ── Training loop ──────────────────────────────────────────────────────
    global_step   = 0
    best_val_kacc = 0.0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_rewards = []
        epoch_r1s     = []
        epoch_kls     = []
        epoch_skips   = 0
        epoch_total   = 0
        epoch_start   = time.time()

        # Accumulators for metrics across the grad_accum window
        accum_pg_loss     = 0.0
        accum_kl          = 0.0
        accum_rewards_sum = 0.0
        accum_r1_sum      = 0.0
        accum_n_useful    = 0
        accum_per_qtype_rewards = defaultdict(list)

        for batch_idx, raw_batch in enumerate(train_loader):
            if config.max_batches > 0 and batch_idx >= config.max_batches:
                break

            batch_start = time.time()
            B_raw = len(raw_batch["question"])

            # ── 1. Encode video ONCE ──────────────────────────────────────
            slow_frames = raw_batch["slow_frames"].to(device)
            fast_frames = raw_batch["fast_frames"].to(device)
            questions   = raw_batch["question"]
            answers     = raw_batch["answer"]
            metas       = raw_batch["meta"]

            visual_tokens = encode_visual_once(
                model, slow_frames, fast_frames, questions,
                slow_mask=raw_batch.get("slow_mask"),
                fast_mask=raw_batch.get("fast_mask"),
            )

            # ── 2. Actual n_vis per sample from the compression output ────
            n_vis_list = [visual_tokens[i].shape[0] for i in range(B_raw)]

            # ── 3. Generate G rollouts per sample ─────────────────────────
            # Gradient checkpointing must be disabled during generation so
            # the KV cache is used. Force-disable at all module levels
            # because PEFT wrapping prevents propagation through
            # gradient_checkpointing_disable().
            _gc_on = config.gradient_checkpointing and not config.mock
            if _gc_on:
                for _m in model.modules():
                    if hasattr(_m, "gradient_checkpointing"):
                        _m.gradient_checkpointing = False

            all_sample_data = []
            G = config.num_rollouts

            # Build prompts for all samples, then generate in ONE batched call
            all_prompt_data = []
            prompt_lens = []
            for i in range(B_raw):
                pdata = prompt_builder.build_prompt_only(
                    question=questions[i],
                    n_visual_tokens=n_vis_list[i],
                )
                all_prompt_data.append(pdata)
                prompt_lens.append(pdata["prompt_len"])

            # Left-pad prompts to max length for batched generation
            max_plen = max(prompt_lens)
            pad_id   = getattr(tokenizer, "pad_token_id", 0) or 0
            total_seqs = B_raw * G
            batched_ids  = torch.full(
                (total_seqs, max_plen), pad_id, dtype=torch.long, device=device
            )
            batched_mask = torch.zeros(
                (total_seqs, max_plen), dtype=torch.long, device=device
            )
            all_vis = []

            for i in range(B_raw):
                p_ids = torch.tensor(
                    all_prompt_data[i]["input_ids"], dtype=torch.long
                )
                p_len = prompt_lens[i]
                offset = max_plen - p_len  # left-pad offset
                for g in range(G):
                    idx = i * G + g
                    batched_ids[idx, offset:] = p_ids
                    batched_mask[idx, offset:] = 1
                    all_vis.append(visual_tokens[i])

            # Single batched generation for all B_raw × G sequences
            with torch.no_grad():
                all_generated = model.generate_from_visual(
                    visual_tokens  = all_vis,
                    input_ids      = batched_ids,
                    attention_mask = batched_mask,
                    max_new_tokens = config.max_new_tokens,
                    do_sample      = True,
                    temperature    = config.temperature,
                    top_p          = config.top_p,
                )  # (B_raw*G, T_gen)

            # ── 4. Decode rollouts + compute rewards per sample ───────────
            eos_id = getattr(tokenizer, "eos_token_id", None)

            for i in range(B_raw):
                qtype        = metas[i].get("question_type", "Q01")
                binary_label = metas[i].get("binary_label")
                prompt_len   = prompt_lens[i]
                # Original (non-padded) prompt for log-prob computation
                prompt_ids_i = torch.tensor(
                    all_prompt_data[i]["input_ids"], dtype=torch.long,
                    device=device,
                )

                rollout_ids   = []
                rollout_texts = []
                for g in range(G):
                    gen_ids = all_generated[i * G + g]
                    if eos_id is not None:
                        eos_pos = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
                        if len(eos_pos) > 0:
                            gen_ids = gen_ids[: eos_pos[0] + 1]
                    rollout_ids.append(gen_ids)
                    rollout_texts.append(
                        tokenizer.decode(gen_ids, skip_special_tokens=True)
                    )

                rewards_data = [
                    compute_reward(
                        prediction    = rollout_texts[g],
                        reference     = answers[i],
                        question_type = qtype,
                        binary_label  = binary_label,
                        config        = config,
                        gen_len       = len(rollout_ids[g]),
                    )
                    for g in range(G)
                ]
                rewards = torch.tensor(
                    [rd["total"] for rd in rewards_data], dtype=torch.float32
                )

                all_sample_data.append({
                    "idx":           i,
                    "qtype":         qtype,
                    "question":      questions[i],
                    "answer":        answers[i],
                    "vis_tokens":    visual_tokens[i],
                    "prompt_ids":    prompt_ids_i,
                    "prompt_len":    prompt_len,
                    "rollout_ids":   rollout_ids,
                    "rollout_texts": rollout_texts,
                    "rewards":       rewards,
                    "rewards_data":  rewards_data,
                })

            # ── 5. Dynamic sampling: keep mixed-reward prompts only ───────
            # A prompt whose G rollouts all score identically yields zero
            # advantage and therefore zero gradient — skip it.
            useful_samples = [
                sd for sd in all_sample_data if sd["rewards"].std() > 1e-8
            ]

            epoch_total += B_raw
            epoch_skips += B_raw - len(useful_samples)

            if not useful_samples:
                if _gc_on:
                    for _m in model.modules():
                        if hasattr(_m, "gradient_checkpointing"):
                            _m.gradient_checkpointing = True
                continue

            # Keep up to batch_size useful samples
            useful_samples = useful_samples[: config.batch_size]

            # ── 6. Advantages + policy update ─────────────────────────────

            # Group-relative advantages (GRPO): normalise each prompt's
            # rewards by their own mean and std.
            for sd in useful_samples:
                rewards = sd["rewards"].to(device)
                sd["advantages"] = (rewards - rewards.mean()) / (
                    rewards.std() + 1e-8
                )

            # ── 6a. Reference log-probs (only if KL is active) ────────────
            _use_kl = config.kl_coeff > 0
            if _use_kl:
                policy_state = swap_to_ref(model, ref_lora_state, device)
                with torch.no_grad():
                    for sd in useful_samples:
                        sd["ref_log_probs"] = compute_batch_log_probs(
                            model, sd["prompt_ids"], sd["rollout_ids"],
                            sd["vis_tokens"], sd["prompt_len"], device,
                        )
                swap_to_policy(model, policy_state, device)

            # Re-enable gradient checkpointing for policy forward passes
            if _gc_on:
                for _m in model.modules():
                    if hasattr(_m, "gradient_checkpointing"):
                        _m.gradient_checkpointing = True

            # ── 6b. Policy log-probs + loss (batched per sample) ──────────
            for sd in useful_samples:
                G_s = len(sd["rollout_ids"])
                advantages = sd["advantages"]

                policy_log_probs = compute_batch_log_probs(
                    model, sd["prompt_ids"], sd["rollout_ids"],
                    sd["vis_tokens"], sd["prompt_len"], device,
                )

                # ── 6c. GRPO loss with DAPO decoupled clipping ────────────
                # In single-pass GRPO (one update per fresh rollout batch),
                # ratio ≈ 1.0 and clipping is inactive — this is correct
                # REINFORCE. The clipping structure is kept for
                # extensibility to multi-update variants.
                old_log_probs = policy_log_probs.detach()
                ratios = torch.exp(policy_log_probs - old_log_probs)

                surr1 = ratios * advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - config.clip_low,
                    1.0 + config.clip_high,
                ) * advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # KL penalty to the frozen SFT reference
                kl = torch.tensor(0.0)
                if _use_kl:
                    kl = (policy_log_probs - sd["ref_log_probs"]).mean()
                kl_loss = config.kl_coeff * kl

                # Q-type inverse-frequency weight — a correct counting
                # answer contributes as much to the gradient as a correct
                # binary answer despite being far rarer in the mixture.
                qtype_w = 1.0
                if dataset_freq and sd["qtype"] in dataset_freq:
                    qtype_w = dataset_freq[sd["qtype"]]

                sample_loss = (pg_loss + kl_loss) * qtype_w

                # Accumulate metrics across the grad_accum window
                accum_pg_loss     += pg_loss.detach().item()
                accum_kl          += kl.detach().item()
                accum_rewards_sum += sd["rewards"].mean().item()
                accum_r1_sum      += sum(rd["r1"] for rd in sd["rewards_data"]) / G_s
                accum_n_useful    += 1
                accum_per_qtype_rewards[sd["qtype"]].append(
                    sd["rewards"].mean().item()
                )

                # Accumulate gradient
                scaled_loss = sample_loss / (
                    config.batch_size * config.grad_accum_steps
                )
                scaled_loss.backward()

            # ── 7. Optimizer step ─────────────────────────────────────────
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Average metrics across the grad_accum window
                n = max(accum_n_useful, 1)
                mean_reward = accum_rewards_sum / n
                mean_r1     = accum_r1_sum / n
                mean_kl     = accum_kl / n
                mean_pg     = accum_pg_loss / n

                epoch_rewards.append(mean_reward)
                epoch_r1s.append(mean_r1)
                epoch_kls.append(mean_kl)

                # ── Logging ───────────────────────────────────────────────
                if global_step % config.log_every_steps == 0:
                    skip_rate = epoch_skips / max(epoch_total, 1)
                    elapsed   = time.time() - epoch_start
                    window = epoch_rewards[-20:]
                    rolling_reward = sum(window) / len(window)
                    window_r1 = epoch_r1s[-20:]
                    rolling_r1 = sum(window_r1) / len(window_r1)
                    logger.info(
                        "step %d | reward=%.3f (roll20=%.3f) | "
                        "r1=%.3f (roll20=%.3f) | kl=%.4f | "
                        "pg_loss=%.4f | skip=%.1f%% | lr=%.2e | "
                        "elapsed=%dm%ds",
                        global_step, mean_reward, rolling_reward,
                        mean_r1, rolling_r1, mean_kl,
                        mean_pg, skip_rate * 100,
                        optimizer.param_groups[0]["lr"],
                        int(elapsed) // 60, int(elapsed) % 60,
                    )

                    qtype_summary = {
                        qt: f"{sum(rs)/len(rs):.3f}"
                        for qt, rs in accum_per_qtype_rewards.items()
                        if rs
                    }

                    with open(metrics_log_path, "a") as f:
                        f.write(json.dumps({
                            "step":      global_step,
                            "epoch":     epoch,
                            "reward":    mean_reward,
                            "r1":        mean_r1,
                            "kl":        mean_kl,
                            "pg_loss":   mean_pg,
                            "skip_rate": skip_rate,
                            "lr":        optimizer.param_groups[0]["lr"],
                            "n_useful":  accum_n_useful,
                            "per_qtype_reward": qtype_summary,
                        }) + "\n")

                # Reset window accumulators
                accum_pg_loss     = 0.0
                accum_kl          = 0.0
                accum_rewards_sum = 0.0
                accum_r1_sum      = 0.0
                accum_n_useful    = 0
                accum_per_qtype_rewards.clear()

                # ── Log sample rollout texts ──────────────────────────────
                if global_step % config.log_samples_every == 0 and useful_samples:
                    sd = useful_samples[0]
                    for g in range(min(2, len(sd["rollout_texts"]))):
                        with open(samples_log_path, "a") as f:
                            f.write(json.dumps({
                                "step":       global_step,
                                "question":   sd["question"],
                                "reference":  sd["answer"],
                                "qtype":      sd["qtype"],
                                "prediction": sd["rollout_texts"][g],
                                "reward":     sd["rewards_data"][g],
                            }) + "\n")

                # ── Periodic checkpoint save ──────────────────────────────
                if (config.save_every_steps > 0
                        and global_step % config.save_every_steps == 0):
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch=epoch,
                        val_loss=0.0,
                        metrics={"step": global_step,
                                 "reward": mean_reward,
                                 "r1": mean_r1},
                        config=config,
                        path=config.checkpoint_dir,
                        is_best=False,
                    )
                    logger.info("Saved step-%d checkpoint to %s",
                                global_step, config.checkpoint_dir)

                # ── Mid-training generative eval ──────────────────────────
                if (config.val_every_steps > 0
                        and global_step % config.val_every_steps == 0):
                    mid_metrics = run_gen_eval(
                        model, val_loader, prompt_builder,
                        config, device, tokenizer,
                        n_samples=min(200, config.val_gen_samples),
                    )
                    mid_kacc = mid_metrics.get("overall_kacc", 0.0)
                    logger.info("Mid-val step %d | gen_kacc=%.4f",
                                global_step, mid_kacc)
                    with open(metrics_log_path, "a") as f:
                        f.write(json.dumps({
                            "step": global_step,
                            "mid_gen_kacc": mid_kacc,
                            **{f"mid_{k}": v for k, v in mid_metrics.items()
                               if isinstance(v, float)},
                        }) + "\n")
                    if mid_kacc > best_val_kacc:
                        best_val_kacc = mid_kacc
                        save_checkpoint(
                            model, optimizer, scheduler,
                            epoch=epoch,
                            val_loss=0.0,
                            metrics={"step": global_step, "gen_kacc": mid_kacc},
                            config=config,
                            path=config.checkpoint_dir,
                            is_best=True,
                        )
                        logger.info(
                            "New best gen_kacc=%.4f at step %d → best_checkpoint.pt",
                            mid_kacc, global_step,
                        )
                    model.train()

            batch_elapsed = time.time() - batch_start
            if batch_idx < 3 or (batch_idx + 1) % 50 == 0:
                logger.info(
                    "  batch %d/%d (%.1fs) | %d useful / %d sampled",
                    batch_idx + 1, batches_per_epoch, batch_elapsed,
                    len(useful_samples), B_raw,
                )

        # ── End of epoch ──────────────────────────────────────────────────
        epoch_elapsed = time.time() - epoch_start
        skip_rate     = epoch_skips / max(epoch_total, 1)
        logger.info(
            "Epoch %d done in %.1fh | mean_reward=%.3f | mean_r1=%.3f | "
            "mean_kl=%.4f | skip_rate=%.1f%% | steps=%d",
            epoch, epoch_elapsed / 3600,
            sum(epoch_rewards) / max(len(epoch_rewards), 1),
            sum(epoch_r1s) / max(len(epoch_r1s), 1),
            sum(epoch_kls) / max(len(epoch_kls), 1),
            skip_rate * 100,
            global_step,
        )

        # ── Epoch-end generative validation ───────────────────────────────
        if epoch % config.val_every_epochs == 0:
            logger.info("Running epoch-end generative validation "
                        "(%d samples)...", config.val_gen_samples)
            val_metrics = run_gen_eval(
                model, val_loader, prompt_builder, config, device, tokenizer,
                n_samples=config.val_gen_samples,
            )
            val_kacc = val_metrics.get("overall_kacc", 0.0)
            logger.info(
                "Val results: gen_kacc=%.4f | gen_primary=%.4f",
                val_kacc, val_metrics.get("overall_primary", 0.0),
            )

            is_best = val_kacc >= best_val_kacc
            if is_best:
                best_val_kacc = val_kacc
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch,
                val_loss=0.0,  # GRPO has no validation loss
                metrics=val_metrics,
                config=config,
                path=config.checkpoint_dir,
                is_best=False,  # avoid a duplicate write; hardlink below
            )
            if is_best:
                best_path = config.checkpoint_dir / "best_checkpoint.pt"
                ckpt_path = config.checkpoint_dir / "checkpoint.pt"
                if best_path.exists():
                    best_path.unlink()
                os.link(ckpt_path, best_path)
                logger.info("Hardlinked best → %s", best_path)

    logger.info("GRPO training complete. Best val kacc: %.4f", best_val_kacc)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION (generative only — reflects the true autoregressive output)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_gen_eval(model, val_loader, prompt_builder, config, device,
                 tokenizer, n_samples: int = 200) -> dict:
    """
    Greedy-decoding generation eval on n_samples from val_loader.
    Used for mid-training checks and as the epoch-end metric.
    """
    from surgqvlm.utils import evaluate_sample, aggregate_metrics

    model.eval()
    _gc_on = config.gradient_checkpointing and not config.mock
    if _gc_on:
        for _m in model.modules():
            if hasattr(_m, "gradient_checkpointing"):
                _m.gradient_checkpointing = False

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    gen_metrics, gen_qtypes, gen_count = [], [], 0

    for raw_batch in val_loader:
        if gen_count >= n_samples:
            break
        B  = len(raw_batch["question"])
        n_gen = min(B, n_samples - gen_count)

        slow2 = raw_batch["slow_frames"].to(device)
        fast2 = raw_batch["fast_frames"].to(device)
        q2, a2, m2 = raw_batch["question"], raw_batch["answer"], raw_batch["meta"]

        vis2   = encode_visual_once(model, slow2, fast2, q2,
                                    slow_mask=raw_batch.get("slow_mask"),
                                    fast_mask=raw_batch.get("fast_mask"))
        n_vis2 = [vis2[i].shape[0] for i in range(B)]

        pdata_list = [prompt_builder.build_prompt_only(q2[i], n_vis2[i])
                      for i in range(n_gen)]
        plens  = [pd["prompt_len"] for pd in pdata_list]
        max_pl = max(plens)

        batch_ids  = torch.full((n_gen, max_pl), pad_id,
                                dtype=torch.long, device=device)
        batch_mask = torch.zeros((n_gen, max_pl),
                                 dtype=torch.long, device=device)
        batch_vis  = []
        for i in range(n_gen):
            p_ids = torch.tensor(pdata_list[i]["input_ids"], dtype=torch.long)
            off   = max_pl - plens[i]
            batch_ids[i, off:]  = p_ids
            batch_mask[i, off:] = 1
            batch_vis.append(vis2[i])

        gen_out = model.generate_from_visual(
            visual_tokens=batch_vis, input_ids=batch_ids,
            attention_mask=batch_mask,
            max_new_tokens=config.max_new_tokens, do_sample=False,
        )
        for i in range(n_gen):
            ids = gen_out[i]
            if eos_id is not None:
                eos_pos = (ids == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos):
                    ids = ids[: eos_pos[0] + 1]
            pred = _strip_think(tokenizer.decode(ids, skip_special_tokens=True))
            gm   = evaluate_sample(pred, a2[i],
                                   m2[i].get("question_type", "Q01"),
                                   m2[i].get("binary_label"))
            gen_metrics.append(gm)
            gen_qtypes.append(m2[i].get("question_type", "Q01"))
            gen_count += 1

    if _gc_on:
        for _m in model.modules():
            if hasattr(_m, "gradient_checkpointing"):
                _m.gradient_checkpointing = True

    model.train()
    return aggregate_metrics(gen_metrics, gen_qtypes)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    from surgqvlm.utils import (
        load_yaml_config, add_config_cli_overrides, apply_cli_overrides,
    )

    parser = argparse.ArgumentParser(
        description="SurgQ-VLM Stage 2: GRPO post-training")
    parser.add_argument("--config", default="configs/grpo.yaml",
                        help="Path to the YAML training config.")
    parser.add_argument("--dry_run", action="store_true",
                        help="A few batches only — shape verification.")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock model (random weights, no download).")
    # Any config field can be overridden: --learning_rate 1e-7, etc.
    add_config_cli_overrides(parser, GRPOConfig,
                             skip=("mock", "description"))
    args = parser.parse_args()

    config = load_yaml_config(GRPOConfig, args.config)
    apply_cli_overrides(config, args)

    if args.dry_run:
        config.num_epochs = 1
        min_batches = max(config.grad_accum_steps + 1, 5)
        config.max_batches = (max(config.max_batches, min_batches)
                              if config.max_batches > 0 else min_batches)
        config.log_every_steps   = 1
        config.log_samples_every = 1
        config.val_every_steps   = 0
        config.save_every_steps  = 0
        config.val_gen_samples   = 8

    if args.mock:
        config.mock = True

    train_grpo(config)


# ═══════════════════════════════════════════════════════════════════════════
# REWARD UNIT TESTS (run: python train_grpo.py --test_rewards)
# ═══════════════════════════════════════════════════════════════════════════

def _test_rewards():
    """Quick sanity checks for the reward functions."""
    # R1: binary
    assert compute_r1("yes, the instrument is present", "yes", "Q06", True) == 1.0
    assert compute_r1("no, I don't see it", "yes", "Q06", True) == 0.0

    # R1: classification
    assert compute_r1(
        "<think>Step 1: ...</think>\ncalot triangle dissection",
        "calot triangle dissection", "Q01"
    ) == 1.0

    # R1: multi_label (continuous)
    r = compute_r1(
        "<think>Step 1: ...</think>\ngrasper, scissors",
        "grasper, scissors, hook", "Q05"
    )
    assert 0.5 < r < 1.0, f"Expected partial credit, got {r}"

    # R1: counting (continuous)
    r = compute_r1("<think>...</think>\n3", "4", "Q12")
    assert 0.5 < r < 1.0, f"Expected partial credit for off-by-1, got {r}"

    # R1: sentence_temporal (blended)
    r = compute_r1(
        "<think>...</think>\nthe grasper appears at second 14",
        "grasper appears at second 14 during calot triangle dissection", "Q08"
    )
    assert r > 0.5, f"Expected decent reward, got {r}"

    # R2: format
    assert compute_r2("<think>Step 1: A\nStep 2: B</think>\nanswer") == 1.0
    assert compute_r2("just an answer without thinking") == 0.0
    assert 0.3 < compute_r2("<think>some reasoning</think>\nanswer") < 0.7
    # R2: anti-hack penalty
    hacky = ("<think>I scan instrument_per_frame at frame index 12. "
             "The annotation records hook.</think>")
    assert compute_r2(hacky) == 0.0, \
        f"Expected hack-floored to 0, got {compute_r2(hacky)}"
    one_hit = "<think>I examine the clip carefully.</think>"
    # 0.66 (tags) - 0.3 (one hit: "I examine") = 0.36
    assert 0.3 < compute_r2(one_hit) < 0.4, \
        f"Expected ~0.36, got {compute_r2(one_hit)}"

    # R3: temporal
    assert compute_r3("Step 1: ...", "Q01") == 1.0  # non-temporal: full credit
    r = compute_r3(
        "<think>At second 14, the grasper appears. Before this, only the "
        "hook was visible.</think>",
        "Q08"
    )
    assert r == 1.0, f"Expected full temporal credit, got {r}"
    assert compute_r3(
        "<think>The phase is calot triangle dissection.</think>", "Q08"
    ) == 0.0

    # R_length: overlong penalty
    assert compute_r_length(200, max_len=512, cache=128) == 0.0
    assert compute_r_length(512, max_len=512, cache=128) == -1.0
    assert -0.6 < compute_r_length(448, max_len=512, cache=128) < -0.4

    print("All reward tests passed!")


if __name__ == "__main__":
    if "--test_rewards" in sys.argv:
        _test_rewards()
        sys.exit(0)
    main()
