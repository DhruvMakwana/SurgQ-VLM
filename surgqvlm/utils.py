"""
utils.py — Shared Utilities for SurgQ-VLM Training and Evaluation
==================================================================

Contents
--------
1. Constants        — system prompt, Q-type → answer type mapping,
                      keyword vocabularies
2. PromptBuilder    — builds tokenized (input_ids, attention_mask, labels)
                      per sample from raw question/answer strings
3. Loss utilities   — Q-type inverse-frequency weights, weighted loss
4. Evaluation       — all metrics dispatched by answer type:
                        binary_accuracy, keyword_accuracy, multilabel_f1,
                        counting_metrics, rouge_l, bleu4, temporal_accuracy
5. Checkpointing    — save/load trainable weights only (compact)
6. EarlyStopping    — patience-based stopping
7. Config loading   — YAML → dataclass with CLI overrides

Design notes
------------
- PromptBuilder handles variable visual token counts per sample (shorter
  clips produce fewer tokens than the default 671). Each sample gets
  exactly the right number of <|video_pad|> placeholders.
- Loss weighting is done in the training script, not inside the model.
- Metrics require no external packages. rouge_l falls back to a built-in
  LCS implementation; bleu4 to a built-in n-gram implementation. Install
  rouge-score / nltk for the (faster) library versions.
"""

import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── 1. Constants ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert surgical video analysis assistant specializing in "
    "laparoscopic procedures. You will be shown a short video clip from a "
    "laparoscopic cholecystectomy and asked a question about it. "
    "Answer accurately and concisely based on what you observe in the video."
)

# Maps each question type to its answer type (drives metric selection)
QTYPE_TO_ANSWER_TYPE: dict[str, str] = {
    "Q01": "classification",    # Surgical phase identification
    "Q02": "binary",            # Phase change
    "Q03": "sentence_temporal", # Phase transition timing
    "Q04": "sentence_temporal", # Phase sequence description
    "Q05": "multi_label",       # All instruments present
    "Q06": "binary",            # Instrument presence
    "Q07": "binary",            # Instrument handoff
    "Q08": "sentence_temporal", # Instrument active duration
    "Q09": "sentence_temporal", # Instrument at probe second
    "Q10": "classification",    # Primary triplet action
    "Q11": "multi_label",       # All triplet actions
    "Q12": "counting",          # Count of distinct actions
    "Q13": "binary",            # Triplet change detection
    "Q14": "classification",    # Target anatomy identification
    "Q15": "classification",    # Primary verb identification
    "Q16": "sentence_temporal", # Instrument at phase transition
    "Q17": "binary",            # Idle period detection
    "Q18": "sentence_temporal", # Action before/after transition
}

# Keyword vocabularies — used by multilabel_f1.
# Canonical natural-language form (spaces, matching the answer strings;
# see data_prep/labels_canonical.py).
PHASE_NAMES = [
    "preparation",
    "calot triangle dissection",
    "clipping cutting",
    "gallbladder dissection",
    "gallbladder packaging",
    "cleaning coagulation",
    "gallbladder retraction",
]
INSTRUMENT_NAMES = [
    "grasper", "bipolar", "hook", "scissors", "clipper",
    "irrigator", "specimen bag",
]
VERB_NAMES = [
    "grasp", "retract", "dissect", "coagulate", "clip",
    "cut", "aspirate", "irrigate", "pack",
]
TARGET_NAMES = [
    "gallbladder", "cystic plate", "cystic duct", "cystic artery",
    "cystic pedicle", "blood vessel", "fluid", "abdominal wall cavity",
    "liver", "adhesion", "omentum", "peritoneum", "gut", "specimen bag",
]


# ── 2. PromptBuilder ──────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Converts (question, answer, n_visual_tokens) triples into the tensor
    format expected by SurgQVLM.forward().

    Each sample gets a prompt structured as:
        [system turn] [user turn: video_pads + question] [assistant turn: answer]

    Labels mask everything before the answer with -100.

    Qwen2.5-VL chat format
    ----------------------
    <|im_start|>system\\n{system}\\n<|im_end|>\\n
    <|im_start|>user\\n{N × <|video_pad|>}\\n{question}\\n<|im_end|>\\n
    <|im_start|>assistant\\n{answer}<|im_end|>

    Parameters
    ----------
    tokenizer      : Qwen AutoTokenizer (or a mock tokenizer in test mode)
    video_token_id : int — token ID for <|video_pad|>
    max_seq_len    : int — hard cap on prompt+answer tokens (truncates the
                     answer if exceeded; should not happen with surgical QA)
    """

    # Qwen2.5-VL chat markers — these must NOT be changed
    _IM_START   = "<|im_start|>"
    _IM_END     = "<|im_end|>"
    _IM_END_NL  = "<|im_end|>\n"

    def __init__(
        self,
        tokenizer,
        video_token_id: int,
        max_seq_len:    int = 2048,
        system_prompt:  str = SYSTEM_PROMPT,
    ):
        self.tokenizer      = tokenizer
        self.video_token_id = video_token_id
        self.max_seq_len    = max_seq_len
        self.system_prompt  = system_prompt

        # Pre-tokenize the fixed parts of the template once
        self._prefix_ids   = self._tok(
            f"{self._IM_START}system\n{system_prompt}\n{self._IM_END_NL}"
            f"{self._IM_START}user\n"
        )
        self._mid_ids      = self._tok(f"\n{self._IM_END_NL}")
        self._asst_ids     = self._tok(f"{self._IM_START}assistant\n")
        self._end_ids      = self._tok(self._IM_END)

    def _tok(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def build_sample(
        self,
        question:        str,
        answer:          str,
        n_visual_tokens: int,
    ) -> dict[str, list[int]]:
        """
        Build one training sample.

        Parameters
        ----------
        question        : raw question string
        answer          : raw ground-truth answer string
        n_visual_tokens : number of <|video_pad|> placeholders to insert.
                          Must match what SurgQVLM._compress() will produce
                          for this clip's frame count.

        Returns
        -------
        dict with keys:
            input_ids      : list[int]
            attention_mask : list[int]  (all 1s before padding)
            labels         : list[int]  (-100 for prompt, target IDs for loss)
            prompt_len     : int        (number of prompt tokens)
        """
        question_ids = self._tok(question)
        answer_ids   = self._tok(answer)
        end_ids      = self._end_ids
        video_ids    = [self.video_token_id] * n_visual_tokens

        # Full sequence: prefix + video + mid + question + asst_prefix + answer + end
        prompt_ids = (
            self._prefix_ids
            + video_ids
            + self._mid_ids
            + question_ids
            + self._asst_ids
        )
        full_ids = prompt_ids + answer_ids + end_ids

        # Truncate to max_seq_len if necessary (never truncate the prompt)
        if len(full_ids) > self.max_seq_len:
            allowed_target = self.max_seq_len - len(prompt_ids) - len(end_ids)
            if allowed_target < 1:
                logger.warning(
                    "Prompt alone (%d tokens) exceeds max_seq_len (%d). "
                    "Consider reducing n_visual_tokens or raising max_seq_len.",
                    len(prompt_ids), self.max_seq_len,
                )
                allowed_target = 1
            answer_ids = answer_ids[:allowed_target]
            full_ids   = prompt_ids + answer_ids + end_ids

        # Labels: -100 for prompt; answer token IDs for loss
        labels = [-100] * len(prompt_ids) + answer_ids + end_ids

        assert len(full_ids) == len(labels), (
            f"Length mismatch: input_ids={len(full_ids)}, labels={len(labels)}"
        )

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
            "prompt_len":     len(prompt_ids),
        }

    def build_prompt_only(
        self,
        question:        str,
        n_visual_tokens: int,
    ) -> dict[str, list[int]]:
        """
        Build the prompt *without* an answer — for generation (evaluation
        and GRPO rollouts).

        Returns the token sequence up to and including the assistant header;
        the model generates from the end of this sequence.
        """
        question_ids = self._tok(question)
        video_ids    = [self.video_token_id] * n_visual_tokens

        prompt_ids = (
            self._prefix_ids
            + video_ids
            + self._mid_ids
            + question_ids
            + self._asst_ids      # ends at "<|im_start|>assistant\n"
        )

        return {
            "input_ids":      prompt_ids,
            "attention_mask": [1] * len(prompt_ids),
            "prompt_len":     len(prompt_ids),
        }

    def collate_batch(
        self,
        samples: list[dict],
        pad_token_id: int,
    ) -> dict[str, torch.Tensor]:
        """
        Pad a list of build_sample outputs to the same length.

        Pads on the RIGHT. Attention mask is 0 for pad positions.
        Labels are -100 for pad positions (ignored in loss).

        Returns
        -------
        dict with tensors of shape (B, L_max): input_ids, attention_mask, labels
        """
        max_len = max(len(s["input_ids"]) for s in samples)

        input_ids_batch = []
        attn_mask_batch = []
        labels_batch    = []

        for s in samples:
            pad_len = max_len - len(s["input_ids"])
            input_ids_batch.append(s["input_ids"]      + [pad_token_id] * pad_len)
            attn_mask_batch.append(s["attention_mask"] + [0]            * pad_len)
            labels_batch.append(   s["labels"]         + [-100]         * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask_batch, dtype=torch.long),
            "labels":         torch.tensor(labels_batch,    dtype=torch.long),
        }


# ── 3. Loss utilities ─────────────────────────────────────────────────────────

def build_dataset_weights(
    split_file: str,
    cap:        float = 5.0,
) -> dict[str, float]:
    """
    Compute dataset-level inverse-frequency weights for each Q-type.

    Reads the split file, counts Q-type occurrences across all QA pairs,
    and returns {Q-type: weight} normalised so the mean weight = 1.0.
    Rare types are capped at cap × median weight.

    Call this ONCE at the start of training, then pass the returned dict
    to compute_qtype_weights() as `dataset_freq`.

    Parameters
    ----------
    split_file : path to train.txt
    cap        : max multiplier for rare types

    Returns
    -------
    dict mapping Q-type string → float weight
    """
    counts: dict[str, int] = defaultdict(int)

    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            qa_path = parts[1] if len(parts) == 2 else parts[0]
            try:
                with open(qa_path) as qf:
                    pairs = json.load(qf)
                for p in pairs:
                    counts[p.get("question_type", "Q01")] += 1
            except Exception:
                continue

    if not counts:
        logger.warning("build_dataset_weights: no Q-types found, returning uniform.")
        return {qt: 1.0 for qt in QTYPE_TO_ANSWER_TYPE}

    raw = {qt: 1.0 / cnt for qt, cnt in counts.items()}

    # Cap at cap × median
    values  = sorted(raw.values())
    median  = values[len(values) // 2]
    ceiling = cap * median
    capped  = {qt: min(w, ceiling) for qt, w in raw.items()}

    # Normalise so mean = 1.0
    mean_w = sum(capped.values()) / len(capped)
    return {qt: w / mean_w for qt, w in capped.items()}


def compute_qtype_weights(
    question_types: list[str],
    mode:           str = "inverse_freq",
    dataset_freq:   Optional[dict] = None,
) -> torch.Tensor:
    """
    Compute per-sample loss weights based on question type frequency.

    Parameters
    ----------
    question_types : Q-type string per sample in the batch
    mode           : "none"         — all weights = 1.0
                     "inverse_freq" — weight ∝ 1/freq(qtype) in the dataset.
                                      Requires the dataset_freq dict from
                                      build_dataset_weights().
    dataset_freq   : dict from build_dataset_weights(). Required for
                     inverse_freq mode (dataset-level frequencies are stable
                     regardless of batch size).

    Returns
    -------
    weights : (B,) float32 tensor, mean ≈ 1.0 (overall loss scale unchanged)
    """
    B = len(question_types)

    if mode == "none":
        return torch.ones(B, dtype=torch.float32)

    if mode == "inverse_freq":
        if dataset_freq is None:
            raise ValueError(
                "inverse_freq weighting requires dataset_freq from "
                "build_dataset_weights()."
            )
        raw_weights = [dataset_freq.get(qt, 1.0) for qt in question_types]
        weights = torch.tensor(raw_weights, dtype=torch.float32)
        weights = weights / weights.mean()
        return weights

    raise ValueError(
        f"Unknown loss_weight_mode '{mode}'. Choose from: none, inverse_freq"
    )


def compute_weighted_loss(
    logits:         torch.Tensor,
    labels:         torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Causal LM cross-entropy loss with optional per-sample weighting.

    Parameters
    ----------
    logits         : (B, L, vocab_size) — raw logits from lm_head
    labels         : (B, L) — -100 for masked positions
    sample_weights : (B,) optional float weights. None → plain mean.

    Returns
    -------
    loss : scalar tensor
    """
    B, L, V = logits.shape

    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()           # (B, L-1, V)
    shift_labels = labels[:, 1:].contiguous()               # (B, L-1)

    if sample_weights is None:
        # Fast path: standard mean reduction
        return F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # Per-sample weighted loss
    per_sample_losses = []
    for i in range(B):
        valid = shift_labels[i] != -100
        if valid.sum() == 0:
            per_sample_losses.append(torch.tensor(0.0, device=logits.device))
            continue
        tok_loss = F.cross_entropy(
            shift_logits[i],          # (L-1, V)
            shift_labels[i],          # (L-1)
            ignore_index=-100,
            reduction="none",
        )
        loss_i = tok_loss[valid].sum() / valid.float().sum().clamp(min=1.0)
        per_sample_losses.append(loss_i)

    losses  = torch.stack(per_sample_losses)   # (B,)
    weights = sample_weights.to(logits.device)
    return (losses * weights).mean()


# ── 4. Evaluation metrics ─────────────────────────────────────────────────────

def _lcs_length(x: list, y: list) -> int:
    """Longest common subsequence length (word-level)."""
    m, n = len(x), len(y)
    # Memory-efficient DP using two rows
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = (prev[j - 1] + 1) if x[i-1] == y[j-1] else max(prev[j], curr[j-1])
        prev = curr
    return prev[n]


def rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L F1 score (word-level LCS).

    Falls back to the rouge_score library if installed (faster + stemming).
    Returns a value in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    except ImportError:
        pass

    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs_len   = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall    = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu4(prediction: str, reference: str) -> float:
    """
    Sentence-level BLEU-4 score with add-1 smoothing.

    Falls back to nltk.translate.bleu_score if installed; otherwise uses a
    built-in implementation. Returns a value in [0, 1].
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens  = reference.lower().split()
        pred_tokens = prediction.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        smooth = SmoothingFunction().method1
        return float(sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth,
        ))
    except ImportError:
        pass

    # Built-in implementation
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    def _ngram_counts(tokens: list[str], n: int) -> dict:
        counts: dict[tuple, int] = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    def _clipped_precision(pred: list[str], ref: list[str], n: int) -> float:
        """Modified n-gram precision with add-1 smoothing for n > 1."""
        pred_counts = _ngram_counts(pred, n)
        ref_counts  = _ngram_counts(ref,  n)

        numerator = sum(
            min(count, ref_counts.get(gram, 0))
            for gram, count in pred_counts.items()
        )
        denominator = sum(pred_counts.values())

        if denominator == 0:
            return 0.0
        if n > 1:
            numerator   += 1
            denominator += 1

        return numerator / denominator

    # Brevity penalty
    bp = min(1.0, math.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1)))

    # Geometric mean of 4 precisions; zero if any precision is 0
    log_sum = 0.0
    for n in range(1, 5):
        p = _clipped_precision(pred_tokens, ref_tokens, n)
        if p == 0.0:
            return 0.0
        log_sum += 0.25 * math.log(p)

    return bp * math.exp(log_sum)


def _extract_seconds(text: str) -> list[int]:
    """
    Extract all numeric second references from a string.

    Handles patterns like:
        "at second 14", "second 14", "14 seconds", "seconds 4 to 25",
        "from second 4 to second 25"
    """
    text = text.lower()
    patterns = [
        r"second[s]?\s+(\d+)",          # "second 14", "seconds 14"
        r"at\s+second\s+(\d+)",         # "at second 14"
        r"from\s+second\s+(\d+)",       # "from second 4"
        r"(\d+)\s+second[s]?(?:\s|$)",  # "14 seconds", "14 second "
        r"(\d+)\s*(?:s\b|sec\b)",       # "14s", "14 sec"
    ]
    seconds = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            val = int(m.group(1))
            if val not in seconds:
                seconds.append(val)
    return sorted(seconds)


def temporal_accuracy(
    prediction: str,
    reference:  str,
    tolerance:  int = 2,
) -> float:
    """
    Temporal Accuracy (±tolerance seconds).

    For each second referenced in the ground-truth answer, the prediction
    scores 1 if any of its second references falls within ±tolerance.

    Returns
    -------
    score in [0, 1]. Returns 1.0 if the ground truth contains no second
    references (temporal grounding is not required for that answer).
    """
    gt_seconds   = _extract_seconds(reference)
    pred_seconds = _extract_seconds(prediction)

    if not gt_seconds:
        return 1.0  # No temporal claim in GT — any answer acceptable
    if not pred_seconds:
        return 0.0  # GT has a temporal claim, prediction has none

    correct = sum(
        any(abs(g - p) <= tolerance for p in pred_seconds)
        for g in gt_seconds
    )
    return correct / len(gt_seconds)


def binary_accuracy(
    prediction:   str,
    binary_label: bool,
) -> float:
    """
    Check if the predicted answer matches the expected yes/no label.

    Searches the prediction for "yes"/"no" keywords. If both are present,
    uses the one that appears first. Neither present counts as wrong.
    """
    pred_lower = prediction.lower()
    has_yes = bool(re.search(r"\byes\b", pred_lower))
    has_no  = bool(re.search(r"\bno\b",  pred_lower))

    if has_yes and not has_no:
        predicted_positive = True
    elif has_no and not has_yes:
        predicted_positive = False
    elif has_yes and has_no:
        yes_pos = pred_lower.index("yes")
        no_pos  = pred_lower.index("no")
        predicted_positive = (yes_pos < no_pos)
    else:
        return 0.0

    return 1.0 if (predicted_positive == binary_label) else 0.0


def keyword_accuracy(
    prediction: str,
    reference:  str,
) -> float:
    """
    Keyword Accuracy (K-Acc) — adopted from SurgViVQA.

    The fraction of content words from the ground-truth answer that appear
    in the prediction. Common stop words (and domain-generic words such as
    "clip", "phase", "instrument") are ignored to avoid false positives.
    Precision-soft by design: extra output is not penalised.
    """
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "and", "or", "but", "not", "no", "yes", "this", "that", "it", "its",
        "there", "here", "from", "into", "out", "up", "down", "during", "clip",
        "video", "surgical", "procedure", "moment", "time", "second", "seconds",
        "instrument", "tool", "phase", "action",
    }
    pred_words = set(re.sub(r"[^\w\s]", "", prediction.lower()).split())
    ref_words  = set(re.sub(r"[^\w\s]", "", reference.lower()).split())
    content_ref = ref_words - STOP_WORDS

    if not content_ref:
        return 1.0  # GT has only stop words — trivially matched
    matches = pred_words & content_ref
    return len(matches) / len(content_ref)


def multilabel_f1(
    prediction: str,
    reference:  str,
    vocabulary: Optional[list[str]] = None,
    question_type: Optional[str] = None,
) -> float:
    """
    Micro-F1 for multi-label answers (Q05 instrument lists, Q11 action lists).

    Extracts known-vocabulary keywords from both strings and computes F1.
    Falls back to word-overlap F1 if the reference contains no vocabulary
    terms.

    Parameters
    ----------
    vocabulary    : explicit keyword list. Overrides question_type routing.
    question_type : routes to the right vocabulary:
                    Q05 → instruments, Q11 → verbs+targets+instruments.
    """
    if vocabulary is None:
        if question_type == "Q05":
            vocabulary = INSTRUMENT_NAMES
        elif question_type == "Q11":
            # Triplet answer "instrument verb target" — union vocabulary so
            # any of the three component types can match.
            vocabulary = INSTRUMENT_NAMES + VERB_NAMES + TARGET_NAMES
        else:
            vocabulary = INSTRUMENT_NAMES

    pred_lower = prediction.lower()
    ref_lower  = reference.lower()

    # Word-boundary matching prevents "grasp" firing inside "grasper" and
    # "clip" firing inside "clipper" (substring collision in the Q11 vocab).
    # "specimen bag" is multi-word — plain substring check for it.
    def _match(text: str, kw: str) -> bool:
        if " " in kw:
            return kw in text
        return bool(re.search(r"\b" + re.escape(kw) + r"\b", text))

    pred_labels = {kw for kw in vocabulary if _match(pred_lower, kw)}
    ref_labels  = {kw for kw in vocabulary if _match(ref_lower,  kw)}

    if not ref_labels:
        # Fall back to word overlap F1
        pred_set = set(pred_lower.split())
        ref_set  = set(ref_lower.split())
        if not ref_set:
            return 1.0
        tp = len(pred_set & ref_set)
        p  = tp / len(pred_set) if pred_set else 0.0
        r  = tp / len(ref_set)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    tp = len(pred_labels & ref_labels)
    fp = len(pred_labels - ref_labels)
    fn = len(ref_labels - pred_labels)

    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def counting_metrics(
    prediction: str,
    reference:  str,
) -> dict[str, float]:
    """
    Metrics for counting answers (Q12 — number of distinct actions).

    Returns
    -------
    dict with keys:
        exact_match : 1.0 if the predicted count equals the GT count
        mae         : absolute error |pred - gt|
        normalized  : max(0, 1 - |pred-gt| / gt) — soft accuracy, used as
                      the GRPO counting reward
    """
    def _extract_int(text: str) -> Optional[int]:
        m = re.search(r"\b(\d+)\b", text)
        return int(m.group(1)) if m else None

    gt_count   = _extract_int(reference)
    pred_count = _extract_int(prediction)

    if gt_count is None:
        return {"exact_match": 0.0, "mae": 0.0, "normalized": 0.0}
    if pred_count is None:
        return {
            "exact_match": 0.0,
            "mae":         float(gt_count),
            "normalized":  0.0,
        }

    mae         = abs(pred_count - gt_count)
    exact_match = 1.0 if pred_count == gt_count else 0.0
    normalized  = max(0.0, 1.0 - mae / max(gt_count, 1))

    return {"exact_match": exact_match, "mae": float(mae), "normalized": normalized}


def evaluate_sample(
    prediction:    str,
    reference:     str,
    question_type: str,
    binary_label:  Optional[bool] = None,
) -> dict[str, float]:
    """
    Compute all applicable metrics for one prediction.

    Dispatches by QTYPE_TO_ANSWER_TYPE. Returns {metric_name: value}.
    Always includes 'kacc' (keyword accuracy) for comparability.

    Parameters
    ----------
    prediction    : model-generated text
    reference     : ground-truth answer text
    question_type : e.g. "Q01"
    binary_label  : required for binary answer types (True=yes, False=no)
    """
    answer_type = QTYPE_TO_ANSWER_TYPE.get(question_type, "classification")
    results: dict[str, float] = {}

    if answer_type == "binary":
        if binary_label is not None:
            results["binary_acc"] = binary_accuracy(prediction, binary_label)
        results["kacc"] = keyword_accuracy(prediction, reference)

    elif answer_type == "classification":
        results["kacc"]    = keyword_accuracy(prediction, reference)
        results["rouge_l"] = rouge_l(prediction, reference)
        results["bleu4"]   = bleu4(prediction, reference)

    elif answer_type == "multi_label":
        results["multilabel_f1"] = multilabel_f1(
            prediction, reference, question_type=question_type
        )
        results["kacc"] = keyword_accuracy(prediction, reference)

    elif answer_type == "counting":
        cnt = counting_metrics(prediction, reference)
        results.update({f"count_{k}": v for k, v in cnt.items()})
        results["kacc"] = keyword_accuracy(prediction, reference)

    elif answer_type == "sentence_temporal":
        results["rouge_l"]       = rouge_l(prediction, reference)
        results["bleu4"]         = bleu4(prediction, reference)
        results["temporal_acc"]  = temporal_accuracy(prediction, reference, tolerance=2)
        results["temporal_acc1"] = temporal_accuracy(prediction, reference, tolerance=1)
        results["temporal_acc3"] = temporal_accuracy(prediction, reference, tolerance=3)
        results["kacc"]          = keyword_accuracy(prediction, reference)

    return results


def aggregate_metrics(
    per_sample: list[dict],
    question_types: list[str],
) -> dict:
    """
    Aggregate per-sample metric dicts into per-Q-type and overall summaries.

    Parameters
    ----------
    per_sample     : list of dicts from evaluate_sample(), one per prediction
    question_types : list of Q-type strings, same length as per_sample

    Returns
    -------
    dict with:
        overall_kacc    : mean K-Acc across all samples
        overall_primary : N-weighted mean of the structure-matched primary
                          metric (binary accuracy for binary Q-types,
                          micro-F1 for multi-label, exact-match for
                          counting, K-Acc otherwise)
        by_qtype        : {Q01: {metric: value}, ...}
        by_answer_type  : {binary: {...}, classification: {...}, ...}
        n_samples       : total samples
    """
    by_qtype:       dict[str, list] = defaultdict(list)
    by_answer_type: dict[str, list] = defaultdict(list)

    for metrics, qt in zip(per_sample, question_types):
        by_qtype[qt].append(metrics)
        at = QTYPE_TO_ANSWER_TYPE.get(qt, "classification")
        by_answer_type[at].append(metrics)

    def _mean_dict(dicts: list[dict]) -> dict[str, float]:
        if not dicts:
            return {}
        all_keys = set(k for d in dicts for k in d)
        return {k: sum(d.get(k, 0.0) for d in dicts) / len(dicts)
                for k in sorted(all_keys)}

    qtype_agg = {qt: _mean_dict(ms) for qt, ms in sorted(by_qtype.items())}
    atype_agg = {at: _mean_dict(ms) for at, ms in sorted(by_answer_type.items())}

    # Overall K-Acc.
    all_kacc = [m.get("kacc", 0.0) for m in per_sample if "kacc" in m]
    overall_kacc = sum(all_kacc) / len(all_kacc) if all_kacc else 0.0

    # Overall PRIMARY metric: binary_acc for binary Q-types (K-Acc ignores
    # polarity on binary questions and systematically inflates those scores),
    # multilabel_f1 for multi-label, count_exact_match for counting, and kacc
    # for everything else. This drives checkpoint selection.
    def _primary_of(q: str, m: dict) -> float:
        at = QTYPE_TO_ANSWER_TYPE.get(q, "classification")
        if at == "binary":
            return m.get("binary_acc", m.get("kacc", 0.0))
        if at == "multi_label":
            return m.get("multilabel_f1", m.get("kacc", 0.0))
        if at == "counting":
            return m.get("count_exact_match", m.get("kacc", 0.0))
        return m.get("kacc", 0.0)

    primary_vals = [
        _primary_of(qt, m) for m, qt in zip(per_sample, question_types)
    ]
    overall_primary = (
        sum(primary_vals) / len(primary_vals) if primary_vals else 0.0
    )

    return {
        "overall_kacc":    overall_kacc,
        "overall_primary": overall_primary,
        "by_qtype":        qtype_agg,
        "by_answer_type":  atype_agg,
        "n_samples":       len(per_sample),
    }


# ── 5. Checkpointing ──────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch:       int,
    val_loss:    float,
    metrics:     dict,
    config,
    path:        Path,
    global_step: int = 0,
    is_best:     bool = False,
) -> None:
    """
    Save only the trainable weights (LoRA adapters + compressor).
    Much smaller than a full model checkpoint — typically 50–200 MB.

    Saves checkpoint.pt and (if is_best=True) a hard link named
    best_checkpoint.pt (zero additional disk space).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Collect trainable state only
    trainable_state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if any(
            tag in k
            for tag in ("lora_", "modules_to_save", "compressor")
        )
    }

    ckpt = {
        "epoch":           epoch,
        "global_step":     global_step,
        "val_loss":        val_loss,
        "metrics":         metrics,
        "trainable_state": trainable_state,
        "optimizer":       optimizer.state_dict(),
        "scheduler":       scheduler.state_dict() if scheduler else None,
    }

    ckpt_path = path / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    logger.info(
        "Checkpoint saved → %s  (epoch %d, step %d, val_loss=%.4f)",
        ckpt_path, epoch, global_step, val_loss,
    )

    if is_best:
        best_path = path / "best_checkpoint.pt"
        if best_path.exists():
            best_path.unlink()
        os.link(ckpt_path, best_path)
        logger.info("New best checkpoint → %s (hard link)", best_path)

    # Save the config alongside the checkpoint for exact reconstruction
    config_path = path / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path: Path,
    device: torch.device,
    best: bool = False,
) -> tuple[int, float, float, Optional[int]]:
    """
    Load a saved checkpoint back into the model.

    Returns
    -------
    (epoch, val_loss, best_primary, global_step)
    """
    path = Path(path)
    filename = "best_checkpoint.pt" if best else "checkpoint.pt"
    ckpt = torch.load(path / filename, map_location=device, weights_only=False)

    # Load only the saved trainable weights — frozen weights untouched
    missing, unexpected = model.load_state_dict(
        ckpt["trainable_state"], strict=False
    )
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])

    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])

    best_metric = ckpt.get("metrics", {}).get(
        "overall_primary",
        ckpt.get("metrics", {}).get("overall_kacc", 0.0),
    )
    global_step = ckpt.get("global_step", None)
    logger.info(
        "Loaded checkpoint from %s (epoch %d, step %s, val_loss=%.4f)",
        path, ckpt["epoch"],
        str(global_step) if global_step is not None else "?",
        ckpt["val_loss"],
    )
    return ckpt["epoch"], ckpt["val_loss"], best_metric, global_step


# ── 6. EarlyStopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Patience-based early stopping.

    Stops training when the tracked value does not improve (decrease) by at
    least `min_delta` for `patience` consecutive checks. To track a metric
    that should increase, pass its negation to step().

    Parameters
    ----------
    patience  : number of checks without improvement before stopping
    min_delta : minimum improvement to count as progress
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Call after each validation run. Returns True to stop training."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d checks "
                "(best=%.4f, current=%.4f)",
                self.counter, self.patience, self.best_loss, val_loss,
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("EarlyStopping triggered.")

        return self.should_stop


# ── 7. Expected visual token count ────────────────────────────────────────────

def expected_visual_tokens(
    n_slow: int,
    n_fast: int,
    config=None,
) -> int:
    """
    Compute the number of compressed visual tokens for given real frame
    counts. Must match what SurgQVLM._compress() produces so the prompt
    carries exactly the right number of <|video_pad|> placeholders.

    Parameters
    ----------
    n_slow : number of slow-stream frames (real, not padded)
    n_fast : number of fast-stream frames (real, not padded)
    config : SurgQCompressorConfig or None (uses defaults)

    Example
    -------
    >>> expected_visual_tokens(15, 30)   # standard 30-second clip
    671
    """
    if config is not None:
        t_k_slow = config.slow_temporal_k
        s_k_slow = config.slow_spatial_k
        t_k_fast = config.fast_temporal_k
        s_k_fast = config.fast_spatial_k
        hw       = config.patch_grid_hw
        fast_hw  = config.fast_spatial_size
        n_g_slow = config.n_global_slow
        n_g_fast = config.n_global_fast
    else:
        # Defaults from SurgQCompressorConfig (Qwen2.5-VL ViT at 384px)
        t_k_slow = 3;  s_k_slow = 3
        t_k_fast = 3;  s_k_fast = 1
        hw       = 27; fast_hw  = 5
        n_g_slow = 8;  n_g_fast = 8

    slow_local = (
        math.ceil(n_slow / t_k_slow)
        * math.ceil(hw / s_k_slow) ** 2
    )
    fast_local = (
        math.ceil(n_fast / t_k_fast)
        * math.ceil(fast_hw / s_k_fast) ** 2
    )
    return slow_local + fast_local + n_g_slow + n_g_fast


# ── 8. Config loading (YAML → dataclass, with CLI overrides) ─────────────────

def load_yaml_config(config_cls, yaml_path: str):
    """
    Instantiate a config dataclass from a YAML file.

    Unknown keys in the YAML raise a ValueError (catches typos early).

    Parameters
    ----------
    config_cls : the dataclass type (SFTConfig or GRPOConfig)
    yaml_path  : path to the YAML file

    Returns
    -------
    An instance of config_cls.
    """
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    valid = {f.name for f in dataclass_fields(config_cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"Unknown keys in {yaml_path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return config_cls(**raw)


def add_config_cli_overrides(parser, config_cls, skip: tuple = ()) -> None:
    """
    Add one --<field> argument per dataclass field so any config value can
    be overridden on the command line (e.g. --learning_rate 1e-4).

    Boolean fields accept "true"/"false". Fields listed in `skip` and fields
    whose type is not int/float/str/bool are not exposed.
    """
    for f in dataclass_fields(config_cls):
        if f.name in skip:
            continue
        ftype = f.type
        if ftype == bool or ftype == "bool":
            parser.add_argument(
                f"--{f.name}", type=lambda x: x.lower() == "true",
                default=None, help=f"Override config.{f.name} (true/false)",
            )
        elif ftype in (int, float, str, "int", "float", "str"):
            pytype = {"int": int, "float": float, "str": str}.get(ftype, ftype)
            parser.add_argument(
                f"--{f.name}", type=pytype, default=None,
                help=f"Override config.{f.name}",
            )


def apply_cli_overrides(config, args) -> None:
    """Apply non-None CLI override values onto a config instance in place."""
    for f in dataclass_fields(type(config)):
        cli_val = getattr(args, f.name, None)
        if cli_val is not None:
            setattr(config, f.name, cli_val)
