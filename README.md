# SurgQ-VLM

**Question-Conditioned Dual-Stream Compression and Answer-Type-Aware
Reinforcement for Procedural Video Reasoning** (WACV 2027)

SurgQ-VLM is a two-stage vision-language model for surgical video question
answering on laparoscopic cholecystectomy:

- **Stage 1 (SFT)** trains a question-conditioned dual-stream (slow/fast)
  compressor that conditions both streams on the question through
  bidirectional cross-attention and reduces each 30-second clip to **671
  visual tokens** (49× fewer than the raw ViT patches) for a LoRA-adapted
  Qwen2.5-VL-7B.
- **Stage 2 (GRPO)** post-trains with Group Relative Policy Optimization
  under an **answer-type-aware reward** that scores each question on the
  metric matching its answer structure: binary match, keyword accuracy for
  classification, micro-F1 for label sets, normalised error for counts, and
  a content–timing blend for temporal sentences.

On a sealed 21,840-question test split over Cholec80 + CholecT50, SurgQ-VLM
reaches **0.779 keyword accuracy**; the answer-type reward adds +2.5 pp
K-Acc over SFT, concentrated in the structurally hard answer types.

```
Architecture:  30 s clip ─┬─ slow stream (0.5 FPS, 15f) ─┐  frozen ViT
                          └─ fast stream (1.0 FPS, 30f) ─┤  (Qwen2.5-VL)
                                                         ▼
               question ──► 4× GuideInjector cross-attention (bidirectional
                            conditioning, per stream × local/global)
                                                         ▼
               671 tokens = 405 slow-local + 250 fast-local + 8+8 global
                                                         ▼
                            LoRA-adapted Qwen2.5-VL-7B LLM → answer
```

## Repository layout

```
configs/            Training configs (the exact paper recipes)
  sft.yaml            Stage 1: supervised fine-tuning
  grpo.yaml           Stage 2: GRPO post-training
data_prep/          Benchmark construction from the raw datasets
  extract_frames.py   Cholec80 videos → 1-FPS PNG frames (ffmpeg)
  extract_clips.py    annotations → 30 s clip JSONs (transition + stable)
  generate_qa.py      clip JSONs → 18-Q-type QA pairs (balanced binaries)
  build_splits.py     video-level train/val/test splits (paper partition)
  labels_canonical.py canonical cross-dataset label vocabulary
scripts/
  download_models.py  pre-fetch Qwen2.5-VL from HuggingFace
  verify_data.py      check the raw dataset layout
surgqvlm/           Model package
  compressor.py       dual-stream question-conditioned compressor (HICom-based)
  model.py            Qwen2.5-VL + LoRA + compressor assembly
  data.py             dataset, SlowFast sampling, collate
  utils.py            prompts, metrics, loss weighting, checkpointing
train_sft.py        Stage 1 entry point
train_grpo.py       Stage 2 entry point
evaluate.py         test-split evaluation (all paper metrics)
```

## 1. Installation

Python 3.11 + CUDA 12.1 (training targets one A100 80 GB):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Optional, 30-40% faster attention on A100/H100:
pip install flash-attn==2.8.3 --no-build-isolation
```

Pre-fetch the backbone (optional — training downloads it on first use):

```bash
python scripts/download_models.py            # Qwen2.5-VL-7B-Instruct, ~16 GB
```

## 2. Data

The benchmark is **generated locally** from two public datasets that require
a signed access request (we cannot redistribute them):

| Dataset   | Request from | Provides |
|-----------|--------------|----------|
| Cholec80  | [CAMMA, University of Strasbourg](http://camma.u-strasbg.fr/datasets) | 80 videos, phase + tool annotations |
| CholecT50 | [CAMMA / CholecT50 challenge](https://github.com/CAMMA-public/cholect50) | 50 videos (1-FPS frames), triplet annotations |

Unpack them into the layout below, then verify:

```
datasets/
  cholec80/
    videos/              video01.mp4 … video80.mp4
    phase_annotations/   video01-phase.txt …
    tool_annotations/    video01-tool.txt …
  CholecT50/
    videos/              VID01/000000.png …   (frames ship pre-extracted)
    labels/              VID01.json …
    label_mapping.txt
```

```bash
python scripts/verify_data.py --data_root datasets
```

Build the benchmark (deterministic; ~80k QA pairs over ~6.7k clips):

```bash
# 1. Cholec80 frames at 1 FPS (needs ffmpeg; several hours, ~80 GB of PNGs)
python data_prep/extract_frames.py

# 2. 30-second clips with canonical per-frame annotations
python data_prep/extract_clips.py

# 3. 18-Q-type QA pairs with dataset-level binary balancing
python data_prep/generate_qa.py

# 4. The paper's video-level splits + the curated validation subset
python data_prep/build_splits.py
```

Expected split sizes (printed by `build_splits.py`):
train 4,212 clips / 52,100 QA pairs · val 575 / 6,696 ·
test 1,914 / 21,840 (the paper's sealed test split).

## 3. Training

**Stage 1 — SFT** (≈1 day on one A100 80 GB):

```bash
python train_sft.py --config configs/sft.yaml
```

Trains the compressor and the LoRA adapters jointly (ViT frozen) with
Q-type inverse-frequency loss weighting. Checkpoints go to
`checkpoints/sft/`; `best_checkpoint.pt` is selected by the generative
primary metric on the curated validation subset.

**Stage 2 — GRPO** (≈1 day on one A100 80 GB):

```bash
python train_grpo.py --config configs/grpo.yaml
```

Warm-starts from `checkpoints/sft/best_checkpoint.pt`, freezes the
compressor, and optimises the LoRA adapters with the answer-type-aware
reward R = R1 + 0.3·R2 + 0.05·R3 − 0.3·R_len, G=8 rollouts per prompt,
DAPO decoupled clipping, KL to the frozen SFT reference, and dynamic
sampling. Output: `checkpoints/grpo/`.

Any config field can be overridden on the command line:

```bash
python train_sft.py --config configs/sft.yaml --batch_size 4 --num_workers 4
```

## 4. Evaluation

```bash
# The paper's headline numbers: GRPO checkpoint on the sealed test split
# (batch 32 and the default 64 generated tokens match the paper's eval runs)
python evaluate.py --experiment_name grpo --split test --eval_batch_size 32

# SFT-only ablation row
python evaluate.py --experiment_name sft --split test --eval_batch_size 32

# Downloaded release weights (no training directory needed)
python evaluate.py --weights surgqvlm_grpo_weights.pt --split test --eval_batch_size 32
```

Writes per-sample predictions (`eval_test_*.jsonl`) and an aggregated
summary (`eval_test_*_summary.json`) with K-Acc, the structure-matched
Primary metric, per-answer-type and per-Q-type breakdowns, ROUGE-L, BLEU-4,
and temporal accuracy at ±1/±2/±3 s.

> Note on reported metrics: `overall_primary` is the N-weighted average of
> binary accuracy, classification K-Acc, multi-label micro-F1, counting
> exact-match, and sentence-temporal K-Acc — computed per sample. Per-group
> rows in the paper's Table 3 report K-Acc for cross-group comparability.

## 5. Pretrained weights

Lean weight files (compressor + LLM LoRA adapters, ~0.8 GB each; produced
by `scripts/export_weights.py`) are released for the SFT and GRPO stages.
Download one and evaluate without any training directory:

```bash
python evaluate.py --weights surgqvlm_grpo_weights.pt \
    --model_config configs/sft.yaml --split test --eval_batch_size 32
```

Expected sealed-test scores (21,840 QA pairs):

| Weights | K-Acc | Primary |
|---------|-------|---------|
| SFT               | 0.754 | 0.722 |
| SFT + GRPO        | 0.779 | 0.743 |

GRPO seed-to-seed variance is ±0.0003 K-Acc (three seeds reported in the
paper's rebuttal), so independent reruns land within 0.1 pp of these values.
Loading uses `load_state_dict(strict=False)`: the frozen Qwen base weights
come from the HuggingFace download, and keys stripped by the exporter (the
frozen embedding table and the identity ViT adapters) keep their
initialisation, which is numerically identical to the training checkpoint.

## 6. Smoke tests (no GPU, no downloads)

Every entry point has a mock mode that verifies shapes end-to-end on CPU:

```bash
python -m surgqvlm.compressor                 # compressor forward shapes
python -m surgqvlm.model                      # full mock model forward/generate
python train_grpo.py --test_rewards           # reward function unit tests
python data_prep/extract_clips.py --mock      # clip pipeline on synthetic data
python data_prep/generate_qa.py --mock        # QA generator on a synthetic clip
python train_sft.py  --config configs/sft.yaml  --mock --dry_run
python train_grpo.py --config configs/grpo.yaml --mock --dry_run
```

(The two `--mock --dry_run` training commands additionally need split files
to exist — run them after `build_splits.py`, or point `--data_root` at any
directory containing `splits/train.txt` and `splits/val.txt`.)

## Hardware & runtime reference

| Stage | Hardware | Time | Peak memory |
|-------|----------|------|-------------|
| SFT (1 epoch, 52.1k QA)  | 1× A100 80 GB | ~16 h | <80 GB (grad ckpt + 8-bit Adam) |
| GRPO (1 epoch, G=8)      | 1× A100 80 GB | ~24 h | <80 GB |
| Inference                | 1× A100 80 GB | 1.7 ± 0.6 s / 30 s clip | 21.2 GB |

## Citation

```bibtex
@inproceedings{surgqvlm2027,
  title     = {SurgQ-VLM: Question-Conditioned Dual-Stream Compression and
               Answer-Type-Aware Reinforcement for Procedural Video Reasoning},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications
               of Computer Vision (WACV)},
  year      = {2027},
}
```

## Acknowledgements

The compressor design builds on
[HICom](https://github.com/lntzm/HICom) (Liu et al., CVPR 2025); the
backbone is [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL); the source
datasets are Cholec80 (Twinanda et al., 2017) and CholecT50 (Nwoye et al.,
2022) from CAMMA, University of Strasbourg.

## License

Code is released under the MIT License (see `LICENSE`). The Cholec80 and
CholecT50 datasets have their own licenses and access conditions — consult
CAMMA's terms before use.
