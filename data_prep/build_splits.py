#!/usr/bin/env python3
"""
build_splits.py — Video-level train/val/test splits for SurgQ-VLM.
===================================================================

Builds the exact split used in the paper. The partition is at the VIDEO
level — no clip from the same procedure ever appears in more than one
split — with the fixed video assignments hard-coded below:

    Cholec80  : 48 train / 9 val / 23 test
    CholecT50 : 35 train / 5 val / 10 test
                (test videos follow the Rendezvous protocol of
                 Nwoye et al., 2022)

Expected result on the full benchmark (after extract_clips.py +
generate_qa.py with the default seed):

    train : 4,212 clips | 52,100 QA pairs
    val   :   575 clips |  6,696 QA pairs
    test  : 1,914 clips | 21,840 QA pairs   (the paper's sealed test split)

Outputs (under --splits_dir)
----------------------------
    train.txt / val.txt / test.txt
        One line per clip: absolute_clip_json_path TAB absolute_qa_json_path
    curated_val.txt + curated_val_qids.json
        A small stratified validation subset (up to --curated_per_qtype QA
        pairs per Q-type, deterministic under --seed) used for fast
        generative monitoring during training. Training works without it,
        but best-checkpoint selection then falls back to validation loss.
    split_summary.json
        Counts per split, per source, and per Q-type.

Usage
-----
    python data_prep/build_splits.py --clips_root datasets/clips \\
                                     --splits_dir datasets/splits
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Fixed video-level split (the paper's partition) ───────────────────────────

SPLIT_VIDEOS: dict[str, dict[str, list[str]]] = {
    "train": {
        "cholec80": [
            "video01", "video02", "video03", "video04", "video05", "video07",
            "video09", "video11", "video13", "video15", "video16", "video17",
            "video18", "video19", "video20", "video21", "video22", "video23",
            "video24", "video25", "video26", "video27", "video28", "video30",
            "video31", "video33", "video34", "video35", "video36", "video37",
            "video38", "video39", "video40", "video43", "video47", "video48",
            "video49", "video52", "video56", "video57", "video60", "video62",
            "video65", "video66", "video68", "video70", "video75", "video79",
        ],
        "cholect50": [
            "VID01", "VID02", "VID04", "VID05", "VID13", "VID15", "VID18",
            "VID22", "VID23", "VID25", "VID26", "VID27", "VID31", "VID35",
            "VID36", "VID40", "VID43", "VID47", "VID48", "VID49", "VID52",
            "VID56", "VID57", "VID60", "VID62", "VID65", "VID66", "VID68",
            "VID70", "VID75", "VID79", "VID92", "VID96", "VID103", "VID110",
        ],
    },
    "val": {
        "cholec80": [
            "video08", "video12", "video29", "video41", "video44", "video45",
            "video46", "video50", "video78",
        ],
        "cholect50": [
            "VID08", "VID12", "VID29", "VID50", "VID78",
        ],
    },
    "test": {
        "cholec80": [
            "video06", "video10", "video14", "video32", "video42", "video51",
            "video53", "video54", "video55", "video58", "video59", "video61",
            "video63", "video64", "video67", "video69", "video71", "video72",
            "video73", "video74", "video76", "video77", "video80",
        ],
        # CholecT50 test videos follow the Rendezvous protocol
        # (Nwoye et al., 2022).
        "cholect50": [
            "VID06", "VID10", "VID14", "VID32", "VID42", "VID51", "VID73",
            "VID74", "VID80", "VID111",
        ],
    },
}


# ── Split assembly ────────────────────────────────────────────────────────────

def collect_split_lines(clips_root: Path, split: str) -> list[tuple[str, str]]:
    """
    Collect (clip_json, qa_json) absolute-path pairs for one split,
    sorted by source, then video, then clip index.
    """
    lines: list[tuple[str, str]] = []
    for source in ("cholec80", "cholect50"):
        for video_id in SPLIT_VIDEOS[split][source]:
            vdir = clips_root / source / video_id
            if not vdir.is_dir():
                logger.warning("%s: missing video directory %s — skipping",
                               split, vdir)
                continue
            for clip_json in sorted(vdir.glob("clip_*.json")):
                name = clip_json.name
                if name.endswith("_qa.json"):
                    continue
                qa_json = clip_json.with_name(name.replace(".json", "_qa.json"))
                if not qa_json.exists():
                    logger.warning("No QA file for %s — skipping", clip_json)
                    continue
                lines.append((str(clip_json.resolve()), str(qa_json.resolve())))
    return lines


def count_qa(lines: list[tuple[str, str]]) -> tuple[int, Counter]:
    """Count the total QA pairs and the per-Q-type breakdown for a split."""
    total = 0
    by_qtype: Counter = Counter()
    for _, qa_path in lines:
        try:
            with open(qa_path) as f:
                pairs = json.load(f)
        except Exception:
            continue
        total += len(pairs)
        for p in pairs:
            by_qtype[p.get("question_type", "?")] += 1
    return total, by_qtype


def verify_no_leakage() -> None:
    """Assert that no video is assigned to more than one split."""
    seen: dict[str, str] = {}
    for split, sources in SPLIT_VIDEOS.items():
        for source, vids in sources.items():
            for v in vids:
                key = f"{source}/{v}"
                if key in seen:
                    raise AssertionError(
                        f"Video-level leakage: {key} in both "
                        f"{seen[key]} and {split}"
                    )
                seen[key] = split
    logger.info("PASS: no video-level leakage between splits "
                "(%d videos total)", len(seen))


# ── Curated validation subset ─────────────────────────────────────────────────

def build_curated_val(
    val_lines: list[tuple[str, str]],
    splits_dir: Path,
    per_qtype: int,
    seed: int,
) -> None:
    """
    Build a small stratified validation subset for fast generative
    monitoring during training: up to `per_qtype` QA pairs per Q-type,
    sampled deterministically from the val split.

    Writes:
        curated_val.txt       — the val-split lines whose clips contribute
                                at least one selected QA pair
        curated_val_qids.json — {question_id: true} whitelist, applied by
                                SurgQDataset(filter_qids=...)
    """
    rng = random.Random(seed)

    # Gather all val QA pairs grouped by Q-type.
    by_qtype: dict[str, list[tuple[str, str]]] = {}   # qtype → [(qid, line_key)]
    line_for_clip: dict[str, tuple[str, str]] = {}
    for clip_path, qa_path in val_lines:
        try:
            with open(qa_path) as f:
                pairs = json.load(f)
        except Exception:
            continue
        for p in pairs:
            qid = p.get("question_id", "")
            qt  = p.get("question_type", "?")
            if qid:
                by_qtype.setdefault(qt, []).append((qid, clip_path))
                line_for_clip[clip_path] = (clip_path, qa_path)

    selected_qids: dict[str, bool] = {}
    selected_clips: set[str] = set()
    for qt in sorted(by_qtype):
        candidates = sorted(by_qtype[qt])     # deterministic order pre-shuffle
        rng.shuffle(candidates)
        for qid, clip_path in candidates[:per_qtype]:
            selected_qids[qid] = True
            selected_clips.add(clip_path)

    curated_lines = [line_for_clip[c] for c in sorted(selected_clips)]

    cval_path = splits_dir / "curated_val.txt"
    with open(cval_path, "w") as f:
        for clip_path, qa_path in curated_lines:
            f.write(f"{clip_path}\t{qa_path}\n")
    qids_path = splits_dir / "curated_val_qids.json"
    with open(qids_path, "w") as f:
        json.dump(selected_qids, f, indent=1)

    logger.info("Curated val subset: %d QA pairs across %d clips → %s, %s",
                len(selected_qids), len(curated_lines), cval_path, qids_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the paper's video-level train/val/test splits."
    )
    parser.add_argument("--clips_root", default="datasets/clips",
                        help="Root of the clip JSONs written by extract_clips.py")
    parser.add_argument("--splits_dir", default="datasets/splits",
                        help="Output directory for the split files.")
    parser.add_argument("--curated_per_qtype", type=int, default=30,
                        help="QA pairs per Q-type in the curated val subset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clips_root = Path(args.clips_root)
    splits_dir = Path(args.splits_dir)
    if not clips_root.is_dir():
        logger.error("Clips root not found: %s — run extract_clips.py and "
                     "generate_qa.py first.", clips_root)
        sys.exit(1)
    splits_dir.mkdir(parents=True, exist_ok=True)

    verify_no_leakage()

    summary = OrderedDict()
    val_lines_for_curated: list[tuple[str, str]] = []

    for split in ("train", "val", "test"):
        lines = collect_split_lines(clips_root, split)
        out_path = splits_dir / f"{split}.txt"
        with open(out_path, "w") as f:
            for clip_path, qa_path in lines:
                f.write(f"{clip_path}\t{qa_path}\n")

        n_qa, by_qtype = count_qa(lines)
        n_videos = sum(len(v) for v in SPLIT_VIDEOS[split].values())
        logger.info("%s: %d clips | %d QA pairs | %d videos → %s",
                    split, len(lines), n_qa, n_videos, out_path)
        summary[split] = {
            "clips":     len(lines),
            "qa_pairs":  n_qa,
            "videos":    n_videos,
            "by_qtype":  dict(sorted(by_qtype.items())),
        }
        if split == "val":
            val_lines_for_curated = lines

    # Curated generative-monitoring subset from the val split.
    build_curated_val(
        val_lines_for_curated, splits_dir,
        per_qtype=args.curated_per_qtype, seed=args.seed,
    )

    summary_path = splits_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary → %s", summary_path)

    logger.info("=" * 60)
    logger.info("SPLIT BUILD COMPLETE")
    for split, s in summary.items():
        logger.info("  %-5s: %5d clips | %6d QA pairs | %2d videos",
                    split, s["clips"], s["qa_pairs"], s["videos"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
