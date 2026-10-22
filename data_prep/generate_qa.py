#!/usr/bin/env python3
"""
generate_qa.py — Unified rule-based QA generator for SurgQ-VLM.
=====================================================================

Reads the canonical clip JSONs written by ``extract_clips.py`` and emits
one ``clip_XXXX_qa.json`` per clip plus a merged ``qa_all.json`` per dataset
and a cross-dataset ``qa_summary.json``.

Design
------
  * Single source of truth for every label — all clip fields are already in
    canonical snake_case (see labels_canonical.py); answers convert to the
    natural-language form (spaces instead of underscores) at render time.
  * Q06 (instrument presence) — each clip yields 2 pairs: 1 positive probe
    and 1 absent-instrument probe (hard negatives preferred: instruments
    absent in the clip but present elsewhere in the same video).
  * Binary-polarity balancing at dataset level — after generation every
    binary Q-type (Q02, Q06, Q07, Q13, Q17) is down-sampled so yes == no,
    removing any "always-no" optimum for the model.
  * Deterministic per-clip RNG — SHA-1 of clip_id combined with --seed.
  * Cholec80 clips skip Q10–Q15 and Q18 (no triplet data); CholecT50 clips
    cover all 18 types.
  * Every answer uses canonical concepts rendered with spaces (e.g.
    ``"the clip primarily shows calot triangle dissection."``).

Question-type registry (unchanged numbers, redesigned generation):
  Q01  phase identification          classification     all clips
  Q02  phase change yes/no           binary             all clips
  Q03  phase transition timing       sentence_temporal  phase-change clips
  Q04  phase sequence description    sequence           phase-change clips
  Q05  instruments present           multi_label        clips with instruments
  Q06  instrument presence (probe)   binary             2 probes per clip
  Q07  instrument handoff yes/no     binary             all clips
  Q08  instrument duration           counting_temporal  1 per unique instrument
  Q09  instrument at second X        classification     3 probes per clip
  Q10  primary action                classification     CholecT50, triplets
  Q11  all actions in sequence       sequence           CholecT50
  Q12  action count                  counting           CholecT50
  Q13  triplet change yes/no         binary             CholecT50
  Q14  primary target anatomy        classification     CholecT50
  Q15  primary verb / maneuver       classification     CholecT50, non-null
  Q16  instrument change at transition sentence_temporal  phase-change clips
  Q17  idle / no-instrument frames   binary             all clips
  Q18  action before/after transition classification    CholecT50, triplet-change
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

import labels_canonical as L

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT  = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logger.addHandler(_console)


def setup_file_logging(output_root: str) -> str:
    os.makedirs(output_root, exist_ok=True)
    log_path = os.path.join(output_root, "generate_qa.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logger.addHandler(fh)
    logger.info(f"Logging to file: {log_path}")
    return log_path


# ===========================================================================
# Question metadata registry
# ===========================================================================

QUESTION_META = {
    "Q01": {"group": "phase",      "answer_type": "classification",    "answer_format": "short_phrase", "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q02": {"group": "phase",      "answer_type": "binary",            "answer_format": "binary",       "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q03": {"group": "phase",      "answer_type": "sentence_temporal", "answer_format": "sentence",     "requires_transition": True,  "temporal_grounding": True,  "novel_vs_sota": True},
    "Q04": {"group": "phase",      "answer_type": "sequence",          "answer_format": "sentence",     "requires_transition": True,  "temporal_grounding": True,  "novel_vs_sota": True},
    "Q05": {"group": "instrument", "answer_type": "multi_label",       "answer_format": "list",         "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q06": {"group": "instrument", "answer_type": "binary",            "answer_format": "binary",       "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q07": {"group": "instrument", "answer_type": "binary",            "answer_format": "sentence",     "requires_transition": False, "temporal_grounding": True,  "novel_vs_sota": False},
    "Q08": {"group": "instrument", "answer_type": "counting_temporal", "answer_format": "sentence",     "requires_transition": False, "temporal_grounding": True,  "novel_vs_sota": True},
    "Q09": {"group": "instrument", "answer_type": "classification",    "answer_format": "short_phrase", "requires_transition": False, "temporal_grounding": True,  "novel_vs_sota": True},
    "Q10": {"group": "triplet",    "answer_type": "classification",    "answer_format": "short_phrase", "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q11": {"group": "triplet",    "answer_type": "sequence",          "answer_format": "sentence",     "requires_transition": False, "temporal_grounding": True,  "novel_vs_sota": True},
    "Q12": {"group": "triplet",    "answer_type": "counting",          "answer_format": "short_phrase", "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q13": {"group": "triplet",    "answer_type": "binary",            "answer_format": "binary",       "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q14": {"group": "triplet",    "answer_type": "classification",    "answer_format": "short_phrase", "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q15": {"group": "triplet",    "answer_type": "classification",    "answer_format": "short_phrase", "requires_transition": False, "temporal_grounding": False, "novel_vs_sota": False},
    "Q16": {"group": "temporal",   "answer_type": "sentence_temporal", "answer_format": "sentence",     "requires_transition": True,  "temporal_grounding": True,  "novel_vs_sota": True},
    "Q17": {"group": "temporal",   "answer_type": "binary",            "answer_format": "sentence",     "requires_transition": False, "temporal_grounding": True,  "novel_vs_sota": False},
    "Q18": {"group": "temporal",   "answer_type": "classification",    "answer_format": "short_phrase", "requires_transition": True,  "temporal_grounding": True,  "novel_vs_sota": True},
}

BINARY_TYPES = [qt for qt, meta in QUESTION_META.items()
                if meta["answer_type"] == "binary"]


# ===========================================================================
# Phrasing templates — 5 variants per Q-type
# ===========================================================================

PHRASINGS = {
    "Q01": [
        "What surgical phase is being performed in this clip?",
        "Which phase of the cholecystectomy procedure is shown here?",
        "What stage of the surgery is taking place in this video?",
        "Identify the operative phase depicted in this clip.",
        "What part of the surgical workflow does this clip show?",
    ],
    "Q02": [
        "Does the surgical phase change at any point during this clip?",
        "Is there a phase transition visible in this clip?",
        "Does this clip span more than one surgical phase?",
        "Are multiple phases of the surgery shown in this clip?",
        "Does the operative phase shift at any moment during this video?",
    ],
    "Q03": [
        "At what point does the surgical phase change in this clip?",
        "When does the phase transition occur during this clip?",
        "At which second does the operative phase shift in this video?",
        "How many seconds into the clip does the surgical phase change?",
        "At what moment during this clip does the phase of surgery change?",
    ],
    "Q04": [
        "What is the sequence of surgical phases in this clip?",
        "Describe the phase progression shown in this clip.",
        "Which phases appear in this clip and in what order?",
        "What phases of surgery occur in this clip, and for how long each?",
        "Walk me through the surgical phases visible in this clip.",
    ],
    "Q05": [
        "Which surgical instruments are used in this clip?",
        "What instruments appear in this video clip?",
        "List the surgical tools used during this clip.",
        "Which tools does the surgeon use in this clip?",
        "What instruments are active at any point in this clip?",
    ],
    "Q06": [
        "Is the {instrument} used at any point in this clip?",
        "Does the {instrument} appear in this video clip?",
        "Is the {instrument} active during this clip?",
        "Can the {instrument} be seen in this clip?",
        "Does the surgeon use the {instrument} in this clip?",
    ],
    "Q07": [
        "Is there a change in the active instrument during this clip?",
        "Does the surgeon switch instruments at any point in this clip?",
        "Is an instrument handoff visible in this clip?",
        "Does the active surgical tool change during this video?",
        "Is the instrument set the same from start to end of this clip?",
    ],
    "Q08": [
        "For how many seconds is the {instrument} used in this clip?",
        "How long is the {instrument} active during this clip?",
        "What is the total duration of {instrument} usage in this clip?",
        "How much time does the {instrument} spend in use during this clip?",
        "Over how many seconds does the {instrument} operate in this clip?",
    ],
    "Q09": [
        "Which instrument is being used at second {offset} of this clip?",
        "What surgical tool is active at second {offset} in this video?",
        "Which instrument does the surgeon have at the {offset}-second mark?",
        "At second {offset}, which instrument is in use?",
        "What is the active instrument {offset} seconds into this clip?",
    ],
    "Q10": [
        "What is the primary surgical action in this clip?",
        "Which action dominates this clip?",
        "What is the main instrument-verb-target action shown here?",
        "Identify the dominant triplet action in this clip.",
        "What is the principal surgical interaction in this video?",
    ],
    "Q11": [
        "What sequence of surgical actions occurs in this clip?",
        "Describe the progression of instrument actions in this clip.",
        "List the surgical actions in the order they appear.",
        "Walk me through the sequence of actions in this clip.",
        "What actions does the surgeon perform, in order?",
    ],
    "Q12": [
        "How many distinct surgical actions occur in this clip?",
        "How many different instrument-verb-target actions are present?",
        "Count the unique surgical triplets visible in this clip.",
        "How many separate actions does the surgeon perform in this clip?",
        "What is the number of distinct actions in this video?",
    ],
    "Q13": [
        "Does the surgical action change during this clip?",
        "Is there an action transition visible in this clip?",
        "Do the active triplets change between frames of this clip?",
        "Does the surgeon transition to a new action during this clip?",
        "Is the surgical action consistent throughout this entire clip?",
    ],
    "Q14": [
        "What anatomical target is the primary focus in this clip?",
        "Which anatomy is being operated on in this clip?",
        "Identify the main target of the surgical actions here.",
        "What organ or tissue is the primary target in this clip?",
        "Which anatomical structure receives the main action?",
    ],
    "Q15": [
        "What is the primary surgical maneuver in this clip?",
        "Which verb best describes the main action in this video?",
        "Identify the dominant surgical maneuver here.",
        "What is the principal action verb in this clip?",
        "Which maneuver characterises this clip?",
    ],
    "Q16": [
        "What instrument change accompanies the phase transition in this clip?",
        "How does the surgical instrument change when the phase transitions in this clip?",
        "What happens to the active instrument when the phase changes in this video?",
        "Which instrument takes over when the surgical phase shifts in this clip?",
        "Describe the instrument handoff that occurs at the phase transition in this clip.",
    ],
    "Q17": [
        "Is there a moment in this clip where no instrument is active?",
        "Does the surgeon have a period of inactivity in this clip?",
        "Is there any time during this clip when no surgical tool is in use?",
        "Are there seconds in this clip where no instrument is being used?",
        "Does this clip contain any idle periods where no instrument is active?",
    ],
    "Q18": [
        "What surgical action appears immediately after the main transition in this clip?",
        "How does the action change across the main transition in this clip?",
        "What action is performed before and after the transition in this clip?",
        "Describe the action change around the clip's transition.",
        "Which triplet appears after the main event in this clip?",
    ],
}


# ===========================================================================
# RNG — deterministic per-clip
# ===========================================================================

def make_rng(seed: int, clip_id: str) -> random.Random:
    """Deterministic RNG seeded from SHA-1 of clip_id combined with ``seed``."""
    h = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()
    sub = int(h[:12], 16)
    return random.Random(seed ^ sub)


def pick_variant(rng: random.Random, qtype: str) -> int:
    return rng.randint(0, len(PHRASINGS[qtype]) - 1)


def _sha1_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


# ===========================================================================
# QA record factory
# ===========================================================================

def make_qa(clip: dict, qtype: str, variant: int, question: str,
             answer: str, binary_label: Optional[bool] = None,
             question_id_suffix: Optional[str] = None) -> dict:
    meta = QUESTION_META[qtype]
    qid = f"{clip['clip_id']}_{qtype}"
    if question_id_suffix is not None:
        qid += f"_{question_id_suffix}"
    qid += f"_v{variant}"
    return OrderedDict([
        ("question_id",         qid),
        ("clip_id",             clip["clip_id"]),
        ("source",              clip["source"]),
        ("video_id",            clip["video_id"]),
        ("question",            question),
        ("answer",              answer),
        ("question_type",       qtype),
        ("question_group",      meta["group"]),
        ("answer_type",         meta["answer_type"]),
        ("answer_format",       meta["answer_format"]),
        ("template_variant",    variant),
        ("requires_transition", meta["requires_transition"]),
        ("temporal_grounding",  meta["temporal_grounding"]),
        ("novel_vs_sota",       meta["novel_vs_sota"]),
        ("binary_label",        binary_label),
    ])


# ===========================================================================
# Per-frame helpers
# ===========================================================================

def instrument_names_pf(clip: dict) -> List[List[str]]:
    return [row if row else [] for row in clip.get("instrument_per_frame", [])]


def phase_names_pf(clip: dict) -> List[Optional[str]]:
    return list(clip.get("phase_per_frame", []))


def triplet_lists_pf(clip: dict) -> List[List[str]]:
    return [row if row else [] for row in clip.get("triplet_per_frame", [])]


def verb_lists_pf(clip: dict) -> List[List[str]]:
    return [row if row else [] for row in clip.get("verb_per_frame", [])]


def target_lists_pf(clip: dict) -> List[List[str]]:
    return [row if row else [] for row in clip.get("target_per_frame", [])]


def unique_instruments(clip: dict) -> List[str]:
    seen = []
    for row in instrument_names_pf(clip):
        for name in row:
            if name not in seen:
                seen.append(name)
    return seen


# ===========================================================================
# Q-specific generators
# ===========================================================================

# ----- Group 1: phase ----------------------------------------------------

def gen_q01(clip: dict, rng: random.Random) -> List[dict]:
    phases = [p for p in phase_names_pf(clip) if p]
    if not phases:
        return []
    dominant = Counter(phases).most_common(1)[0][0]
    v = pick_variant(rng, "Q01")
    a = f"The clip primarily shows {L.natural_phase(dominant)}."
    return [make_qa(clip, "Q01", v, PHRASINGS["Q01"][v], a)]


def gen_q02(clip: dict, rng: random.Random) -> List[dict]:
    phases = {p for p in phase_names_pf(clip) if p}
    has_change = len(phases) > 1
    v = pick_variant(rng, "Q02")
    a = ("Yes, the surgical phase changes during this clip."
         if has_change else
         "No, the surgical phase remains constant throughout this clip.")
    return [make_qa(clip, "Q02", v, PHRASINGS["Q02"][v], a,
                    binary_label=has_change)]


def gen_q03(clip: dict, rng: random.Random) -> List[dict]:
    """Detect every consecutive phase-change in the clip."""
    pfs = phase_names_pf(clip)
    transitions = []
    for i in range(1, len(pfs)):
        a, b = pfs[i - 1], pfs[i]
        if a and b and a != b:
            transitions.append((i, a, b))
    if not transitions:
        return []
    v = pick_variant(rng, "Q03")
    if len(transitions) == 1:
        sec, pb, pa = transitions[0]
        a = (f"The phase changes from {L.natural_phase(pb)} to "
             f"{L.natural_phase(pa)} at second {sec} of the clip.")
    else:
        parts = [f"from {L.natural_phase(pb)} to {L.natural_phase(pa)} "
                 f"at second {sec}" for sec, pb, pa in transitions]
        a = "The phase changes " + ", and again ".join(parts) + "."
    return [make_qa(clip, "Q03", v, PHRASINGS["Q03"][v], a)]


def gen_q04(clip: dict, rng: random.Random) -> List[dict]:
    pfs = phase_names_pf(clip)
    segments: List[Tuple[str, int, int]] = []
    current, start = None, 0
    for i, p in enumerate(pfs):
        if p != current:
            if current is not None:
                segments.append((current, start, i - 1))
            current, start = p, i
    if current is not None:
        segments.append((current, start, len(pfs) - 1))
    segments = [(p, s, e) for p, s, e in segments if p is not None]
    if len(segments) <= 1:
        return []
    v = pick_variant(rng, "Q04")
    parts = []
    for idx, (p, s, e) in enumerate(segments):
        if idx == len(segments) - 1:
            parts.append(f"{L.natural_phase(p)} from second {s} to the end of the clip")
        else:
            parts.append(f"{L.natural_phase(p)} from second {s} to second {e}")
    a = ", then ".join(parts) + "."
    return [make_qa(clip, "Q04", v, PHRASINGS["Q04"][v], a)]


# ----- Group 2: instrument -----------------------------------------------

def gen_q05(clip: dict, rng: random.Random) -> List[dict]:
    seen = unique_instruments(clip)
    if not seen:
        return []
    v = pick_variant(rng, "Q05")
    a = ", ".join(L.natural_instrument(i) for i in seen)
    return [make_qa(clip, "Q05", v, PHRASINGS["Q05"][v], a)]


def gen_q06(clip: dict, rng: random.Random,
             video_instrument_vocab: List[str]) -> List[dict]:
    """1 positive probe + 1 absent-in-clip probe.  Prefer instruments that
    appear elsewhere in the same video as hard negatives."""
    present = set(unique_instruments(clip))
    absent_in_clip = [i for i in L.INSTRUMENTS if i not in present]
    if not present or not absent_in_clip:
        # Edge case: clip has every instrument, or none.  Emit a single pair.
        qas: List[dict] = []
        if present:
            target = rng.choice(sorted(present))
            qas.extend(_make_q06(clip, rng, target, True))
        elif absent_in_clip:
            target = rng.choice(absent_in_clip)
            qas.extend(_make_q06(clip, rng, target, False))
        return qas

    pos_target = rng.choice(sorted(present))
    # Hard-neg preference: instruments absent in clip but present in same video.
    video_absent_present = [i for i in absent_in_clip
                              if i in video_instrument_vocab]
    if video_absent_present:
        neg_target = rng.choice(video_absent_present)
    else:
        neg_target = rng.choice(absent_in_clip)

    qas = []
    qas.extend(_make_q06(clip, rng, pos_target, True))
    qas.extend(_make_q06(clip, rng, neg_target, False))
    return qas


def _make_q06(clip: dict, rng: random.Random, instrument: str,
               positive: bool) -> List[dict]:
    v = pick_variant(rng, "Q06")
    natural = L.natural_instrument(instrument)
    q = PHRASINGS["Q06"][v].format(instrument=natural)
    a = (f"Yes, the {natural} is used in this clip."
         if positive else
         f"No, the {natural} is not used in this clip.")
    return [make_qa(clip, "Q06", v, q, a, binary_label=positive,
                    question_id_suffix=instrument)]


def gen_q07(clip: dict, rng: random.Random) -> List[dict]:
    """Handoff: the active instrument set differs between two consecutive
    frames AND both frames are non-empty."""
    pf = instrument_names_pf(clip)
    handoffs = []
    for i in range(1, len(pf)):
        prev_set = set(pf[i - 1])
        curr_set = set(pf[i])
        if prev_set and curr_set and prev_set != curr_set:
            outgoing = sorted(prev_set - curr_set)
            incoming = sorted(curr_set - prev_set)
            if outgoing or incoming:
                handoffs.append((i, outgoing, incoming))
    v = pick_variant(rng, "Q07")
    if handoffs:
        sec, out_list, in_list = handoffs[0]
        if in_list and out_list:
            a = (f"Yes, the active instrument changes from "
                 f"{L.natural_instrument(out_list[0])} to "
                 f"{L.natural_instrument(in_list[0])} at second {sec}.")
        elif in_list:
            a = (f"Yes, the {L.natural_instrument(in_list[0])} joins the "
                 f"active set at second {sec}.")
        else:
            a = (f"Yes, the {L.natural_instrument(out_list[0])} leaves the "
                 f"active set at second {sec}.")
        return [make_qa(clip, "Q07", v, PHRASINGS["Q07"][v], a,
                        binary_label=True)]
    else:
        all_instr = unique_instruments(clip)
        if len(all_instr) == 1:
            a = (f"No, the {L.natural_instrument(all_instr[0])} is used "
                 f"throughout the clip.")
        elif len(all_instr) == 0:
            a = "No, no instrument is active during this clip."
        else:
            a = ("No, the active instrument set stays the same across "
                 "the clip.")
        return [make_qa(clip, "Q07", v, PHRASINGS["Q07"][v], a,
                        binary_label=False)]


def gen_q08(clip: dict, rng: random.Random) -> List[dict]:
    """One pair per unique instrument in the clip; report count + first/last
    seconds of activity."""
    pf = instrument_names_pf(clip)
    stats: Dict[str, dict] = {}
    for sec, row in enumerate(pf):
        for name in row:
            if name not in stats:
                stats[name] = {"count": 0, "first": sec, "last": sec}
            stats[name]["count"] += 1
            stats[name]["last"] = sec
    qas = []
    for name, s in stats.items():
        v = pick_variant(rng, "Q08")
        natural = L.natural_instrument(name)
        q = PHRASINGS["Q08"][v].format(instrument=natural)
        a = (f"The {natural} is used for {s['count']} seconds, from "
             f"second {s['first']} to second {s['last']} of the clip.")
        qas.append(make_qa(clip, "Q08", v, q, a, question_id_suffix=name))
    return qas


def gen_q09(clip: dict, rng: random.Random) -> List[dict]:
    """Two temporally-spread probes per clip: sample 2 of 3 thirds
    (early / mid / late) per clip to keep Q09 share commensurate with the
    other high-count instrument questions while preserving temporal coverage
    at the corpus level (each third is sampled in roughly 2/3 of clips)."""
    pf = instrument_names_pf(clip)
    n = len(pf)
    if n < 3:
        return []
    third = n // 3
    all_offsets = [
        rng.randint(0,           third - 1),
        rng.randint(third,       2 * third - 1),
        rng.randint(2 * third,   n - 1),
    ]
    all_probe_idx = [0, 1, 2]
    keep = sorted(rng.sample(range(3), 2))
    offsets = [all_offsets[k] for k in keep]
    probe_indices = [all_probe_idx[k] for k in keep]
    qas = []
    for probe_idx, o in zip(probe_indices, offsets):
        v = pick_variant(rng, "Q09")
        q = PHRASINGS["Q09"][v].format(offset=o)
        if pf[o]:
            a = ", ".join(L.natural_instrument(i) for i in pf[o])
        else:
            a = "No instrument is active at this moment."
        qas.append(make_qa(clip, "Q09", v, q, a,
                           question_id_suffix=f"p{probe_idx}"))
    return qas


# ----- Group 3: triplet (CholecT50 only) ---------------------------------

def _is_meaningful_triplet(t: str) -> bool:
    """Skip triplets that end in null_verb or null_target — they read as
    'hook no identifiable action no specific target' which is ungrammatical
    and not a useful 'action' for Q10 / Q11 / Q18."""
    return "null_verb" not in t and "null_target" not in t


def gen_q10(clip: dict, rng: random.Random) -> List[dict]:
    """Primary triplet = most frequent *meaningful* triplet (skipping null
    verbs / null targets)."""
    counts: Counter = Counter()
    for row in triplet_lists_pf(clip):
        for t in row:
            if _is_meaningful_triplet(t):
                counts[t] += 1
    if not counts:
        return []
    primary = counts.most_common(1)[0][0]
    v = pick_variant(rng, "Q10")
    a = (f"The primary action in this clip is "
         f"{L.natural_triplet(primary)}.")
    return [make_qa(clip, "Q10", v, PHRASINGS["Q10"][v], a)]


def gen_q11(clip: dict, rng: random.Random) -> List[dict]:
    """Sequence of distinct *meaningful* triplets in first-appearance order."""
    seen = []
    for row in triplet_lists_pf(clip):
        for t in row:
            if t not in seen and _is_meaningful_triplet(t):
                seen.append(t)
    if not seen:
        return []
    v = pick_variant(rng, "Q11")
    parts = [L.natural_triplet(t) for t in seen]
    if len(parts) == 1:
        a = f"The only action in this clip is {parts[0]}."
    else:
        a = "The actions in this clip are, in order of first appearance: " \
            + ", ".join(parts) + "."
    return [make_qa(clip, "Q11", v, PHRASINGS["Q11"][v], a)]


def gen_q12(clip: dict, rng: random.Random) -> List[dict]:
    """Count of distinct triplets."""
    unique = set()
    for row in triplet_lists_pf(clip):
        for t in row:
            unique.add(t)
    n = len(unique)
    v = pick_variant(rng, "Q12")
    if n == 0:
        a = "0"
    elif n == 1:
        a = "1 action"
    else:
        a = f"{n} actions"
    return [make_qa(clip, "Q12", v, PHRASINGS["Q12"][v], a)]


def gen_q13(clip: dict, rng: random.Random) -> List[dict]:
    changed = bool(clip.get("has_triplet_change", False))
    v = pick_variant(rng, "Q13")
    a = ("Yes, the active triplets change during this clip."
         if changed else
         "No, the active triplets stay the same across the clip.")
    return [make_qa(clip, "Q13", v, PHRASINGS["Q13"][v], a,
                    binary_label=changed)]


def gen_q14(clip: dict, rng: random.Random) -> List[dict]:
    """Primary target = most frequent non-null target."""
    counts: Counter = Counter()
    for row in target_lists_pf(clip):
        for t in row:
            if t != "null_target":
                counts[t] += 1
    if not counts:
        return []
    primary = counts.most_common(1)[0][0]
    v = pick_variant(rng, "Q14")
    a = (f"The primary anatomical target in this clip is "
         f"{L.natural_target(primary)}.")
    return [make_qa(clip, "Q14", v, PHRASINGS["Q14"][v], a)]


def gen_q15(clip: dict, rng: random.Random) -> List[dict]:
    """Primary verb = most frequent non-null verb."""
    counts: Counter = Counter()
    for row in verb_lists_pf(clip):
        for v_name in row:
            if v_name != "null_verb":
                counts[v_name] += 1
    if not counts:
        return []
    primary = counts.most_common(1)[0][0]
    v = pick_variant(rng, "Q15")
    a = (f"The primary surgical maneuver in this clip is "
         f"{L.natural_verb(primary)}.")
    return [make_qa(clip, "Q15", v, PHRASINGS["Q15"][v], a)]


# ----- Group 4: temporal -------------------------------------------------

def gen_q16(clip: dict, rng: random.Random) -> List[dict]:
    """Instrument change around the first internal phase transition."""
    pfs = phase_names_pf(clip)
    pf = instrument_names_pf(clip)
    for i in range(1, len(pfs)):
        if pfs[i - 1] and pfs[i] and pfs[i - 1] != pfs[i]:
            # 5-frame window before and after.
            a_before = set()
            for j in range(max(0, i - 5), i):
                a_before.update(pf[j])
            a_after = set()
            for j in range(i, min(len(pf), i + 5)):
                a_after.update(pf[j])
            if a_before and a_after and a_before != a_after:
                # Prefer an instrument newly appearing after the transition.
                incoming = sorted(a_after - a_before)
                outgoing = sorted(a_before - a_after)
                i_after  = incoming[0] if incoming else sorted(a_after)[0]
                i_before = outgoing[0] if outgoing else sorted(a_before)[0]
                v = pick_variant(rng, "Q16")
                a = (f"When the phase changes from {L.natural_phase(pfs[i-1])} "
                     f"to {L.natural_phase(pfs[i])} at second {i}, the "
                     f"{L.natural_instrument(i_before)} is replaced by the "
                     f"{L.natural_instrument(i_after)}.")
                return [make_qa(clip, "Q16", v, PHRASINGS["Q16"][v], a)]
    return []


def gen_q17(clip: dict, rng: random.Random) -> List[dict]:
    """Idle frames: longest run of consecutive frames with no active instrument."""
    pf = instrument_names_pf(clip)
    runs = []
    start = None
    for sec, row in enumerate(pf):
        idle = len(row) == 0
        if idle and start is None:
            start = sec
        elif not idle and start is not None:
            runs.append((start, sec - 1))
            start = None
    if start is not None:
        runs.append((start, len(pf) - 1))
    v = pick_variant(rng, "Q17")
    if runs:
        longest = max(runs, key=lambda r: r[1] - r[0] + 1)
        dur = longest[1] - longest[0] + 1
        a = (f"Yes, there is a period of {dur} seconds with no active "
             f"instrument, from second {longest[0]} to second {longest[1]} "
             f"of the clip.")
        return [make_qa(clip, "Q17", v, PHRASINGS["Q17"][v], a,
                        binary_label=True)]
    else:
        a = "No, an instrument is active throughout the entire clip."
        return [make_qa(clip, "Q17", v, PHRASINGS["Q17"][v], a,
                        binary_label=False)]


def gen_q18(clip: dict, rng: random.Random) -> List[dict]:
    """Dominant *meaningful* triplet before vs. after the first internal
    triplet change.  Requires both sides to contain at least one meaningful
    triplet (neither side is entirely null_verb / null_target)."""
    rows = triplet_lists_pf(clip)
    for i in range(1, len(rows)):
        prev = set(rows[i - 1])
        curr = set(rows[i])
        if not (prev and curr and prev != curr):
            continue
        before_meaningful = sorted(t for t in prev if _is_meaningful_triplet(t))
        after_meaningful  = sorted(t for t in curr if _is_meaningful_triplet(t))
        if not (before_meaningful and after_meaningful):
            continue
        before = before_meaningful[0]
        after  = after_meaningful[0]
        if before == after:
            continue
        v = pick_variant(rng, "Q18")
        a = (f"Before the transition the action is "
             f"{L.natural_triplet(before)}; after the transition the "
             f"action is {L.natural_triplet(after)}.")
        return [make_qa(clip, "Q18", v, PHRASINGS["Q18"][v], a)]
    return []


# ===========================================================================
# Master generator per clip
# ===========================================================================

def generate_for_clip(clip: dict, seed: int,
                      video_instrument_vocab: List[str]) -> List[dict]:
    rng = make_rng(seed, clip["clip_id"])
    source = clip["source"]
    out: List[dict] = []
    out.extend(gen_q01(clip, rng))
    out.extend(gen_q02(clip, rng))
    out.extend(gen_q03(clip, rng))
    out.extend(gen_q04(clip, rng))
    out.extend(gen_q05(clip, rng))
    out.extend(gen_q06(clip, rng, video_instrument_vocab))
    out.extend(gen_q07(clip, rng))
    out.extend(gen_q08(clip, rng))
    out.extend(gen_q09(clip, rng))
    if source == "cholect50":
        out.extend(gen_q10(clip, rng))
        out.extend(gen_q11(clip, rng))
        out.extend(gen_q12(clip, rng))
        out.extend(gen_q13(clip, rng))
        out.extend(gen_q14(clip, rng))
        out.extend(gen_q15(clip, rng))
    out.extend(gen_q16(clip, rng))
    out.extend(gen_q17(clip, rng))
    if source == "cholect50":
        out.extend(gen_q18(clip, rng))
    return out


# ===========================================================================
# Binary balancing (strict 50/50 per Q-type)
# ===========================================================================

def balance_binaries(all_pairs: List[dict], rng: random.Random
                      ) -> Tuple[List[dict], Dict[str, dict]]:
    non_binary = [p for p in all_pairs if p["question_type"] not in BINARY_TYPES]
    by_qt: Dict[str, List[dict]] = {}
    for p in all_pairs:
        if p["question_type"] in BINARY_TYPES:
            by_qt.setdefault(p["question_type"], []).append(p)

    balanced = list(non_binary)
    report: Dict[str, dict] = {}
    for qt, rows in by_qt.items():
        yes = [r for r in rows if r["binary_label"] is True]
        no  = [r for r in rows if r["binary_label"] is False]
        n_y, n_n = len(yes), len(no)
        k = min(n_y, n_n)
        rng.shuffle(yes); rng.shuffle(no)
        kept_yes = yes[:k]
        kept_no  = no[:k]
        balanced.extend(kept_yes)
        balanced.extend(kept_no)
        report[qt] = {
            "raw_yes": n_y, "raw_no": n_n,
            "kept_yes": len(kept_yes), "kept_no": len(kept_no),
        }
        logger.info(
            f"  {qt}: {n_y} yes / {n_n} no → kept {k}/{k} (50/50)"
        )
    return balanced, report


# ===========================================================================
# Clip discovery
# ===========================================================================

def discover_clip_files(clips_root: str) -> List[str]:
    """Return sorted list of clip-metadata JSONs (clip_XXXX.json, not _qa/_cot)."""
    out = []
    for source in ("cholec80", "cholect50"):
        src_root = os.path.join(clips_root, source)
        if not os.path.isdir(src_root):
            continue
        for vdir in sorted(os.listdir(src_root)):
            vpath = os.path.join(src_root, vdir)
            if not os.path.isdir(vpath) or vdir.startswith("_"):
                continue
            for fn in sorted(os.listdir(vpath)):
                if (fn.startswith("clip_") and fn.endswith(".json")
                        and "_qa" not in fn and "_cot" not in fn):
                    out.append(os.path.join(vpath, fn))
    return out


def build_video_instrument_vocab(clip_files: List[str]) -> Dict[str, List[str]]:
    """For each video (key = "{source}/{video_id}"), return the union of
    instruments seen in any clip of that video — used as hard-negative source
    for Q06."""
    vocab: Dict[str, set] = {}
    for cp in clip_files:
        d = json.load(open(cp))
        key = f"{d['source']}/{d['video_id']}"
        s = vocab.setdefault(key, set())
        for row in d.get("instrument_per_frame", []):
            if row:
                s.update(row)
    return {k: sorted(v) for k, v in vocab.items()}


# ===========================================================================
# Mock
# ===========================================================================

def run_mock_smoke_test() -> None:
    logger.info("MOCK smoke test — synthetic clip")
    clip = {
        "clip_id": "mock_video_clip_0001",
        "source":  "cholect50",
        "video_id": "mock_video",
        "fps": 1,
        "clip_duration_seconds": 30,
        "clip_start_frame": 0,
        "clip_end_frame":   29,
        "frame_indices":    list(range(30)),
        "frames":           [f"/nonexistent/{i:06d}.png" for i in range(30)],
        "is_transition_clip": True,
        "has_phase_change":   True,
        "has_triplet_change": True,
        "phase_before": "preparation",
        "phase_after":  "calot_triangle_dissection",
        "phase_per_frame":
            ["preparation"] * 15 + ["calot_triangle_dissection"] * 15,
        # Instrument set changes at second 15 (aligned with phase change) so
        # Q16 has a qualifying event.
        "instrument_per_frame":
            [["grasper"]] * 15 + [["hook"]] * 15,
        "triplet_per_frame":
            [["grasper_grasp_gallbladder"]] * 15 +
            [["hook_dissect_cystic_plate"]] * 15,
        "verb_per_frame":
            [["grasp"]] * 15 + [["dissect"]] * 15,
        "target_per_frame":
            [["gallbladder"]] * 15 + [["cystic_plate"]] * 15,
        "tool_vocabulary": list(L.INSTRUMENTS),
    }
    pairs = generate_for_clip(clip, seed=42,
                               video_instrument_vocab=["grasper", "hook"])
    types = Counter(p["question_type"] for p in pairs)
    logger.info(f"  mock generated {len(pairs)} pairs: {dict(types)}")
    for p in pairs:
        assert "question" in p and "answer" in p, p
        assert len(p["answer"]) > 0
    assert types["Q01"] == 1
    assert types["Q02"] == 1
    assert types["Q03"] == 1
    assert types["Q04"] == 1
    assert types["Q06"] == 2           # 1 pos + 1 neg
    assert types["Q09"] == 2           # 2 of 3 temporal thirds sampled
    assert types["Q10"] == 1
    assert types["Q13"] == 1
    assert types["Q14"] == 1
    assert types["Q15"] == 1
    assert types["Q16"] == 1
    assert types["Q18"] == 1
    logger.info("  mock: smoke test passed")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based QA generator for SurgQ-VLM."
    )
    parser.add_argument("--clips_root", type=str,
                        default="datasets/clips")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Process only clip files belonging to these video IDs.")
    parser.add_argument("--mock",    action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate everything but do not write output files.")
    args = parser.parse_args()

    if args.mock:
        run_mock_smoke_test()
        return

    log_path = setup_file_logging(args.clips_root)
    logger.debug(f"CLI args: {vars(args)}")

    clip_files = discover_clip_files(args.clips_root)
    if args.videos:
        keep = set(args.videos)
        clip_files = [c for c in clip_files
                      if any(f"/{v}/" in c for v in keep)]
    if not clip_files:
        logger.error("No clip files found.")
        sys.exit(1)
    logger.info(f"Found {len(clip_files)} clip metadata files")

    # Per-video instrument vocab for Q06 hard-negatives.
    logger.info("Building per-video instrument vocabularies ...")
    video_vocab = build_video_instrument_vocab(clip_files)
    logger.info(f"  vocab for {len(video_vocab)} video(s) built")

    # Pass 1: generate raw pairs per clip.
    clip_to_path: Dict[str, str] = {}
    raw_pairs: List[dict] = []
    for cp in clip_files:
        clip = json.load(open(cp))
        key  = f"{clip['source']}/{clip['video_id']}"
        pairs = generate_for_clip(clip, args.seed, video_vocab.get(key, []))
        raw_pairs.extend(pairs)
        qa_path = cp.replace(".json", "_qa.json")
        clip_to_path[clip["clip_id"]] = qa_path

    logger.info(f"Raw QA pairs: {len(raw_pairs)}")

    # Pass 2: strict 50/50 binary balancing.
    logger.info("Binary balancing (strict 50/50):")
    rng_bal = random.Random(args.seed)
    balanced, bal_report = balance_binaries(raw_pairs, rng_bal)
    logger.info(f"After balancing: {len(balanced)} pairs "
                f"({len(raw_pairs) - len(balanced)} removed)")

    # Pass 3: write per-clip QA JSONs and per-source merged files.
    if not args.dry_run:
        # Delete old _qa.json files under clips_root first.
        logger.info("Cleaning old *_qa.json files ...")
        removed = 0
        for src in ("cholec80", "cholect50"):
            sroot = os.path.join(args.clips_root, src)
            if not os.path.isdir(sroot):
                continue
            for vdir in os.listdir(sroot):
                vpath = os.path.join(sroot, vdir)
                if not os.path.isdir(vpath):
                    continue
                for fn in os.listdir(vpath):
                    if fn.endswith("_qa.json"):
                        os.remove(os.path.join(vpath, fn))
                        removed += 1
        logger.info(f"  removed {removed} old QA files")

        # Group balanced pairs by clip_id.
        by_clip: Dict[str, list] = {}
        for p in balanced:
            by_clip.setdefault(p["clip_id"], []).append(p)
        written = 0
        for cid, qa_path in clip_to_path.items():
            pairs = by_clip.get(cid, [])
            with open(qa_path, "w") as f:
                json.dump(pairs, f, indent=2)
            written += 1
        logger.info(f"Wrote {written} per-clip QA files "
                    f"({len(balanced)} total pairs)")

        # Per-source merged files.
        for src in ("cholec80", "cholect50"):
            merged = [p for p in balanced if p["source"] == src]
            out = os.path.join(args.clips_root, src, "qa_all.json")
            with open(out, "w") as f:
                json.dump(merged, f, indent=2)
            logger.info(f"  {src} merged: {len(merged)} pairs -> {out}")

        # Summary.
        type_counts   = Counter(p["question_type"]  for p in balanced)
        group_counts  = Counter(p["question_group"] for p in balanced)
        format_counts = Counter(p["answer_format"]  for p in balanced)
        novel_count   = sum(1 for p in balanced if p["novel_vs_sota"])
        temporal_cnt  = sum(1 for p in balanced if p["temporal_grounding"])
        polarity = {}
        for qt in BINARY_TYPES:
            subset = [p for p in balanced if p["question_type"] == qt]
            y = sum(1 for p in subset if p["binary_label"] is True)
            n = sum(1 for p in subset if p["binary_label"] is False)
            polarity[qt] = {"yes": y, "no": n, "total": y + n,
                             "yes_ratio": round(y / max(1, y + n), 3)}
        summary = OrderedDict([
                        ("total_clips",               len(clip_files)),
            ("total_pairs_raw",           len(raw_pairs)),
            ("total_pairs_balanced",      len(balanced)),
            ("avg_pairs_per_clip",
             round(len(balanced) / max(1, len(clip_files)), 2)),
            ("by_question_type",          dict(sorted(type_counts.items()))),
            ("by_question_group",         dict(sorted(group_counts.items()))),
            ("by_answer_format",          dict(sorted(format_counts.items()))),
            ("novel_pairs",               novel_count),
            ("temporal_grounded_pairs",   temporal_cnt),
            ("binary_polarity",           polarity),
            ("balance_report",            bal_report),
            ("config", {
                "seed":        args.seed,
                "clips_root":  args.clips_root,
            }),
        ])
        out = os.path.join(args.clips_root, "qa_summary.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary: {out}")

    logger.info(f"\n{'='*60}")
    logger.info(f"QA GENERATION COMPLETE")
    logger.info(f"  Clips:              {len(clip_files)}")
    logger.info(f"  Raw pairs:          {len(raw_pairs)}")
    logger.info(f"  Balanced pairs:     {len(balanced)}")
    logger.info(f"  Avg pairs/clip:     "
                f"{round(len(balanced) / max(1, len(clip_files)), 2)}")
    logger.info(f"  Log:                {log_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
