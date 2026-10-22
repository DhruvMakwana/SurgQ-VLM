#!/usr/bin/env python3
"""
extract_clips.py — Unified 30-second clip extractor for SurgQ-VLM.
========================================================================

Produces fixed 30-second clips at 1 FPS for both Cholec80 and CholecT50.

Design
------
  * Fixed 30-second window.
  * Native-FPS transition detection — Cholec80 transitions are detected on
    the 25 FPS phase stream and snapped to the first 1-FPS second that
    reflects the new phase, preserving sub-second boundary precision.
  * Canonical vocabulary — every phase/instrument/verb/target/triplet string
    written to the clip JSON is the canonical snake_case form defined in
    ``labels_canonical.py``, eliminating cross-dataset label mismatches.
  * CholecT50 ``triplet_before`` / ``triplet_after`` are the full
    canonical-string lists; the symmetric difference (outgoing / incoming)
    is recorded separately.
  * Two clip classes:
      - Transition clip (TC): one per phase or triplet transition, window
        centred on the transition second.
      - Stable clip (SC): uniformly sampled every ``stable_interval`` seconds
        inside single-phase regions.
  * Deterministic ordering — clips are sorted by start frame per video.

Output layout
-------------
  output_root/
    {video_id}/
      clip_0001.json
      clip_0002.json
      …
      summary.json      ← per-video summary
    extract_clips.log
    summary.json        ← global summary across all videos

Clip JSON schema
----------------
  clip_id                str    "{video_id}_clip_{NNNN}"
  source                 str    "cholec80" | "cholect50"
  video_id               str
  fps                    int    always 1
  clip_duration_seconds  int    always 30 (actual frame count)
  clip_duration_type     str    always "standard"        (kept for back-compat)
  clip_start_frame       int    0-based inclusive
  clip_end_frame         int    0-based inclusive
  frame_indices          list   [start, …, end]
  frames                 list   absolute PNG paths
  is_transition_clip     bool
  has_phase_change       bool
  has_triplet_change     bool
  transition_frame       int?   clip-local second where the transition lands
                                (always equals 15 for interior transitions;
                                 may differ if the window was clamped)
  transition_type        str?   "phase" | "triplet" | "both"
  phase_before           str?   canonical phase name (or None)
  phase_after            str?
  triplet_before         list?  canonical triplet strings present at the
                                frame immediately before the transition
  triplet_after          list?  canonical triplet strings present at the
                                frame immediately after the transition
  triplet_outgoing       list?  triplet_before − triplet_after (set difference)
  triplet_incoming       list?  triplet_after  − triplet_before
  phase_per_frame        list   canonical phase string per frame (or None)
  triplet_per_frame      list   list-per-frame of canonical triplet strings
                                (or None if no triplet is active)
  instrument_per_frame   list   list-per-frame of canonical instrument names
                                (or None if no instrument is active)
  verb_per_frame         list   list-per-frame of canonical verb names
                                (or None if no verb is active)
  target_per_frame       list   list-per-frame of canonical target names
                                (or None if no target is active)
  tool_vocabulary        list   full canonical INSTRUMENTS list (7 entries)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

# Local import — same directory.
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
    log_path = os.path.join(output_root, "extract_clips.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    logger.addHandler(fh)
    logger.info(f"Logging to file: {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIP_DURATION_S    = 30     # fixed clip length in seconds / 1-FPS frames
STABLE_INTERVAL_S  = 45     # spacing between consecutive stable-clip centres
MIN_GAP_S          = 30     # min gap between TC centres (= CLIP_DURATION_S;
                            # ensures transition clips do not overlap each other)
CHOLEC80_FPS       = 25     # original phase-file frame rate
HALF_WINDOW_S      = CLIP_DURATION_S // 2   # =15

# ===========================================================================
# Cholec80 loaders
# ===========================================================================

def load_cholec80_phases_native(dataset_root: str, video_id: str) -> List[str]:
    """Return the Cholec80 phase stream at native 25 FPS, one canonical
    phase string per original frame (row).  Header is skipped."""
    path = os.path.join(dataset_root, "phase_annotations",
                        f"{video_id}-phase.txt")
    phases = []
    with open(path) as f:
        f.readline()  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, raw = line.split("\t")
            phases.append(L.canonicalize_phase(raw))
    return phases


def load_cholec80_tools(dataset_root: str, video_id: str
                        ) -> Tuple[List[str], List[Dict[str, int]]]:
    """Return (tool_names_canonical, per-1fps-second dicts).  Tool annotations
    in Cholec80 are already stored at 1 FPS (one row per extracted second)."""
    path = os.path.join(dataset_root, "tool_annotations",
                        f"{video_id}-tool.txt")
    with open(path) as f:
        header     = f.readline().strip().split("\t")
        raw_names  = header[1:]  # drop "Frame"
        tool_names = [L.canonicalize_instrument(n) for n in raw_names]
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")[1:]  # drop frame column
            rows.append({name: int(v) for name, v in zip(tool_names, parts)})
    return tool_names, rows


# ===========================================================================
# CholecT50 loaders
# ===========================================================================

IDX_TRIPLET_ID    = 0
IDX_INSTRUMENT_ID = 1
IDX_VERB_ID       = 7
IDX_TARGET_ID     = 8
IDX_PHASE_ID      = 14
NULL_VALUE        = -1


def load_label_mapping(dataset_root: str) -> Dict[int, Dict[str, int]]:
    """Load ``label_mapping.txt`` → {triplet_id: {instrument_id, verb_id,
    target_id}}.  Used to decode triplet IDs to (i,v,t) components."""
    path = os.path.join(dataset_root, "label_mapping.txt")
    mapping: Dict[int, Dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            mapping[int(parts[0])] = {
                "instrument_id": int(parts[1]),
                "verb_id":       int(parts[2]),
                "target_id":     int(parts[3]),
            }
    return mapping


def load_cholect50_video(dataset_root: str, video_id: str) -> dict:
    with open(os.path.join(dataset_root, "labels", f"{video_id}.json")) as f:
        return json.load(f)


# ===========================================================================
# Unified frame-table construction
# ===========================================================================

def build_frame_table_cholec80(phases_1fps: List[str],
                                tool_rows: List[Dict[str, int]],
                                tool_names: List[str]) -> List[dict]:
    """Align 1-FPS phase and tool streams; produce one entry per extracted
    frame with canonical labels.  Cholec80 has no triplets/verbs/targets."""
    n = min(len(phases_1fps), len(tool_rows))
    if len(phases_1fps) != len(tool_rows):
        logger.debug(
            f"  phase entries ({len(phases_1fps)}) != tool entries "
            f"({len(tool_rows)}); using min={n}"
        )
    table = []
    for i in range(n):
        active = sorted(name for name in tool_names
                        if tool_rows[i].get(name, 0) == 1)
        table.append({
            "phase":       phases_1fps[i],
            "instruments": active,
            "triplets":    [],
            "verbs":       [],
            "targets":     [],
        })
    return table


def build_frame_table_cholect50(video_data: dict,
                                 label_map: Dict[int, Dict[str, int]]
                                 ) -> List[dict]:
    """Parse the CholecT50 annotation dict into canonical per-frame entries.

    Missing frames (annotations dict doesn't cover every frame index) are
    filled with empty entries so the table is dense from 0 to num_frames-1.
    """
    categories = video_data["categories"]
    cat_instr  = {int(k): v for k, v in categories["instrument"].items()}
    cat_verb   = {int(k): v for k, v in categories["verb"].items()}
    cat_target = {int(k): v for k, v in categories["target"].items()}
    cat_phase  = {int(k): v for k, v in categories["phase"].items()}

    num_frames = video_data["num_frames"]
    table: List[Optional[dict]] = [None] * num_frames

    for frame_str, instances in video_data["annotations"].items():
        fidx = int(frame_str)
        if not (0 <= fidx < num_frames):
            continue

        phase_canon: Optional[str] = None
        instr_set: set  = set()
        verb_set: set   = set()
        target_set: set = set()
        triplet_set: set = set()

        for inst in instances:
            p_id = int(inst[IDX_PHASE_ID])
            t_id = int(inst[IDX_TRIPLET_ID])
            i_id = int(inst[IDX_INSTRUMENT_ID])
            v_id = int(inst[IDX_VERB_ID])
            g_id = int(inst[IDX_TARGET_ID])

            if p_id != NULL_VALUE and phase_canon is None:
                phase_canon = L.canonicalize_phase(cat_phase[p_id])
            if i_id != NULL_VALUE:
                instr_set.add(L.canonicalize_instrument(cat_instr[i_id]))
            if v_id != NULL_VALUE:
                verb_set.add(L.canonicalize_verb(cat_verb[v_id]))
            if g_id != NULL_VALUE:
                target_set.add(L.canonicalize_target(cat_target[g_id]))
            if t_id != NULL_VALUE and t_id in label_map:
                m = label_map[t_id]
                triplet_set.add(L.canonicalize_triplet(
                    cat_instr[m["instrument_id"]],
                    cat_verb[m["verb_id"]],
                    cat_target[m["target_id"]],
                ))
            elif t_id != NULL_VALUE:
                # Unknown triplet ID — record as bare token.
                triplet_set.add(f"triplet_{t_id}")

        table[fidx] = {
            "phase":       phase_canon,
            "instruments": sorted(instr_set),
            "triplets":    sorted(triplet_set),
            "verbs":       sorted(verb_set),
            "targets":     sorted(target_set),
        }

    # Fill remaining gaps.
    for fidx in range(num_frames):
        if table[fidx] is None:
            table[fidx] = {
                "phase": None, "instruments": [], "triplets": [],
                "verbs": [], "targets": [],
            }
    return table  # type: ignore[return-value]


# ===========================================================================
# Transition detection
# ===========================================================================

def detect_cholec80_transitions_25fps_to_1fps(phases_25fps: List[str],
                                               num_1fps_frames: int
                                               ) -> List[dict]:
    """Detect every phase-change frame at native 25 FPS, then map to the
    first 1-FPS second that reflects the new phase.

    Mapping rule: if the new phase first appears at original frame ``f``,
    the first 1-FPS frame to show that phase is ``ceil(f / 25)``.
    """
    events = []
    for f in range(1, len(phases_25fps)):
        prev = phases_25fps[f - 1]
        curr = phases_25fps[f]
        if prev != curr:
            sec_1fps = math.ceil(f / CHOLEC80_FPS)
            if sec_1fps < num_1fps_frames:
                events.append({
                    "second":        sec_1fps,
                    "type":          "phase",
                    "phase_before":  prev,
                    "phase_after":   curr,
                    "triplet_before": None,
                    "triplet_after":  None,
                })
    # Deduplicate by 1-FPS second (many 25-FPS frames may map to same second).
    dedup = {}
    for e in events:
        dedup.setdefault(e["second"], e)
    return sorted(dedup.values(), key=lambda e: e["second"])


def detect_cholect50_transitions(frame_table: List[dict]) -> List[dict]:
    """Scan consecutive 1-FPS frames for phase or triplet-set changes.

    * Phase change: both sides non-null and differ.
    * Triplet change: both sides non-empty and the sorted tuple differs.

    The event's ``triplet_before`` / ``triplet_after`` are the full sorted
    triplet lists at the two frames, and ``triplet_outgoing`` /
    ``triplet_incoming`` record the symmetric-difference (fixes the v1 bug
    that took only the first sorted element).
    """
    events = []
    for i in range(1, len(frame_table)):
        prev = frame_table[i - 1]
        curr = frame_table[i]

        phase_changed = bool(
            prev["phase"] is not None and curr["phase"] is not None
            and prev["phase"] != curr["phase"]
        )
        t_before = tuple(prev["triplets"])
        t_after  = tuple(curr["triplets"])
        triplet_changed = bool(t_before and t_after and t_before != t_after)

        if not (phase_changed or triplet_changed):
            continue

        t_before_set = set(t_before)
        t_after_set  = set(t_after)
        outgoing = sorted(t_before_set - t_after_set)
        incoming = sorted(t_after_set - t_before_set)

        ttype = (
            "both" if (phase_changed and triplet_changed)
            else ("phase" if phase_changed else "triplet")
        )
        events.append({
            "second":         i,
            "type":           ttype,
            "phase_before":   prev["phase"] if phase_changed else None,
            "phase_after":    curr["phase"] if phase_changed else None,
            "triplet_before": list(t_before) if triplet_changed else None,
            "triplet_after":  list(t_after)  if triplet_changed else None,
            "triplet_outgoing": outgoing    if triplet_changed else None,
            "triplet_incoming": incoming    if triplet_changed else None,
        })
    return events


# ===========================================================================
# Clip window computation
# ===========================================================================

def compute_window(centre: int, num_frames: int) -> Tuple[int, int]:
    """Return (start, end_exclusive) for a 30-frame window centred on
    ``centre``, clamped to [0, num_frames).  Guarantees window width stays
    equal to ``CLIP_DURATION_S`` whenever the video is long enough; for
    shorter videos the window is shrunk to fit."""
    start = centre - HALF_WINDOW_S
    end   = start + CLIP_DURATION_S
    if start < 0:
        start = 0
        end   = min(CLIP_DURATION_S, num_frames)
    if end > num_frames:
        end   = num_frames
        start = max(num_frames - CLIP_DURATION_S, 0)
    return start, end


# ===========================================================================
# Clip generation — transition pass then stable pass
# ===========================================================================

def generate_transition_clips(events: List[dict],
                               num_frames: int,
                               frame_table: List[dict]
                               ) -> List[dict]:
    """Emit one TC per transition event, in input order, skipping events
    whose centre sits within ``MIN_GAP_S`` of an already-accepted centre."""
    clips: List[dict] = []
    used: List[int]   = []
    for ev in events:
        c = ev["second"]
        if any(abs(c - u) < MIN_GAP_S for u in used):
            continue
        # Additionally skip events that fall within the first / last
        # HALF_WINDOW_S seconds if the video is too short for a centred 30-s
        # window — we *still* emit the clip (clamped) because excluding
        # near-edge transitions would bias the dataset.  Only skip if the
        # video itself is shorter than one clip.
        if num_frames < CLIP_DURATION_S:
            continue
        start, end = compute_window(c, num_frames)
        clips.append({
            "centre":        c,
            "start":         start,
            "end":           end,
            "is_transition": True,
            "event":         ev,
        })
        used.append(c)
    return clips


def generate_stable_clips(frame_table: List[dict],
                           num_frames: int,
                           used_centres: List[int]) -> List[dict]:
    """Uniformly sample SC clips every ``STABLE_INTERVAL_S`` seconds in
    regions where (a) the phase is constant across the 30-s window, (b) at
    least one frame contains an active instrument, and (c) no already-used
    centre sits within ``MIN_GAP_S``."""
    if num_frames < CLIP_DURATION_S:
        return []

    forbidden: set = set()
    for c in used_centres:
        for f in range(c - MIN_GAP_S + 1, c + MIN_GAP_S):
            forbidden.add(f)

    clips: List[dict] = []
    candidate = HALF_WINDOW_S
    last_centre = candidate - STABLE_INTERVAL_S
    while candidate < num_frames - HALF_WINDOW_S:
        if candidate in forbidden:
            candidate += 1
            continue
        start, end = compute_window(candidate, num_frames)

        phases_in_window = {frame_table[f]["phase"]
                             for f in range(start, end)}
        has_any_instr = any(frame_table[f]["instruments"]
                             for f in range(start, end))

        # Treat uniformly-unknown phase as a single "phase" (Cholec80 is fully
        # annotated anyway; CholecT50 may have gaps).
        single_phase = len(phases_in_window - {None}) <= 1

        if single_phase and has_any_instr \
                and candidate - last_centre >= STABLE_INTERVAL_S:
            clips.append({
                "centre":        candidate,
                "start":         start,
                "end":           end,
                "is_transition": False,
                "event":         None,
            })
            last_centre = candidate
            for f in range(candidate - MIN_GAP_S + 1,
                           candidate + MIN_GAP_S):
                forbidden.add(f)
            candidate += STABLE_INTERVAL_S
        else:
            candidate += 1
    return clips


# ===========================================================================
# Frame path resolution
# ===========================================================================

def resolve_frame_path(source: str, dataset_root: str, video_id: str,
                        fidx: int) -> str:
    if source == "cholec80":
        # Cholec80 frames are 1-based on disk: frame_idx 0 -> *_000001.png
        return os.path.join(dataset_root, "frames", video_id,
                            f"{video_id}_{fidx + 1:06d}.png")
    # CholecT50: 0-based, 6-digit.
    return os.path.join(dataset_root, "videos", video_id, f"{fidx:06d}.png")


# ===========================================================================
# Per-clip JSON assembly
# ===========================================================================

def assemble_clip_json(clip: dict,
                        video_id: str,
                        source: str,
                        dataset_root: str,
                        frame_table: List[dict],
                        clip_idx: int) -> dict:
    start = clip["start"]
    end   = clip["end"]
    frame_indices = list(range(start, end))
    frames = [resolve_frame_path(source, dataset_root, video_id, f)
              for f in frame_indices]

    # Per-frame arrays, all from canonical frame_table.
    phase_pf   = []
    triplet_pf = []
    instr_pf   = []
    verb_pf    = []
    target_pf  = []
    all_phases:  set = set()
    for f in frame_indices:
        entry = frame_table[f] if f < len(frame_table) else {
            "phase": None, "instruments": [], "triplets": [],
            "verbs": [], "targets": [],
        }
        phase_pf.append(entry["phase"])
        if entry["phase"] is not None:
            all_phases.add(entry["phase"])
        triplet_pf.append(entry["triplets"] if entry["triplets"] else None)
        instr_pf.append(entry["instruments"] if entry["instruments"] else None)
        verb_pf.append(entry["verbs"] if entry["verbs"] else None)
        target_pf.append(entry["targets"] if entry["targets"] else None)

    # has_triplet_change: sequential set comparison within the clip.
    has_triplet_change = False
    prev_set = None
    for f in frame_indices:
        cs = frozenset(frame_table[f]["triplets"]
                       if f < len(frame_table) else [])
        if prev_set is not None and cs != prev_set:
            has_triplet_change = True
            break
        prev_set = cs

    has_phase_change = len(all_phases) > 1

    # Transition metadata (clip-local second).
    ev = clip["event"]
    if ev is not None:
        # Centre frame (global) translated to clip-local second.
        local_sec = ev["second"] - start
        trans_frame = local_sec if 0 <= local_sec < (end - start) else None
        trans_type = ev["type"]
        phase_before = ev.get("phase_before")
        phase_after  = ev.get("phase_after")
        triplet_before = ev.get("triplet_before")
        triplet_after  = ev.get("triplet_after")
        triplet_outgoing = ev.get("triplet_outgoing")
        triplet_incoming = ev.get("triplet_incoming")
    else:
        trans_frame = None
        trans_type  = None
        phase_before = phase_after = None
        triplet_before = triplet_after = None
        triplet_outgoing = triplet_incoming = None

    clip_id = f"{video_id}_clip_{clip_idx:04d}"

    return OrderedDict([
        ("clip_id",                clip_id),
        ("source",                 source),
        ("video_id",               video_id),
        ("fps",                    1),
        ("clip_duration_seconds",  len(frame_indices)),
        ("clip_duration_type",     "standard"),
        ("clip_start_frame",       start),
        ("clip_end_frame",         end - 1),
        ("frame_indices",          frame_indices),
        ("frames",                 frames),
        ("is_transition_clip",     clip["is_transition"]),
        ("has_phase_change",       has_phase_change),
        ("has_triplet_change",     has_triplet_change),
        ("transition_frame",       trans_frame),
        ("transition_type",        trans_type),
        ("phase_before",           phase_before),
        ("phase_after",            phase_after),
        ("triplet_before",         triplet_before),
        ("triplet_after",          triplet_after),
        ("triplet_outgoing",       triplet_outgoing),
        ("triplet_incoming",       triplet_incoming),
        ("phase_per_frame",        phase_pf),
        ("triplet_per_frame",      triplet_pf),
        ("instrument_per_frame",   instr_pf),
        ("verb_per_frame",         verb_pf),
        ("target_per_frame",       target_pf),
        ("tool_vocabulary",        list(L.INSTRUMENTS)),
    ])


# ===========================================================================
# Per-video driver
# ===========================================================================

def process_cholec80_video(video_id: str,
                            dataset_root: str,
                            output_root: str,
                            dry_run: bool) -> dict:
    try:
        phases_25fps = load_cholec80_phases_native(dataset_root, video_id)
        tool_names, tool_rows = load_cholec80_tools(dataset_root, video_id)
    except FileNotFoundError as e:
        logger.warning(f"Missing file for {video_id}: {e} — skipping")
        return {"video_id": video_id, "status": "skipped", "reason": str(e)}

    # Subsample phases to 1 FPS for the per-frame table.
    phases_1fps = [phases_25fps[i]
                   for i in range(0, len(phases_25fps), CHOLEC80_FPS)]
    frame_table = build_frame_table_cholec80(phases_1fps, tool_rows, tool_names)
    num_frames  = len(frame_table)

    events = detect_cholec80_transitions_25fps_to_1fps(phases_25fps, num_frames)
    logger.info(
        f"  {video_id}: {num_frames} frames @ 1 FPS  "
        f"({len(phases_25fps)} @ 25 FPS), {len(events)} phase transitions"
    )

    return _process_common(
        video_id, "cholec80", dataset_root, output_root,
        frame_table, events, num_frames, dry_run,
    )


def process_cholect50_video(video_id: str,
                              dataset_root: str,
                              output_root: str,
                              label_map: Dict[int, Dict[str, int]],
                              dry_run: bool) -> dict:
    try:
        video_data  = load_cholect50_video(dataset_root, video_id)
    except FileNotFoundError:
        logger.warning(f"No label file for {video_id} — skipping")
        return {"video_id": video_id, "status": "skipped",
                "reason": "no_label_file"}

    frame_table = build_frame_table_cholect50(video_data, label_map)
    num_frames  = len(frame_table)
    events = detect_cholect50_transitions(frame_table)
    logger.info(
        f"  {video_id}: {num_frames} frames, {len(events)} transitions"
    )
    return _process_common(
        video_id, "cholect50", dataset_root, output_root,
        frame_table, events, num_frames, dry_run,
    )


def _process_common(video_id: str, source: str, dataset_root: str,
                     output_root: str,
                     frame_table: List[dict], events: List[dict],
                     num_frames: int, dry_run: bool) -> dict:
    tc = generate_transition_clips(events, num_frames, frame_table)
    used = [c["centre"] for c in tc]
    sc = generate_stable_clips(frame_table, num_frames, used)

    all_clips = sorted(tc + sc, key=lambda c: c["start"])
    logger.info(f"  {video_id}: {len(tc)} TC + {len(sc)} SC = "
                f"{len(all_clips)} clips")

    output_dir = os.path.join(output_root, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # Remove any stale clip_XXXX*.json files from a previous run.  We only
    # touch files that match the naming pattern; user-authored files and the
    # top-level log are left alone.
    if not dry_run:
        for fn in os.listdir(output_dir):
            if fn.startswith("clip_") and fn.endswith(".json"):
                os.remove(os.path.join(output_dir, fn))

    clip_summaries = []
    for idx, clip in enumerate(all_clips, start=1):
        cj = assemble_clip_json(
            clip, video_id, source, dataset_root, frame_table, idx
        )
        if not dry_run:
            out_path = os.path.join(output_dir, f"clip_{idx:04d}.json")
            with open(out_path, "w") as f:
                json.dump(cj, f, indent=2)
        clip_summaries.append({
            "clip_id":            cj["clip_id"],
            "clip_file":          f"clip_{idx:04d}.json",
            "start_frame":        cj["clip_start_frame"],
            "end_frame":          cj["clip_end_frame"],
            "is_transition_clip": cj["is_transition_clip"],
            "has_phase_change":   cj["has_phase_change"],
            "has_triplet_change": cj["has_triplet_change"],
            "transition_type":    cj["transition_type"],
        })

    video_summary = OrderedDict([
        ("video_id",          video_id),
        ("source",            source),
        ("num_frames",        num_frames),
        ("total_clips",       len(clip_summaries)),
        ("transition_clips",  sum(1 for c in clip_summaries
                                  if c["is_transition_clip"])),
        ("stable_clips",      sum(1 for c in clip_summaries
                                  if not c["is_transition_clip"])),
        ("raw_transitions_detected", len(events)),
        ("clips",             clip_summaries),
    ])
    if not dry_run:
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(video_summary, f, indent=2)

    return video_summary


# ===========================================================================
# Video discovery
# ===========================================================================

def discover_cholec80_videos(dataset_root: str) -> List[str]:
    pdir = os.path.join(dataset_root, "phase_annotations")
    return sorted(f.replace("-phase.txt", "")
                  for f in os.listdir(pdir) if f.endswith("-phase.txt"))


def discover_cholect50_videos(dataset_root: str) -> List[str]:
    ldir = os.path.join(dataset_root, "labels")
    return sorted(f.replace(".json", "")
                  for f in os.listdir(ldir) if f.endswith(".json"))


# ===========================================================================
# Mock mode — purely synthetic, no disk access
# ===========================================================================

def run_mock_smoke_test() -> None:
    """Build a fake 60-frame Cholec80-style frame table with one phase
    transition and two instruments, run the full clip pipeline, and verify
    output invariants."""
    logger.info("MOCK smoke test — synthetic 60-frame video")
    # Build 25-FPS phase stream: preparation for 40 orig-frames, then calot.
    phases_25fps = ["Preparation"] * 40 + ["CalotTriangleDissection"] * 1460
    phases_25fps = [L.canonicalize_phase(p) for p in phases_25fps]
    # 1-FPS = 60 frames (1500 / 25).
    phases_1fps = phases_25fps[::CHOLEC80_FPS]
    tool_rows = []
    for i in range(60):
        r = {name: 0 for name in L.INSTRUMENTS}
        r["grasper"] = 1 if i < 30 else 0
        r["hook"]    = 1 if i >= 20 else 0
        tool_rows.append(r)
    frame_table = build_frame_table_cholec80(phases_1fps, tool_rows,
                                              list(L.INSTRUMENTS))
    events = detect_cholec80_transitions_25fps_to_1fps(
        phases_25fps, len(frame_table)
    )
    logger.info(f"  mock: {len(events)} transitions detected: {events}")
    tc = generate_transition_clips(events, len(frame_table), frame_table)
    sc = generate_stable_clips(frame_table, len(frame_table),
                                [c["centre"] for c in tc])
    logger.info(f"  mock: TC={len(tc)}  SC={len(sc)}")
    assert len(tc) >= 1, "expected at least one transition clip"
    # Assemble and check JSON.
    clip_json = assemble_clip_json(tc[0], "mock_video", "cholec80",
                                    "/nonexistent", frame_table, 1)
    assert clip_json["clip_duration_seconds"] == 30
    assert clip_json["is_transition_clip"] is True
    assert clip_json["has_phase_change"] is True
    assert clip_json["phase_before"] == "preparation"
    assert clip_json["phase_after"]  == "calot_triangle_dissection"
    assert len(clip_json["phase_per_frame"]) == 30
    logger.info("  mock: smoke test passed")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="30-second clip extractor for SurgQ-VLM."
    )
    parser.add_argument("--dataset", choices=["cholec80", "cholect50", "both"],
                        default="both")
    parser.add_argument("--cholec80_root", type=str,
                        default="datasets/cholec80")
    parser.add_argument("--cholect50_root", type=str,
                        default="datasets/CholecT50")
    parser.add_argument("--output_root", type=str,
                        default="datasets/clips")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Process only these video IDs.")
    parser.add_argument("--mock",    action="store_true",
                        help="Run a synthetic smoke test and exit.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process inputs but don't write clip JSONs.")
    args = parser.parse_args()

    if args.mock:
        run_mock_smoke_test()
        return

    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)
    log_path = setup_file_logging(output_root)
    logger.debug(f"CLI args: {vars(args)}")

    all_summaries = []

    # ---- Cholec80 ----
    if args.dataset in ("cholec80", "both"):
        c80_out = os.path.join(output_root, "cholec80")
        os.makedirs(c80_out, exist_ok=True)
        vids = (args.videos
                if args.videos else discover_cholec80_videos(args.cholec80_root))
        vids = [v for v in vids if v.startswith("video")]
        logger.info(f"Cholec80: processing {len(vids)} videos "
                    f"{'(dry run)' if args.dry_run else ''}")
        for v in vids:
            s = process_cholec80_video(
                v, args.cholec80_root, c80_out, args.dry_run
            )
            all_summaries.append(s)
            if args.dry_run and len(all_summaries) >= 2:
                logger.info("dry run: processed 2 Cholec80 videos, stopping")
                break

    # ---- CholecT50 ----
    if args.dataset in ("cholect50", "both"):
        t50_out = os.path.join(output_root, "cholect50")
        os.makedirs(t50_out, exist_ok=True)
        label_map = load_label_mapping(args.cholect50_root)
        logger.info(f"CholecT50 label_map: {len(label_map)} triplet entries")
        vids = (args.videos
                if args.videos else discover_cholect50_videos(args.cholect50_root))
        vids = [v for v in vids if v.startswith("VID")]
        logger.info(f"CholecT50: processing {len(vids)} videos "
                    f"{'(dry run)' if args.dry_run else ''}")
        for v in vids:
            s = process_cholect50_video(
                v, args.cholect50_root, t50_out, label_map, args.dry_run
            )
            all_summaries.append(s)
            if args.dry_run and len(all_summaries) >= 4:
                logger.info("dry run: processed total 4 videos, stopping")
                break

    # ---- Global summary ----
    processed = [s for s in all_summaries if s.get("status") != "skipped"]
    total_clips = sum(s.get("total_clips", 0) for s in processed)
    total_tc    = sum(s.get("transition_clips", 0) for s in processed)
    total_sc    = sum(s.get("stable_clips", 0)     for s in processed)
    global_summary = OrderedDict([
                ("config", {
            "clip_duration_seconds": CLIP_DURATION_S,
            "stable_interval_s":     STABLE_INTERVAL_S,
            "min_gap_s":             MIN_GAP_S,
            "cholec80_native_fps":   CHOLEC80_FPS,
        }),
        ("total_videos",      len(all_summaries)),
        ("total_processed",   len(processed)),
        ("total_skipped",     len(all_summaries) - len(processed)),
        ("total_clips",       total_clips),
        ("transition_clips",  total_tc),
        ("stable_clips",      total_sc),
        ("per_video", [{"video_id":         s.get("video_id"),
                         "source":           s.get("source"),
                         "status":           s.get("status", "processed"),
                         "total_clips":      s.get("total_clips", 0),
                         "transition_clips": s.get("transition_clips", 0),
                         "stable_clips":     s.get("stable_clips", 0)}
                         for s in all_summaries]),
    ])
    if not args.dry_run:
        gs_path = os.path.join(output_root, "summary.json")
        with open(gs_path, "w") as f:
            json.dump(global_summary, f, indent=2)
        logger.info(f"Global summary: {gs_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE — {total_clips} clips across {len(processed)} videos")
    logger.info(f"  Transition: {total_tc}  |  Stable: {total_sc}")
    logger.info(f"  Log: {log_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
