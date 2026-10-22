#!/usr/bin/env python3
"""
verify_data.py — Check the raw dataset layout before running the pipeline.
===========================================================================

Cholec80 and CholecT50 must be requested from CAMMA (University of
Strasbourg) — see the README for links. After downloading and unpacking,
run this script to confirm the directory layout matches what
data_prep/extract_frames.py and data_prep/extract_clips.py expect:

    datasets/
      cholec80/
        videos/              video01.mp4 … video80.mp4   (for extract_frames)
        frames/              video01/video01_000001.png … (after extract_frames)
        phase_annotations/   video01-phase.txt …
        tool_annotations/    video01-tool.txt …
      CholecT50/
        videos/              VID01/000000.png …           (frames ship with it)
        labels/              VID01.json …
        label_mapping.txt

Usage
-----
    python scripts/verify_data.py --data_root datasets
"""

import argparse
import sys
from pathlib import Path


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK " if ok else "MISSING"
    print(f"  [{mark:7s}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify raw dataset layout.")
    parser.add_argument("--data_root", default="datasets")
    args = parser.parse_args()

    root = Path(args.data_root)
    all_ok = True

    # ── Cholec80 ───────────────────────────────────────────────────────────
    print("\nCholec80:")
    c80 = root / "cholec80"
    phase = c80 / "phase_annotations"
    tools = c80 / "tool_annotations"
    vids  = c80 / "videos"
    frames = c80 / "frames"

    n_phase = len(list(phase.glob("*-phase.txt"))) if phase.is_dir() else 0
    n_tool  = len(list(tools.glob("*-tool.txt")))  if tools.is_dir() else 0
    n_vid   = len(list(vids.glob("video*.mp4")))   if vids.is_dir() else 0
    n_fdirs = len([d for d in frames.iterdir() if d.is_dir()]) \
              if frames.is_dir() else 0

    all_ok &= check("phase_annotations/ (80 expected)", n_phase == 80,
                    f"{n_phase} found")
    all_ok &= check("tool_annotations/ (80 expected)", n_tool == 80,
                    f"{n_tool} found")
    videos_ok = n_vid == 80
    frames_ok = n_fdirs == 80
    check("videos/ (80 .mp4)", videos_ok, f"{n_vid} found")
    check("frames/ (80 video dirs; created by extract_frames.py)",
          frames_ok, f"{n_fdirs} found")
    if not (videos_ok or frames_ok):
        print("         → need either videos/ (then run extract_frames.py) "
              "or pre-extracted frames/")
        all_ok = False

    # ── CholecT50 ──────────────────────────────────────────────────────────
    print("\nCholecT50:")
    t50 = root / "CholecT50"
    labels = t50 / "labels"
    t50vid = t50 / "videos"
    lmap   = t50 / "label_mapping.txt"

    n_labels = len(list(labels.glob("VID*.json"))) if labels.is_dir() else 0
    n_t50v   = len([d for d in t50vid.iterdir() if d.is_dir()]) \
               if t50vid.is_dir() else 0

    all_ok &= check("labels/ (50 expected)", n_labels == 50,
                    f"{n_labels} found")
    all_ok &= check("videos/ (50 frame dirs)", n_t50v == 50,
                    f"{n_t50v} found")
    all_ok &= check("label_mapping.txt", lmap.is_file())

    # Spot-check one frame dir naming convention (0-based 6-digit PNGs)
    if n_t50v > 0:
        sample_dir = sorted(d for d in t50vid.iterdir() if d.is_dir())[0]
        has_zero = (sample_dir / "000000.png").exists()
        check(f"{sample_dir.name}/000000.png (0-based naming)", has_zero)
        all_ok &= has_zero

    print()
    if all_ok:
        print("Layout verified. Next steps:")
        print("  1. python data_prep/extract_frames.py        (if frames/ is empty)")
        print("  2. python data_prep/extract_clips.py")
        print("  3. python data_prep/generate_qa.py")
        print("  4. python data_prep/build_splits.py")
        sys.exit(0)
    else:
        print("Layout incomplete — fix the items marked MISSING above. "
              "See the README's Data section.")
        sys.exit(1)


if __name__ == "__main__":
    main()
