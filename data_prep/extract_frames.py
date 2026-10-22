#!/usr/bin/env python3
"""
extract_frames.py — 1-FPS frame extraction for Cholec80 videos.
================================================================

Cholec80 ships 25-FPS .mp4 videos; the SurgQ-VLM pipeline consumes frames
extracted at exactly 1 FPS. This script wraps ffmpeg to produce the layout
that extract_clips.py and the dataset loader expect:

    datasets/cholec80/frames/video01/video01_000001.png   (1-based index)
    datasets/cholec80/frames/video01/video01_000002.png
    ...

CholecT50 needs NO frame extraction — it ships pre-extracted 1-FPS frames
under CholecT50/videos/VIDxx/000000.png (0-based index).

Requirements: ffmpeg on PATH.

Usage
-----
    # All 80 videos (several hours; ~80 GB of PNGs)
    python data_prep/extract_frames.py \\
        --videos_dir datasets/cholec80/videos \\
        --output_dir datasets/cholec80/frames

    # A subset
    python data_prep/extract_frames.py --only video01 video02
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_video(video_path: Path, output_dir: Path, overwrite: bool) -> bool:
    """
    Extract one video to 1-FPS PNG frames named {video_id}_{NNNNNN}.png.

    ffmpeg's image2 muxer numbers frames starting at 1, which matches the
    1-based frame naming that extract_clips.py resolves for Cholec80
    (frame index 0 → *_000001.png).

    Returns True on success.
    """
    video_id  = video_path.stem                      # "video01"
    frame_dir = output_dir / video_id
    if frame_dir.exists() and any(frame_dir.glob("*.png")) and not overwrite:
        logger.info("%s: frames already present — skipping (use --overwrite)",
                    video_id)
        return True
    frame_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(frame_dir / f"{video_id}_%06d.png")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", "fps=1",          # one frame per second of video
        "-start_number", "1",    # 1-based, matching the expected layout
        pattern,
    ]
    logger.info("%s: extracting → %s", video_id, frame_dir)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("%s: ffmpeg failed (exit %d)", video_id, result.returncode)
        return False

    n = len(list(frame_dir.glob("*.png")))
    logger.info("%s: %d frames written", video_id, n)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 1-FPS frames from Cholec80 videos via ffmpeg."
    )
    parser.add_argument("--videos_dir", default="datasets/cholec80/videos",
                        help="Directory containing video01.mp4 … video80.mp4")
    parser.add_argument("--output_dir", default="datasets/cholec80/frames",
                        help="Output root; one sub-directory per video.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Extract only these video IDs (e.g. video01).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if frames already exist.")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not found on PATH. Install it first "
                     "(e.g. apt install ffmpeg).")
        sys.exit(1)

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        logger.error("Videos directory not found: %s", videos_dir)
        sys.exit(1)

    videos = sorted(videos_dir.glob("video*.mp4"))
    if args.only:
        keep = set(args.only)
        videos = [v for v in videos if v.stem in keep]
    if not videos:
        logger.error("No videos found in %s", videos_dir)
        sys.exit(1)

    logger.info("Extracting %d video(s) at 1 FPS ...", len(videos))
    failures = 0
    for v in videos:
        if not extract_video(v, Path(args.output_dir), args.overwrite):
            failures += 1

    if failures:
        logger.error("DONE with %d failure(s).", failures)
        sys.exit(1)
    logger.info("DONE — all %d videos extracted.", len(videos))


if __name__ == "__main__":
    main()
