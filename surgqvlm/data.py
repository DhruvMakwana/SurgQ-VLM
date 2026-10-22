"""
data.py — SurgQ-VLM Dataset Handler
====================================

Each sample in the dataset is one (clip, question, answer) triple.
A clip JSON holds frame paths and per-frame annotations (produced by
data_prep/extract_clips.py). A QA JSON holds the list of QA pairs
generated from that clip (produced by data_prep/generate_qa.py).

At training time:
    - One clip JSON is paired with N QA pairs.
    - Each QA pair is returned as a separate sample
      (so one clip may yield many training samples).
    - Frame loading and sampling happen on-the-fly.

SlowFast sampling
-----------------
Clips are extracted at 1 FPS (one frame per second of surgery).
Two sampling streams provide different temporal granularities:

    Slow stream : 0.5 FPS → 1 frame every 2 seconds
    Fast stream : 1.0 FPS → 1 frame per second (same as extraction rate)

For a standard 30-second clip this gives:
    15 slow frames  (indices 0, 2, 4, …, 28)
    30 fast frames  (indices 0, 1, 2, …, 29)

For a boundary clip shorter than 30 s the counts scale proportionally.
The training script builds the prompt dynamically from the actual clip
length (see surgqvlm.utils.expected_visual_tokens).

Mock mode (CPU testing)
-----------------------
Set use_mock_encoder=True to skip PIL/torchvision entirely. Returns random
tensors of the correct shape instead of real image pixels — verify the
Dataset → DataLoader → model forward path on any CPU machine.

Split files
-----------
    datasets/splits/{train,val,test}.txt — one line per clip:
        absolute_clip_json_path TAB absolute_qa_json_path
    Produced by data_prep/build_splits.py.

Smoke test:

    python -m surgqvlm.data --mock --split datasets/splits/val.txt
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

# PIL / torchvision are imported lazily so this module can be imported on
# machines without them (e.g. for mock-mode testing or building split files).
try:
    from PIL import Image
    import torchvision.transforms as T
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# SlowFast sampling rates (frames per second).
# Clips are extracted at 1 FPS so FAST_FPS=1.0 uses every available frame.
# SLOW_FPS=0.5 takes every other frame for coarser temporal coverage.
SLOW_FPS = 0.5   # 1 frame every 2 seconds → 15 frames for a 30 s clip
FAST_FPS = 1.0   # 1 frame per second      → 30 frames for a 30 s clip

# Default image resolution for Qwen2.5-VL's ViT encoder.
# At 384 px with patch_size=14, floor(384/14)=27 patches per side —
# consistent with patch_grid_hw=27 in surgqvlm.model.
DEFAULT_IMAGE_SIZE = 384

# Qwen2.5-VL's vision tower was pre-trained with CLIP / OpenAI-CLIP
# normalisation values (inherited from Qwen2-VL's image processor). These
# differ from ImageNet values — using ImageNet stats on a Qwen-VL ViT
# produces a subtle but measurable colour-statistics shift in every patch.
# Source: transformers.models.qwen2_vl.image_processing_qwen2_vl
QWEN_VL_IMAGE_MEAN = [0.48145466, 0.4578275,  0.40821073]
QWEN_VL_IMAGE_STD  = [0.26862954, 0.26130258, 0.27577711]


# ── Split file reader ─────────────────────────────────────────────────────────

def read_split_file(split_path: str) -> list[tuple[str, str]]:
    """
    Read a split index file produced by data_prep/build_splits.py.

    Each non-empty line must contain exactly two tab-separated fields:

        clip_json_path TAB qa_json_path

    Malformed lines are logged as warnings and skipped.

    Returns
    -------
    list of (clip_path, qa_path) tuples — one per valid line.

    Raises
    ------
    FileNotFoundError if the split file does not exist.
    """
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    pairs = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                logger.warning(
                    "Skipping malformed line %d in %s: %r",
                    lineno, split_path, line
                )
                continue
            pairs.append((parts[0], parts[1]))

    logger.info("Loaded %d clip-QA pairs from %s", len(pairs), split_path)
    return pairs


# ── Frame sampling ────────────────────────────────────────────────────────────

def sample_frame_indices(
    total_frames: int,
    fps: float,
    clip_duration_seconds: int,
) -> list[int]:
    """
    Return frame indices to sample from a 1-FPS extracted clip.

    The number of frames to sample is
        n_frames = round(clip_duration_seconds × fps)
    and the indices are uniformly spaced across [0, total_frames - 1].

    For FAST_FPS=1.0 on a 30-second clip this returns the 30 available
    frames unchanged. For SLOW_FPS=0.5 it returns every other frame.

    Parameters
    ----------
    total_frames          : actual number of frames in the clip
    fps                   : desired sampling rate (SLOW_FPS or FAST_FPS)
    clip_duration_seconds : clip length in seconds (= total_frames for
                            1-FPS clips)

    Returns
    -------
    List of integer indices into frames[], clamped to [0, total_frames-1].

    Raises
    ------
    ValueError if fps <= 0.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    n_frames = max(1, round(clip_duration_seconds * fps))

    # Evenly spaced indices over the available frame range
    step    = total_frames / n_frames
    indices = [min(int(i * step), total_frames - 1) for i in range(n_frames)]
    return indices


# ── Image loading and transform ───────────────────────────────────────────────

def build_transform(image_size: int) -> "T.Compose":
    """
    Build the standard preprocessing pipeline for the vision encoder.

    Steps:
        1. Resize shorter edge to image_size (bicubic, keeps aspect ratio)
        2. Centre-crop to image_size × image_size
        3. Convert to float tensor in [0, 1]
        4. Normalise with Qwen-VL image mean and std

    Raises
    ------
    RuntimeError if torchvision is not installed.
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError(
            "torchvision is required for real image loading. "
            "Install it or use use_mock_encoder=True for CPU testing."
        )
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=QWEN_VL_IMAGE_MEAN,
            std =QWEN_VL_IMAGE_STD,
        ),
    ])


def load_frame(path: str, transform: "T.Compose") -> torch.Tensor:
    """
    Load one PNG frame and apply the vision encoder transform.

    Returns
    -------
    FloatTensor of shape (3, H, W).
    """
    img = Image.open(path).convert("RGB")
    return transform(img)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SurgQDataset(Dataset):
    """
    PyTorch Dataset for SurgQ-VLM.

    The logical unit is one (clip, question, answer) sample. Since one clip
    produces many QA pairs, the effective dataset size equals the total
    number of balanced QA pairs across all clips — not the number of clips.

    At construction time all clip and QA JSONs are loaded into memory
    (they are small, O(KB) per file). Frame images are loaded lazily
    in __getitem__.

    Parameters
    ----------
    split_file : str
        Path to train.txt / val.txt / test.txt from build_splits.py.
    image_size : int
        Square target resolution for the vision encoder (default 384).
    slow_fps : float
        Sampling rate for the slow stream (default 0.5 FPS).
    fast_fps : float
        Sampling rate for the fast stream (default 1.0 FPS).
    use_mock_encoder : bool
        If True, skip image loading and return random tensors.
    max_qa_per_clip : int | None
        If set, randomly subsample at most this many QA pairs per clip.
        Useful for quick development runs. Reproducible via seed.
    seed : int
        RNG seed for max_qa_per_clip subsampling (default 42).
    filter_qids : set | None
        When provided, only QA pairs whose ``question_id`` is in this set
        are retained. Used by the curated validation subset.
    """

    def __init__(
        self,
        split_file: str,
        image_size: int = DEFAULT_IMAGE_SIZE,
        slow_fps: float = SLOW_FPS,
        fast_fps: float = FAST_FPS,
        use_mock_encoder: bool = False,
        max_qa_per_clip: Optional[int] = None,
        seed: int = 42,
        filter_qids: Optional[set] = None,
    ) -> None:
        self.image_size       = image_size
        self.slow_fps         = slow_fps
        self.fast_fps         = fast_fps
        self.use_mock_encoder = use_mock_encoder
        self.filter_qids      = filter_qids

        # Build the image transform once and reuse across all __getitem__ calls
        self.transform = (
            None if use_mock_encoder else build_transform(image_size)
        )

        # Read split index → list of (clip_path, qa_path)
        raw_pairs = read_split_file(split_file)

        # Expand into a flat (clip_data, qa_pair) sample list.
        # All JSONs are loaded now; frame files are opened on demand.
        rng = random.Random(seed)
        self._samples: list[tuple[dict, dict]] = []

        n_clips_loaded   = 0   # clips that contributed at least one sample
        n_clips_missing  = 0   # clip or QA JSON could not be read
        n_clips_empty_qa = 0   # QA list was empty (valid: balanced to zero)

        for clip_path, qa_path in raw_pairs:

            # -- Load clip metadata --
            try:
                with open(clip_path) as f:
                    clip_data = json.load(f)
            except Exception as exc:
                logger.warning(
                    "Cannot load clip JSON %s: %s — skipping", clip_path, exc
                )
                n_clips_missing += 1
                continue

            # -- Load QA pairs --
            try:
                with open(qa_path) as f:
                    qa_pairs = json.load(f)
            except Exception as exc:
                logger.warning(
                    "Cannot load QA JSON %s: %s — skipping", qa_path, exc
                )
                n_clips_missing += 1
                continue

            # Optional question-id whitelist (curated val subset).
            if self.filter_qids is not None:
                qa_pairs = [q for q in qa_pairs
                            if q.get("question_id", "") in self.filter_qids]

            # An empty QA list is valid: binary balancing may have removed
            # all pairs for a clip. Not an error — log at DEBUG only.
            if not qa_pairs:
                logger.debug(
                    "Empty QA list for clip %s (balanced to zero) — skipping",
                    clip_path
                )
                n_clips_empty_qa += 1
                continue

            # Optionally cap QA pairs per clip for quick subset runs
            if max_qa_per_clip is not None and len(qa_pairs) > max_qa_per_clip:
                qa_pairs = rng.sample(qa_pairs, max_qa_per_clip)

            for qa in qa_pairs:
                self._samples.append((clip_data, qa))

            n_clips_loaded += 1

        logger.info(
            "Dataset ready: %d samples from %d clips "
            "| %d clips missing/unreadable "
            "| %d clips with empty QA (balanced to zero)",
            len(self._samples), n_clips_loaded,
            n_clips_missing, n_clips_empty_qa,
        )

    # ── Core interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one (clip, question, answer) sample as a dict.

        Keys
        ----
        slow_frames : FloatTensor (N_slow, 3, H, W)
        fast_frames : FloatTensor (N_fast, 3, H, W)
        question    : str
        answer      : str  — lower-cased for dataset consistency
        meta        : dict — question_id, clip_id, video_id, source,
                      question_type, question_group, answer_type,
                      answer_format, template_variant, requires_transition,
                      temporal_grounding, binary_label,
                      clip_duration_seconds
        """
        clip_data, qa = self._samples[idx]

        # -- Frame list and clip duration --
        frames_list   = clip_data.get("frames", [])
        clip_duration = clip_data.get("clip_duration_seconds", 30)
        total_frames  = len(frames_list)

        if total_frames == 0:
            # Defensive fallback: should not happen if the extractor ran correctly
            logger.warning(
                "Clip %s has no frames — padding with zero tensors",
                clip_data.get("clip_id", "?")
            )
            total_frames = clip_duration
            frames_list  = [""] * total_frames

        # -- Sample frame indices for each stream --
        slow_indices = sample_frame_indices(
            total_frames, self.slow_fps, clip_duration
        )
        fast_indices = sample_frame_indices(
            total_frames, self.fast_fps, clip_duration
        )

        # -- Load frames (or return mocks) --
        clip_id     = clip_data.get("clip_id", "?")
        slow_frames = self._load_frames(frames_list, slow_indices, clip_id)
        fast_frames = self._load_frames(frames_list, fast_indices, clip_id)

        # -- QA text --
        question = qa.get("question", "")
        answer   = qa.get("answer", "")

        # Normalise the answer to lower-case so CholecT50 (already lowercase)
        # and Cholec80 (Title-Case instrument names) match at eval time.
        answer = answer.lower() if answer else answer

        # -- Metadata for evaluation and stratified analysis --
        meta = {
            "question_id":           qa.get("question_id", ""),
            "clip_id":               clip_data.get("clip_id", ""),
            "video_id":              clip_data.get("video_id", ""),
            "source":                clip_data.get("source", ""),
            "question_type":         qa.get("question_type", ""),
            "question_group":        qa.get("question_group", ""),
            "answer_type":           qa.get("answer_type", ""),
            "answer_format":         qa.get("answer_format", ""),
            "template_variant":      qa.get("template_variant", 0),
            "requires_transition":   qa.get("requires_transition", False),
            "temporal_grounding":    qa.get("temporal_grounding", False),
            "binary_label":          qa.get("binary_label", None),
            "clip_duration_seconds": clip_duration,
        }

        return {
            "slow_frames": slow_frames,                    # (N_slow, 3, H, W)
            "fast_frames": fast_frames,                    # (N_fast, 3, H, W)
            "question":    question,
            "answer":      answer,
            "meta":        meta,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _load_frames(
        self,
        frames_list: list[str],
        indices: list[int],
        clip_id: str,
    ) -> torch.Tensor:
        """
        Load the frames at the given indices from frames_list.

        In mock mode returns random float tensors without touching disk.
        In real mode opens each PNG via PIL and applies self.transform.
        Missing or unreadable frames are replaced with zero tensors and
        logged at DEBUG / WARNING level respectively.

        Returns
        -------
        FloatTensor of shape (N, 3, H, W) where N = len(indices).
        """
        N = len(indices)
        H = W = self.image_size

        if self.use_mock_encoder:
            return torch.randn(N, 3, H, W)

        tensors = []
        for i in indices:
            path = frames_list[i] if i < len(frames_list) else ""
            if not path or not Path(path).exists():
                logger.debug(
                    "Frame %d missing for clip %s — using zeros", i, clip_id
                )
                tensors.append(torch.zeros(3, H, W))
            else:
                try:
                    tensors.append(load_frame(path, self.transform))
                except Exception as exc:
                    logger.warning(
                        "Failed to load frame %s: %s — using zeros", path, exc
                    )
                    tensors.append(torch.zeros(3, H, W))

        return torch.stack(tensors)   # (N, 3, H, W)


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for the DataLoader.

    Clips within the same batch can have different lengths (e.g. a boundary
    clip shorter than 30 s mixed with standard 30 s clips). slow_frames and
    fast_frames are zero-padded along the frame dimension to the longest
    clip in the batch, and boolean masks record which positions are real.

    Returns
    -------
    dict with keys:
        slow_frames : FloatTensor (B, N_slow_max, 3, H, W)
        fast_frames : FloatTensor (B, N_fast_max, 3, H, W)
        slow_mask   : BoolTensor  (B, N_slow_max)  — True = real frame
        fast_mask   : BoolTensor  (B, N_fast_max)  — True = real frame
        question    : list[str]  length B
        answer      : list[str]  length B
        meta        : list[dict] length B
    """
    slow_list = [s["slow_frames"] for s in batch]   # each (N_s, 3, H, W)
    fast_list = [s["fast_frames"] for s in batch]   # each (N_f, 3, H, W)

    slow_frames, slow_mask = _pad_frame_sequence(slow_list)
    fast_frames, fast_mask = _pad_frame_sequence(fast_list)

    return {
        "slow_frames": slow_frames,   # (B, N_slow_max, 3, H, W)
        "fast_frames": fast_frames,   # (B, N_fast_max, 3, H, W)
        "slow_mask":   slow_mask,     # (B, N_slow_max)
        "fast_mask":   fast_mask,     # (B, N_fast_max)
        "question":    [s["question"] for s in batch],
        "answer":      [s["answer"]   for s in batch],
        "meta":        [s["meta"]     for s in batch],
    }


def _pad_frame_sequence(
    tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of (N_i, 3, H, W) tensors to a common length N_max.

    Returns
    -------
    padded : FloatTensor (B, N_max, 3, H, W) — zero-padded on the right
    mask   : BoolTensor  (B, N_max)           — True for real frames
    """
    n_max       = max(t.shape[0] for t in tensors)
    B           = len(tensors)
    _, C, H, W  = tensors[0].shape

    padded = torch.zeros(B, n_max, C, H, W, dtype=tensors[0].dtype)
    mask   = torch.zeros(B, n_max, dtype=torch.bool)

    for i, t in enumerate(tensors):
        n = t.shape[0]
        padded[i, :n] = t
        mask[i, :n]   = True

    return padded, mask


# ── DataLoader builder ────────────────────────────────────────────────────────

def build_dataloader(
    dataset:     SurgQDataset,
    batch_size:  int  = 2,
    shuffle:     bool = True,
    num_workers: int  = 4,
    pin_memory:  bool = True,
) -> DataLoader:
    """
    Build a DataLoader wrapping SurgQDataset.

    Parameters
    ----------
    dataset     : SurgQDataset instance
    batch_size  : samples per batch
    shuffle     : True for training splits, False for val/test
    num_workers : worker processes for parallel frame loading.
                  0 for single-process debugging or CPU-only machines,
                  8+ recommended on A100-class nodes.
    pin_memory  : True speeds up host-to-GPU transfers; False on CPU-only.
    """
    # Perf knobs for A100-class GPUs (only apply when num_workers > 0):
    #   persistent_workers=True — reuse worker processes across epochs.
    #   prefetch_factor=4       — decouple CPU decode from GPU forward.
    loader_kwargs = dict(
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        collate_fn  = collate_fn,
        drop_last   = False,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"]    = 4
    return DataLoader(dataset, **loader_kwargs)


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone smoke test. Run from the repository root:

        python -m surgqvlm.data --mock --split datasets/splits/val.txt

    Prints batch shapes, one sample's Q/A, metadata, and frame statistics.
    Exits with a non-zero code if any assertion fails.
    """
    import argparse
    import sys

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="SurgQDataset smoke test")
    parser.add_argument(
        "--split",
        default = "datasets/splits/val.txt",
        help    = "Path to a split index file (train.txt / val.txt / test.txt)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock encoder — no image files needed, random tensors returned",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--max_qa_per_clip", type=int, default=3,
        help="Cap QA pairs per clip for a quick test (default 3)",
    )
    args = parser.parse_args()

    print(f"\nLoading dataset from : {args.split}")
    print(f"Mock encoder         : {args.mock}\n")

    ds = SurgQDataset(
        split_file       = args.split,
        use_mock_encoder = args.mock,
        max_qa_per_clip  = args.max_qa_per_clip,
    )
    print(f"Total samples : {len(ds):,}\n")

    dl = build_dataloader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 0,        # single-process for debugging
        pin_memory  = False,
    )

    print("Fetching first batch ...")
    batch = next(iter(dl))

    print(f"\n{'='*54}")
    print("  BATCH SHAPES")
    print(f"{'='*54}")
    print(f"  slow_frames : {tuple(batch['slow_frames'].shape)}  (B, N_slow, C, H, W)")
    print(f"  fast_frames : {tuple(batch['fast_frames'].shape)}  (B, N_fast, C, H, W)")
    print(f"  slow_mask   : {tuple(batch['slow_mask'].shape)}")
    print(f"  fast_mask   : {tuple(batch['fast_mask'].shape)}")

    print(f"\n{'='*54}")
    print("  SAMPLE 0")
    print(f"{'='*54}")
    print(f"  Question  : {batch['question'][0]}")
    print(f"  Answer    : {batch['answer'][0]}")
    m = batch["meta"][0]
    print(f"  Q-type    : {m['question_type']}  ({m['answer_type']})")
    print(f"  Source    : {m['source']}  /  {m['video_id']}")
    print(f"  Duration  : {m['clip_duration_seconds']} s")

    # Basic shape assertions
    H = W = DEFAULT_IMAGE_SIZE
    assert batch["slow_frames"].shape[2:] == (3, H, W), \
        f"Unexpected slow_frames spatial shape: {batch['slow_frames'].shape}"
    assert batch["fast_frames"].shape[2:] == (3, H, W), \
        f"Unexpected fast_frames spatial shape: {batch['fast_frames'].shape}"
    assert batch["slow_frames"].shape[0] <= args.batch_size
    assert len(batch["question"]) == batch["slow_frames"].shape[0]
    assert len(batch["meta"])     == batch["slow_frames"].shape[0]

    print(f"\nAll assertions passed.")
    print(f"Smoke test complete.\n")
    sys.exit(0)
