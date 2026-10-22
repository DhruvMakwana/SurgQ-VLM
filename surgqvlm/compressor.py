"""
compressor.py — Dual-Stream Question-Conditioned Visual Token Compressor
=========================================================================

Implements the visual token compression pipeline for SurgQ-VLM, adapted
from HICom (Hybrid-Level Instruction Injection for Video Token Compression,
Liu et al., CVPR 2025).

Two compression levels are applied independently to the slow and fast streams:

  Local Compressor  — spatiotemporal windowed attention. Each output token
                      summarises a (temporal_k × spatial_k × spatial_k)
                      cube of patch tokens. The query is derived from the
                      visual features themselves, then conditioned on the
                      question via cross-attention (GuideInjector).

  Global Compressor — a small set of learnable query tokens attend to ALL
                      flattened frame tokens from the full clip, conditioned
                      on the question. Captures cross-clip temporal patterns.

Pipeline (per forward call)
---------------------------
  slow_frames  (B, N_slow, 3, H, W)
  fast_frames  (B, N_fast, 3, H, W)
  questions    list[str]  length B

  Step 1 : ViT encoding  → slow (B, N_slow, 27, 27, 1280)
                         → fast (B, N_fast, 27, 27, 1280)
  Step 2 : Fast spatial pool 27×27 → 5×5
                         → fast (B, N_fast, 5, 5, 1280)
  Step 3 : Qwen embed_tokens → question embeddings (B, L, llm_embed_dim)
           → projected to encoder_dim via text_proj MLP
  Step 4 : Per-sample local + global compression on slow and fast
  Step 5 : Concatenate [slow_local | fast_local | slow_global | fast_global]
                         → (B, 671, llm_dim)

Token counts (30-second clip, default config)
---------------------------------------------
  Slow local  : ceil(15/3) × ceil(27/3) × ceil(27/3) = 5×9×9  = 405
  Fast local  : ceil(30/3) × ceil(5/1)  × ceil(5/1)  = 10×5×5 = 250
  Slow global : n_global_slow = 8
  Fast global : n_global_fast = 8
  Total       : 671 tokens → LLM

ViT encoder (Qwen2.5-VL, both 3B and 7B share the same backbone)
-----------------------------------------------------------------
  embed_dim     : 1280
  patch_size    : 14
  input size    : 384×384  →  floor(384/14) = 27 patches per side
  patch_grid_hw : 27×27 = 729 patch tokens per frame

Per-sample processing loop
--------------------------
LocalCompressor and GlobalCompressor operate on a single sample
(t, h, w, d) without a batch dimension. We process each sample in
the batch with a Python for-loop in SurgQCompressor.forward().

For typical training batch sizes (2–8), this adds negligible overhead:
the bottleneck is ViT encoding and the LLM forward, not the compressor loop.

Mock mode
---------
Set use_mock_encoder=True to skip all model loading. Returns random tensors
of the correct output shape. Use for CPU smoke tests and shape verification:

    python -m surgqvlm.compressor

References
----------
  HICom : https://github.com/lntzm/HICom  (Liu et al., CVPR 2025)
  SlowFast : Feichtenhofer et al., ICCV 2019
"""

import math
import logging
from dataclasses import dataclass
from functools import partial
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import trunc_normal_

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SurgQCompressorConfig:
    """
    All hyperparameters for SurgQCompressor in one place.

    Attributes
    ----------
    encoder_dim : int
        Hidden dimension of the frozen ViT encoder.
        Qwen2.5-VL ViT (3B and 7B share the same backbone) → 1280.
    llm_dim : int
        Hidden dimension of the LLM backbone.
        Qwen2.5-VL-3B → 2048.  Qwen2.5-VL-7B → 3584.
    patch_grid_hw : int
        Number of patches per side from the ViT.
        Qwen2.5-VL ViT at 384px input, patch_size=14 → floor(384/14) = 27.
    fast_spatial_size : int
        Target spatial size for the fast stream after pooling (27×27 → 5×5).
    slow_temporal_k / slow_spatial_k : int
        Kernel sizes for slow-stream local compression. Each output token
        summarises a (temporal_k × spatial_k × spatial_k) window.
    fast_temporal_k / fast_spatial_k : int
        Kernel sizes for fast-stream local compression. fast_spatial_k=1
        because the fast stream is already spatially pooled to 5×5.
    n_global_slow / n_global_fast : int
        Number of learnable global query tokens per stream.
    num_attn_heads : int
        Attention heads in MultiheadAttention. Must divide encoder_dim.
    max_num_frames : int
        Maximum frame count for the 3D positional embedding cache.
    slow_max_hw / fast_max_hw : int
        Maximum spatial grid per stream for the positional embedding cache
        (27 for slow, 5 for fast — avoids a wasteful 27×27 fast cache).
    llm_embed_dim : int
        Output dimension of Qwen2.5-VL's embed_tokens layer, used as the
        text encoder for question conditioning (2048 for 3B, 3584 for 7B).
    use_mock_encoder : bool
        Skip all model loading and return random tensors. For CPU testing.
    mlp_depth : int
        Depth of the readout MLP inside compressors (encoder_dim → llm_dim).
    """

    # Dimensions
    encoder_dim:       int   = 1280      # Qwen2.5-VL ViT; overridden by SurgQVLMConfig
    llm_dim:           int   = 2048      # Overridden to 3584 for 7B
    patch_grid_hw:     int   = 27        # floor(384/14) = 27 patches per side

    # Fast stream spatial pooling
    fast_spatial_size: int   = 5

    # Local compressor kernel sizes
    slow_temporal_k:   int   = 3
    slow_spatial_k:    int   = 3
    fast_temporal_k:   int   = 3
    fast_spatial_k:    int   = 1

    # Global compressor query counts
    n_global_slow:     int   = 8
    n_global_fast:     int   = 8

    # Attention
    num_attn_heads:    int   = 10        # 1280 // 128; overridden by SurgQVLMConfig

    # Positional embedding cache sizes — set per stream so we do not
    # wastefully allocate a 27×27 cache for the 5×5 fast stream.
    max_num_frames:    int   = 256
    slow_max_hw:       int   = 27        # slow stream patch grid (27×27)
    fast_max_hw:       int   = 5         # fast stream after spatial pool (5×5)

    # Question embedding source: Qwen2.5-VL embed_tokens output dimension.
    # Projected down to encoder_dim inside GuideInjector.
    llm_embed_dim:     int   = 2048

    # Testing
    use_mock_encoder:  bool  = False

    # MLP depth inside readout projectors
    mlp_depth:         int   = 2


# ── Helper: MLP builder ───────────────────────────────────────────────────────

def build_mlp(depth: int, in_dim: int, out_dim: int) -> nn.Sequential:
    """
    Build a depth-layer MLP: Linear → (GELU → Linear) × (depth-1).

    Parameters
    ----------
    depth   : number of Linear layers
    in_dim  : input dimension
    out_dim : output dimension of every layer
    """
    modules: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(out_dim, out_dim))
    return nn.Sequential(*modules)


# ── 3D positional embedding (numpy, then registered as buffer) ────────────────

def get_3d_position_embedding(
    t: int, h: int, w: int, d_model: int
) -> np.ndarray:
    """
    Sinusoidal 3D positional embedding.

    Returns ndarray of shape (t, h, w, d_model) computed as the sum of
    independent 1-D sinusoidal embeddings along each spatial-temporal axis.
    Follows the approach used in HICom (Liu et al., CVPR 2025).

    Parameters
    ----------
    t, h, w  : temporal, height, and width extents
    d_model  : embedding dimension (must match encoder_dim)
    """
    def get_angles(pos: np.ndarray, i: np.ndarray, d: int) -> np.ndarray:
        return pos / np.power(10000, (2 * (i // 2)) / np.float32(d))

    def sinusoidal_1d(length: int) -> np.ndarray:
        angles = get_angles(
            np.arange(length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model,
        )
        enc = np.zeros_like(angles)
        enc[:, 0::2] = np.sin(angles[:, 0::2])
        enc[:, 1::2] = np.cos(angles[:, 1::2])
        return enc                           # (length, d_model)

    enc_t = sinusoidal_1d(t)[:, np.newaxis, np.newaxis, :]   # (t, 1, 1, d)
    enc_h = sinusoidal_1d(h)[np.newaxis, :, np.newaxis, :]   # (1, h, 1, d)
    enc_w = sinusoidal_1d(w)[np.newaxis, np.newaxis, :, :]   # (1, 1, w, d)

    return enc_t + enc_h + enc_w                              # (t, h, w, d)


# ── MultiheadAttention ────────────────────────────────────────────────────────

class MultiheadAttention(nn.Module):
    """
    Multi-head attention used inside the compressors.

    Implementation note
    -------------------
    The attention math is routed through PyTorch's
    ``F.scaled_dot_product_attention`` which auto-selects Flash-Attention-2
    or memory-efficient attention depending on hardware and input shapes.
    On A100 this gives a 2-3x speedup on the GlobalCompressor's long-KV
    cross-attention (Q of ~8 tokens over KV of ~7000 frame tokens) and
    on GuideInjector cross-attention with no change in numerics.

    The attention-weight tensor is not returned (SDPA does not expose it);
    callers all ignore the second return value, so we return ``None`` to
    keep the tuple interface.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.dropout   = dropout

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        query:          torch.Tensor,
        key:            torch.Tensor,
        value:          torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        query          : (B, Q, D)
        key            : (B, K, D)
        value          : (B, K, D)
        attention_mask : (B, 1, Q, K) additive mask, optional

        Returns
        -------
        attn_output    : (B, Q, D)
        attn_weights   : always ``None`` — SDPA does not expose weights.
        """
        B, q_len, _ = query.shape
        k_len = key.shape[1]

        def project_and_split(proj: nn.Linear, x: torch.Tensor, seq: int):
            return (proj(x)
                    .view(B, seq, self.num_heads, self.head_dim)
                    .transpose(1, 2))          # (B, H, seq, head_dim)

        Q = project_and_split(self.q_proj, query, q_len)
        K = project_and_split(self.k_proj, key,   k_len)
        V = project_and_split(self.v_proj, value, k_len)

        # Flash-Attention-2 / memory-efficient attention via PyTorch SDPA.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask = attention_mask,
            dropout_p = self.dropout if self.training else 0.0,
            scale     = self.scale,
        )

        out = out.transpose(1, 2).contiguous()          # (B, Q, H, head_dim)
        out = out.reshape(B, q_len, self.embed_dim)     # (B, Q, D)
        out = self.out_proj(out)

        return out, None


# ── GuideInjector ─────────────────────────────────────────────────────────────

class GuideInjector(nn.Module):
    """
    Injects question (guide) conditioning into visual features.

    Each visual token independently attends to the full question token
    sequence via cross-attention (the "fine" mode of HICom, Liu et al.,
    CVPR 2025). This gives position-specific conditioning — the scissors
    patch attends to "cutting", the anatomy patch attends to "cystic duct"
    — rather than a single global scale/shift applied uniformly across
    all positions.

    Parameters
    ----------
    visual_dim : dimension of visual tokens (encoder_dim)
    text_dim   : dimension of incoming text tokens. For Qwen2.5-VL-3B this
                 is 2048 (llm_embed_dim); for 7B it is 3584. Projected to
                 visual_dim internally via the text_proj MLP.
    num_heads  : attention heads for cross-attention
    mlp_depth  : depth of text projection MLP
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim:   int,
        num_heads:  int,
        mlp_depth:  int = 2,
        norm_layer  = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim   = text_dim

        # Project question text tokens to visual_dim if dimensions differ
        if text_dim != visual_dim:
            self.text_proj = build_mlp(mlp_depth, text_dim, visual_dim)
        else:
            self.text_proj = nn.Identity()

        # Cross-attention: visual tokens (Q) attend to text tokens (K, V)
        self.cross_attn = MultiheadAttention(visual_dim, num_heads)
        self.norm       = norm_layer(visual_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply fine-grained question conditioning to visual tokens.

        Parameters
        ----------
        visual_tokens : (N, D_visual)  — flattened spatial positions
                        OR (D_visual,) for the single-global-query case
        text_tokens   : (1, L, D_text)  — question token sequence

        Returns
        -------
        conditioned   : same shape as visual_tokens

        Notes
        -----
        visual_tokens is treated as a single sequence regardless of its
        original spatial layout. The caller is responsible for reshaping
        before and after.
        """
        # Project text tokens to visual dimension
        text_projected = self.text_proj(text_tokens)          # (1, L, D_vis)

        original_shape = visual_tokens.shape

        # Ensure visual tokens are (1, N, D) for attention
        if visual_tokens.ndim == 1:
            # Global query: single token
            q = visual_tokens.unsqueeze(0).unsqueeze(0)       # (1, 1, D)
        elif visual_tokens.ndim == 2:
            q = visual_tokens.unsqueeze(0)                    # (1, N, D)
        else:
            raise ValueError(
                f"visual_tokens must be 1-D or 2-D, got shape "
                f"{visual_tokens.shape}"
            )

        kv = text_projected                                   # (1, L, D)
        attn_out, _ = self.cross_attn(q, kv, kv)              # (1, N, D)
        attn_out = attn_out.squeeze(0)                        # (N, D) or (D,)

        # Residual + LayerNorm
        conditioned = self.norm(visual_tokens + attn_out)

        return conditioned.reshape(original_shape)


# ── LocalCompressor ───────────────────────────────────────────────────────────

class LocalCompressor(nn.Module):
    """
    Spatiotemporal windowed attention compressor.

    Adapted from HICom LocalCompressor (Liu et al., CVPR 2025) with the
    following changes:
      - Config-driven dimensions (no string parsing)
      - GuideInjector cross-attention conditioning
      - No logit_scale / logit_bias (CLIP contrastive scale) — not needed
        when using Qwen's embed_tokens rather than a SigLIP alignment head
      - Accepts explicit temporal_k and spatial_k arguments at construction

    How it works
    ------------
    Given frame features (t, h, w, d):

    1. Downsample via trilinear interpolation → query (t_out, h_out, w_out, d)
       t_out = ceil(t / temporal_k),  h_out = w_out = ceil(hw / spatial_k)

    2. Inject question conditioning into the query via GuideInjector:
       each output position attends to the full question token sequence.

    3. Divide original features into (temporal_k × spatial_k × spatial_k)
       windows → reshape to (n_groups, group_size, d).

    4. Attention: each query token (one per group) attends to its window.
       output = softmax(query @ key.T / sqrt(d)) @ value

    5. Readout MLP: project encoder_dim → llm_dim.

    Output token count
    ------------------
    ceil(t / temporal_k) × ceil(h / spatial_k) × ceil(w / spatial_k)

    Parameters
    ----------
    encoder_dim    : ViT hidden dimension (1280 for Qwen2.5-VL ViT)
    llm_dim        : LLM hidden dimension (output of readout MLP)
    temporal_k     : temporal kernel size
    spatial_k      : spatial kernel size (applied to both H and W)
    guide_injector : pre-built GuideInjector instance
    mlp_depth      : depth of readout MLP
    """

    def __init__(
        self,
        encoder_dim:    int,
        llm_dim:        int,
        temporal_k:     int,
        spatial_k:      int,
        guide_injector: GuideInjector,
        mlp_depth:      int = 2,
        norm_layer      = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.temporal_k  = temporal_k
        self.spatial_k   = spatial_k

        self.guide_injector = guide_injector

        # Readout: project encoder_dim → llm_dim
        self.readout = build_mlp(mlp_depth, encoder_dim, llm_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ── Feature division helpers (from HICom) ─────────────────────────────

    def _balance_divide(
        self, x: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        """
        Handle the case where the dimension length is not divisible by
        kernel_size. Splits into ceil(t / kernel_size) groups of size
        kernel_size, with boundary overlap rather than padding.

        Adapted directly from HICom's balance_divide_feature.
        """
        t = x.shape[0]
        split_size = math.ceil(t / kernel_size)

        no_repeat_num = t % split_size
        if no_repeat_num == 0:
            no_repeat_num = split_size

        split_idx_list = [
            kernel_size - (0 if i < no_repeat_num else 1)
            for i in range(split_size)
        ]

        start_idx = 0
        split_parts = []
        for i in range(split_size):
            end_idx = start_idx + split_idx_list[i]
            if split_idx_list[i] < kernel_size:
                start_idx -= 1
            split_parts.append(x[start_idx:end_idx, ...])
            start_idx = end_idx

        return torch.stack(split_parts, dim=1)

    def _divide_feature(
        self,
        x:          torch.Tensor,
        kernel_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Divide a (t, h, w, d) feature tensor into non-overlapping windows.

        Returns (n_groups, group_size, d) where:
            n_groups   = ceil(t/kt) × ceil(h/kh) × ceil(w/kw)
            group_size = kt × kh × kw

        Adapted directly from HICom's divide_feature.

        Parameters
        ----------
        x           : (t, h, w, d)
        kernel_size : (kt, kh, kw)
        """
        t, h, w = x.shape[:3]
        kt, kh, kw = kernel_size

        # ── Temporal division ─────────────────────────────────────────────
        if t % kt == 0:
            t_split = einops.rearrange(
                x, '(t1 t2) h w d -> t2 t1 h w d', t2=kt
            )
        else:
            t_split = self._balance_divide(x, kt)

        # ── Height division ───────────────────────────────────────────────
        t_split = einops.rearrange(t_split, 't2 t1 h w d -> h t2 t1 w d')
        if h % kh == 0:
            th_split = einops.rearrange(
                t_split, '(h1 h2) t2 t1 w d -> h2 h1 t2 t1 w d', h2=kh
            )
        else:
            th_split = self._balance_divide(t_split, kh)

        # ── Width division ────────────────────────────────────────────────
        th_split = einops.rearrange(
            th_split, 'h2 h1 t2 t1 w d -> w h2 h1 t2 t1 d'
        )
        if w % kw == 0:
            thw_split = einops.rearrange(
                th_split, '(w1 w2) h2 h1 t2 t1 d -> w2 w1 h2 h1 t2 t1 d',
                w2=kw
            )
        else:
            thw_split = self._balance_divide(th_split, kw)

        # Reshape to (n_groups, group_size, d)
        return einops.rearrange(
            thw_split,
            'w2 w1 h2 h1 t2 t1 d -> (t1 h1 w1) (t2 h2 w2) d'
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        frames_feature: torch.Tensor,
        text_tokens:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress a single video's frame features.

        Parameters
        ----------
        frames_feature : (t, h, w, encoder_dim)
            Patch token grid for all frames. Single sample, no batch dim.
        text_tokens    : (1, L, text_dim)
            Question token embeddings from Qwen2.5-VL embed_tokens (frozen),
            projected to encoder_dim inside GuideInjector.

        Returns
        -------
        compressed : (t_out, h_out, w_out, llm_dim)
            t_out = ceil(t / temporal_k)
            h_out = ceil(h / spatial_k)
            w_out = ceil(w / spatial_k)
        """
        t, h, w, d = frames_feature.shape

        # ── Step 1: Create query by downsampling visual features ──────────
        # Trilinear interpolation matches HICom's approach.
        t_out = math.ceil(t / self.temporal_k)
        h_out = math.ceil(h / self.spatial_k)
        w_out = math.ceil(w / self.spatial_k)

        query = F.interpolate(
            einops.rearrange(frames_feature, 't h w d -> 1 d t h w'),
            size=(t_out, h_out, w_out),
            mode='trilinear',
            align_corners=False,
        )                                                  # (1, d, t_out, h_out, w_out)
        query = einops.rearrange(
            query, '1 d t h w -> t h w d'
        )                                                  # (t_out, h_out, w_out, d)

        # ── Step 2: Inject question into query via GuideInjector ──────────
        # Flatten spatial dims for attention, then restore
        query_flat = einops.rearrange(
            query, 't h w d -> (t h w) d'
        )                                                  # (n_out, d)
        query_flat = self.guide_injector(
            query_flat, text_tokens
        )                                                  # (n_out, d)
        query = einops.rearrange(
            query_flat, '(t h w) d -> t h w d',
            t=t_out, h=h_out, w=w_out
        )

        # ── Step 3: Divide key and value into windows ─────────────────────
        # key and value come from the original full-resolution features
        key_groups   = self._divide_feature(
            frames_feature,
            (self.temporal_k, self.spatial_k, self.spatial_k)
        )                                                  # (n_groups, group_size, d)
        value_groups = key_groups                          # same features for K and V

        # ── Step 4: Query groups (one query per output position) ──────────
        # Use kernel (1,1,1) to get one-token "groups" — each is just one
        # output query token in the right order
        query_groups = self._divide_feature(
            query, (1, 1, 1)
        )                                                  # (n_groups, 1, d)

        # ── Step 5: Windowed attention ────────────────────────────────────
        # Each query attends only to its corresponding window
        # Shapes: (n_groups, 1, d) × (n_groups, d, group_size) → (n_groups, 1, group_size)
        attn_weights = torch.softmax(
            torch.bmm(query_groups, key_groups.permute(0, 2, 1))
            / math.sqrt(self.encoder_dim),
            dim=-1,
        )                                                  # (n_groups, 1, group_size)

        out = torch.bmm(attn_weights, value_groups)        # (n_groups, 1, d)

        # ── Step 6: Reshape back to spatial grid and project ──────────────
        out = einops.rearrange(
            out,
            '(t h w) 1 d -> t h w d',
            t=t_out, h=h_out, w=w_out
        )

        return self.readout(out)                           # (t_out, h_out, w_out, llm_dim)


# ── GlobalCompressor ──────────────────────────────────────────────────────────

class GlobalCompressor(nn.Module):
    """
    Global cross-clip compressor with learnable query tokens.

    Adapted from HICom GlobalCompressor (Liu et al., CVPR 2025).

    A small fixed set of learnable tokens attend to ALL flattened frame
    tokens from the entire clip. Each learnable token is first conditioned
    on the question via GuideInjector, then attends to the full
    (t × h × w) token sequence.

    3D sinusoidal positional embeddings are added to the frame tokens
    before attention so the model can distinguish temporal and spatial
    positions across the clip.

    For surgical video, this module captures:
      - Which phase dominates the clip
      - Whether a transition occurs and roughly when
      - The overall instrument usage pattern

    Parameters
    ----------
    encoder_dim    : ViT hidden dimension
    llm_dim        : LLM hidden dimension (readout output)
    num_queries    : number of learnable global query tokens
    guide_injector : pre-built GuideInjector instance
    num_heads      : attention heads
    max_num_frames : maximum frames for positional embedding cache
    max_spatial_hw : maximum spatial grid size for positional embedding cache.
                     Pass patch_grid_hw (27) for the slow stream,
                     fast_spatial_size (5) for the fast stream.
    mlp_depth      : readout MLP depth
    """

    def __init__(
        self,
        encoder_dim:    int,
        llm_dim:        int,
        num_queries:    int,
        guide_injector: GuideInjector,
        num_heads:      int,
        max_num_frames: int  = 256,
        max_spatial_hw: int  = 27,
        mlp_depth:      int  = 2,
        norm_layer       = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.encoder_dim    = encoder_dim
        self.num_queries    = num_queries
        self.guide_injector = guide_injector

        # Learnable global query tokens — initialised to zeros (HICom convention)
        self.query_tokens = nn.Parameter(
            torch.zeros(num_queries, encoder_dim)
        )

        # Cross-attention: learnable queries attend to all frame tokens
        self.attn_layer = MultiheadAttention(encoder_dim, num_heads)

        # Readout: project encoder_dim → llm_dim
        self.readout = build_mlp(mlp_depth, encoder_dim, llm_dim)

        # 3D positional embedding cache (per-stream spatial extent).
        self.max_size = [max_num_frames, max_spatial_hw, max_spatial_hw]
        self._build_pos_cache(max_num_frames, max_spatial_hw, max_spatial_hw)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _build_pos_cache(self, max_t: int, max_h: int, max_w: int) -> None:
        """Build and register the 3D positional embedding cache as a buffer."""
        pos_embed = torch.from_numpy(
            get_3d_position_embedding(max_t, max_h, max_w, self.encoder_dim)
        ).float()
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _maybe_resize_pos_cache(
        self, t: int, h: int, w: int, device: torch.device
    ) -> None:
        """Expand the positional embedding cache if needed."""
        needs_rebuild = (
            t > self.max_size[0]
            or h > self.max_size[1]
            or w > self.max_size[2]
        )
        if needs_rebuild:
            self.max_size = [
                max(t, self.max_size[0]),
                max(h, self.max_size[1]),
                max(w, self.max_size[2]),
            ]
            self._build_pos_cache(*self.max_size)
            self.pos_embed = self.pos_embed.to(device)

    def forward(
        self,
        frames_feature: torch.Tensor,
        text_tokens:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress a single video into a fixed number of global tokens.

        Parameters
        ----------
        frames_feature : (t, h, w, encoder_dim)
        text_tokens    : (1, L, text_dim)

        Returns
        -------
        global_tokens : (num_queries, llm_dim)
        """
        t, h, w, d = frames_feature.shape

        # ── Add 3D positional embeddings ──────────────────────────────────
        self._maybe_resize_pos_cache(t, h, w, frames_feature.device)
        pos = self.pos_embed[:t, :h, :w, :].to(frames_feature.dtype)
        frames_pos = frames_feature + pos                  # (t, h, w, d)

        # ── Condition learnable queries on the question ───────────────────
        conditioned_queries = self.guide_injector(
            self.query_tokens, text_tokens
        )                                                  # (num_queries, d)

        # ── Cross-attention over all frame tokens ─────────────────────────
        batch_q  = conditioned_queries.unsqueeze(0)        # (1, num_queries, d)
        batch_kv = einops.rearrange(
            frames_pos, 't h w d -> 1 (t h w) d'
        )                                                  # (1, t*h*w, d)

        attn_out, _ = self.attn_layer(
            batch_q, batch_kv, batch_kv
        )                                                  # (1, num_queries, d)
        attn_out = attn_out.squeeze(0)                     # (num_queries, d)

        # Residual connection (HICom: query + attn_out) then readout
        global_tokens = self.readout(
            conditioned_queries + attn_out
        )                                                  # (num_queries, llm_dim)

        return global_tokens


# ── SurgQCompressor ───────────────────────────────────────────────────────────

class SurgQCompressor(nn.Module):
    """
    Full visual token compression module for SurgQ-VLM.

    Orchestrates the complete pipeline from raw frames + question strings
    to a single flat sequence of compressed visual tokens ready for the LLM.

    Architecture
    ------------
    ViT encoder (frozen, Qwen2.5-VL, embed_dim=1280)
        ↓
    Fast stream spatial pooling 27×27 → 5×5
        ↓
    Qwen2.5-VL embed_tokens (frozen) + projection → question embeddings
        ↓
    Per-sample loop:
        slow LocalCompressor  → (5, 9, 9, llm_dim)   = 405 tokens
        fast LocalCompressor  → (10, 5, 5, llm_dim)  = 250 tokens
        slow GlobalCompressor → (8,  llm_dim)
        fast GlobalCompressor → (8,  llm_dim)
        ↓
    Concatenate all → (671, llm_dim) per sample
        ↓
    Stack batch → (B, 671, llm_dim)

    The GuideInjector instances are NOT shared between slow/fast streams —
    each stream has its own set so they can learn stream-specific
    question-visual relationships (four independent injectors in total).

    Parameters
    ----------
    config : SurgQCompressorConfig
    """

    def __init__(self, config: SurgQCompressorConfig):
        super().__init__()
        self.config           = config
        self.use_mock_encoder = config.use_mock_encoder

        d  = config.encoder_dim
        ld = config.llm_dim
        h  = config.num_attn_heads
        md = config.mlp_depth

        # ── Question text embedding source ────────────────────────────────
        # We use Qwen2.5-VL's own embed_tokens layer (frozen) to encode
        # questions. This keeps text and vision in the same representational
        # family as the LLM that generates the answer.
        #
        # In mock mode we skip loading any model and use random tensors of
        # shape (B, L, encoder_dim) so no projection is needed.
        #
        # In real mode embed_tokens outputs (B, L, llm_embed_dim) where
        # llm_embed_dim is 2048 (3B) or 3584 (7B). GuideInjector projects
        # this down to encoder_dim (1280) internally via text_proj.
        if not config.use_mock_encoder:
            # embed_tokens and tokenizer are passed in at training time via
            # set_text_encoder() rather than loaded here, so that this module
            # can be constructed before the full Qwen model is loaded.
            self._text_embed_dim = config.llm_embed_dim
            self.qwen_tokenizer  = None   # set via set_text_encoder()
            self.qwen_embed      = None   # set via set_text_encoder()
        else:
            # Mock: produce text tokens with same dim as encoder (no projection).
            self._text_embed_dim = d

        # ── GuideInjectors (one per stream × one per compressor level) ────
        # Each stream gets independent injectors so slow and fast can learn
        # different question-visual relationships.
        self.guide_slow_local  = GuideInjector(d, self._text_embed_dim, h, md)
        self.guide_fast_local  = GuideInjector(d, self._text_embed_dim, h, md)
        self.guide_slow_global = (
            GuideInjector(d, self._text_embed_dim, h, md)
            if config.n_global_slow > 0 else None
        )
        self.guide_fast_global = (
            GuideInjector(d, self._text_embed_dim, h, md)
            if config.n_global_fast > 0 else None
        )

        # ── Local compressors ─────────────────────────────────────────────
        self.slow_local = LocalCompressor(
            encoder_dim    = d,
            llm_dim        = ld,
            temporal_k     = config.slow_temporal_k,
            spatial_k      = config.slow_spatial_k,
            guide_injector = self.guide_slow_local,
            mlp_depth      = md,
        )
        self.fast_local = LocalCompressor(
            encoder_dim    = d,
            llm_dim        = ld,
            temporal_k     = config.fast_temporal_k,
            spatial_k      = config.fast_spatial_k,
            guide_injector = self.guide_fast_local,
            mlp_depth      = md,
        )

        # ── Global compressors ────────────────────────────────────────────
        # max_spatial_hw is stream-specific: slow → 27 (full ViT patch grid),
        # fast → 5 (after fast_spatial_size pooling). This prevents building
        # a wasteful 256×27×27×1280 pos-embed buffer for the fast stream.
        if config.n_global_slow > 0:
            self.slow_global = GlobalCompressor(
                encoder_dim    = d,
                llm_dim        = ld,
                num_queries    = config.n_global_slow,
                guide_injector = self.guide_slow_global,
                num_heads      = h,
                max_num_frames = config.max_num_frames,
                max_spatial_hw = config.slow_max_hw,
                mlp_depth      = md,
            )
        else:
            self.slow_global = None
        if config.n_global_fast > 0:
            self.fast_global = GlobalCompressor(
                encoder_dim    = d,
                llm_dim        = ld,
                num_queries    = config.n_global_fast,
                guide_injector = self.guide_fast_global,
                num_heads      = h,
                max_num_frames = config.max_num_frames,
                max_spatial_hw = config.fast_max_hw,
                mlp_depth      = md,
            )
        else:
            self.fast_global = None

        logger.info(
            "SurgQCompressor built | encoder_dim=%d | llm_dim=%d | "
            "slow_k=(%d,%d) | fast_k=(%d,%d) | global=(%d+%d) | "
            "text_dim=%d | mock=%s",
            d, ld,
            config.slow_temporal_k, config.slow_spatial_k,
            config.fast_temporal_k, config.fast_spatial_k,
            config.n_global_slow, config.n_global_fast,
            self._text_embed_dim,
            config.use_mock_encoder,
        )

    # ── Text encoder setup ────────────────────────────────────────────────

    def set_text_encoder(
        self,
        tokenizer,
        embed_tokens: nn.Embedding,
    ) -> None:
        """
        Attach Qwen2.5-VL's tokenizer and embed_tokens layer for question
        encoding. Called automatically by SurgQVLM._build_real_model() after
        both Qwen and the compressor are constructed.

        embed_tokens is kept frozen — gradients do not flow into it.
        The text_proj MLP inside each GuideInjector is the only trainable
        parameter on the text side.

        Parameters
        ----------
        tokenizer    : Qwen2.5-VL tokenizer (AutoTokenizer)
        embed_tokens : nn.Embedding from the Qwen text backbone
        """
        self.qwen_tokenizer = tokenizer
        self.qwen_embed     = embed_tokens
        self.qwen_embed.requires_grad_(False)
        logger.info(
            "Text encoder set: embed_tokens dim=%d",
            embed_tokens.embedding_dim,
        )

    # ── Encoding helpers ──────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_text(
        self, questions: list[str], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Encode a batch of question strings using Qwen's embed_tokens layer.

        This produces raw token embeddings — just the lookup table output,
        no transformer layers. The GuideInjector's text_proj MLP then
        projects these from llm_embed_dim → encoder_dim before cross-attention.

        Using Qwen's own vocabulary ensures the text representation used for
        compression is in the same semantic family as the text representation
        used for answer generation.

        Parameters
        ----------
        questions : list[str] length B
        device    : target device
        dtype     : target dtype

        Returns
        -------
        text_tokens : (B, L, llm_embed_dim)
            Raw token embeddings. L ≤ 64 (surgical QA questions peak at
            ~30 subword tokens).
        """
        if self.qwen_tokenizer is None or self.qwen_embed is None:
            raise RuntimeError(
                "Text encoder not initialised. Call set_text_encoder() "
                "with the Qwen tokenizer and embed_tokens before forward()."
            )

        inputs = self.qwen_tokenizer(
            questions,
            return_tensors  = "pt",
            padding         = True,
            truncation      = True,
            max_length      = 64,       # generous for surgical QA (~30 tokens max)
        ).to(device)

        # embed_tokens: token_ids (B, L) → embeddings (B, L, llm_embed_dim)
        embeddings = self.qwen_embed(inputs.input_ids)
        return embeddings.to(dtype)

    @torch.no_grad()
    def _encode_frames_mock(
        self,
        B: int, N: int, hw: int, d: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return random frame tokens for mock/testing mode."""
        return torch.randn(B, N, hw, hw, d, device=device, dtype=dtype)

    # ── Fast stream spatial pooling ───────────────────────────────────────

    def _pool_fast_spatial(
        self, fast_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool fast stream patch tokens spatially from 27×27 to 5×5.

        Uses trilinear interpolation (same as HICom's visual_compressor).
        Operates on the spatial dims only; the temporal dim is unchanged.

        Parameters
        ----------
        fast_tokens : (B, N_fast, h, w, d)

        Returns
        -------
        pooled : (B, N_fast, target_s, target_s, d)
        """
        B, N, h, w, d = fast_tokens.shape
        s = self.config.fast_spatial_size

        # Reshape for F.interpolate which expects (B, C, D, H, W)
        x = einops.rearrange(fast_tokens, 'b n h w d -> b d n h w')
        x = F.interpolate(
            x, size=(N, s, s), mode='trilinear', align_corners=False
        )
        return einops.rearrange(x, 'b d n h w -> b n h w d')

    # ── Main forward ──────────────────────────────────────────────────────

    def forward(
        self,
        slow_frames: torch.Tensor,
        fast_frames: torch.Tensor,
        questions:   list[str],
    ) -> torch.Tensor:
        """
        Compress slow + fast video streams conditioned on questions.

        This standalone path (mock ViT encoding) is used by the smoke test.
        In real training, ViT encoding happens inside SurgQVLM and the
        pre-encoded tokens are routed through SurgQVLM._compress(), which
        calls this module's sub-components directly.

        Parameters
        ----------
        slow_frames : (B, N_slow, 3, H, W)
        fast_frames : (B, N_fast, 3, H, W)
        questions   : list[str] length B

        Returns
        -------
        visual_tokens : (B, T_total, llm_dim)
            T_total = slow_local + fast_local + n_global_slow + n_global_fast
                    = 405 + 250 + 8 + 8 = 671  (default 30s clip config)
        """
        B      = slow_frames.shape[0]
        device = slow_frames.device
        dtype  = slow_frames.dtype
        d      = self.config.encoder_dim

        # ── Step 1: Encode frames ─────────────────────────────────────────
        N_slow = slow_frames.shape[1]
        N_fast = fast_frames.shape[1]
        hw     = self.config.patch_grid_hw

        if self.use_mock_encoder:
            slow_tokens = self._encode_frames_mock(B, N_slow, hw, d, device, dtype)
            fast_tokens = self._encode_frames_mock(B, N_fast, hw, d, device, dtype)
        else:
            raise NotImplementedError(
                "Real ViT encoding is handled by Qwen2.5-VL in the full "
                "model assembly. Use use_mock_encoder=True for standalone "
                "testing, or call this module from within SurgQVLM."
            )

        # ── Step 2: Pool fast stream spatially (27×27 → 5×5) ─────────────
        fast_tokens = self._pool_fast_spatial(fast_tokens)

        # ── Step 3: Encode questions ──────────────────────────────────────
        if self.use_mock_encoder:
            text_tokens = torch.randn(B, 10, d, device=device, dtype=dtype)
        else:
            text_tokens = self._encode_text(questions, device, dtype)

        # ── Step 4: Per-sample compression loop ───────────────────────────
        # LocalCompressor and GlobalCompressor operate on (t, h, w, d)
        # without a batch dimension (inherited from HICom's design).
        all_tokens: list[torch.Tensor] = []

        for i in range(B):
            slow_feat = slow_tokens[i]                # (N_slow, hw, hw, d)
            fast_feat = fast_tokens[i]                # (N_fast, fast_s, fast_s, d)
            txt       = text_tokens[i].unsqueeze(0)   # (1, L, text_dim)

            # Local compression — question-conditioned windowed attention
            sl = self.slow_local(slow_feat, txt)
            fl = self.fast_local(fast_feat, txt)

            # Global compression — learnable queries over the full clip
            sl_flat = einops.rearrange(sl, 't h w d -> (t h w) d')
            fl_flat = einops.rearrange(fl, 't h w d -> (t h w) d')
            ld = sl_flat.shape[-1]
            if self.slow_global is not None:
                sg = self.slow_global(slow_feat, txt)   # (n_global_slow, ld)
            else:
                sg = torch.zeros(0, ld, dtype=sl_flat.dtype, device=sl_flat.device)
            if self.fast_global is not None:
                fg = self.fast_global(fast_feat, txt)   # (n_global_fast, ld)
            else:
                fg = torch.zeros(0, ld, dtype=fl_flat.dtype, device=fl_flat.device)

            # [slow_local | fast_local | slow_global | fast_global]
            sample_tokens = torch.cat([sl_flat, fl_flat, sg, fg], dim=0)
            all_tokens.append(sample_tokens)

        return torch.stack(all_tokens, dim=0)   # (B, T_total, llm_dim)

    # ── Utility ───────────────────────────────────────────────────────────

    def count_trainable_params(self) -> int:
        """Number of trainable parameters (embed_tokens is frozen externally)."""
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def expected_output_tokens(self) -> int:
        """Expected number of output tokens for a standard 30-second clip."""
        cfg = self.config
        slow_local = (
            math.ceil(15 / cfg.slow_temporal_k)
            * math.ceil(cfg.patch_grid_hw / cfg.slow_spatial_k) ** 2
        )
        fast_local = (
            math.ceil(30 / cfg.fast_temporal_k)
            * cfg.fast_spatial_size ** 2
        )
        return slow_local + fast_local + cfg.n_global_slow + cfg.n_global_fast


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone smoke test. Run from the repository root:

        python -m surgqvlm.compressor

    Verifies:
      1. Config instantiation
      2. Module construction
      3. Forward pass shapes (mock encoder — no GPU or model weights needed)
      4. Trainable parameter count
      5. Token budget computation
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="SurgQCompressor smoke test")
    parser.add_argument("--llm_dim",    type=int,  default=3584,
                        help="LLM hidden dim: 2048 (3B) or 3584 (7B)")
    parser.add_argument("--batch_size", type=int,  default=2)
    parser.add_argument("--n_slow",     type=int,  default=15,
                        help="Number of slow stream frames (30s clip @ 0.5fps)")
    parser.add_argument("--n_fast",     type=int,  default=30,
                        help="Number of fast stream frames (30s clip @ 1fps)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  SurgQCompressor Smoke Test (mock encoder mode)")
    print(f"{'='*60}\n")

    cfg = SurgQCompressorConfig(
        llm_dim          = args.llm_dim,
        use_mock_encoder = True,
    )

    print("Config:")
    print(f"  encoder_dim     : {cfg.encoder_dim}")
    print(f"  llm_dim         : {cfg.llm_dim}")
    print(f"  patch_grid_hw   : {cfg.patch_grid_hw}")
    print(f"  slow_kernel     : temporal={cfg.slow_temporal_k}  "
          f"spatial={cfg.slow_spatial_k}")
    print(f"  fast_kernel     : temporal={cfg.fast_temporal_k}  "
          f"spatial={cfg.fast_spatial_k}")
    print(f"  fast_pool_size  : {cfg.fast_spatial_size}x{cfg.fast_spatial_size}")
    print(f"  global_queries  : slow={cfg.n_global_slow}  "
          f"fast={cfg.n_global_fast}")

    model = SurgQCompressor(cfg)
    model.eval()

    n_params = model.count_trainable_params()
    expected_tokens = model.expected_output_tokens()

    print(f"\nModel built successfully.")
    print(f"  Trainable parameters : {n_params:,}")
    print(f"  Expected output tokens (30s clip) : {expected_tokens}")

    B  = args.batch_size
    H  = W = 384
    Ns = args.n_slow
    Nf = args.n_fast

    slow_frames = torch.randn(B, Ns, 3, H, W)
    fast_frames = torch.randn(B, Nf, 3, H, W)
    questions   = [
        "What instrument is being used to dissect the cystic duct?",
        "At what second does the phase transition occur in this clip?",
    ][:B]

    print(f"\nInput shapes:")
    print(f"  slow_frames : {tuple(slow_frames.shape)}  (B, N_slow, C, H, W)")
    print(f"  fast_frames : {tuple(fast_frames.shape)}  (B, N_fast, C, H, W)")

    print("\nRunning forward pass...")
    with torch.no_grad():
        visual_tokens = model(slow_frames, fast_frames, questions)

    print(f"\n  visual_tokens : {tuple(visual_tokens.shape)}  (B, T_total, llm_dim)")

    slow_local_count = (
        math.ceil(Ns / cfg.slow_temporal_k)
        * math.ceil(cfg.patch_grid_hw / cfg.slow_spatial_k) ** 2
    )
    fast_local_count = (
        math.ceil(Nf / cfg.fast_temporal_k)
        * cfg.fast_spatial_size ** 2
    )
    total = (slow_local_count + fast_local_count
             + cfg.n_global_slow + cfg.n_global_fast)

    print(f"\nToken budget breakdown:")
    print(f"  Slow local  : {slow_local_count}")
    print(f"  Fast local  : {fast_local_count}")
    print(f"  Global      : {cfg.n_global_slow} + {cfg.n_global_fast}")
    print(f"  Total       : {total}")
    assert visual_tokens.shape == (B, total, cfg.llm_dim), \
        f"Shape mismatch: {visual_tokens.shape} != {(B, total, cfg.llm_dim)}"
    print(f"  Output shape matches: OK")

    vt = visual_tokens[0]
    assert not torch.isnan(vt).any(), "NaN in output"
    assert not torch.isinf(vt).any(), "Inf in output"
    print(f"\nValue check (sample 0): mean={vt.mean():.4f}  std={vt.std():.4f}  "
          f"NaN/Inf: none")

    print(f"\nSmoke test passed.\n")
