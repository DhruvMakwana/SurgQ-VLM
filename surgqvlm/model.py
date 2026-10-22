"""
model.py — SurgQVLM Full Model Assembly
========================================

SurgQVLM wraps Qwen2.5-VL and intercepts the visual token pipeline to insert
the SurgQCompressor between the ViT encoder and the LLM, replacing Qwen's
PatchMerger with our question-conditioned dual-stream compression module.

Architecture
------------

  Qwen2.5-VL normal flow:
    pixel_values
        → model.visual (ViT, embed_dim=1280)         [frozen]
        → PatchMerger (5120 → llm_dim)               [bypassed]
        → image_embeds (N_tokens, llm_dim)
        → masked_scatter into input_embeds
        → LLM layers                                  [LoRA adapters]

  SurgQVLM flow:
    slow_frames (B, N_slow, 3, H, W)
    fast_frames (B, N_fast, 3, H, W)
        → model.visual (ViT, embed_dim=1280)         [frozen, shared]
        → SurgQCompressor                            [trainable]
              ├── slow LocalCompressor
              ├── fast LocalCompressor
              ├── slow GlobalCompressor
              └── fast GlobalCompressor
        → compressed tokens (671 per sample, llm_dim)
        → injected into input_embeds via the video-pad token mask
        → LLM layers                                  [LoRA adapters]

Key design decisions
--------------------
1. ViT interception via forward hook: We register a pre-forward hook on
   Qwen's PatchMerger, letting the ViT run its exact intended forward path
   (packed-sequence attention, window attention, rotary embeddings) and
   capturing the 1280-dim patch tokens at the merger input. The merger
   itself is bypassed; the SurgQCompressor readout MLP acts as the new
   projector (1280 → llm_dim). This is safer than calling ViT blocks
   directly, which risks mismatching cu_seqlens handling and window
   attention internals.

2. Token injection: We build a flat token sequence matching the number of
   <|video_pad|> tokens in the input_ids, replacing them with compressed
   tokens using masked_scatter — identical to Qwen's own flow.

3. LoRA: Applied to Qwen's LLM attention and FFN layers. The ViT and the
   compressor are frozen/trainable respectively. After PEFT wrapping, all
   "visual.*" parameters are re-frozen — get_peft_model matches module
   names (q_proj, k_proj, ...) inside the ViT too and would otherwise add
   trainable LoRA adapters there.

4. Dtype: Qwen2.5-VL is loaded in bfloat16, so the compressor is cast to
   bfloat16 after construction to match the ViT output dtype.

5. Device placement: device_map="auto" is NOT used during training because
   it shards the model in ways that conflict with the custom forward pass.
   Use .to_device(device) for single-GPU training.

Smoke test (CPU, no weights downloaded):

    python -m surgqvlm.model

References
----------
  HICom : https://github.com/lntzm/HICom  (Liu et al., CVPR 2025)
  Qwen2.5-VL : https://github.com/QwenLM/Qwen2.5-VL
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from surgqvlm.compressor import SurgQCompressor, SurgQCompressorConfig

logger = logging.getLogger(__name__)


# ── Model size presets ────────────────────────────────────────────────────────

# Verified from Qwen2.5-VL config.json.
# ViT embed_dim is 1280 for both 3B and 7B (same ViT backbone).
# PatchMerger projects: embed_dim×merge_size² → llm_hidden_size
#   3B: 1280×4 = 5120 → 2048
#   7B: 1280×4 = 5120 → 3584
QWEN_MODEL_CONFIGS = {
    "3b": {
        "hf_name":       "Qwen/Qwen2.5-VL-3B-Instruct",
        "llm_dim":        2048,
        "vit_embed_dim":  1280,
        "num_layers":     36,
    },
    "7b": {
        "hf_name":       "Qwen/Qwen2.5-VL-7B-Instruct",
        "llm_dim":        3584,
        "vit_embed_dim":  1280,
        "num_layers":     28,
    },
}


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    """
    LoRA adapter configuration for the Qwen2.5-VL LLM backbone.

    Attributes
    ----------
    lora_mode : str
        "attention_only"    — q, k, v, o projections. Fewer params, faster.
        "attention_and_ffn" — attention + gate/up/down FFN projections.
                              Used for all paper results.
    r : int
        LoRA rank (16 for all paper results).
    lora_alpha : float
        Scaling factor. Convention: alpha = 2 × r.
    lora_dropout : float
        Dropout on LoRA weights (0.0 for the single-epoch paper runs).
    modules_to_save : list[str]
        Modules to keep fully trainable outside the LoRA adapters.
        Default empty — note that Qwen2.5-VL ties embed_tokens and lm_head,
        so putting "lm_head" here would clone and decouple the tied tensor,
        wasting ~4 GB of optimizer memory with no standard-practice benefit.
    """

    lora_mode:        str        = "attention_and_ffn"
    r:                int        = 16
    lora_alpha:       float      = 32.0
    lora_dropout:     float      = 0.0
    modules_to_save:  list[str]  = field(default_factory=lambda: [])

    # Target modules per mode — resolved at build time
    _ATTENTION_MODULES: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        repr=False,
    )
    _FFN_MODULES: list[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"],
        repr=False,
    )

    def target_modules(self) -> list[str]:
        """Return the list of module names to apply LoRA to."""
        if self.lora_mode == "attention_only":
            return self._ATTENTION_MODULES
        elif self.lora_mode == "attention_and_ffn":
            return self._ATTENTION_MODULES + self._FFN_MODULES
        else:
            raise ValueError(
                f"Unknown lora_mode '{self.lora_mode}'. "
                f"Choose from: attention_only, attention_and_ffn"
            )


@dataclass
class SurgQVLMConfig:
    """
    Top-level configuration for SurgQVLM.

    Attributes
    ----------
    model_size : str
        Qwen2.5-VL size: "7b" (paper default) or "3b".
        Controls llm_dim, hf_name automatically.
    compressor : SurgQCompressorConfig
        Configuration for the dual-stream compressor.
        encoder_dim and llm_dim are auto-filled from model_size.
    lora : LoRAConfig
        LoRA adapter configuration for the LLM backbone.
    use_mock_encoder : bool
        If True, skip all HuggingFace model loading and return random
        tensors. For smoke tests and shape verification on CPU.
    freeze_vit : bool
        Whether to freeze Qwen2.5-VL's ViT encoder. Always True in the
        paper. When True, a post-LoRA re-freeze pass also suppresses the
        LoRA adapters that get_peft_model would otherwise add to ViT
        attention layers.
    patch_grid_hw : int
        ViT patches per side at the fixed input resolution.
        384×384 input → floor(384/14) = 27.
    max_new_tokens : int
        Default maximum tokens to generate during inference.
    """

    model_size:       str                   = "7b"
    compressor:       SurgQCompressorConfig = field(
        default_factory=SurgQCompressorConfig
    )
    lora:             LoRAConfig            = field(
        default_factory=LoRAConfig
    )
    use_mock_encoder: bool                  = False
    freeze_vit:       bool                  = True
    patch_grid_hw:    int                   = 27
    max_new_tokens:   int                   = 512

    def __post_init__(self):
        """Auto-fill compressor dims from model_size after init."""
        if self.model_size not in QWEN_MODEL_CONFIGS:
            raise ValueError(
                f"model_size must be one of {list(QWEN_MODEL_CONFIGS)}, "
                f"got '{self.model_size}'"
            )
        preset = QWEN_MODEL_CONFIGS[self.model_size]

        # Sync compressor dims with the selected model size
        self.compressor.encoder_dim      = preset["vit_embed_dim"]  # 1280
        self.compressor.llm_dim          = preset["llm_dim"]        # 2048/3584
        self.compressor.llm_embed_dim    = preset["llm_dim"]        # same
        self.compressor.patch_grid_hw    = self.patch_grid_hw       # 27
        self.compressor.use_mock_encoder = self.use_mock_encoder

        # Re-derive num_attn_heads for 1280-dim: 1280 // 128 = 10
        self.compressor.num_attn_heads = (
            self.compressor.encoder_dim // 128
        )

    @property
    def hf_name(self) -> str:
        return QWEN_MODEL_CONFIGS[self.model_size]["hf_name"]

    @property
    def llm_dim(self) -> int:
        return QWEN_MODEL_CONFIGS[self.model_size]["llm_dim"]

    @property
    def vit_embed_dim(self) -> int:
        return QWEN_MODEL_CONFIGS[self.model_size]["vit_embed_dim"]


# ── SurgQVLM ──────────────────────────────────────────────────────────────────

class SurgQVLM(nn.Module):
    """
    SurgQ Vision-Language Model.

    Wraps Qwen2.5-VL with the dual-stream compressor inserted between the
    ViT encoder and the LLM. The compressor replaces Qwen's PatchMerger.

    Training flow
    -------------
    SurgQVLM.forward(slow_frames, fast_frames, input_ids, attention_mask,
                     questions, labels)
        1. Encode slow + fast frames through the frozen Qwen ViT
        2. Run SurgQCompressor → per-sample visual tokens (T_vis_i, llm_dim)
        3. Build input_embeds: embed input_ids, scatter visual tokens at
           <|video_pad|> positions
        4. Run the Qwen LLM forward with input_embeds (+ labels → loss)

    Inference flow
    --------------
    SurgQVLM.generate(...) → generated token IDs (B, ≤max_new_tokens)

    Parameters
    ----------
    config : SurgQVLMConfig
    """

    def __init__(self, config: SurgQVLMConfig):
        super().__init__()
        self.config = config

        if config.use_mock_encoder:
            self._build_mock_model(config)
        else:
            self._build_real_model(config)

        self._log_param_counts()

    # ── Model construction ────────────────────────────────────────────────

    def _build_mock_model(self, config: SurgQVLMConfig) -> None:
        """
        Build a minimal mock model for smoke testing on CPU.
        No HuggingFace weights loaded. All forward passes use random tensors.
        """
        logger.info("Building SurgQVLM in MOCK mode (no weights loaded)")

        d  = config.vit_embed_dim   # 1280
        ld = config.llm_dim         # 2048/3584

        # Minimal text embedding (vocabulary of 1000 for mock)
        self.embed_tokens = nn.Embedding(1000, ld)

        # SurgQCompressor (already in mock mode via config)
        self.compressor = SurgQCompressor(config.compressor)

        # Attach mock text encoder to compressor
        self.compressor.qwen_tokenizer = _MockTokenizer()
        self.compressor.qwen_embed     = self.embed_tokens
        self.compressor.qwen_embed.requires_grad_(False)

        # Minimal LLM: single linear layer for mock loss computation
        self.lm_head = nn.Linear(ld, 1000, bias=False)

        # Placeholders
        self.qwen_model     = None
        self.video_token_id = 42    # arbitrary mock token id

        logger.info("Mock model built | llm_dim=%d | encoder_dim=%d", ld, d)

    def _build_real_model(self, config: SurgQVLMConfig) -> None:
        """
        Load Qwen2.5-VL from HuggingFace, build the SurgQCompressor,
        apply LoRA to the LLM backbone.
        """
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoTokenizer,
        )
        from peft import LoraConfig, get_peft_model, TaskType

        logger.info("Loading Qwen2.5-VL: %s", config.hf_name)

        # ── Attention implementation selection ────────────────────────────
        #   flash_attention_2 : ~30-40 % faster forward + backward on A100.
        #                       Requires ``pip install flash-attn``.
        #   sdpa              : PyTorch 2 scaled_dot_product_attention —
        #                       always-available fallback.
        try:
            import flash_attn  # noqa: F401
            _attn_impl = "flash_attention_2"
        except Exception:
            _attn_impl = "sdpa"
        logger.info("Qwen attention implementation: %s", _attn_impl)

        # Do NOT use device_map="auto" here: it shards the model for
        # inference convenience but breaks the custom training forward
        # (direct access to model.visual submodules + a separate compressor
        # module on one device + LoRA wrapping after loading).
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.hf_name,
            torch_dtype         = torch.bfloat16,
            attn_implementation = _attn_impl,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_name)

        # ── Freeze ViT if configured ──────────────────────────────────────
        if config.freeze_vit:
            self.qwen_model.model.visual.requires_grad_(False)
            logger.info("ViT encoder frozen.")

        # Qwen's PatchMerger lives inside model.visual — already frozen
        # above; it is bypassed entirely in the forward pass.

        # ── Apply LoRA to the LLM backbone ────────────────────────────────
        lora_cfg = config.lora
        peft_config = LoraConfig(
            task_type        = TaskType.CAUSAL_LM,
            r                = lora_cfg.r,
            lora_alpha       = lora_cfg.lora_alpha,
            lora_dropout     = lora_cfg.lora_dropout,
            target_modules   = lora_cfg.target_modules(),
            modules_to_save  = lora_cfg.modules_to_save,
        )
        self.qwen_model = get_peft_model(self.qwen_model, peft_config)
        logger.info(
            "LoRA applied | mode=%s | r=%d | alpha=%.1f | targets=%s",
            lora_cfg.lora_mode, lora_cfg.r, lora_cfg.lora_alpha,
            lora_cfg.target_modules(),
        )

        # ── Re-freeze ViT after LoRA wrapping ─────────────────────────────
        # get_peft_model creates new trainable LoRA A/B matrices for every
        # module whose name matches target_modules. Qwen2.5-VL's ViT
        # attention layers use the same names (q_proj, ...) as the LLM
        # layers, so PEFT silently adds trainable adapters to the ViT even
        # though the base ViT weights are frozen. Suppress them here.
        if config.freeze_vit:
            n_suppressed = 0
            for name, param in self.qwen_model.named_parameters():
                if "visual" in name:
                    param.requires_grad_(False)
                    n_suppressed += 1
            logger.info(
                "ViT re-frozen after LoRA wrapping "
                "(%d parameters suppressed, including any LoRA A/B on ViT).",
                n_suppressed,
            )

        # ── Build the SurgQCompressor ─────────────────────────────────────
        self.compressor = SurgQCompressor(config.compressor)

        # Cast the compressor to match Qwen's dtype: the ViT outputs
        # bfloat16 patch tokens that flow directly into the compressor's
        # Linear layers; a float32 compressor would raise a RuntimeError on
        # the first real forward pass.
        qwen_dtype = next(self.qwen_model.parameters()).dtype
        self.compressor = self.compressor.to(dtype=qwen_dtype)
        logger.info("Compressor cast to %s to match Qwen dtype.", qwen_dtype)

        # ── Resolve the inner model path after LoRA wrapping ──────────────
        # get_peft_model wraps the model as:
        #   PeftModelForCausalLM
        #       └── base_model (LoraModel)
        #               └── model  (Qwen2_5_VLForConditionalGeneration)
        #                       └── model (Qwen2_5_VLModel)
        #                               └── visual, language_model, ...
        try:
            inner = self.qwen_model.base_model.model.model
        except AttributeError:
            try:
                inner = self.qwen_model.model
            except AttributeError:
                raise RuntimeError(
                    "Cannot resolve Qwen2.5-VL inner model path. "
                    "Expected either qwen_model.base_model.model.model "
                    "or qwen_model.model. Check your transformers version."
                )

        # embed_tokens lives on the text backbone:
        # inner.language_model.embed_tokens.
        text_model = inner.language_model

        # Attach Qwen's tokenizer + embed_tokens for question encoding
        self.compressor.set_text_encoder(
            tokenizer    = self.tokenizer,
            embed_tokens = text_model.embed_tokens,
        )

        # Convenience references used in forward/generate
        self.embed_tokens   = text_model.embed_tokens
        self.lm_head        = self.qwen_model.lm_head
        self.video_token_id = self.qwen_model.config.video_token_id
        self._vit_inner     = inner        # contains .visual for hook encoding
        self._llm_inner     = text_model   # Qwen2Model — returns last_hidden_state

        logger.info(
            "SurgQVLM built | model=%s | video_token_id=%d",
            config.hf_name, self.video_token_id,
        )

    # ── ViT frame encoding via forward hook ───────────────────────────────

    def _encode_frames_vit(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of frames through Qwen2.5-VL's frozen ViT,
        capturing 1280-dim patch tokens BEFORE Qwen's PatchMerger.

        Strategy: forward hook on visual.merger
        ----------------------------------------
        We let Qwen's visual encoder run its complete, unmodified forward
        pass — including window attention, cu_seqlens packing, and rotary
        embeddings — and intercept the tensor at the merger's input via a
        pre-forward hook. After the hook fires, the hook is removed and the
        merger's output is discarded. The captured tokens are reshaped to
        (BN, hw, hw, embed_dim) for the compressor.

        Gradient behaviour
        ------------------
        No @torch.no_grad() decorator here so gradients can flow back
        through the ViT when freeze_vit=False. When freeze_vit=True (the
        default) the ViT parameters have requires_grad=False and the
        captured tensor is detached, which also releases the ViT
        computation graph.

        Parameters
        ----------
        frames : (B*N, 3, H, W)
            All frames flattened. Pre-normalised with the Qwen-VL image
            mean/std by the dataloader transform.

        Returns
        -------
        patch_tokens : (B*N, patch_hw, patch_hw, vit_embed_dim)
            Raw ViT patch tokens before the PatchMerger:
            (BN, 27, 27, 1280) for 384×384 input.
        """
        visual = self._vit_inner.visual
        BN     = frames.shape[0]

        # Outward-facing spatial grid size (27) — kept unchanged for compressor.
        out_hw  = self.config.patch_grid_hw          # 27

        # Qwen's window attention groups positions in 2×2 blocks, so the
        # spatial patch grid must be even. 27 is odd → pad to 28, crop after.
        qwen_hw = out_hw if (out_hw % 2 == 0) else (out_hw + 1)   # 28
        patch   = 14
        target  = qwen_hw * patch                    # 392

        # Crop/pad frames to the exact spatial size Qwen expects.
        # Input frames are (BN, 3, H, W), typically 384×384.
        frames = frames[:, :, :target, :target].contiguous()
        pad_h  = max(0, target - frames.shape[-2])
        pad_w  = max(0, target - frames.shape[-1])
        if pad_h > 0 or pad_w > 0:
            frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        # Qwen's patch_embed uses temporal_patch_size=2 — duplicate frames.
        frames_5d = torch.stack([frames, frames], dim=2)   # (BN, 3, 2, H, W)

        # Patchify to Qwen's expected layout:
        # visual.forward() expects (total_patches, C*T*ph*pw), NOT raw images.
        pixel_values = einops.rearrange(
            frames_5d,
            "bn c t (h ph) (w pw) -> (bn h w) (c t ph pw)",
            h=qwen_hw, w=qwen_hw, ph=patch, pw=patch,
        ).contiguous()
        # pixel_values: (BN * qwen_hw * qwen_hw, 3 * 2 * 14 * 14)

        # grid_thw: one row per sample — [temporal_groups=1, h_patches, w_patches]
        grid_thw = torch.tensor(
            [[1, qwen_hw, qwen_hw]] * BN,
            dtype  = torch.long,
            device = frames.device,
        )

        # ── Register a pre-forward hook on the merger ─────────────────────
        _captured: dict = {}

        def _pre_merger_hook(module, args):
            raw = args[0]
            _captured["pre_merger"] = (
                raw.detach() if self.config.freeze_vit else raw
            )

        hook = visual.merger.register_forward_pre_hook(_pre_merger_hook)

        try:
            if self.config.freeze_vit:
                with torch.no_grad():
                    _ = visual(pixel_values, grid_thw=grid_thw)
            else:
                _ = visual(pixel_values, grid_thw=grid_thw)
        finally:
            hook.remove()

        if "pre_merger" not in _captured:
            raise RuntimeError(
                "Forward hook did not fire. The merger may not have been "
                "called during visual.forward(). Check Qwen2.5-VL internals."
            )

        # ── Reshape flat tokens to the spatial grid ───────────────────────
        flat         = _captured["pre_merger"]
        patch_tokens = flat.view(BN, qwen_hw * qwen_hw, -1)      # (BN, P, d)
        patch_tokens = einops.rearrange(
            patch_tokens, "bn (h w) d -> bn h w d", h=qwen_hw, w=qwen_hw
        )

        # Crop back to the configured 27×27 grid — the padded column/row
        # (from the 27→28 extension) consisted of zeros and can be dropped.
        if qwen_hw != out_hw:
            patch_tokens = patch_tokens[:, :out_hw, :out_hw, :].contiguous()

        return patch_tokens   # (BN, out_hw, out_hw, embed_dim)

    def _encode_frames_mock(
        self,
        B: int, N: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return random patch tokens for mock mode."""
        hw = self.config.patch_grid_hw
        d  = self.config.vit_embed_dim
        return torch.randn(B, N, hw, hw, d, device=device, dtype=dtype)

    # ── Token injection ───────────────────────────────────────────────────

    def _build_inputs_embeds(
        self,
        input_ids:      torch.Tensor,
        visual_tokens:  list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build the full input embedding sequence by replacing <|video_pad|>
        placeholder tokens with the compressed visual tokens.

        Mirrors Qwen's own masked_scatter logic.

        Parameters
        ----------
        input_ids     : (B, L) — tokenized prompt with <|video_pad|> placeholders
        visual_tokens : list of B tensors, each (T_vis_i, llm_dim).
                        T_vis_i may differ across samples (variable-length clips).

        Returns
        -------
        inputs_embeds : (B, L, llm_dim)

        Notes
        -----
        The number of <|video_pad|> tokens in input_ids[i] must equal
        visual_tokens[i].shape[0]. masked_scatter fills True positions in
        row-major order across the whole (B, L, ld) tensor, so concatenating
        the per-sample token tensors in order maps sample i's tokens to
        sample i's placeholder positions even with variable T_vis.
        """
        B, L = input_ids.shape

        # Start with text token embeddings
        inputs_embeds = self.embed_tokens(input_ids)          # (B, L, ld)

        # Mask of video placeholder positions
        video_mask = (input_ids == self.video_token_id)       # (B, L)

        # Verify per-sample token count matches placeholder count
        n_placeholders = video_mask.sum(dim=1)
        for i, vt in enumerate(visual_tokens):
            if vt.shape[0] != n_placeholders[i].item():
                raise ValueError(
                    f"Sample {i}: input_ids has {n_placeholders[i].item()} "
                    f"<|video_pad|> tokens but visual_tokens[{i}] has "
                    f"{vt.shape[0]} tokens."
                )

        video_mask_expanded = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        flat_visual = torch.cat(visual_tokens, dim=0)         # (sum(T_vis_i), ld)
        # In real training, autocast wraps the compressor Linear layers and
        # emits bf16 visual tokens, but embed_tokens is an nn.Embedding
        # lookup that autocast does NOT wrap — cast to match before scatter.
        if flat_visual.dtype != inputs_embeds.dtype:
            flat_visual = flat_visual.to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(
            video_mask_expanded, flat_visual
        )

        return inputs_embeds

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        slow_frames:    torch.Tensor,
        fast_frames:    torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        questions:      list[str],
        labels:         Optional[torch.Tensor] = None,
        slow_mask:      Optional[torch.Tensor] = None,
        fast_mask:      Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full forward pass for training.

        Parameters
        ----------
        slow_frames    : (B, N_slow, 3, H, W)
        fast_frames    : (B, N_fast, 3, H, W)
        input_ids      : (B, L) tokenized [system + video_pads + question + answer]
        attention_mask : (B, L)
        questions      : list[str] length B — raw question text
        labels         : (B, L) optional, -100 for non-answer tokens
        slow_mask      : (B, N_slow) bool — True for real (unpadded) frames
        fast_mask      : (B, N_fast) bool

        Returns
        -------
        dict with keys:
            loss          : scalar (if labels provided)
            logits        : (B, L, vocab_size)
            visual_tokens : list of B tensors (T_vis_i, llm_dim)
        """
        B      = slow_frames.shape[0]
        device = slow_frames.device
        dtype  = slow_frames.dtype

        # ── Step 1: Encode frames ─────────────────────────────────────────
        N_slow = slow_frames.shape[1]
        N_fast = fast_frames.shape[1]

        if self.config.use_mock_encoder:
            slow_tokens = self._encode_frames_mock(B, N_slow, device, dtype)
            fast_tokens = self._encode_frames_mock(B, N_fast, device, dtype)
        else:
            # Flatten batch and frame dims for efficient single ViT pass
            slow_flat = einops.rearrange(slow_frames, 'b n c h w -> (b n) c h w')
            fast_flat = einops.rearrange(fast_frames, 'b n c h w -> (b n) c h w')

            slow_enc = self._encode_frames_vit(slow_flat)    # (B*N_slow, hw, hw, d)
            fast_enc = self._encode_frames_vit(fast_flat)    # (B*N_fast, hw, hw, d)

            slow_tokens = einops.rearrange(
                slow_enc, '(b n) h w d -> b n h w d', b=B, n=N_slow
            )
            fast_tokens = einops.rearrange(
                fast_enc, '(b n) h w d -> b n h w d', b=B, n=N_fast
            )

        # ── Step 2: Question-conditioned compression ──────────────────────
        # Per-sample real frame counts from masks let _compress() trim
        # padding frames before compression. Without this, T_vis would be
        # computed from the padded batch-max length and mismatch the n_vis
        # already embedded in input_ids (which used the real frame count).
        slow_real_n = (
            [int(slow_mask[i].sum().item()) for i in range(B)]
            if slow_mask is not None else None
        )
        fast_real_n = (
            [int(fast_mask[i].sum().item()) for i in range(B)]
            if fast_mask is not None else None
        )
        visual_tokens = self._compress(
            slow_tokens, fast_tokens, questions,
            slow_real_n=slow_real_n, fast_real_n=fast_real_n,
        )
        # → list of B tensors, each (T_vis_i, llm_dim)

        # ── Step 3: Build input embeddings ────────────────────────────────
        inputs_embeds = self._build_inputs_embeds(input_ids, visual_tokens)

        # ── Step 4: LLM forward ───────────────────────────────────────────
        if self.config.use_mock_encoder:
            logits = self.lm_head(inputs_embeds)             # (B, L, vocab)
            loss   = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )
        else:
            # Use the inner transformer reference resolved once at
            # construction time — avoids re-navigating the PEFT wrapper
            # hierarchy on every forward call.
            outputs = self._llm_inner(
                inputs_embeds  = inputs_embeds,
                attention_mask = attention_mask,
                use_cache      = False,
            )
            hidden_states = outputs.last_hidden_state        # (B, L, llm_dim)
            logits        = self.lm_head(hidden_states)      # (B, L, vocab)

            loss = None
            if labels is not None:
                # Shift for causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        return {
            "loss":          loss,
            "logits":        logits,
            "visual_tokens": visual_tokens,
        }

    def _compress(
        self,
        slow_tokens:  torch.Tensor,
        fast_tokens:  torch.Tensor,
        questions:    list[str],
        slow_real_n:  list[int] | None = None,
        fast_real_n:  list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Run the SurgQCompressor on pre-encoded frame tokens.

        Bypasses the compressor's own (mock-only) ViT encoding step and
        feeds the already-encoded patch tokens directly to the local and
        global sub-compressors.

        Parameters
        ----------
        slow_tokens : (B, N_slow, hw, hw, vit_embed_dim) — may include padding frames
        fast_tokens : (B, N_fast, hw, hw, vit_embed_dim) — may include padding frames
        questions   : list[str] length B
        slow_real_n : real (unpadded) slow-frame count per sample, or None for all
        fast_real_n : real (unpadded) fast-frame count per sample, or None for all

        Returns
        -------
        list of B tensors, each (T_vis_i, llm_dim).
        T_vis_i may differ across samples when clips have different real lengths.
        """
        B      = slow_tokens.shape[0]
        device = slow_tokens.device
        dtype  = slow_tokens.dtype
        d      = self.config.vit_embed_dim

        # Encode questions
        if self.config.use_mock_encoder:
            text_tokens = torch.randn(B, 10, d, device=device, dtype=dtype)
        else:
            text_tokens = self.compressor._encode_text(questions, device, dtype)

        # Pool fast stream spatially: hw×hw → fast_spatial_size²
        fast_pooled = self.compressor._pool_fast_spatial(fast_tokens)

        # Per-sample compression loop. Trim to the real frame count before
        # compression so T_vis_i reflects the actual clip length, not the
        # padded batch-max length.
        all_tokens = []
        for i in range(B):
            n_s = slow_real_n[i] if slow_real_n is not None else slow_tokens.shape[1]
            n_f = fast_real_n[i] if fast_real_n is not None else fast_pooled.shape[1]
            slow_feat = slow_tokens[i, :n_s]
            fast_feat = fast_pooled[i, :n_f]
            txt       = text_tokens[i].unsqueeze(0)

            sl = self.compressor.slow_local(slow_feat, txt)
            fl = self.compressor.fast_local(fast_feat, txt)
            sl_flat = einops.rearrange(sl, 't h w d -> (t h w) d')
            fl_flat = einops.rearrange(fl, 't h w d -> (t h w) d')
            ld = sl_flat.shape[-1]
            if self.compressor.slow_global is not None:
                sg = self.compressor.slow_global(slow_feat, txt)
            else:
                sg = torch.zeros(0, ld, dtype=sl_flat.dtype, device=sl_flat.device)
            if self.compressor.fast_global is not None:
                fg = self.compressor.fast_global(fast_feat, txt)
            else:
                fg = torch.zeros(0, ld, dtype=fl_flat.dtype, device=fl_flat.device)

            all_tokens.append(torch.cat([sl_flat, fl_flat, sg, fg], dim=0))

        return all_tokens                                     # list of B tensors

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        slow_frames:    torch.Tensor,
        fast_frames:    torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        questions:      list[str],
        max_new_tokens: Optional[int] = None,
        slow_mask:      Optional[torch.Tensor] = None,
        fast_mask:      Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate text conditioned on video + question.

        Parameters
        ----------
        slow_frames    : (B, N_slow, 3, H, W)
        fast_frames    : (B, N_fast, 3, H, W)
        input_ids      : (B, L) — prompt with video pad tokens
        attention_mask : (B, L)
        questions      : list[str] length B
        max_new_tokens : int, optional. Defaults to config.max_new_tokens.

        Returns
        -------
        generated_ids : (B, ≤max_new_tokens) — new tokens only
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        B      = slow_frames.shape[0]
        device = slow_frames.device
        dtype  = slow_frames.dtype
        N_slow = slow_frames.shape[1]
        N_fast = fast_frames.shape[1]

        # Encode frames
        if self.config.use_mock_encoder:
            slow_tokens = self._encode_frames_mock(B, N_slow, device, dtype)
            fast_tokens = self._encode_frames_mock(B, N_fast, device, dtype)
        else:
            slow_flat = einops.rearrange(slow_frames, 'b n c h w -> (b n) c h w')
            fast_flat = einops.rearrange(fast_frames, 'b n c h w -> (b n) c h w')
            slow_enc  = self._encode_frames_vit(slow_flat)
            fast_enc  = self._encode_frames_vit(fast_flat)
            slow_tokens = einops.rearrange(
                slow_enc, '(b n) h w d -> b n h w d', b=B, n=N_slow
            )
            fast_tokens = einops.rearrange(
                fast_enc, '(b n) h w d -> b n h w d', b=B, n=N_fast
            )

        # Compress — trim to real frame counts to match n_vis in input_ids
        slow_real_n = (
            [int(slow_mask[i].sum().item()) for i in range(B)]
            if slow_mask is not None else None
        )
        fast_real_n = (
            [int(fast_mask[i].sum().item()) for i in range(B)]
            if fast_mask is not None else None
        )
        visual_tokens = self._compress(
            slow_tokens, fast_tokens, questions,
            slow_real_n=slow_real_n, fast_real_n=fast_real_n,
        )

        # Build inputs_embeds
        inputs_embeds = self._build_inputs_embeds(input_ids, visual_tokens)

        if self.config.use_mock_encoder:
            # Mock generation: return random token ids
            return torch.randint(
                0, 1000, (B, max_new_tokens), device=device
            )

        # Real generation via Qwen
        generated = self.qwen_model.generate(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            **generate_kwargs,
        )
        # When inputs_embeds is passed (not input_ids), HF generate() already
        # returns only the newly generated token IDs — the prompt embeddings
        # are not reversible to token IDs so they are never prepended to the
        # output. Do NOT slice by input_ids.shape[1]: that would strip all
        # generated tokens and return an empty tensor.
        return generated

    def generate_from_visual(
        self,
        visual_tokens:  list[torch.Tensor],
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate text from pre-computed visual tokens (skip ViT + compress).

        Used in GRPO: the frozen ViT + compressor run once per sample, then
        all G rollouts reuse the same visual tokens.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        inputs_embeds = self._build_inputs_embeds(input_ids, visual_tokens)

        if self.config.use_mock_encoder:
            B = input_ids.shape[0]
            return torch.randint(0, 1000, (B, max_new_tokens),
                                 device=input_ids.device)

        generated = self.qwen_model.generate(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            **generate_kwargs,
        )
        return generated

    # ── Utilities ─────────────────────────────────────────────────────────

    def _log_param_counts(self) -> None:
        """Log trainable vs frozen parameter counts per component."""
        def count(module):
            total     = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters()
                           if p.requires_grad)
            return total, trainable

        comp_total, comp_train = count(self.compressor)

        if self.config.use_mock_encoder:
            llm_total, llm_train = count(self.lm_head)
            logger.info(
                "Param counts (MOCK) | compressor: %dM trainable / %dM total "
                "| lm_head: %dM trainable / %dM total",
                comp_train // 1_000_000, comp_total // 1_000_000,
                llm_train  // 1_000_000, llm_total  // 1_000_000,
            )
        else:
            qwen_total, qwen_train = count(self.qwen_model)
            logger.info(
                "Param counts | compressor: %dM trainable / %dM total "
                "| qwen+lora: %dM trainable / %dM total",
                comp_train // 1_000_000, comp_total // 1_000_000,
                qwen_train // 1_000_000, qwen_total // 1_000_000,
            )

    def count_trainable_params(self) -> int:
        """Total trainable parameters across all components."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def expected_visual_tokens(self) -> int:
        """Expected number of compressed visual tokens per 30-second clip."""
        return self.compressor.expected_output_tokens()

    def to_device(self, device: torch.device) -> "SurgQVLM":
        """
        Move the full model to a single device for training.

        Use this instead of device_map="auto". For multi-GPU training, use
        DeepSpeed ZeRO-2 or FSDP in the training script instead.
        """
        if self.qwen_model is not None:
            self.qwen_model = self.qwen_model.to(device)
        self.compressor = self.compressor.to(device)
        if hasattr(self, "embed_tokens"):
            self.embed_tokens = self.embed_tokens.to(device)
        return self

    def verify_hook(self) -> bool:
        """
        Verify that the forward hook on visual.merger fires correctly by
        running a tiny dummy forward pass and checking the captured tensor.

        Call once after loading the real model to confirm the hook approach
        works with the installed transformers version.
        """
        if self.config.use_mock_encoder:
            logger.info("verify_hook: skipped in mock mode.")
            return True

        if not hasattr(self, "_vit_inner") or self._vit_inner is None:
            raise RuntimeError(
                "verify_hook called before _build_real_model completed."
            )

        logger.info("Running hook verification...")
        hw     = self.config.patch_grid_hw
        device = next(self._vit_inner.parameters()).device
        dtype  = next(self._vit_inner.parameters()).dtype

        dummy = torch.zeros(2, 3, hw * 14, hw * 14,
                            device=device, dtype=dtype)
        try:
            tokens = self._encode_frames_vit(dummy)
        except Exception as exc:
            raise RuntimeError(
                f"Hook verification failed: {exc}. "
                f"Check that model.visual.merger exists in your "
                f"transformers version."
            ) from exc

        expected = (2, hw, hw, self.config.vit_embed_dim)
        if tokens.shape != expected:
            raise RuntimeError(
                f"Hook verification: expected shape {expected}, "
                f"got {tokens.shape}."
            )

        logger.info("Hook verification passed: pre-merger tokens shape %s",
                    tuple(tokens.shape))
        return True


# ── Mock tokenizer for smoke test ─────────────────────────────────────────────

class _MockTokenizer:
    """Minimal tokenizer interface for mock mode."""
    def __call__(
        self, texts, return_tensors=None, padding=True,
        truncation=True, max_length=64,
    ):
        class _Inputs:
            def __init__(self, ids):
                self.input_ids = ids
            def to(self, device):
                self.input_ids = self.input_ids.to(device)
                return self
        B = len(texts) if isinstance(texts, list) else 1
        return _Inputs(torch.randint(0, 1000, (B, 10)))


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone smoke test. Run from the repository root:

        python -m surgqvlm.model                    # 7B mock
        python -m surgqvlm.model --model_size 3b    # 3B mock

    Verifies config construction, mock model build, forward/generate shapes,
    and the visual token budget.
    """
    import argparse

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="SurgQVLM smoke test")
    parser.add_argument("--model_size",  default="7b", choices=["3b", "7b"])
    parser.add_argument("--lora_mode",   default="attention_and_ffn",
                        choices=["attention_only", "attention_and_ffn"])
    parser.add_argument("--lora_r",      type=int, default=16)
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--n_slow",      type=int, default=15)
    parser.add_argument("--n_fast",      type=int, default=30)
    parser.add_argument("--seq_len",     type=int, default=700,
                        help="Input sequence length (must be >= n_visual_tokens)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SurgQVLM Smoke Test — {args.model_size.upper()} Mock Mode")
    print(f"{'='*60}\n")

    cfg = SurgQVLMConfig(
        model_size       = args.model_size,
        use_mock_encoder = True,
        lora             = LoRAConfig(
            lora_mode  = args.lora_mode,
            r          = args.lora_r,
            lora_alpha = float(args.lora_r * 2),
        ),
    )

    print("Config:")
    print(f"  model_size    : {cfg.model_size}")
    print(f"  llm_dim       : {cfg.llm_dim}")
    print(f"  vit_embed_dim : {cfg.vit_embed_dim}")
    print(f"  patch_grid_hw : {cfg.patch_grid_hw}")
    print(f"  lora_mode     : {cfg.lora.lora_mode}  r={cfg.lora.r}")

    model = SurgQVLM(cfg)
    model.eval()

    T_vis    = model.expected_visual_tokens()
    n_params = model.count_trainable_params()

    print(f"\nModel built successfully.")
    print(f"  Expected visual tokens : {T_vis}")
    print(f"  Trainable parameters   : {n_params:,}")

    B   = args.batch_size
    H   = W = 384      # matches the dataset's frame transform size
    Ns  = args.n_slow
    Nf  = args.n_fast
    L   = args.seq_len

    slow_frames    = torch.randn(B, Ns, 3, H, W)
    fast_frames    = torch.randn(B, Nf, 3, H, W)
    questions      = [
        "What instrument is dissecting the cystic duct?",
        "At what second does the phase transition occur?",
    ][:B]

    # Build input_ids: T_vis video pad tokens + (L - T_vis) text tokens
    # (video_token_id = 42 in mock mode)
    video_pad      = torch.full((B, T_vis), 42, dtype=torch.long)
    text_part      = torch.randint(1, 40, (B, L - T_vis), dtype=torch.long)
    input_ids      = torch.cat([video_pad, text_part], dim=1)  # (B, L)
    attention_mask = torch.ones(B, L, dtype=torch.long)

    # Labels: -100 for visual tokens, real ids for text
    labels         = torch.full((B, L), -100, dtype=torch.long)
    labels[:, T_vis:] = text_part

    print(f"\nInput shapes:")
    print(f"  slow_frames : {tuple(slow_frames.shape)}")
    print(f"  fast_frames : {tuple(fast_frames.shape)}")
    print(f"  input_ids   : {tuple(input_ids.shape)}")
    print(f"  video_pads  : {T_vis} per sample")

    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(
            slow_frames    = slow_frames,
            fast_frames    = fast_frames,
            input_ids      = input_ids,
            attention_mask = attention_mask,
            questions      = questions,
            labels         = labels,
        )

    vis_list = outputs["visual_tokens"]   # list of B tensors
    print(f"\n  loss          : {outputs['loss'].item():.4f}")
    print(f"  logits        : {tuple(outputs['logits'].shape)}  (B, L, vocab)")
    print(f"  visual_tokens : list of {len(vis_list)} tensors, "
          f"each {tuple(vis_list[0].shape)}  (T_vis, llm_dim)")

    print("\nRunning generate...")
    with torch.no_grad():
        generated = model.generate(
            slow_frames    = slow_frames,
            fast_frames    = fast_frames,
            input_ids      = input_ids,
            attention_mask = attention_mask,
            questions      = questions,
            max_new_tokens = 20,
        )
    print(f"  generated_ids : {tuple(generated.shape)}  (B, max_new_tokens)")

    # ── Token budget check ────────────────────────────────────────────────
    cfg_c = cfg.compressor
    sl    = math.ceil(Ns / cfg_c.slow_temporal_k) * math.ceil(
        cfg_c.patch_grid_hw / cfg_c.slow_spatial_k) ** 2
    fl    = math.ceil(Nf / cfg_c.fast_temporal_k) * cfg_c.fast_spatial_size ** 2
    total = sl + fl + cfg_c.n_global_slow + cfg_c.n_global_fast

    print(f"\nToken budget: slow_local={sl} + fast_local={fl} + "
          f"global={cfg_c.n_global_slow}+{cfg_c.n_global_fast} = {total}")
    for i, vt in enumerate(vis_list):
        assert vt.shape == (total, cfg.llm_dim), \
            f"Sample {i} visual token shape mismatch: {vt.shape}"
    print("  Visual token shapes match: OK")

    vt = vis_list[0]
    assert not torch.isnan(vt).any() and not torch.isinf(vt).any()
    print(f"\nValue check (sample 0): mean={vt.mean():.4f}  std={vt.std():.4f}")

    print(f"\nSmoke test passed.\n")
