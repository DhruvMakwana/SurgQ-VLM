"""SurgQ-VLM: question-conditioned dual-stream compression and
answer-type-aware reinforcement for surgical video question answering.

Package layout
--------------
    surgqvlm.compressor : SurgQCompressor — dual-stream question-conditioned
                          visual token compressor (HICom-based).
    surgqvlm.model      : SurgQVLM — Qwen2.5-VL + LoRA + compressor assembly.
    surgqvlm.data       : SurgQDataset, collate_fn, build_dataloader.
    surgqvlm.utils      : PromptBuilder, evaluation metrics, loss weighting,
                          checkpointing, config loading.
"""

__version__ = "1.0.0"
