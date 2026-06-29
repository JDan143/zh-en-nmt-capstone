"""
models/arch4_transformer.py — Architecture 4: vanilla Transformer (from scratch).

d_model 256; d_ff 1024; heads 8; 6 encoder + 6 decoder layers; dropout 0.1;
max len 128; sinusoidal absolute PE; NO label smoothing (that is Arch 5);
Adam + Noam warmup 4000; token-batching ~4096; gradient accumulation 4;
inference beam 5, length penalty 0.6.

All hyperparameters come from config.ARCH_CONFIGS['arch4_transformer'].
The implementation lives in transformer_common.TransformerModel with rel_pos_k=0.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_common import TransformerModel  # noqa: E402


def build(cfg, vocab_size):
    assert cfg.rel_pos_k == 0, "Arch 4 must use absolute PE (rel_pos_k=0)"
    return TransformerModel(cfg, vocab_size)
