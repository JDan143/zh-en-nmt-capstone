"""
models/arch5_improved_transformer.py — Architecture 5: improved Transformer.

Identical to Arch 4 (d_model 256, d_ff 1024, 8 heads, 6+6 layers, dropout 0.1,
Noam warmup 4000, token-batching, beam 5 / lp 0.6) plus two changes:
  (1) Relative positional encoding — learned, clipped to max distance k=16,
      embeddings shared across heads (Shaw et al., 2018). Implemented in
      transformer_common.MultiHeadAttention; absolute sinusoidal PE is dropped
      (standard Shaw configuration).
  (2) Label smoothing eps=0.1 — applied in the loss (config.label_smoothing=0.1;
      see models/common.build_criterion), not inside the model.

Hypothesis (for the report):
  Arch 5 should beat Arch 4 by >= 0.5 BLEU in both directions (ZH->EN and
  EN->ZH) with no extra compute — relative PE adds only two small embedding
  tables (2k+1 x d_k) and no extra layers, so params/throughput are ~unchanged.
  Tested directly via the paired bootstrap in evaluate.py (--significance).
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_common import TransformerModel  # noqa: E402


def build(cfg, vocab_size):
    assert cfg.rel_pos_k > 0, "Arch 5 must enable relative PE (rel_pos_k>0)"
    return TransformerModel(cfg, vocab_size)
