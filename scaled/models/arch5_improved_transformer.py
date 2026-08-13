from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_common import TransformerModel  # noqa: E402


def build(cfg, vocab_size):
    assert cfg.rel_pos_k > 0, "Arch 5 must enable relative PE (rel_pos_k>0)"
    return TransformerModel(cfg, vocab_size)
