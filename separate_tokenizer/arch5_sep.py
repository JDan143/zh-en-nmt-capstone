from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_common_sep import SeparateVocabTransformer  # noqa: E402


def build(cfg, src_vocab, tgt_vocab):
    assert cfg.rel_pos_k > 0, "Arch 5 must enable relative PE (rel_pos_k>0)"
    return SeparateVocabTransformer(cfg, src_vocab, tgt_vocab)
