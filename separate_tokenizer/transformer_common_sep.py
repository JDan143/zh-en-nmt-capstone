from __future__ import annotations
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as C  # noqa: E402  (study config, repo root — constants only, never edited)
from models.common import pad_mask, subsequent_mask, Seq2SeqBase  # noqa: E402
# Reuse the study's building blocks unchanged — this is what keeps the backbone identical.
from models.transformer_common import (SinusoidalPE, EncoderLayer,  # noqa: E402
                                       DecoderLayer)


class SeparateVocabTransformer(Seq2SeqBase):
    def __init__(self, cfg, src_vocab: int, tgt_vocab: int):
        super().__init__()
        self.cfg = cfg
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        d = cfg.d_model
        self.d_model = d
        self.rel_k = cfg.rel_pos_k
        self.use_abs_pe = cfg.rel_pos_k == 0
        self.scale = math.sqrt(d)

        # ── the ONLY architectural change vs the study model ──────────────────
        # Source input embedding (its own table, NOT tied to anything).
        self.src_embed = nn.Embedding(src_vocab, d, padding_idx=C.PAD_ID)
        # Target input embedding, TIED to the output projection (Press & Wolf, 2017;
        # standard MT tie). Source is separate, target stays honest against the output.
        self.tgt_embed = nn.Embedding(tgt_vocab, d, padding_idx=C.PAD_ID)
        self.out = nn.Linear(d, tgt_vocab, bias=False)
        self.out.weight = self.tgt_embed.weight  # tied (target side only)
        # ──────────────────────────────────────────────────────────────────────
        # beam_search / greedy_decode read model.vocab_size to size the output space.
        # For a separate-tokenizer model the OUTPUT space is the target vocab.
        self.vocab_size = tgt_vocab

        self.pe = SinusoidalPE(d, max_len=max(cfg.max_len, 512))
        self.drop = nn.Dropout(cfg.dropout)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout, self.rel_k)
            for _ in range(cfg.enc_layers)])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout, self.rel_k)
            for _ in range(cfg.dec_layers)])
        self.norm_enc = nn.LayerNorm(d)
        self.norm_dec = nn.LayerNorm(d)
        self._reset_parameters()

    def _reset_parameters(self):
        # Same scheme as the study model: Xavier for weight matrices, then set both
        # embedding tables to std = d_model**-0.5 (so inputs ~unit scale and the tied
        # target projection gives ~unit-scale logits). Skip Shaw rel tables (kept small
        # in MultiHeadAttention) so Arch 5 dynamics are untouched.
        for name, p in self.named_parameters():
            if p.dim() > 1 and not (name.endswith("rel_key") or name.endswith("rel_val")):
                nn.init.xavier_uniform_(p)
        for emb in (self.src_embed, self.tgt_embed):
            nn.init.normal_(emb.weight, mean=0.0, std=self.d_model ** -0.5)
            with torch.no_grad():
                emb.weight[C.PAD_ID].zero_()

    # ── source vs target embedding (the split that separate tokenizers require) ──
    def _embed_src(self, x):
        e = self.src_embed(x) * self.scale
        if self.use_abs_pe:
            e = self.pe(e)
        return self.drop(e)

    def _embed_tgt(self, x):
        e = self.tgt_embed(x) * self.scale
        if self.use_abs_pe:
            e = self.pe(e)
        return self.drop(e)

    def encode(self, src, src_len):
        src_pad = pad_mask(src)
        x = self._embed_src(src)                       # SOURCE embedding
        for layer in self.enc_layers:
            x = layer(x, src_pad)
        x = self.norm_enc(x)
        return {"memory": x, "src_mask": src_pad}

    def _decode_full(self, memory, ys):
        mem, src_pad = memory["memory"], memory["src_mask"]
        tgt_mask = subsequent_mask(ys.size(1), ys.device)
        x = self._embed_tgt(ys)                         # TARGET embedding
        for layer in self.dec_layers:
            x = layer(x, mem, tgt_mask, src_pad)
        x = self.norm_dec(x)
        return self.out(x)

    def forward(self, src, src_len, tgt_in, tf_ratio: float = 1.0):
        # Transformer training is fully teacher-forced; tf_ratio kept for loop-API parity
        # with the RNN archs (same signature train.run_epoch calls). Matches study model.
        memory = self.encode(src, src_len)
        return self._decode_full(memory, tgt_in)

    # ── incremental decoding (KV cache) — identical to the study model, but the
    #    per-step token embedding uses the TARGET table ──
    supports_incremental = True

    def incremental_init(self, memory, batch_size, device):
        mem, src_pad = memory["memory"], memory["src_mask"]
        return {
            "enc_kv": [layer.cross_attn.compute_kv(mem) for layer in self.dec_layers],
            "self": [{"k": None, "v": None} for _ in self.dec_layers],
            "src_pad": src_pad,
            "t": 0,
        }

    def incremental_step(self, memory, state, last_tok):
        t = state["t"]
        x = self.tgt_embed(last_tok).unsqueeze(1) * self.scale     # TARGET embedding
        if self.use_abs_pe:
            x = x + self.pe.pe[:, t:t + 1]
        for i, layer in enumerate(self.dec_layers):
            x = layer.step(x, state["self"][i], state["enc_kv"][i], state["src_pad"])
        x = self.norm_dec(x)
        state["t"] = t + 1
        return self.out(x)[:, -1, :]

    def incremental_reorder(self, state, order):
        for c in state["self"]:
            if c["k"] is not None:
                c["k"] = c["k"][order]
                c["v"] = c["v"][order]
        for ek in state["enc_kv"]:
            ek["k"] = ek["k"][order]
            ek["v"] = ek["v"][order]
        state["src_pad"] = state["src_pad"][order]
