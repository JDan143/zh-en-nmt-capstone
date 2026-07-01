"""
models/arch3_convs2s.py — Architecture 3: Convolutional Seq2Seq (Gehring 2017).

Spec: emb 256 + learned absolute position emb (max_len 1024); conv channels 256;
kernel 3; 6 encoder blocks; 6 decoder blocks; block = Conv1d(2C) -> GLU -> residual;
decoder convs causally masked via F.pad(x,(k-1,0)); multi-step attention between
each decoder block and encoder outputs; dropout 0.1 (inputs)/0.2 (attention);
NAG (lr 0.25, momentum 0.99); batch 64; clip 0.1.

Compact reimplementation: residual connections are scaled
by sqrt(0.5) (the ConvS2S stabiliser), attention values are (encoder output +
source embedding), and each decoder block has its own attention ("multi-step").
The decoder runs in parallel under teacher forcing (causal left-padding), so
_decode_full returns [B, L, V] in a single pass.

Weight normalization (Salimans & Kingma, 2016) is applied to ALL layers except the
embedding lookup tables, exactly as Gehring et al. (2017) specify. This is what
makes NAG at lr 0.25 stable: without it the raw conv weights grow unboundedly and
training diverges.
"""
from __future__ import annotations
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm as _weight_norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.common import Seq2SeqBase, pad_mask  # noqa: E402

SCALE = math.sqrt(0.5)


def _wn(module):
    """Weight-normalize a conv/linear layer (Gehring uses it on all non-lookup layers)."""
    return _weight_norm(module)


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel, dropout, causal):
        super().__init__()
        self.kernel = kernel
        self.causal = causal
        self.conv = _wn(nn.Conv1d(channels, 2 * channels, kernel,
                                  padding=0 if causal else kernel // 2))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, L]
        x = self.drop(x)
        if self.causal:
            x = F.pad(x, (self.kernel - 1, 0))  # left-pad only -> no peeking ahead
        x = self.conv(x)
        x = F.glu(x, dim=1)                      # [B, C, L]
        return x


class Arch3ConvS2S(Seq2SeqBase):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        E, Cc = cfg.emb_dim, cfg.conv_channels
        self.Cc = Cc
        self.max_pos = cfg.conv_max_pos
        self.attn_dropout = nn.Dropout(0.2)

        self.tok_emb = nn.Embedding(vocab_size, E, padding_idx=C.PAD_ID)
        self.pos_emb = nn.Embedding(cfg.conv_max_pos, E)
        self.in_drop = nn.Dropout(cfg.dropout)  # 0.1
        self.emb2c = _wn(nn.Linear(E, Cc))
        self.c2emb = _wn(nn.Linear(Cc, E))

        self.enc_blocks = nn.ModuleList(
            [ConvBlock(Cc, cfg.kernel_size, cfg.dropout, causal=False)
             for _ in range(cfg.conv_blocks)])
        self.dec_blocks = nn.ModuleList(
            [ConvBlock(Cc, cfg.kernel_size, cfg.dropout, causal=True)
             for _ in range(cfg.conv_blocks)])
        # per-decoder-block attention projections (multi-step attention)
        self.attn_q = nn.ModuleList([_wn(nn.Linear(Cc, E)) for _ in range(cfg.conv_blocks)])
        self.attn_o = nn.ModuleList([_wn(nn.Linear(E, Cc)) for _ in range(cfg.conv_blocks)])
        self.enc_out2emb = _wn(nn.Linear(Cc, E))
        self.out = _wn(nn.Linear(Cc, vocab_size))

        # Gehring et al. init: lookup tables ~ N(0, 0.1); pad row zeroed.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.tok_emb.weight[C.PAD_ID].zero_()

    def _positions(self, x):
        L = x.size(1)
        if L > self.max_pos:
            raise ValueError(f"sequence length {L} exceeds conv_max_pos {self.max_pos}")
        return torch.arange(L, device=x.device).unsqueeze(0).expand(x.size(0), L)

    def _embed(self, ids):
        e = self.tok_emb(ids) + self.pos_emb(self._positions(ids))
        return self.in_drop(e)

    def encode(self, src, src_len):
        emb = self._embed(src)                       # [B, S, E]
        x = self.emb2c(emb).transpose(1, 2)          # [B, C, S]
        for blk in self.enc_blocks:
            res = x
            x = (blk(x) + res) * SCALE
        enc = x.transpose(1, 2)                       # [B, S, C]
        enc_emb = self.enc_out2emb(enc)               # keys in emb space
        enc_val = enc_emb + emb                       # attention values (z + e)
        return {"enc_keys": enc_emb, "enc_values": enc_val,
                "src_mask": pad_mask(src)}

    def _decode_full(self, memory, ys):
        keys, vals, src_mask = memory["enc_keys"], memory["enc_values"], memory["src_mask"]
        g = self._embed(ys)                           # target embedding [B, L, E]
        x = self.emb2c(g).transpose(1, 2)             # [B, C, L]
        for i, blk in enumerate(self.dec_blocks):
            res = x
            h = blk(x)                                # [B, C, L]
            # multi-step attention
            q = self.attn_q[i](h.transpose(1, 2)) + g  # [B, L, E]
            scores = torch.bmm(q, keys.transpose(1, 2))  # [B, L, S]
            scores = scores.masked_fill(src_mask.unsqueeze(1), float("-inf"))
            attn = self.attn_dropout(torch.softmax(scores, dim=-1))
            ctx = torch.bmm(attn, vals)               # [B, L, E]
            h = h + self.attn_o[i](ctx).transpose(1, 2)
            x = (h + res) * SCALE
        out = x.transpose(1, 2)                        # [B, L, C]
        return self.out(out)


    # ── incremental decoding (cache each block's last k-1 inputs) ───────────
    supports_incremental = True

    def incremental_init(self, memory, batch_size, device):
        return {"buf": [None] * len(self.dec_blocks), "t": 0}

    def incremental_step(self, memory, state, last_tok):
        keys, vals, src_mask = memory["enc_keys"], memory["enc_values"], memory["src_mask"]
        t = state["t"]
        B = last_tok.size(0)
        k = self.cfg.kernel_size
        pos = torch.full((B,), t, dtype=torch.long, device=last_tok.device)
        g = self.tok_emb(last_tok) + self.pos_emb(pos)        # [B,E] (in_drop identity)
        x = self.emb2c(g).unsqueeze(-1)                       # [B,C,1]
        for i, blk in enumerate(self.dec_blocks):
            res = x
            buf = state["buf"][i]
            if buf is None:
                pad = torch.zeros(B, self.Cc, k - 1, device=x.device)
                window = torch.cat([pad, x], dim=2)          # [B,C,k]
            else:
                window = torch.cat([buf, x], dim=2)
            state["buf"][i] = window[:, :, 1:].contiguous()  # last k-1 inputs
            h = torch.nn.functional.glu(blk.conv(window), dim=1)   # [B,C,1]
            q = self.attn_q[i](h.transpose(1, 2)) + g.unsqueeze(1)  # [B,1,E]
            scores = torch.bmm(q, keys.transpose(1, 2))            # [B,1,S]
            scores = scores.masked_fill(src_mask.unsqueeze(1), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.bmm(attn, vals)                           # [B,1,E]
            h = h + self.attn_o[i](ctx).transpose(1, 2)
            x = (h + res) * SCALE
        state["t"] = t + 1
        return self.out(x.transpose(1, 2))[:, -1, :]             # [B,V]

    def incremental_reorder(self, state, order):
        state["buf"] = [b[order].contiguous() if b is not None else None
                        for b in state["buf"]]


def build(cfg, vocab_size):
    return Arch3ConvS2S(cfg, vocab_size)
