"""
models/transformer_common.py — Transformer built from scratch (no nn.Transformer),
so Arch 5 can inject Shaw-style relative positional encoding into self-attention.

Shared by Arch 4 (vanilla, sinusoidal absolute PE) and Arch 5 (learned relative
PE, clipped distance k, shared across heads + label smoothing handled in the loss).

Design choices:
  * Post-norm layers + Noam warmup (the vanilla Vaswani et al. 2017 recipe).
  * Shared source/target embedding (single joint vocab) with tied output
    projection (Press & Wolf, 2017) to keep the model small enough for a free T4.
  * Arch 5 uses relative self-attention and drops absolute PE (standard Shaw
    et al. 2018 setup); cross-attention stays plain.
"""
from __future__ import annotations
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.common import Seq2SeqBase, pad_mask, subsequent_mask  # noqa: E402


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Scaled dot-product MHA. If rel_k>0 and self-attention, adds Shaw relative
    position embeddings (shared across heads) to keys and values."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, rel_k: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rel_k = rel_k
        if rel_k > 0:
            # tables shared across heads: [2k+1, dk]
            self.rel_key = nn.Parameter(torch.zeros(2 * rel_k + 1, self.dk))
            self.rel_val = nn.Parameter(torch.zeros(2 * rel_k + 1, self.dk))
            nn.init.normal_(self.rel_key, std=0.02)
            nn.init.normal_(self.rel_val, std=0.02)

    def _rel_index(self, lq, lk, device):
        q_idx = torch.arange(lq, device=device).unsqueeze(1)
        k_idx = torch.arange(lk, device=device).unsqueeze(0)
        dist = (k_idx - q_idx).clamp(-self.rel_k, self.rel_k) + self.rel_k
        return dist  # [lq, lk]

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        B, Lq, _ = query.shape
        Lk = key.size(1)
        q = self.q(query).view(B, Lq, self.h, self.dk).transpose(1, 2)  # [B,h,Lq,dk]
        k = self.k(key).view(B, Lk, self.h, self.dk).transpose(1, 2)
        v = self.v(value).view(B, Lk, self.h, self.dk).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))                   # [B,h,Lq,Lk]
        if self.rel_k > 0 and Lq == Lk:  # relative only for self-attention
            ridx = self._rel_index(Lq, Lk, query.device)
            rel_k = self.rel_key[ridx]                                  # [Lq,Lk,dk]
            scores = scores + torch.einsum("bhqd,qkd->bhqk", q, rel_k)
        scores = scores / math.sqrt(self.dk)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        attn = self.drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)                                    # [B,h,Lq,dk]
        if self.rel_k > 0 and Lq == Lk:
            ridx = self._rel_index(Lq, Lk, query.device)
            rel_v = self.rel_val[ridx]                                 # [Lq,Lk,dk]
            out = out + torch.einsum("bhqk,qkd->bhqd", attn, rel_v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.h * self.dk)
        return self.o(out)

    # ── incremental decoding (KV cache) ────────────────────────────────────
    def _heads(self, x, B, L):
        return x.view(B, L, self.h, self.dk).transpose(1, 2)  # [B,h,L,dk]

    def incr_self(self, x_step, cache):
        """One-step causal self-attention with a growing K/V cache.
        x_step: [B,1,d]; cache: dict with 'k','v' (or None). Returns [B,1,d]."""
        B = x_step.size(0)
        q = self._heads(self.q(x_step), B, 1)
        k = self._heads(self.k(x_step), B, 1)
        v = self._heads(self.v(x_step), B, 1)
        if cache.get("k") is None:
            cache["k"], cache["v"] = k, v
        else:
            cache["k"] = torch.cat([cache["k"], k], dim=2)
            cache["v"] = torch.cat([cache["v"], v], dim=2)
        K, V = cache["k"], cache["v"]
        T = K.size(2)
        scores = torch.matmul(q, K.transpose(-2, -1))            # [B,h,1,T]
        if self.rel_k > 0:
            j = torch.arange(T, device=x_step.device)
            dist = (j - (T - 1)).clamp(-self.rel_k, self.rel_k) + self.rel_k  # [T]
            rk = self.rel_key[dist]                              # [T,dk]
            scores = scores + torch.einsum("bhqd,kd->bhqk", q, rk)
        scores = scores / math.sqrt(self.dk)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                             # [B,h,1,dk]
        if self.rel_k > 0:
            rv = self.rel_val[dist]
            out = out + torch.einsum("bhqk,kd->bhqd", attn, rv)
        out = out.transpose(1, 2).contiguous().view(B, 1, self.h * self.dk)
        return self.o(out)

    def compute_kv(self, mem):
        """Precompute encoder K/V once for cross-attention."""
        B, S, _ = mem.shape
        return {"k": self._heads(self.k(mem), B, S),
                "v": self._heads(self.v(mem), B, S)}

    def incr_cross(self, x_step, enc_kv, key_padding_mask):
        """One-step cross-attention against cached encoder K/V."""
        B = x_step.size(0)
        q = self._heads(self.q(x_step), B, 1)
        K, V = enc_kv["k"], enc_kv["v"]
        scores = torch.matmul(q, K.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,h,1,S]
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, 1, self.h * self.dk)
        return self.o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, rel_k=0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, rel_k)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_pad_mask):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, key_padding_mask=src_pad_mask)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, rel_k=0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, rel_k)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, rel_k=0)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_pad_mask):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, attn_mask=tgt_mask)))
        x = self.n2(x + self.drop(self.cross_attn(x, memory, memory,
                                                  key_padding_mask=src_pad_mask)))
        x = self.n3(x + self.drop(self.ff(x)))
        return x

    def step(self, x_step, self_cache, enc_kv, src_pad_mask):
        """Incremental one-token decode (dropout is identity in eval)."""
        x = self.n1(x_step + self.self_attn.incr_self(x_step, self_cache))
        x = self.n2(x + self.cross_attn.incr_cross(x, enc_kv, src_pad_mask))
        x = self.n3(x + self.ff(x))
        return x


class TransformerModel(Seq2SeqBase):
    """Shared Transformer. rel_k>0 -> Arch 5 (relative PE, no absolute PE)."""

    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        d = cfg.d_model
        self.d_model = d
        self.rel_k = cfg.rel_pos_k
        self.use_abs_pe = cfg.rel_pos_k == 0
        self.scale = math.sqrt(d)

        self.embed = nn.Embedding(vocab_size, d, padding_idx=C.PAD_ID)  # shared src/tgt
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
        self.out = nn.Linear(d, vocab_size, bias=False)
        self.out.weight = self.embed.weight  # tied
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier for all weight matrices, then give the (tied, √d-scaled) embedding
        # std = d_model**-0.5 so input embeddings are ~unit scale (matching the PE)
        # AND output logits are ~unit scale. Without this the tied projection makes
        # logits std ~= sqrt(d_model), so the initial loss is huge (~200 not ~ln V)
        # and early gradients are oversized.
        #
        # IMPORTANT: skip the Shaw relative-position tables (rel_key/rel_val). They
        # are deliberately initialized small (std=0.02) in MultiHeadAttention; a
        # blanket xavier_uniform here would overwrite that (~9x larger) and quietly
        # change ONLY Arch 5's dynamics — the exact model the hypothesis rests on.
        for name, p in self.named_parameters():
            if p.dim() > 1 and not (name.endswith("rel_key") or name.endswith("rel_val")):
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[C.PAD_ID].zero_()

    def _embed(self, x):
        e = self.embed(x) * self.scale
        if self.use_abs_pe:
            e = self.pe(e)
        return self.drop(e)

    def encode(self, src, src_len):
        src_pad = pad_mask(src)
        x = self._embed(src)
        for layer in self.enc_layers:
            x = layer(x, src_pad)
        x = self.norm_enc(x)
        return {"memory": x, "src_mask": src_pad}

    def _decode_full(self, memory, ys):
        mem, src_pad = memory["memory"], memory["src_mask"]
        tgt_mask = subsequent_mask(ys.size(1), ys.device)
        x = self._embed(ys)
        for layer in self.dec_layers:
            x = layer(x, mem, tgt_mask, src_pad)
        x = self.norm_dec(x)
        return self.out(x)

    # ── incremental decoding (KV cache) ────────────────────────────────────
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
        x = self.embed(last_tok).unsqueeze(1) * self.scale          # [B,1,d]
        if self.use_abs_pe:
            x = x + self.pe.pe[:, t:t + 1]
        for i, layer in enumerate(self.dec_layers):
            x = layer.step(x, state["self"][i], state["enc_kv"][i], state["src_pad"])
        x = self.norm_dec(x)
        state["t"] = t + 1
        return self.out(x)[:, -1, :]                                # [B,V]

    def incremental_reorder(self, state, order):
        for c in state["self"]:
            if c["k"] is not None:
                c["k"] = c["k"][order]
                c["v"] = c["v"][order]
        for ek in state["enc_kv"]:
            ek["k"] = ek["k"][order]
            ek["v"] = ek["v"][order]
        state["src_pad"] = state["src_pad"][order]
