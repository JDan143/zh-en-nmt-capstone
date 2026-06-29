"""
models/arch1_gru.py — Architecture 1: GRU Seq2Seq + Bahdanau (additive) attention.

Config: emb 256; 2-layer BiGRU encoder (hidden 512); 2-layer GRU decoder (hidden
512); attention dim 256; dropout 0.3; Adam lr 1e-3; batch 128; clip 1.0; teacher
forcing with scheduled annealing.

Bahdanau score:  e(s_t, h_i) = vᵀ tanh(W_s·s_t + W_h·h_i + b)
context = Σ_i softmax(e)_i · h_i ;  decoder input = concat(prev_emb, context).
"""
from __future__ import annotations
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.common import Seq2SeqBase, pad_mask  # noqa: E402


class BahdanauAttention(nn.Module):
    def __init__(self, dec_hidden: int, enc_dim: int, attn_dim: int):
        super().__init__()
        self.W_s = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(attn_dim))
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, s_t, enc_outputs, src_mask):
        # s_t: [B, H]  enc_outputs: [B, S, enc_dim]  src_mask: [B, S] (True=pad)
        e = self.v(torch.tanh(self.W_s(s_t).unsqueeze(1) + self.W_h(enc_outputs) + self.b))
        e = e.squeeze(-1)                                  # [B, S]
        e = e.masked_fill(src_mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)                   # [B, S]
        ctx = torch.bmm(alpha.unsqueeze(1), enc_outputs).squeeze(1)  # [B, enc_dim]
        return ctx, alpha


class Arch1GRU(Seq2SeqBase):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        E, Hh, L = cfg.emb_dim, cfg.enc_hidden, cfg.enc_layers
        self.enc_dim = 2 * Hh
        self.dec_hidden = cfg.dec_hidden
        self.dec_layers = cfg.dec_layers

        self.src_emb = nn.Embedding(vocab_size, E, padding_idx=C.PAD_ID)
        self.tgt_emb = nn.Embedding(vocab_size, E, padding_idx=C.PAD_ID)
        self.dropout = nn.Dropout(cfg.dropout)

        self.encoder = nn.GRU(E, Hh, num_layers=L, bidirectional=True,
                              dropout=cfg.dropout if L > 1 else 0.0, batch_first=True)
        self.bridge = nn.Linear(self.enc_dim, self.dec_hidden)
        self.attn = BahdanauAttention(self.dec_hidden, self.enc_dim, cfg.attn_dim)
        self.decoder = nn.GRU(E + self.enc_dim, self.dec_hidden,
                              num_layers=cfg.dec_layers,
                              dropout=cfg.dropout if cfg.dec_layers > 1 else 0.0,
                              batch_first=True)
        self.out = nn.Linear(self.dec_hidden + self.enc_dim, vocab_size)

    # --- encode -------------------------------------------------------------
    def encode(self, src, src_len):
        emb = self.dropout(self.src_emb(src))              # [B, S, E]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        enc_outputs, hidden = self.encoder(packed)
        enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            enc_outputs, batch_first=True, total_length=src.size(1))
        # combine last fwd/bwd hidden -> decoder init (replicated over layers)
        last = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, 2H]
        dec0 = torch.tanh(self.bridge(last))               # [B, H]
        dec_init = dec0.unsqueeze(1).repeat(1, self.dec_layers, 1)  # [B, layers, H]
        return {
            "enc_outputs": enc_outputs,                    # [B, S, 2H]
            "src_mask": pad_mask(src),                      # [B, S]
            "dec_init": dec_init,                           # [B, layers, H] (batch dim 0)
        }

    # --- single decoder step over a full ys (teacher forced) ----------------
    def _decode_full(self, memory, ys):
        enc, mask = memory["enc_outputs"], memory["src_mask"]
        h = memory["dec_init"].transpose(0, 1).contiguous()  # [layers, B, H]
        emb = self.dropout(self.tgt_emb(ys))                 # [B, L, E]
        outs = []
        for t in range(ys.size(1)):
            ctx, _ = self.attn(h[-1], enc, mask)             # [B, 2H]
            rnn_in = torch.cat([emb[:, t], ctx], dim=-1).unsqueeze(1)
            out, h = self.decoder(rnn_in, h)                 # out [B,1,H]
            logit = self.out(torch.cat([out.squeeze(1), ctx], dim=-1))
            outs.append(logit)
        return torch.stack(outs, dim=1)                      # [B, L, V]

    # --- training forward with scheduled-sampling teacher forcing -----------
    def forward(self, src, src_len, tgt_in, tf_ratio: float = 1.0):
        memory = self.encode(src, src_len)
        if tf_ratio >= 1.0:
            return self._decode_full(memory, tgt_in)
        enc, mask = memory["enc_outputs"], memory["src_mask"]
        h = memory["dec_init"].transpose(0, 1).contiguous()
        B, L = tgt_in.shape
        device = src.device
        inp = tgt_in[:, 0]
        outs = []
        for t in range(L):
            emb = self.dropout(self.tgt_emb(inp))            # [B, E]
            ctx, _ = self.attn(h[-1], enc, mask)
            rnn_in = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
            out, h = self.decoder(rnn_in, h)
            logit = self.out(torch.cat([out.squeeze(1), ctx], dim=-1))
            outs.append(logit)
            use_gold = torch.rand(B, device=device) < tf_ratio
            gold = tgt_in[:, t + 1] if t + 1 < L else inp
            pred = logit.argmax(-1)
            inp = torch.where(use_gold, gold, pred)
        return torch.stack(outs, dim=1)


    # ── incremental decoding (O(1) per step instead of re-running from t=0) ──
    supports_incremental = True

    def incremental_init(self, memory, batch_size, device):
        return {"h": memory["dec_init"].transpose(0, 1).contiguous()}  # [layers,B,H]

    def incremental_step(self, memory, state, last_tok):
        enc, mask = memory["enc_outputs"], memory["src_mask"]
        h = state["h"]
        emb = self.tgt_emb(last_tok)                      # dropout is identity in eval
        ctx, _ = self.attn(h[-1], enc, mask)
        rnn_in = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
        out, h = self.decoder(rnn_in, h)
        state["h"] = h
        return self.out(torch.cat([out.squeeze(1), ctx], dim=-1))

    def incremental_reorder(self, state, order):
        state["h"] = state["h"][:, order, :].contiguous()


def build(cfg, vocab_size):
    return Arch1GRU(cfg, vocab_size)
