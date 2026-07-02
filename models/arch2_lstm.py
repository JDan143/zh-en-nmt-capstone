"""
models/arch2_lstm.py — Architecture 2: LSTM Seq2Seq + Luong attention.

Config: emb 256; 2-layer BiLSTM encoder (hidden 512/dir), 2-layer LSTM decoder
(hidden 512), dropout 0.3, Adam lr 1e-3, batch 128, clip 1.0.

Luong (2015) global attention with all three score functions:
    dot     : score(h_t, h_s) = h_tᵀ h_s
    general : score(h_t, h_s) = h_tᵀ W_a h_s
    concat  : score(h_t, h_s) = vᵀ tanh(W_a [h_t; h_s])
`general` is the primary variant reported; the others are an ablation (set
cfg via SPEECHBRIDGE_LUONG=dot|general|concat or attn_score arg).

Input-feeding: the attentional vector  h~_t = tanh(W_c [c_t; h_t])  is fed as
extra decoder input at the next step (concat with the next token embedding).

Encoder outputs are projected to the decoder hidden size so all three score
functions operate in one consistent space.
"""
from __future__ import annotations
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.common import Seq2SeqBase, pad_mask  # noqa: E402

LUONG_VARIANT = os.environ.get("SPEECHBRIDGE_LUONG", "general")  # dot|general|concat


class LuongAttention(nn.Module):
    def __init__(self, dim: int, variant: str = "general", attn_dim: int = 256):
        super().__init__()
        self.variant = variant
        if variant == "general":
            self.W_a = nn.Linear(dim, dim, bias=False)
        elif variant == "concat":
            self.W_a = nn.Linear(2 * dim, attn_dim, bias=False)
            self.v = nn.Linear(attn_dim, 1, bias=False)
        elif variant != "dot":
            raise ValueError(f"unknown Luong variant {variant}")

    def forward(self, h_t, enc, src_mask):
        # h_t: [B, H]  enc: [B, S, H]  src_mask: [B, S] (True = pad)
        if self.variant == "dot":
            score = torch.bmm(enc, h_t.unsqueeze(-1)).squeeze(-1)        # [B, S]
        elif self.variant == "general":
            score = torch.bmm(self.W_a(enc), h_t.unsqueeze(-1)).squeeze(-1)
        else:  # concat
            S = enc.size(1)
            h_rep = h_t.unsqueeze(1).expand(-1, S, -1)
            score = self.v(torch.tanh(self.W_a(torch.cat([h_rep, enc], -1)))).squeeze(-1)
        score = score.masked_fill(src_mask, float("-inf"))
        alpha = torch.softmax(score, dim=-1)                            # [B, S]
        ctx = torch.bmm(alpha.unsqueeze(1), enc).squeeze(1)             # [B, H]
        return ctx, alpha


class Arch2LSTM(Seq2SeqBase):
    def __init__(self, cfg, vocab_size: int, attn_score: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.variant = attn_score or LUONG_VARIANT
        E, Hh, L = cfg.emb_dim, cfg.enc_hidden, cfg.enc_layers
        self.H = cfg.dec_hidden
        self.dec_layers = cfg.dec_layers

        self.src_emb = nn.Embedding(vocab_size, E, padding_idx=C.PAD_ID)
        self.tgt_emb = nn.Embedding(vocab_size, E, padding_idx=C.PAD_ID)
        self.dropout = nn.Dropout(cfg.dropout)

        self.encoder = nn.LSTM(E, Hh, num_layers=L, bidirectional=True,
                               dropout=cfg.dropout if L > 1 else 0.0, batch_first=True)
        self.enc_proj = nn.Linear(2 * Hh, self.H)        # -> decoder space
        self.bridge_h = nn.Linear(2 * Hh, self.H)
        self.bridge_c = nn.Linear(2 * Hh, self.H)
        # input-feeding: decoder input = emb + previous attentional vector (H)
        self.decoder = nn.LSTM(E + self.H, self.H, num_layers=cfg.dec_layers,
                               dropout=cfg.dropout if cfg.dec_layers > 1 else 0.0,
                               batch_first=True)
        self.attn = LuongAttention(self.H, self.variant, cfg.attn_dim)
        self.W_c = nn.Linear(2 * self.H, self.H)         # attentional vector
        self.attn_ln = nn.LayerNorm(self.H)
        self.out = nn.Linear(self.H, vocab_size)

        # Forget-gate bias = 1.0 (Jozefowicz et al., 2015; Gers et al., 2000). With
        # input-feeding over long sequences the decoder is a very deep recurrence,
        # the default ~0 forget bias lets the cell state decay, gradients vanish, and
        # the model collapses to the unigram distribution. Biasing the forget gate
        # toward "remember" keeps the state (and gradients) alive. PyTorch LSTM bias
        # layout per gate is [input, forget, cell, output], total bias = ih + hh, so
        # we set ih's forget slice to 1 and zero hh's forget slice for a net bias 1.
        for lstm in (self.encoder, self.decoder):
            for name, param in lstm.named_parameters():
                if "bias_ih" in name:
                    n = param.size(0) // 4
                    with torch.no_grad():
                        param[n:2 * n].fill_(1.0)
                elif "bias_hh" in name:
                    n = param.size(0) // 4
                    with torch.no_grad():
                        param[n:2 * n].zero_()

    def encode(self, src, src_len):
        emb = self.dropout(self.src_emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        enc_out, (h, c) = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(
            enc_out, batch_first=True, total_length=src.size(1))
        enc_out = self.enc_proj(enc_out)                  # [B, S, H]
        last_h = torch.cat([h[-2], h[-1]], dim=1)
        last_c = torch.cat([c[-2], c[-1]], dim=1)
        h0 = torch.tanh(self.bridge_h(last_h)).unsqueeze(1).repeat(1, self.dec_layers, 1)
        c0 = torch.tanh(self.bridge_c(last_c)).unsqueeze(1).repeat(1, self.dec_layers, 1)
        return {"enc_outputs": enc_out, "src_mask": pad_mask(src),
                "dec_h0": h0, "dec_c0": c0}

    def _run(self, memory, ys, tf_ratio=1.0, gold=None):
        enc, mask = memory["enc_outputs"], memory["src_mask"]
        h = memory["dec_h0"].transpose(0, 1).contiguous()
        c = memory["dec_c0"].transpose(0, 1).contiguous()
        B = ys.size(0) if gold is None else gold.size(0)
        device = enc.device
        attn_vec = torch.zeros(B, self.H, device=device)
        if gold is None:
            emb_all = self.dropout(self.tgt_emb(ys))
            L = ys.size(1)
        else:
            L = gold.size(1)
        inp_tok = (ys[:, 0] if gold is None else gold[:, 0])
        outs = []
        for t in range(L):
            if gold is None:
                emb = emb_all[:, t]
            else:
                emb = self.dropout(self.tgt_emb(inp_tok))
            dec_in = torch.cat([emb, attn_vec], dim=-1).unsqueeze(1)
            out, (h, c) = self.decoder(dec_in, (h, c))
            h_t = out.squeeze(1)
            ctx, _ = self.attn(h_t, enc, mask)
            attn_vec = torch.tanh(self.attn_ln(self.W_c(torch.cat([ctx, h_t], dim=-1))))
            logit = self.out(attn_vec)
            outs.append(logit)
            if gold is not None:
                use_gold = torch.rand(B, device=device) < tf_ratio
                nxt_gold = gold[:, t + 1] if t + 1 < L else inp_tok
                inp_tok = torch.where(use_gold, nxt_gold, logit.argmax(-1))
        return torch.stack(outs, dim=1)

    def _decode_full(self, memory, ys):
        return self._run(memory, ys, gold=None)

    def forward(self, src, src_len, tgt_in, tf_ratio: float = 1.0):
        memory = self.encode(src, src_len)
        if tf_ratio >= 1.0:
            return self._run(memory, tgt_in, gold=None)
        return self._run(memory, ys=None, tf_ratio=tf_ratio, gold=tgt_in)


    # ── incremental decoding (carry (h,c) + attentional vector across steps) ──
    supports_incremental = True

    def incremental_init(self, memory, batch_size, device):
        return {"h": memory["dec_h0"].transpose(0, 1).contiguous(),
                "c": memory["dec_c0"].transpose(0, 1).contiguous(),
                "attn_vec": torch.zeros(batch_size, self.H, device=device)}

    def incremental_step(self, memory, state, last_tok):
        enc, mask = memory["enc_outputs"], memory["src_mask"]
        emb = self.tgt_emb(last_tok)                      # dropout identity in eval
        dec_in = torch.cat([emb, state["attn_vec"]], dim=-1).unsqueeze(1)
        out, (h, c) = self.decoder(dec_in, (state["h"], state["c"]))
        h_t = out.squeeze(1)
        ctx, _ = self.attn(h_t, enc, mask)
        attn_vec = torch.tanh(self.attn_ln(self.W_c(torch.cat([ctx, h_t], dim=-1))))
        state["h"], state["c"], state["attn_vec"] = h, c, attn_vec
        return self.out(attn_vec)

    def incremental_reorder(self, state, order):
        state["h"] = state["h"][:, order, :].contiguous()
        state["c"] = state["c"][:, order, :].contiguous()
        state["attn_vec"] = state["attn_vec"][order].contiguous()


def build(cfg, vocab_size):
    return Arch2LSTM(cfg, vocab_size)
