"""
models/common.py — shared building blocks reused by all five architectures.

Uniform model contract (every architecture implements `encode` + `_decode_full`):
    encode(src, src_len)            -> memory: dict[str, Tensor]   (batch dim = 0)
    _decode_full(memory, ys)        -> logits [B, L, V]  (teacher-forced)
The base class derives:
    forward(src, src_len, tgt_in)   -> logits [B, L, V]
    decode_logits(memory, ys)       -> logits [B, V]     (last step; for decoding)
    expand_memory(memory, k)        -> memory repeated k times along batch dim

Decoders (greedy + batched beam search with length penalty) live here so all
architectures share identical inference behaviour.

Convention: anything stored in `memory` keeps batch at dim 0 so expand_memory is
generic. (The RNN model stores its decoder init state as [B, layers, H].)
"""
from __future__ import annotations
import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# masks
# ─────────────────────────────────────────────────────────────────────────────
def pad_mask(x: torch.Tensor) -> torch.Tensor:
    """True where token is padding. x: [B, T] -> [B, T] bool."""
    return x == C.PAD_ID


def subsequent_mask(size: int, device) -> torch.Tensor:
    """[size, size] bool, True where j>i (positions to block in self-attn)."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


# ─────────────────────────────────────────────────────────────────────────────
# losses
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    """KL-style label smoothing (Szegedy 2016); ignores pad positions."""

    def __init__(self, vocab_size: int, epsilon: float = 0.1, pad_idx: int = C.PAD_ID):
        super().__init__()
        self.epsilon = epsilon
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [N, V], target: [N]
        log_prob = F.log_softmax(logits, dim=-1)
        smooth = torch.full_like(log_prob, self.epsilon / (self.vocab_size - 1))
        smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
        mask = (target != self.pad_idx).unsqueeze(1).float()
        loss = -(smooth * log_prob * mask).sum() / mask.sum().clamp(min=1.0)
        return loss


def build_criterion(vocab_size: int, label_smoothing: float):
    if label_smoothing and label_smoothing > 0:
        return LabelSmoothingLoss(vocab_size, label_smoothing, C.PAD_ID)
    return nn.CrossEntropyLoss(ignore_index=C.PAD_ID)


# ─────────────────────────────────────────────────────────────────────────────
# scheduler
# ─────────────────────────────────────────────────────────────────────────────
def noam_lambda(d_model: int, warmup: int):
    def f(step: int):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
    return f


def build_optimizer_scheduler(model, cfg):
    if cfg.optimizer == "nag":
        # Nesterov accelerated gradient, the ConvS2S optimizer (Gehring et al.,
        # 2017). lr 0.25 / momentum 0.99 are the NAG settings; under Adam-family
        # optimizers lr 0.25 diverges because they normalize the gradient.
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                              momentum=cfg.nadam_momentum, nesterov=True)
    elif cfg.optimizer == "nadam":
        opt = torch.optim.NAdam(model.parameters(), lr=cfg.lr,
                                betas=cfg.betas, eps=cfg.eps,
                                momentum_decay=4e-3)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                               betas=cfg.betas, eps=cfg.eps)
    sched = None
    if cfg.scheduler == "noam":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, noam_lambda(cfg.d_model, cfg.warmup_steps))
    elif cfg.scheduler == "plateau":
        # Adaptive LR: scale the LR by `factor` when dev loss stalls for
        # `patience` epochs. Reduction patience (3) is shorter than early-stop
        # patience (10) so the LR drops a couple of times before training halts;
        # those drops often unstick dev loss and buy more useful training.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3, threshold=1e-4, min_lr=1e-6)
    return opt, sched


# ─────────────────────────────────────────────────────────────────────────────
# base model
# ─────────────────────────────────────────────────────────────────────────────
class Seq2SeqBase(nn.Module):
    """Implements forward/decode_logits/expand_memory from encode + _decode_full.

    Optional fast inference: a subclass may set `supports_incremental = True` and
    implement incremental_init / incremental_step / incremental_reorder for O(1)
    per-step decoding (KV cache for the Transformer; carried hidden state for the
    RNNs; cached causal buffers for ConvS2S). Decoders below use it when present
    and fall back to the (always-correct) full-recompute path otherwise.
    """

    vocab_size: int
    supports_incremental = False

    def encode(self, src, src_len):
        raise NotImplementedError

    def _decode_full(self, memory, ys):
        raise NotImplementedError

    # incremental hooks (overridden by models that support it)
    def incremental_init(self, memory, batch_size, device):
        raise NotImplementedError

    def incremental_step(self, memory, state, last_tok):
        raise NotImplementedError

    def incremental_reorder(self, state, order):
        raise NotImplementedError

    def forward(self, src, src_len, tgt_in, tf_ratio: float = 1.0):
        memory = self.encode(src, src_len)
        return self._decode_full(memory, tgt_in)

    @torch.no_grad()
    def decode_logits(self, memory, ys):
        return self._decode_full(memory, ys)[:, -1, :]

    @staticmethod
    def expand_memory(memory: dict, k: int) -> dict:
        out = {}
        for key, v in memory.items():
            if torch.is_tensor(v):
                out[key] = v.repeat_interleave(k, dim=0)
            else:
                out[key] = v
        return out


# ─────────────────────────────────────────────────────────────────────────────
# decoding
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def greedy_decode(model: Seq2SeqBase, src, src_len, max_len: int = 128):
    model.eval()
    device = src.device
    B = src.size(0)
    memory = model.encode(src, src_len)
    incr = getattr(model, "supports_incremental", False)
    state = model.incremental_init(memory, B, device) if incr else None
    ys = torch.full((B, 1), C.BOS_ID, dtype=torch.long, device=device)
    last = ys[:, 0]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        if incr:
            logits = model.incremental_step(memory, state, last)   # [B, V]
        else:
            logits = model.decode_logits(memory, ys)
        nxt = logits.argmax(-1).masked_fill(finished, C.PAD_ID)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        last = nxt
        finished |= nxt == C.EOS_ID
        if finished.all():
            break
    return ys[:, 1:]  # drop BOS


@torch.no_grad()
def beam_search(model: Seq2SeqBase, src, src_len, max_len: int = 128,
                beam: int = 5, length_penalty: float = 0.6):
    """Batched beam search with Wu et al. (2016) length penalty. Uses incremental
    decoding + cache reorder when the model supports it; otherwise recompute."""
    model.eval()
    device = src.device
    B = src.size(0)
    V = model.vocab_size
    memory = model.encode(src, src_len)
    memory = model.expand_memory(memory, beam)           # [B*beam, ...]
    incr = getattr(model, "supports_incremental", False)
    state = model.incremental_init(memory, B * beam, device) if incr else None

    ys = torch.full((B * beam, 1), C.BOS_ID, dtype=torch.long, device=device)
    last = ys[:, 0]
    beam_scores = torch.full((B, beam), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(-1)                   # [B*beam]
    done = torch.zeros(B * beam, dtype=torch.bool, device=device)

    def lp(length):
        return ((5 + length) / 6) ** length_penalty

    for step in range(1, max_len + 1):
        if incr:
            logits = model.incremental_step(memory, state, last)
        else:
            logits = model.decode_logits(memory, ys)
        logp = F.log_softmax(logits, dim=-1)
        # finished beams contribute no new score and only extend with PAD
        logp[done] = float("-inf")
        logp[done, C.PAD_ID] = 0.0
        scores = beam_scores.unsqueeze(1) + logp         # [B*beam, V]
        scores = scores.view(B, beam * V)
        top_scores, top_idx = scores.topk(beam, dim=-1)  # [B, beam]
        beam_id = top_idx // V
        tok_id = top_idx % V
        base = (torch.arange(B, device=device) * beam).unsqueeze(1)
        flat_beam = (base + beam_id).view(-1)            # [B*beam]
        ys = ys[flat_beam]
        ys = torch.cat([ys, tok_id.view(-1, 1)], dim=1)
        beam_scores = top_scores.view(-1)
        done = done[flat_beam] | (tok_id.view(-1) == C.EOS_ID)
        last = tok_id.view(-1)
        if incr:
            model.incremental_reorder(state, flat_beam)  # keep caches aligned to beams
        if done.all():
            break

    lengths = (ys != C.PAD_ID).sum(-1).float().clamp(min=1)
    final = (beam_scores / lp(lengths)).view(B, beam)
    best = final.argmax(-1)
    base = torch.arange(B, device=device) * beam
    chosen = ys[base + best]
    return chosen[:, 1:]  # drop BOS
