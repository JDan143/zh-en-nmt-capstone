"""
WHAT CHANGES HERE:
  * DATA: OpenSubtitles2018 (capped 2.5M) + News Commentary v18 (all) + TED2020 (all)
  * the tokenizer is retrained on that mixture (vocab still 32,000, so the deployable
    model size is unchanged)
"""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility / shared constants
# ─────────────────────────────────────────────────────────────────────────────
SEED = 191
EVAL_SEEDS = [191]
SELECTED_ARCH = "arch4_transformer"
DEV_SIZE = 3000
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
LANG_TOKENS = ["<zh>", "<en>"]
VOCAB_SIZE = 32000
CHAR_COVERAGE = 0.9999
MODEL_TYPE = "bpe"
MAX_TOKENS_PER_SIDE = 100


# ─────────────────────────────────────────────────────────────────────────────
# Path handling — no hardcoded absolute paths; works on Colab and Kaggle.
# Override any of these with environment variables.
# ─────────────────────────────────────────────────────────────────────────────
def _default_root() -> str:
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/zh-en-nmt"
    if os.path.isdir("/content"):
        return "/content/zh-en-nmt"
    return os.path.abspath("./zh-en-nmt-run")


ROOT = os.environ.get("SPEECHBRIDGE_ROOT", _default_root())
DATA_DIR = os.environ.get("SPEECHBRIDGE_DATA", os.path.join(ROOT, "data"))
CKPT_DIR = os.environ.get("SPEECHBRIDGE_CKPT", os.path.join(ROOT, "checkpoints"))
RESULTS_DIR = os.environ.get("SPEECHBRIDGE_RESULTS", os.path.join(ROOT, "results"))
TOKENIZER_PREFIX = os.path.join(DATA_DIR, "bpe_zh_en")  # -> bpe_zh_en.model/.vocab

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration (verified sources — see README)
# ─────────────────────────────────────────────────────────────────────────────
DATA = {
    # OPUS OpenSubtitles2018, ZH-EN. Pulled as a moses zip (direct download).
    # URL scheme: https://object.pouta.csc.fi/OPUS-<Corpus>/<version>/moses/<l1>-<l2>.txt.zip
    "opensubtitles": {
        "corpus": "OpenSubtitles",
        "version": "v2018",
        "l1": "en",
        "l2": "zh_cn",
        "sample": 3_500_000,
    },
    # TED2020 v1 (OPUS) a spoken register, volunteer-translated TED/TEDx transcripts.
    "ted2020": {
        "corpus": "TED2020",
        "version": "v1",
        "l1": "en",
        "l2_candidates": ["zh_cn", "zh", "zh-cn"],
        "sample": None,
    },
    # News Commentary - latest is v18 (statmt). Two-column TSV (en \t zh).
    "news_commentary": {
        "version": "v18",
        "url": "https://data.statmt.org/news-commentary/v18/training/news-commentary-v18.en-zh.tsv.gz",
        # fallback if the v18 path ever moves:
        "url_fallback": "https://data.statmt.org/news-commentary/v16/training/news-commentary-v16.en-zh.tsv.gz",
    },
    # WMT newstest test sets are fetched via sacrebleu --echo (see evaluate.py).
    "wmt_testset": "wmt19",         # newstest2019
    "wmt_testset_alt": "wmt20",     # newstest2020
    # FLORES+ (eval only). openlanguagedata/flores_plus, devtest split.
    "flores": {
        "hf_id": "openlanguagedata/flores_plus",
        "split": "devtest",   # dev=997, devtest=1012
        "zh_code": "cmn_Hans",
        "en_code": "eng_Latn",
    },
    "max_len_filter": MAX_TOKENS_PER_SIDE,
    "length_ratio_max": 9.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Training length: we don't fix the epoch count. Early stopping ends training
# once dev loss stalls for EARLY_STOP_PATIENCE epochs. EPOCH_CAP is just a
# runaway guard, not a target; raise it if a model is still improving at the cap.
# Patience 10 sits at the safe end of the usual 5-10 range, since from-scratch
# NMT dev-loss curves are noisy.
# ─────────────────────────────────────────────────────────────────────────────
EPOCH_CAP = 200
EARLY_STOP_PATIENCE = 10


# ─────────────────────────────────────────────────────────────────────────────
# Shared training-harness defaults (overridden per-arch below)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ArchConfig:
    name: str
    family: str                       # "rnn" | "conv" | "transformer"
    # model dims
    emb_dim: int = 256
    enc_hidden: int = 512
    dec_hidden: int = 512
    attn_dim: int = 256
    enc_layers: int = 2
    dec_layers: int = 2
    dropout: float = 0.3
    # transformer-specific (ignored by rnn/conv)
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 8
    max_len: int = 128
    rel_pos_k: int = 0                 # >0 enables relative PE (Arch5: 16)
    label_smoothing: float = 0.0
    # conv-specific
    conv_channels: int = 256
    kernel_size: int = 3
    conv_blocks: int = 6
    conv_max_pos: int = 1024
    # optimization
    optimizer: str = "adam"           # "adam" | "nadam" | "nag" (SGD+Nesterov)
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    nadam_momentum: float = 0.99
    scheduler: str = "none"           # "none" | "noam" | "plateau"
    warmup_steps: int = 4000
    grad_clip: float = 1.0
    amp_safe: bool = True
    # batching
    batch_size: int = 128
    batch_by_tokens: bool = False
    max_tokens: int = 4096
    accumulation_steps: int = 1
    # loop
    epochs: int = EPOCH_CAP
    patience: int = EARLY_STOP_PATIENCE
    teacher_forcing: bool = False
    tf_start: float = 1.0
    tf_end: float = 0.5
    tf_decay_epochs: int = 20
    # inference
    beam_size: int = 5
    length_penalty: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ARCH_CONFIGS: dict[str, ArchConfig] = {
    # Arch 1 — GRU Seq2Seq + Bahdanau (additive) attention
    "arch1_gru": ArchConfig(
        name="arch1_gru", family="rnn",
        emb_dim=256, enc_hidden=512, dec_hidden=512, attn_dim=256,
        enc_layers=2, dec_layers=2, dropout=0.3,
        optimizer="adam", lr=1e-3, grad_clip=1.0,
        scheduler="plateau",          # adaptive LR: halve on dev-loss plateau
        batch_size=128,               # epochs/patience use the open-ended defaults
        teacher_forcing=True, tf_start=1.0, tf_end=0.5, tf_decay_epochs=20,
    ),
    # Arch 2 — LSTM Seq2Seq + Luong attention (dot/general/concat; general primary)
    "arch2_lstm": ArchConfig(
        name="arch2_lstm", family="rnn",
        emb_dim=256, enc_hidden=512, dec_hidden=512, attn_dim=256,
        enc_layers=2, dec_layers=2, dropout=0.3,
        optimizer="adam", lr=1e-3, grad_clip=1.0,
        scheduler="plateau",          # adaptive LR: halve on dev-loss plateau
        batch_size=128,
        teacher_forcing=True, tf_start=1.0, tf_end=0.5, tf_decay_epochs=20,
    ),
    # Arch 3 — Convolutional Seq2Seq (ConvS2S)
    "arch3_convs2s": ArchConfig(
        name="arch3_convs2s", family="conv",
        emb_dim=256, conv_channels=256, kernel_size=3, conv_blocks=6,
        conv_max_pos=1024, dropout=0.1,        # 0.1 inputs / 0.2 attention (in-model)
        optimizer="nag", lr=0.25, nadam_momentum=0.99, grad_clip=0.1,
        scheduler="plateau",          # adaptive LR: halve on dev-loss plateau
        amp_safe=False,               # fp16 overflows ConvS2S -> NaN; train in fp32
        batch_size=64,
        teacher_forcing=False,
    ),
    # Arch 4 — Vanilla Transformer
    "arch4_transformer": ArchConfig(
        name="arch4_transformer", family="transformer",
        d_model=256, d_ff=1024, n_heads=8, enc_layers=6, dec_layers=6,
        dropout=0.1, max_len=128, rel_pos_k=0, label_smoothing=0.0,
        optimizer="adam", lr=1.0, betas=(0.9, 0.98), eps=1e-9,
        scheduler="noam", warmup_steps=4000, grad_clip=1.0,
        batch_by_tokens=True, max_tokens=4096, accumulation_steps=4,
        teacher_forcing=False,        # epochs/patience use open-ended defaults
        beam_size=5, length_penalty=0.6,
    ),
    # Arch 5 — Improved Transformer (relative PE k=16 + label smoothing 0.1)
    "arch5_improved_transformer": ArchConfig(
        name="arch5_improved_transformer", family="transformer",
        d_model=256, d_ff=1024, n_heads=8, enc_layers=6, dec_layers=6,
        dropout=0.1, max_len=128, rel_pos_k=16, label_smoothing=0.1,
        optimizer="adam", lr=1.0, betas=(0.9, 0.98), eps=1e-9,
        scheduler="noam", warmup_steps=4000, grad_clip=1.0,
        batch_by_tokens=True, max_tokens=4096, accumulation_steps=4,
        teacher_forcing=False,
        beam_size=5, length_penalty=0.6,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Smoke profile — shrink everything so the full path runs in minutes on CPU/T4.
# ─────────────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("SPEECHBRIDGE_SMOKE", "0") == "1"

SMOKE_OVERRIDES = dict(
    sample_pairs=4000,     # tiny subset for preprocess
    vocab_size=2000,       # SentencePiece needs vocab <= types available
    epochs=10,
    patience=10,           # = epochs so the smoke run actually completes 10 epochs
    max_tokens=1024,
    accumulation_steps=1,
)


def get_arch_config(name: str) -> ArchConfig:
    if name not in ARCH_CONFIGS:
        raise KeyError(f"Unknown architecture '{name}'. Options: {list(ARCH_CONFIGS)}")
    cfg = ARCH_CONFIGS[name]
    if SMOKE:
        cfg.epochs = SMOKE_OVERRIDES["epochs"]
        cfg.patience = SMOKE_OVERRIDES["patience"]
        cfg.max_tokens = SMOKE_OVERRIDES["max_tokens"]
        cfg.accumulation_steps = SMOKE_OVERRIDES["accumulation_steps"]
        cfg.tf_decay_epochs = 5
        # shrink model so it trains fast and so SentencePiece's small vocab fits
        if cfg.family == "transformer":
            cfg.enc_layers = cfg.dec_layers = 2
            cfg.d_ff = 512
        elif cfg.family == "conv":
            cfg.conv_blocks = 2
        else:
            cfg.enc_layers = cfg.dec_layers = 1
    return cfg


# convenience bundle some scripts import
CONFIG = {
    "seed": SEED,
    "eval_seeds": EVAL_SEEDS,
    "pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID,
    "lang_tokens": LANG_TOKENS,
    "vocab_size": SMOKE_OVERRIDES["vocab_size"] if SMOKE else VOCAB_SIZE,
    "char_coverage": CHAR_COVERAGE,
    "model_type": MODEL_TYPE,
    "data": DATA,
    "paths": {
        "root": ROOT, "data": DATA_DIR, "ckpt": CKPT_DIR,
        "results": RESULTS_DIR, "tokenizer_prefix": TOKENIZER_PREFIX,
    },
    "smoke": SMOKE,
}
