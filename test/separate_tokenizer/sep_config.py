from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config as C  # study config at the repo root (constants only; never edited)

ARCHS = ["arch4_sep", "arch5_sep"]                 # thin wrappers in this folder
ARCH_TO_STUDY = {"arch4_sep": "arch4_transformer",  # for cfg lookup (rel_pos_k etc.)
                 "arch5_sep": "arch5_improved_transformer"}
DIRECTIONS = ["enzh", "zhen"]
DIRECTION_LANGS = {"enzh": ("en", "zh"), "zhen": ("zh", "en")}  # (src_lang, tgt_lang)

VOCAB_ZH = 32000
VOCAB_EN = 32000


def check_arch(arch: str) -> str:
    if arch not in ARCHS:
        raise SystemExit(f"--arch must be one of {ARCHS}, got {arch!r}")
    return arch


def check_direction(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise SystemExit(f"--direction must be one of {DIRECTIONS}, got {direction!r}")
    return direction


def sep_root() -> str:
    """Everything for this experiment lives under speechbridge/sep_tok/."""
    return os.environ.get("SPEECHBRIDGE_SEP_ROOT", os.path.join(C.ROOT, "sep_tok"))


def tokenizer_prefix(lang: str) -> str:
    """Monolingual tokenizer prefix, shared by both directions.
    speechbridge/sep_tok/data/spm_zh (.model/.vocab) and spm_en."""
    assert lang in ("zh", "en")
    data = os.path.join(sep_root(), "data")
    os.makedirs(data, exist_ok=True)
    return os.path.join(data, f"spm_{lang}")


def direction_paths(direction: str) -> dict:
    """Per-direction Drive layout, all under sep_tok/:

        sep_tok/data/            train.tsv.gz, dev.tsv.gz (shared 800k),
                                 spm_zh.model, spm_en.model (the two tokenizers)
        sep_tok/ENZH/{checkpoints,results}/
        sep_tok/ZHEN/{checkpoints,results}/
    """
    check_direction(direction)
    root = sep_root()
    sub = "ENZH" if direction == "enzh" else "ZHEN"
    data = os.path.join(root, "data")
    ckpt = os.environ.get(f"SPEECHBRIDGE_SEP_CKPT_{sub}",
                          os.path.join(root, sub, "checkpoints"))
    results = os.environ.get(f"SPEECHBRIDGE_SEP_RESULTS_{sub}",
                             os.path.join(root, sub, "results"))
    for d in (data, ckpt, results):
        os.makedirs(d, exist_ok=True)
    return {"data": data, "ckpt": ckpt, "results": results}


def data_file(name: str) -> str:
    """Shared 800k train/dev live in sep_tok/data/ (copy or symlink the study's)."""
    return os.path.join(sep_root(), "data", name)


def stem(arch: str, direction: str, seed: int) -> str:
    """Filename stem: arch + direction + seed, e.g. arch4_sep_enzh_seed191."""
    return f"{arch}_{direction}_seed{seed}"
