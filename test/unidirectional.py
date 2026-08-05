from __future__ import annotations
import os

import config as C

ARCHS = ["arch4_transformer", "arch5_improved_transformer"]
DIRECTIONS = ["enzh", "zhen"]

DIRECTION_TO_PAIR = {"enzh": "en-zh", "zhen": "zh-en"}


def check_direction(direction: str) -> str:
    if direction not in DIRECTIONS:
        raise SystemExit(f"--direction must be one of {DIRECTIONS}, got {direction!r}")
    return direction


def check_arch(arch: str) -> str:
    if arch not in ARCHS:
        raise SystemExit(f"this ablation is arch4/arch5 only; got {arch!r}. Allowed: {ARCHS}")
    return arch


def direction_paths(direction: str) -> dict:
    """Per-direction Drive layout. Data + tokenizer are SHARED (identical for both
    directions), only checkpoints/results are separated so the two runs never collide:

        speechbridge/data/           train.tsv.gz, dev.tsv.gz, bpe_zh_en.model  (shared)
        speechbridge/ENZH/{checkpoints,results}/
        speechbridge/ZHEN/{checkpoints,results}/
    """
    check_direction(direction)
    sub = "ENZH" if direction == "enzh" else "ZHEN"
    ckpt = os.environ.get(f"SPEECHBRIDGE_CKPT_{sub}",
                          os.path.join(C.ROOT, sub, "checkpoints"))
    results = os.environ.get(f"SPEECHBRIDGE_RESULTS_{sub}",
                             os.path.join(C.ROOT, sub, "results"))
    for d in (ckpt, results):
        os.makedirs(d, exist_ok=True)
    return {"data": C.DATA_DIR, "ckpt": ckpt, "results": results,
            "tokenizer_prefix": C.TOKENIZER_PREFIX}


def stem(arch: str, direction: str, seed: int) -> str:
    """Filename stem: arch + direction + seed, e.g. arch4_transformer_enzh_seed191."""
    return f"{arch}_{direction}_seed{seed}"
