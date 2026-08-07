from __future__ import annotations
import gzip
import os
import sys
from collections import Counter

import sentencepiece as spm

# repo root on path for the study config (constants)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402  (study constants: PAD/UNK/BOS/EOS ids)
import sep_config as S  # noqa: E402


def _write_monolingual(train_gz: str, side: str, out_path: str) -> int:
    assert side in ("zh", "en")
    col = 0 if side == "zh" else 1  # study tsv is zh<TAB>en
    n = 0
    with gzip.open(train_gz, "rt", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            fout.write(parts[col] + "\n")
            n += 1
    return n


def _required_chars(path: str, coverage: float) -> int:
    cnt: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            cnt.update(ch for ch in line.rstrip("\n") if not ch.isspace())
    total = sum(cnt.values())
    if total == 0:
        return 0
    acc, needed = 0, 0
    for _ch, c in cnt.most_common():
        acc += c
        needed += 1
        if acc / total >= coverage:
            break
    return needed


def train_one(side: str, vocab_size: int, coverage: float) -> str:
    train_gz = S.data_file("train.tsv.gz")
    if not os.path.exists(train_gz):
        raise SystemExit(f"[tok] {train_gz} not found. Copy the study's 800k "
                         f"train.tsv.gz into sep_tok/data/ first.")
    prefix = S.tokenizer_prefix(side)
    mono = prefix + "_corpus.txt"
    n = _write_monolingual(train_gz, side, mono)
    print(f"[tok] {side}: wrote {n:,} lines -> {mono}")

    # HARD-FAIL guard (Q1=b): if the monolingual alphabet + merge headroom can't support
    # the requested vocab, SentencePiece would clamp it. We refuse to silently shrink.
    need_chars = _required_chars(mono, coverage)
    floor = need_chars + 4  # + the 4 special tokens
    if vocab_size < floor:
        raise SystemExit(f"[tok] {side}: requested vocab {vocab_size} < required {floor} "
                         f"(chars {need_chars} + 4 specials). Impossible.")
    try:
        spm.SentencePieceTrainer.train(
            input=mono, model_prefix=prefix, vocab_size=vocab_size,
            model_type="bpe", character_coverage=coverage,
            pad_id=C.PAD_ID, unk_id=C.UNK_ID, bos_id=C.BOS_ID, eos_id=C.EOS_ID,
            # NO user_defined_symbols — separate tokenizers need no <zh>/<en> tag.
            hard_vocab_limit=True,  # <- do NOT auto-shrink; error if 32k unreachable
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"[tok] {side}: SentencePiece could not build a {vocab_size} vocab from the "
            f"800k {side} side (hard_vocab_limit=True). This is the Q1 hard-fail: the "
            f"monolingual corpus does not support 32k. Options: lower this side's vocab, "
            f"or reconsider Option B. Underlying error: {e}")

    # verify the achieved size + special ids
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    got = sp.get_piece_size()
    print(f"[tok] {side}: built vocab={got} at {prefix}.model")
    if got != vocab_size:
        raise SystemExit(f"[tok] {side}: achieved vocab {got} != requested {vocab_size} "
                         f"(hard-fail — 32k not reachable from 800k {side}).")
    # SentencePiece stores control symbols by ID (the piece strings default to <pad>/<unk>/
    # <s>/</s>); verify the IDs directly, which is what collate/loss actually use.
    got_ids = {"pad": sp.pad_id(), "unk": sp.unk_id(),
               "bos": sp.bos_id(), "eos": sp.eos_id()}
    want = {"pad": C.PAD_ID, "unk": C.UNK_ID, "bos": C.BOS_ID, "eos": C.EOS_ID}
    for k in want:
        if got_ids[k] != want[k]:
            raise SystemExit(f"[tok] {side}: special id mismatch {k}={got_ids[k]} "
                             f"(want {want[k]})")
    print(f"[tok] {side}: special ids OK (pad{got_ids['pad']}/unk{got_ids['unk']}/"
          f"bos{got_ids['bos']}/eos{got_ids['eos']})")
    return prefix + ".model"


def main():
    cov_zh = 0.9999  # Chinese needs high coverage (large alphabet)
    cov_en = 1.0     # English alphabet is small; full coverage is fine
    print("=== training TWO monolingual tokenizers (Option B: 32k + 32k) ===")
    zh = train_one("zh", S.VOCAB_ZH, cov_zh)
    en = train_one("en", S.VOCAB_EN, cov_en)
    print(f"\n[tok] done.\n  ZH -> {zh}\n  EN -> {en}")
    print("[tok] special IDs verified identical (pad0/unk1/bos2/eos3) across both. "
          "No <zh>/<en> tag (not needed with separate tokenizers).")


if __name__ == "__main__":
    main()
