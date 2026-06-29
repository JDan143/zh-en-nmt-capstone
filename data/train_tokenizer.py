"""
data/train_tokenizer.py — train one shared SentencePiece BPE model.

The same model file (bpe_zh_en.model) is reused by all five architectures, so
they share one vocabulary.

Config:
  vocab_size=32000, character_coverage=0.9999, model_type='bpe'
  pad=0, unk=1, bos=2, eos=3, user_defined_symbols=['<zh>','<en>']

Chinese and English text are concatenated into one training file so a single
subword vocabulary covers both languages.

Small-corpus / smoke handling:
  SentencePiece needs vocab_size >= required_chars + special_tokens, where
  required_chars is the set of distinct characters needed to hit
  character_coverage. Chinese has thousands of hanzi, so a tiny smoke subset can
  already need >2500 slots. So we measure the corpus alphabet and raise
  vocab_size to fit (plus headroom for BPE merges). At the full vocab_size=32000
  this never triggers.
"""
from __future__ import annotations
import gzip
import os
import sys
from collections import Counter

import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

MERGE_HEADROOM = 256  # extra slots above the alphabet so BPE has merges to learn


def build_corpus_file() -> str:
    """Write combined (zh + en) training text to a single plain-text file."""
    train_gz = os.path.join(C.DATA_DIR, "train.tsv.gz")
    if not os.path.exists(train_gz):
        raise FileNotFoundError(
            f"{train_gz} not found. Run data/preprocess.py first.")
    combined = os.path.join(C.DATA_DIR, "spm_train_combined.txt")
    n = 0
    with gzip.open(train_gz, "rt", encoding="utf-8") as fin, \
            open(combined, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            zh, en = parts
            fout.write(zh + "\n")
            fout.write(en + "\n")
            n += 2
    print(f"[tokenizer] wrote {n:,} lines -> {combined}")
    return combined


def required_char_count(path: str, coverage: float) -> int:
    """Smallest #distinct chars whose cumulative frequency >= coverage
    (mirrors how SentencePiece computes required_chars)."""
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


def train() -> str:
    combined = build_corpus_file()
    requested = C.CONFIG["vocab_size"]  # 32000, or smaller in smoke mode
    model_prefix = C.TOKENIZER_PREFIX

    # auto-fit vocab to the corpus alphabet (+ special tokens + merge headroom)
    n_special = 4 + len(C.LANG_TOKENS)  # pad/unk/bos/eos + <zh>,<en>
    required = required_char_count(combined, C.CHAR_COVERAGE)
    floor = required + n_special + MERGE_HEADROOM
    vocab_size = max(requested, floor)
    if vocab_size != requested:
        print(f"[tokenizer] requested vocab_size={requested} is below the corpus "
              f"alphabet floor ({required} chars + {n_special} special + "
              f"{MERGE_HEADROOM} merges) -> using {vocab_size}")

    spm.SentencePieceTrainer.train(
        input=combined,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=C.CHAR_COVERAGE,
        model_type=C.MODEL_TYPE,
        pad_id=C.PAD_ID, unk_id=C.UNK_ID, bos_id=C.BOS_ID, eos_id=C.EOS_ID,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        user_defined_symbols=C.LANG_TOKENS,   # <zh>, <en>
        hard_vocab_limit=False,               # robust on small corpora
        input_sentence_size=2_000_000,        # cap for memory on big corpora
        shuffle_input_sentence=True,
    )
    model_path = model_prefix + ".model"
    print(f"[tokenizer] trained -> {model_path}")

    # sanity check: special-token ids and language tags resolve correctly
    sp = spm.SentencePieceProcessor(model_file=model_path)
    assert sp.pad_id() == C.PAD_ID and sp.unk_id() == C.UNK_ID
    assert sp.bos_id() == C.BOS_ID and sp.eos_id() == C.EOS_ID
    for tag in C.LANG_TOKENS:
        tid = sp.piece_to_id(tag)
        assert tid != C.UNK_ID, f"language tag {tag} did not register"
        print(f"[tokenizer] {tag} -> id {tid}")
    print(f"[tokenizer] vocab_size={sp.get_piece_size()}")
    return model_path


if __name__ == "__main__":
    train()
