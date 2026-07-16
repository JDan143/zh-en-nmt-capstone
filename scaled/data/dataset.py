"""
data/dataset.py — turn cleaned TSV + the shared tokenizer into tensors.

Bidirectional trick (Johnson et al., 2017): prepend the TARGET-language tag to
the SOURCE sequence. From each (zh, en) pair we create up to two examples:

    ZH->EN :  src = [<en>] + spm(zh) + [eos]   tgt = [bos] + spm(en) + [eos]
    EN->ZH :  src = [<zh>] + spm(en) + [eos]   tgt = [bos] + spm(zh) + [eos]

`directions` controls which are produced (both for training; one for directional
evaluation). Raw target text is retained so evaluate.py has detokenized refs.
"""
from __future__ import annotations
import gzip
import hashlib
import os
import sys
from array import array
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Sampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402


@dataclass
class Example:
    src_ids: list[int]
    tgt_ids: list[int]
    direction: str       # "zh-en" or "en-zh"
    tgt_text: str        # raw reference (detokenized)


def read_tsv_gz(path: str) -> list[tuple[str, str]]:
    pairs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))  # (zh, en)
    return pairs


class TranslationDataset(Dataset):
    """Lazy: stores raw text + direction; tokenizes in __getitem__. This avoids
    eagerly building ~3M id-lists at startup (slow + memory-heavy on ~1.6M pairs).
    The token-batch sampler needs lengths, so lengths() does a single tokenization
    pass and caches the result (only triggered for the Transformers)."""

    def __init__(self, tsv_gz_path: str, sp, max_subword_len: int = 100,
                 directions=("zh-en", "en-zh")):
        self.sp = sp
        self.src_path = tsv_gz_path
        self.max_len = max_subword_len
        self.tag = {lang: sp.piece_to_id(tok)
                    for lang, tok in (("zh", "<zh>"), ("en", "<en>"))}
        pairs = read_tsv_gz(tsv_gz_path)
        # items: (tgt_lang_tag, source_text, target_text, direction)  — NOT tokenized
        self.items: list[tuple[int, str, str, str]] = []
        for zh, en in pairs:
            if "zh-en" in directions:
                self.items.append((self.tag["en"], zh, en, "zh-en"))
            if "en-zh" in directions:
                self.items.append((self.tag["zh"], en, zh, "en-zh"))
        self._lengths: list[int] | None = None

    def _encode(self, tag, src_text, tgt_text, direction) -> Example | None:
        src_ids = self.sp.encode(src_text, out_type=int)[: self.max_len - 2]
        tgt_ids = self.sp.encode(tgt_text, out_type=int)[: self.max_len - 2]
        if not src_ids or not tgt_ids:
            return None
        src = [tag] + src_ids + [C.EOS_ID]
        tgt = [C.BOS_ID] + tgt_ids + [C.EOS_ID]
        return Example(src, tgt, direction, tgt_text)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tag, src_text, tgt_text, direction = self.items[i]
        ex = self._encode(tag, src_text, tgt_text, direction)
        if ex is None:  # empty after tokenization — return a 1-token fallback
            ex = Example([tag, C.EOS_ID], [C.BOS_ID, C.EOS_ID], direction, tgt_text)
        return ex

    def lengths(self) -> list[int]:
        """max(src_len, tgt_len) per item; tokenizes once and caches."""
        if self._lengths is not None:
            return self._lengths
        cache = self._lengths_cache_path()
        if cache and os.path.exists(cache):
            try:
                with open(cache, "rb") as f:
                    lens = array("i"); lens.frombytes(f.read())
                if len(lens) == len(self.items):
                    print(f"[dataset] lengths cache hit ({len(lens):,}) -> {cache}")
                    self._lengths = list(lens)
                    return self._lengths
                print("[dataset] lengths cache size mismatch; recomputing")
            except Exception as e:  # noqa: BLE001
                print(f"[dataset] lengths cache unreadable ({e}); recomputing")

        print(f"[dataset] computing lengths for {len(self.items):,} examples "
              f"(one-time; cached for later sessions)...", flush=True)
        out: list[int] = []
        for i, (tag, s, t, d) in enumerate(self.items):
            if i and i % 500000 == 0:
                print(f"[dataset]   {i:,}/{len(self.items):,}", flush=True)
            ex = self._encode(tag, s, t, d)
            out.append(max(len(ex.src_ids), len(ex.tgt_ids)) if ex else 2)
        self._lengths = out
        if cache:
            try:
                tmp = cache + ".part"
                with open(tmp, "wb") as f:
                    array("i", out).tofile(f)
                os.replace(tmp, cache)
                print(f"[dataset] lengths cached -> {cache}")
            except Exception as e:  # noqa: BLE001
                print(f"[dataset] could not write lengths cache ({e}); continuing")
        return self._lengths

    def _lengths_cache_path(self) -> str | None:
        if not getattr(self, "src_path", None):
            return None
        try:
            st = os.stat(self.src_path)
            tok = getattr(self.sp, "serialized_model_proto", None)
            tok_id = hashlib.sha256(tok()).hexdigest()[:8] if tok else "na"
            key = (f"{os.path.basename(self.src_path)}|{st.st_size}|"
                   f"{int(st.st_mtime)}|{tok_id}|{self.max_len}")
            h = hashlib.sha256(key.encode()).hexdigest()[:16]
            return os.path.join(os.path.dirname(self.src_path), f".lengths_{h}.bin")
        except Exception:  # noqa: BLE001
            return None


def collate(batch: list[Example]):
    """Pad to batch-max. Returns src, src_len, tgt (full, bos..eos), refs, dirs."""
    src_max = max(len(b.src_ids) for b in batch)
    tgt_max = max(len(b.tgt_ids) for b in batch)
    n = len(batch)
    src = torch.full((n, src_max), C.PAD_ID, dtype=torch.long)
    tgt = torch.full((n, tgt_max), C.PAD_ID, dtype=torch.long)
    src_len = torch.zeros(n, dtype=torch.long)
    for i, b in enumerate(batch):
        src[i, : len(b.src_ids)] = torch.tensor(b.src_ids)
        tgt[i, : len(b.tgt_ids)] = torch.tensor(b.tgt_ids)
        src_len[i] = len(b.src_ids)
    refs = [b.tgt_text for b in batch]
    dirs = [b.direction for b in batch]
    return src, src_len, tgt, refs, dirs


class MaxTokensBatchSampler(Sampler):
    """Length-bucketed batches that target ~max_tokens per batch (fairseq-style).

    tokens(batch) ~= num_sentences * max_seq_len_in_batch. Used for the
    Transformers (Arch 4/5) to approximate the ~4096 tokens/batch spec.
    """

    def __init__(self, dataset: TranslationDataset, max_tokens: int,
                 shuffle: bool = True, seed: int = 123):
        self.lengths = dataset.lengths()
        self.order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._batches = self._make_batches()

    def set_epoch(self, epoch: int):
        """Call once per epoch so batch order varies across epochs (the RNG is
        seeded with seed+epoch — deterministic per epoch, different each epoch)."""
        self.epoch = epoch

    def _make_batches(self):
        batches, cur, cur_max = [], [], 0
        for idx in self.order:
            l = self.lengths[idx]
            new_max = max(cur_max, l)
            if cur and new_max * (len(cur) + 1) > self.max_tokens:
                batches.append(cur)
                cur, cur_max = [idx], l
            else:
                cur.append(idx)
                cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            import random
            random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches

    def __len__(self):
        return len(self._batches)
