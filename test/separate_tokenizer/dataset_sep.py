from __future__ import annotations
import gzip
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config as C  # noqa: E402
from data.dataset import Example, read_tsv_gz  # noqa: E402


class SeparateTokenizerDataset(torch.utils.data.Dataset):

    def __init__(self, tsv_gz_path: str, sp_src, sp_tgt, direction: str,
                 max_subword_len: int = 100):
        assert direction in ("en-zh", "zh-en")
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.direction = direction
        self.max_len = max_subword_len
        pairs = read_tsv_gz(tsv_gz_path)
        self.items: list[tuple[str, str]] = []
        for zh, en in pairs:
            if direction == "en-zh":
                self.items.append((en, zh))
            else:
                self.items.append((zh, en))
        self._lengths: list[int] | None = None

    def _encode(self, src_text, tgt_text) -> Example | None:
        src_ids = self.sp_src.encode(src_text, out_type=int)[: self.max_len - 2]
        tgt_ids = self.sp_tgt.encode(tgt_text, out_type=int)[: self.max_len - 2]
        if not src_ids or not tgt_ids:
            return None
        src = src_ids + [C.EOS_ID]
        tgt = [C.BOS_ID] + tgt_ids + [C.EOS_ID]
        return Example(src, tgt, self.direction, tgt_text)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src_text, tgt_text = self.items[i]
        ex = self._encode(src_text, tgt_text)
        if ex is None:  # empty after tokenization — 1-token fallback (matches study)
            ex = Example([C.EOS_ID], [C.BOS_ID, C.EOS_ID], self.direction, tgt_text)
        return ex

    def lengths(self) -> list[int]:
        if self._lengths is None:
            self._lengths = []
            for s, t in self.items:
                ex = self._encode(s, t)
                self._lengths.append(max(len(ex.src_ids), len(ex.tgt_ids)) if ex else 2)
        return self._lengths
