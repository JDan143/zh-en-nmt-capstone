"""
data/preprocess.py — build train and dev from raw corpora.

Pipeline (seeded SEED=123):
  1. Length filter:        1 <= len(src) <= 100  and  1 <= len(tgt) <= 100
  2. Length-ratio filter:  drop if max(|s|,|t|)/min(|s|,|t|) > 9
  3. Language-ID:          verify ZH and EN (langdetect); drop mismatches
  4. Normalize:            Unicode NFC; strip control chars; collapse whitespace
  5. Deduplicate:          exact match on the (normalized) source (zh) string
Normalize runs before dedup so duplicates are caught after whitespace/Unicode
are canonicalized. Output: gzip TSV (zh <TAB> en) -> train.tsv.gz, dev.tsv.gz.

Length is measured pre-subword: English by whitespace tokens, Chinese by
characters (Chinese is unsegmented). This is an approximation; the real length
cap is reapplied after SentencePiece.

Data sources (see README):
  * OpenSubtitles2018 ZH-EN : OPUS moses zip (object.pouta.csc.fi)
  * News Commentary v18   : statmt .tsv.gz
  * WMT newstest (test set) : fetched via sacrebleu --echo

Run:
    python data/preprocess.py            # full
    SPEECHBRIDGE_SMOKE=1 python data/preprocess.py   # tiny subset, fast
"""
from __future__ import annotations
import gzip
import io
import os
import random
import sys
import unicodedata
import zipfile
from typing import Iterable

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from utils import seed_everything  # noqa: E402

CONTROL_CHARS = {c for c in map(chr, range(0x20)) if c not in "\t\n"}


# ─────────────────────────────────────────────────────────────────────────────
# download helpers
# ─────────────────────────────────────────────────────────────────────────────
def _download(url: str, dest: str, timeout: int = 60) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[cache] {dest}")
        return dest
    print(f"[download] {url}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dest)
    return dest


def opus_moses_url(corpus: str, version: str, l1: str, l2: str) -> str:
    # OPUS expects the pair in alphabetical order: en before zh_cn.
    a, b = sorted([l1, l2])
    return f"https://object.pouta.csc.fi/OPUS-{corpus}/{version}/moses/{a}-{b}.txt.zip"


def load_opensubtitles(sample: int, seed: int) -> list[tuple[str, str]]:
    """Return list of (zh, en) pairs sampled from OpenSubtitles2018."""
    d = C.DATA["opensubtitles"]
    url = opus_moses_url(d["corpus"], d["version"], d["l1"], d["l2"])
    zpath = os.path.join(C.DATA_DIR, "opensubtitles_en_zh.zip")
    _download(url, zpath)
    a, b = sorted([d["l1"], d["l2"]])  # e.g. en, zh_cn
    en_name = f"{d['corpus']}.{a}-{b}.{d['l1']}"
    zh_name = f"{d['corpus']}.{a}-{b}.{d['l2']}"
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        en_member = next(n for n in names if n.endswith(en_name))
        zh_member = next(n for n in names if n.endswith(zh_name))
        with z.open(en_member) as fe, z.open(zh_member) as fz:
            en_lines = io.TextIOWrapper(fe, encoding="utf-8")
            zh_lines = io.TextIOWrapper(fz, encoding="utf-8")
            pairs = [(zh.rstrip("\n"), en.rstrip("\n")) for zh, en in zip(zh_lines, en_lines)]
    print(f"[opensubtitles] {len(pairs):,} raw pairs")
    if sample and len(pairs) > sample:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample)
        print(f"[opensubtitles] sampled {len(pairs):,}")
    return pairs


def _cjk_count(s: str) -> int:
    """Number of CJK Unified Ideographs (a reliable zh-vs-en signal)."""
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")


def load_news_commentary() -> list[tuple[str, str]]:
    """Return list of (zh, en) pairs from News Commentary v18 (a 2-column TSV).

    The column order is NOT assumed from the filename. We sniff the first rows by
    CJK density and map the Chinese/English columns accordingly, so if statmt ever
    ships the columns swapped it self-corrects — and if NO Chinese is found (wrong
    file / corrupt download), it raises instead of silently poisoning training with
    reversed pairs.
    """
    d = C.DATA["news_commentary"]
    gz = os.path.join(C.DATA_DIR, "news_commentary_en_zh.tsv.gz")
    try:
        _download(d["url"], gz)
    except Exception as e:  # noqa: BLE001
        print(f"[news_commentary] primary URL failed ({e}); trying fallback")
        _download(d["url_fallback"], gz)

    rows = []
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                rows.append(parts)

    # --- detect which column is Chinese, by CJK density over the first rows ---
    probe = rows[: min(500, len(rows))]
    cjk0 = sum(_cjk_count(r[0]) for r in probe)
    cjk1 = sum(_cjk_count(r[1]) for r in probe)
    if max(cjk0, cjk1) == 0:
        raise RuntimeError(
            "[news_commentary] no Chinese characters found in either column of "
            f"{gz} — wrong file or corrupt download. Refusing to proceed (a silent "
            "zh/en swap would poison training). Check the URL in config.DATA.")
    zh_idx, en_idx = (0, 1) if cjk0 > cjk1 else (1, 0)
    if zh_idx == 0:
        print("[news_commentary] WARNING: columns are swapped vs the 'en-zh' "
              "filename (col0=zh, col1=en); auto-correcting.")

    pairs = [(r[zh_idx], r[en_idx]) for r in rows]
    print(f"[news_commentary] {len(pairs):,} pairs (zh=col{zh_idx}, en=col{en_idx})")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# cleaning steps
# ─────────────────────────────────────────────────────────────────────────────
def seg_len(text: str, lang: str) -> int:
    return len(text) if lang == "zh" else len(text.split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in CONTROL_CHARS)
    return " ".join(text.split())


def passes_length(zh: str, en: str, max_len: int) -> bool:
    lz, le = seg_len(zh, "zh"), seg_len(en, "en")
    return 1 <= lz <= max_len and 1 <= le <= max_len


def passes_ratio(zh: str, en: str, ratio_max: float) -> bool:
    lz, le = seg_len(zh, "zh"), seg_len(en, "en")
    if min(lz, le) == 0:
        return False
    return max(lz, le) / min(lz, le) <= ratio_max


def langid_ok(zh: str, en: str, detector) -> bool:
    try:
        return detector(zh).startswith("zh") and detector(en) == "en"
    except Exception:
        return False


def clean(pairs: Iterable[tuple[str, str]], do_langid: bool) -> list[tuple[str, str]]:
    max_len = C.DATA["max_len_filter"]
    ratio_max = C.DATA["length_ratio_max"]

    detector = None
    if do_langid:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = C.SEED
        detector = detect

    pairs = list(pairs)
    total = len(pairs)
    # langid runs detect() twice per pair and is the slow step (minutes-to-hours on the full corpus), so emit a heartbeat instead of going silent.
    print(f"[clean] starting on {total:,} pairs "
          f"(langid={'ON — this is the slow part' if do_langid else 'off'})...",
          flush=True)

    seen_src: set[str] = set()
    out: list[tuple[str, str]] = []
    n0 = 0
    for zh, en in pairs:
        n0 += 1
        if n0 % 20000 == 0:
            pct = 100.0 * n0 / total
            print(f"[clean] {n0:,}/{total:,} ({pct:4.1f}%) processed, "
                  f"{len(out):,} kept", flush=True)
        zh, en = zh.strip(), en.strip()
        if not zh or not en:
            continue
        # 1. length
        if not passes_length(zh, en, max_len):
            continue
        # 2. ratio
        if not passes_ratio(zh, en, ratio_max):
            continue
        # 3. language id
        if do_langid and not langid_ok(zh, en, detector):
            continue
        # 4. normalize (NFC + strip control + collapse whitespace)
        zh_n, en_n = normalize(zh), normalize(en)
        if not zh_n or not en_n:
            continue
        # 5. dedup on the normalized source (zh) -> also removes whitespace/Unicode-equivalent dups
        if zh_n in seen_src:
            continue
        seen_src.add(zh_n)
        out.append((zh_n, en_n))
    print(f"[clean] {n0:,} -> {len(out):,} after filtering")
    return out


def write_tsv_gz(pairs: list[tuple[str, str]], path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for zh, en in pairs:
            f.write(f"{zh}\t{en}\n")
    print(f"[write] {path}  ({len(pairs):,} pairs)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    seed_everything(C.SEED)
    smoke = C.SMOKE
    sample = C.SMOKE_OVERRIDES["sample_pairs"] if smoke else C.DATA["opensubtitles"]["sample"]

    # --- training corpora ---
    pairs = load_opensubtitles(sample=sample, seed=C.SEED)
    pairs += load_news_commentary()
    random.Random(C.SEED).shuffle(pairs)
    if smoke:
        pairs = pairs[: C.SMOKE_OVERRIDES["sample_pairs"]]

    cleaned = clean(pairs, do_langid=not smoke)  # langid is the slow step; skip in smoke

    # train/dev split (dev ~2000 or 2% for smoke)
    rng = random.Random(C.SEED)
    rng.shuffle(cleaned)
    dev_n = min(2000, max(50, int(len(cleaned) * 0.02)))
    dev, train = cleaned[:dev_n], cleaned[dev_n:]

    write_tsv_gz(train, os.path.join(C.DATA_DIR, "train.tsv.gz"))
    write_tsv_gz(dev, os.path.join(C.DATA_DIR, "dev.tsv.gz"))

    print("[done] preprocessing complete.")


if __name__ == "__main__":
    main()
