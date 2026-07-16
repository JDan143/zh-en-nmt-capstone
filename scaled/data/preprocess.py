"""
Mixture (per-corpus caps are deliberate):
  * OpenSubtitles2018   capped at 2.5M   — conversational, ~11M raw available but the
                                           noisiest source, so capped to protect the
                                           register balance
  * News Commentary v18 all (~300-450k)  — news, matches the WMT newstest evaluation
  * TED2020             all (~few 100k)  — spoken

Pipeline (seeded SEED=191), applied sequentially:
  1. Length filter:        1 <= len(src) <= 100  and  1 <= len(tgt) <= 100
  2. Length-ratio filter:  drop if max(|s|,|t|)/min(|s|,|t|) > 9
  3. Language-ID:          OpenSubtitles ONLY (it is the only dirty source; at ~1-2 ms
                           per pair this would cost hours across 2.5M pairs, and NC /
                           TED2020 are already language-clean)
  4. Normalize:            Unicode NFC; strip control chars; collapse whitespace
  4b. Leakage guard:       drop anything appearing in the WMT / FLORES+ test sets
  5. Deduplicate:          GLOBAL across all three corpora (one shared seen-set)

Deliberate deviations from the design doc, carried over from the 5-arch study:
  * normalize BEFORE dedup, so dedup also collapses whitespace/Unicode-equivalent
    duplicates (strictly more duplicates removed; nothing extra dropped).

Usage:
    python data/preprocess.py            # skips if the data already exists
    python data/preprocess.py --force    # rebuild (invalidates others checkpoints)
    SPEECHBRIDGE_SMOKE=1 python data/preprocess.py   # tiny subset, fast plumbing test

Data sources:
  * OpenSubtitles2018 : OPUS moses zip (object.pouta.csc.fi), code 'zh_cn'
  * News Commentary   : statmt .tsv.gz (column order auto-detected by CJK density)
  * TED2020           : OPUS moses zip, code tried as zh_cn -> zh -> zh-cn
  * WMT / FLORES+     : TEST ONLY — fetched by evaluate.py, never trained on
"""
from __future__ import annotations
import argparse
import hashlib
import json
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


def _read_opus_moses_zip(zpath: str, corpus: str, l1: str, l2: str) -> list[tuple[str, str]]:
    a, b = sorted([l1, l2])
    en_name = f"{corpus}.{a}-{b}.{l1}"
    zh_name = f"{corpus}.{a}-{b}.{l2}"
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        en_member = next(n for n in names if n.endswith(en_name))
        zh_member = next(n for n in names if n.endswith(zh_name))
        with z.open(en_member) as fe, z.open(zh_member) as fz:
            en_lines = io.TextIOWrapper(fe, encoding="utf-8")
            zh_lines = io.TextIOWrapper(fz, encoding="utf-8")
            return [(zh.rstrip("\n"), en.rstrip("\n")) for zh, en in zip(zh_lines, en_lines)]


def load_opus_corpus(key: str, sample: int | None, seed: int,
                     l2_candidates: list[str] | None = None) -> list[tuple[str, str]]:
    d = C.DATA[key]
    corpus, version, l1 = d["corpus"], d["version"], d["l1"]
    candidates = l2_candidates or [d["l2"]]
    pairs, last_err = None, None
    for l2 in candidates:
        url = opus_moses_url(corpus, version, l1, l2)
        zpath = os.path.join(C.DATA_DIR, f"{key}_{l1}_{l2}.zip")
        try:
            _download(url, zpath)
            pairs = _read_opus_moses_zip(zpath, corpus, l1, l2)
            print(f"[{key}] {len(pairs):,} raw pairs (OPUS code '{l2}')")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[{key}] code '{l2}' unavailable ({type(e).__name__}); trying next")
            last_err = e
    if pairs is None:
        raise RuntimeError(f"[{key}] could not download {corpus} {version} for any of "
                           f"{candidates}. Last error: {last_err}")
    if sample and len(pairs) > sample:
        pairs = random.Random(seed).sample(pairs, sample)
        print(f"[{key}] capped to {len(pairs):,}")
    return pairs


def load_opensubtitles(sample: int, seed: int) -> list[tuple[str, str]]:
    return load_opus_corpus("opensubtitles", sample, seed)


def load_ted2020(seed: int) -> list[tuple[str, str]]:
    d = C.DATA["ted2020"]
    return load_opus_corpus("ted2020", d.get("sample"), seed,
                            l2_candidates=d["l2_candidates"])


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


def clean(pairs: Iterable[tuple[str, str]], do_langid: bool,
          seen_src: set[str] | None = None,
          exclude_src: set[str] | None = None,
          tag: str = "clean") -> list[tuple[str, str]]:
    max_len = C.DATA["max_len_filter"]
    ratio_max = C.DATA["length_ratio_max"]

    detector = None
    if do_langid:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = C.SEED
        detector = detect

    pairs = list(pairs)
    total = len(pairs)
    # langid runs detect() twice per pair and is the slow step (hours at this scale),
    # so emit a heartbeat instead of going silent.
    print(f"[{tag}] starting on {total:,} pairs "
          f"(langid={'ON — the slow step' if do_langid else 'off'})...", flush=True)

    if seen_src is None:
        seen_src = set()
    out: list[tuple[str, str]] = []
    n0 = n_leak = 0
    for zh, en in pairs:
        n0 += 1
        if n0 % 100000 == 0:
            print(f"[{tag}] {n0:,}/{total:,} ({100.0*n0/total:4.1f}%), "
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
        # 4. normalize (NFC + strip control + collapse whitespace) — BEFORE dedup
        zh_n, en_n = normalize(zh), normalize(en)
        if not zh_n or not en_n:
            continue
        # 4b. never train on a benchmark test sentence
        if exclude_src and (zh_n in exclude_src or en_n in exclude_src):
            n_leak += 1
            continue
        # 5. dedup on the NORMALIZED source (GLOBAL when seen_src is shared)
        if zh_n in seen_src:
            continue
        seen_src.add(zh_n)
        out.append((zh_n, en_n))
    msg = f"[{tag}] {n0:,} -> {len(out):,} after filtering"
    if exclude_src:
        msg += f" ({n_leak:,} dropped as test-set overlap)"
    print(msg)
    return out


def build_test_source_index() -> set[str]:
    idx: set[str] = set()
    try:
        from sacrebleu.utils import get_source_file, get_reference_files
        for ts in (C.DATA["wmt_testset"], C.DATA["wmt_testset_alt"]):
            for lp in ("zh-en", "en-zh"):
                try:
                    files = [get_source_file(ts, lp)] + list(get_reference_files(ts, lp))
                    for fp in files:
                        with open(fp, encoding="utf-8") as f:
                            for line in f:
                                s = normalize(line.strip())
                                if s:
                                    idx.add(s)
                except Exception:  # noqa: BLE001, S112
                    continue
    except Exception as e:  # noqa: BLE001
        print(f"[leakage] could not index WMT test sets ({e}); skipping")
    try:
        from datasets import load_dataset
        f = C.DATA["flores"]
        for code in (f["zh_code"], f["en_code"]):
            ds = load_dataset(f["hf_id"], code, split=f["split"])
            for row in ds:
                s = normalize((row.get("text") or "").strip())
                if s:
                    idx.add(s)
    except Exception as e:  # noqa: BLE001
        print(f"[leakage] could not index FLORES+ ({e}); skipping "
              f"(gated — run `huggingface-cli login` to enable this check)")
    print(f"[leakage] indexed {len(idx):,} test sentences to exclude")
    return idx


def write_tsv_gz(pairs: list[tuple[str, str]], path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for zh, en in pairs:
            f.write(f"{zh}\t{en}\n")
    print(f"[write] {path}  ({len(pairs):,} pairs)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_manifest(train_path: str, dev_path: str, counts: dict) -> None:
    man = {
        "profile": "scaled-smoke" if C.SMOKE else "scaled",
        "seed": C.SEED,
        "mixture": counts,
        "train_pairs": counts.get("_train"),
        "dev_pairs": counts.get("_dev"),
        "train_sha256_16": fingerprint(train_path),
        "dev_sha256_16": fingerprint(dev_path),
    }
    with open(os.path.join(C.DATA_DIR, "data_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print(" DATA MANIFEST — share train_sha256_16 with your team.")
    print(" Every member MUST see the same hash, or their checkpoints are incompatible.")
    print("=" * 70)
    for k, v in man.items():
        print(f"  {k}: {v}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaled-run preprocessing (Phase 2).")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if train/dev exist. DANGER: if the data changes, "
                         "every teammate's checkpoint becomes invalid.")
    args = ap.parse_args()

    seed_everything(C.SEED)
    train_path = os.path.join(C.DATA_DIR, "train.tsv.gz")
    dev_path = os.path.join(C.DATA_DIR, "dev.tsv.gz")

    if os.path.exists(train_path) and os.path.exists(dev_path) and not args.force:
        print(f"[skip] {train_path} and {dev_path} exist — reusing the shared data.")
        man = os.path.join(C.DATA_DIR, "data_manifest.json")
        if os.path.exists(man):
            print(open(man, encoding="utf-8").read())
        print("[skip] --force rebuilds (only if the whole team restarts together).")
        return

    counts: dict = {}

    if C.SMOKE:
        n = C.SMOKE_OVERRIDES["sample_pairs"]
        sources = [("opensubtitles", load_opensubtitles(n, C.SEED), False)]
        counts["opensubtitles"] = len(sources[0][1])
    else:
        os_pairs = load_opensubtitles(C.DATA["opensubtitles"]["sample"], C.SEED)
        counts["opensubtitles"] = len(os_pairs)
        nc_pairs = load_news_commentary()
        counts["news_commentary"] = len(nc_pairs)
        ted_pairs = load_ted2020(C.SEED)
        counts["ted2020"] = len(ted_pairs)
        sources = [("opensubtitles", os_pairs, True),
                   ("news_commentary", nc_pairs, False),
                   ("ted2020", ted_pairs, False)]

    exclude = set() if C.SMOKE else build_test_source_index()

    seen_src: set[str] = set()
    cleaned: list[tuple[str, str]] = []
    for name, pairs, needs_langid in sources:
        kept = clean(pairs, do_langid=needs_langid, seen_src=seen_src,
                     exclude_src=exclude, tag=name)
        counts[f"{name}_clean"] = len(kept)
        cleaned += kept

    rng = random.Random(C.SEED)
    rng.shuffle(cleaned)
    dev_n = min(C.DEV_SIZE, max(50, int(len(cleaned) * 0.02)))
    dev, train = cleaned[:dev_n], cleaned[dev_n:]
    counts["_train"], counts["_dev"] = len(train), len(dev)

    write_tsv_gz(train, train_path)
    write_tsv_gz(dev, dev_path)

    print("\n[mix] register balance of the final training set:")
    for name, _, _ in sources:
        raw_n, cl_n = counts.get(name, 0), counts.get(f"{name}_clean", 0)
        pct = 100.0 * cl_n / max(len(cleaned), 1)
        print(f"   {name:18s} {raw_n:>10,} raw -> {cl_n:>10,} clean ({pct:5.1f}% of mixture)")
    write_manifest(train_path, dev_path, counts)
    print("[done] scaled preprocessing complete.")


if __name__ == "__main__":
    main()
