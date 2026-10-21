"""
extras around finetune.py. nothing here trains, picks or changes a model.

run from scaled/:
    python ft_extras.py near-dups       lists ft_test rows that are near-copies of an ft_train
                                        sentence. run it once before training, it saves
                                        results/ft_test_near_dups.json
    python ft_extras.py rescore-test    ft_test scores again without those rows, from the
                                        outputs the test step already saved (no new
                                        translation, so it isn't another look at the test)

near-copy = same source sentence once digits are masked and punctuation and spaces are
dropped, or a character 4-gram jaccard of 0.8 or more against some ft_train source.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import finetune as F  # noqa: E402  (sets the fine-tune dirs before config is imported)

NGRAM = 4
JACCARD = 0.8
OUT_NAME = "ft_test_near_dups.json"


def _norm(s: str) -> str:
    s = re.sub(r"\d+", "0", s.lower())
    return re.sub(r"[\W_]+", "", s)


def _grams(s: str) -> set:
    s = _norm(s)
    return {s[i:i + NGRAM] for i in range(max(1, len(s) - NGRAM + 1))}


def find_near_dups(train_pairs, test_pairs, test_meta) -> dict:
    """returns {direction: [0-based ft_test row numbers]}. compares the natural source
    side only, since that is what the model reads at test time."""
    out = {}
    for direction, lang, side in (("en-zh", "en", 1), ("zh-en", "zh", 0)):
        train_norm = {_norm(p[side]) for p in train_pairs}
        train_grams = [_grams(p[side]) for p in train_pairs]
        index = collections.defaultdict(list)
        for i, g in enumerate(train_grams):
            for x in g:
                index[x].append(i)
        rows = []
        for r, m in enumerate(test_meta):
            if m["source_lang"].strip() != lang:
                continue
            src = test_pairs[r][side]
            if _norm(src) in train_norm:
                rows.append(r)
                continue
            g = _grams(src)
            hits = collections.Counter()
            for x in g:
                if len(index[x]) < 2000:   # very common grams say nothing and cost a lot
                    hits.update(index[x])
            best = max((c / len(g | train_grams[j]) for j, c in hits.most_common(30)), default=0.0)
            if best >= JACCARD:
                rows.append(r)
        out[direction] = rows
    return out


def cmd_near_dups(args):
    d = F.ft_dirs()
    train = F.read_pairs(os.path.join(d["data"], "ft_train.tsv.gz"))
    test = F.read_pairs(os.path.join(d["data"], "ft_test.tsv.gz"))
    meta = F.read_meta(os.path.join(d["data"], "test_meta.tsv"), len(test))
    found = find_near_dups(train, test, meta)
    info = {"rows": found, "rule": f"masked-digit exact match or char {NGRAM}-gram jaccard >= "
                                    f"{JACCARD} against an ft_train source, natural source side",
            "ft_train_sha16": F.content_sha16(os.path.join(d["data"], "ft_train.tsv.gz")),
            "ft_test_sha16": F.content_sha16(os.path.join(d["data"], "ft_test.tsv.gz")),
            "made_at": datetime.now().isoformat(timespec="seconds"), "made_by": args.owner}
    p = os.path.join(d["results"], OUT_NAME)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            old = json.load(f)
        if old["rows"] == found:
            print(f"[extras] {OUT_NAME} already there and still matches the data")
            return
        if os.path.exists(os.path.join(d["results"], "test_runs.jsonl")):
            raise SystemExit(f"[extras] {OUT_NAME} differs from the data now, but the test was "
                             f"already scored. leaving the old list in place")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    for direction, rows in found.items():
        n = sum(1 for m in meta if m["source_lang"].strip() == direction[:2])
        print(f"[extras] {direction}: {len(rows)} of {n} test rows are near-copies of ft_train")
    print(f"[extras] saved {p}")


def _ft_test_refs(d, prereview: bool) -> dict:
    """{set key: (refs, out_lang, test row numbers)} in the same order the test step used"""
    pairs = F.read_pairs(os.path.join(d["data"], "ft_test.tsv.gz"))
    meta = F.read_meta(os.path.join(d["data"], "test_meta.tsv"), len(pairs))
    nat = F.natural_source_split(pairs, meta)
    out = {"ft_test:en-zh": (nat["en-zh"][1], "zh", nat["en-zh"][2]),
           "ft_test:zh-en": (nat["zh-en"][1], "en", nat["zh-en"][2])}
    pre_path = os.path.join(d["data"], "ft_test_prereview.tsv.gz")
    if prereview and os.path.exists(pre_path):
        pre = F.read_pairs(pre_path)
        for direction, ref_side in (("en-zh", 0), ("zh-en", 1)):
            rows = nat[direction][2]
            lang = "zh" if direction == "en-zh" else "en"
            out[f"ft_test:{direction}:prereview"] = ([pre[i][ref_side] for i in rows], lang, rows)
    return out


def cmd_rescore_test(args):
    d = F.ft_dirs()
    res_p = os.path.join(d["results"], "test_results.json")
    dup_p = os.path.join(d["results"], OUT_NAME)
    for p in (res_p, dup_p):
        if not os.path.exists(p):
            raise SystemExit(f"[extras] {p} not there yet")
    with open(res_p, encoding="utf-8") as f:
        results = json.load(f)
    with open(dup_p, encoding="utf-8") as f:
        dups = json.load(f)
    if os.path.getmtime(dup_p) > os.path.getmtime(res_p):
        print("[extras] the near-copy list is newer than the test results. it was meant to be "
              "fixed before the test, worth mentioning if you report these numbers")
    seeds = results["seeds"]
    names = ["base"] + [f"seed{s}" for s in seeds]
    hyps = {}
    for n in names:
        with open(os.path.join(d["results"], "test_hyps", f"{n}.json"), encoding="utf-8") as f:
            hyps[n] = json.load(f)
    out = {"winner": results["winner"], "seeds": seeds, "rule": dups["rule"], "sets": {}}
    final = f"seed{seeds[0]}"
    for key, (refs, lang, rows) in _ft_test_refs(d, prereview=True).items():
        hkey = key.replace(":prereview", "")
        if hkey not in hyps["base"]:
            continue
        n_h = len(hyps["base"][hkey])        # shorter than refs in a smoke run with --limit
        drop = set(dups["rows"][hkey.split(":")[1]])
        keep = [i for i in range(n_h) if rows[i] not in drop]
        r_keep = [refs[i] for i in keep]
        entry = {"n": len(keep), "n_dropped": n_h - len(keep)}
        for n in names:
            entry[n] = F.score_texts([hyps[n][hkey][i] for i in keep], r_keep, lang)
        entry["bootstrap"] = F.paired_bootstrap([hyps["base"][hkey][i] for i in keep],
                                                [hyps[final][hkey][i] for i in keep],
                                                r_keep, lang, n_samples=args.bootstrap)
        out["sets"][key] = entry
    p = os.path.join(d["results"], "test_results_no_near_dups.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nft_test without near-copies of ft_train, base vs {final} "
          f"(compare with the full rows in test_results.csv)")
    for key, e in out["sets"].items():
        for m in ("BLEU", "chrF++"):
            b = e["bootstrap"][m]
            print(f"{key:<28} n={e['n']:<4} (-{e['n_dropped']:<3}) {m:<6} base {e['base'][m]:6.2f}  "
                  f"final {e[final][m]:6.2f}  diff {b['delta']:+6.2f} "
                  f"[{b['delta_ci95'][0]:+.2f}, {b['delta_ci95'][1]:+.2f}]")
    print(f"[extras] saved {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="extras around finetune.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("near-dups")
    p.add_argument("--owner", default=os.environ.get("SPEECHBRIDGE_OWNER", "unknown"))
    p = sub.add_parser("rescore-test")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    {"near-dups": cmd_near_dups, "rescore-test": cmd_rescore_test}[args.cmd](args)


if __name__ == "__main__":
    main()
