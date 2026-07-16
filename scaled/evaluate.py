"""
Benchmarks (both directions ZH->EN and EN->ZH):
  * WMT newstest  (fetched via sacrebleu --echo)
  * FLORES+       (openlanguagedata/flores_plus, devtest, cmn_Hans/eng_Latn)
                  eval-only, multi-parallel; never trained on.

Metrics:
  BLEU-4 (sacrebleu; tok=zh for Chinese output, 13a for English)
  chrF++ (sacrebleu CHRF word_order=2)
  TER    (sacrebleu TER; lower is better)
  spBLEU (sacrebleu tok=flores200) — reported for FLORES+
  Perplexity (exp of token CE on the benchmark, teacher-forced)
  Params(M), Train time/epoch(s)*, Peak VRAM(GB)*, Inference speed(tok/s), BLEU/M-params
  (* read back from results/metrics_log_seed{seed}.csv)
"""
from __future__ import annotations
import argparse
import csv
import importlib
import os
import sys
import time
from statistics import mean, pstdev

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from utils import seed_everything, get_device, count_params, human_millions  # noqa: E402
from data.dataset import collate  # noqa: E402 (unused but keeps import graph explicit)
from models.common import greedy_decode, beam_search  # noqa: E402


MODEL_MODULES = {
    "arch4_transformer": "models.arch4_transformer",
}


# ─────────────────────────────────────────────────────────────────────────────
# benchmark loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_wmt(testset: str) -> dict[str, tuple[list[str], list[str]]]:
    """Return {'zh-en': (src_zh, ref_en), 'en-zh': (src_en, ref_zh)}."""
    import subprocess

    def echo(langpair, which):
        out = subprocess.run(
            [sys.executable, "-m", "sacrebleu", "-t", testset, "-l", langpair,
             "--echo", which], capture_output=True, text=True, check=True)
        return out.stdout.rstrip("\n").split("\n")

    data = {}
    for lp in ("zh-en", "en-zh"):
        try:
            data[lp] = (echo(lp, "src"), echo(lp, "ref"))
        except Exception as e:  # noqa: BLE001
            print(f"[wmt] {testset} {lp} unavailable: {e}")
    return data


def load_flores() -> dict[str, tuple[list[str], list[str]]]:
    """FLORES+ devtest, aligned by id. Returns same dict shape as load_wmt."""
    from datasets import load_dataset
    f = C.DATA["flores"]

    def lang_text(code):
        ds = load_dataset(f["hf_id"], code, split=f["split"])
        cols = ds.column_names
        # sentence column is 'text' in FLORES+; fall back to a 'sentence*' column
        col = "text" if "text" in cols else \
            next(c for c in cols if c.startswith("sentence"))
        if "id" in cols:
            return {row["id"]: row[col] for row in ds}
        return {i: row[col] for i, row in enumerate(ds)}  # fall back to row order

    zh, en = lang_text(f["zh_code"]), lang_text(f["en_code"])
    ids = sorted(set(zh) & set(en))
    zh_t = [zh[i] for i in ids]
    en_t = [en[i] for i in ids]
    return {"zh-en": (zh_t, en_t), "en-zh": (en_t, zh_t)}


# ─────────────────────────────────────────────────────────────────────────────
# model loading + translation
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, sp, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    # Rebuild the exact architecture the checkpoint was trained with (from the
    # stored cfg), not the live config — otherwise a SMOKE/full mismatch or any
    # later edit to config.py would change the shapes and break state_dict load.
    from config import ArchConfig
    cfg = ArchConfig(**ckpt["cfg"]) if "cfg" in ckpt else C.get_arch_config(arch)
    mod = importlib.import_module(MODEL_MODULES[arch])
    model = mod.build(cfg, ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, arch


def encode_source(sp, text, target_lang, max_len):
    tag = sp.piece_to_id("<en>" if target_lang == "en" else "<zh>")
    ids = sp.encode(text, out_type=int)[: max_len - 2]
    return [tag] + ids + [C.EOS_ID]


def strip_decode(sp, ids):
    out = []
    for i in ids:
        if i == C.EOS_ID:
            break
        if i in (C.PAD_ID, C.BOS_ID):
            continue
        out.append(int(i))
    return sp.decode(out)


@torch.no_grad()
def translate_corpus(model, cfg, sp, src_texts, target_lang, device,
                     beam=5, batch_size=32, max_eval=None):
    if max_eval:
        src_texts = src_texts[:max_eval]
    max_len = cfg.max_len if cfg.family == "transformer" else C.MAX_TOKENS_PER_SIDE
    hyps, n_out_tok, t_total = [], 0, 0.0
    for s in range(0, len(src_texts), batch_size):
        chunk = src_texts[s:s + batch_size]
        seqs = [encode_source(sp, t, target_lang, max_len) for t in chunk]
        smax = max(len(x) for x in seqs)
        src = torch.full((len(seqs), smax), C.PAD_ID, dtype=torch.long, device=device)
        src_len = torch.zeros(len(seqs), dtype=torch.long, device=device)
        for i, x in enumerate(seqs):
            src[i, :len(x)] = torch.tensor(x, device=device)
            src_len[i] = len(x)
        t0 = time.time()
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and cfg.amp_safe)):
            if beam and beam > 1:
                out = beam_search(model, src, src_len, max_len=max_len,
                                  beam=beam, length_penalty=cfg.length_penalty)
            else:
                out = greedy_decode(model, src, src_len, max_len=max_len)
        t_total += time.time() - t0
        for row in out.tolist():
            hyps.append(strip_decode(sp, row))
            n_out_tok += sum(1 for i in row if i not in (C.PAD_ID, C.BOS_ID, C.EOS_ID))
    tok_per_s = n_out_tok / t_total if t_total > 0 else 0.0
    return hyps, tok_per_s


# ─────────────────────────────────────────────────────────────────────────────
# scoring
# ─────────────────────────────────────────────────────────────────────────────
def score(hyps, refs, output_lang, with_spbleu=False):
    from sacrebleu.metrics import BLEU, CHRF, TER
    if output_lang == "zh":
        cseg = lambda s: " ".join(s.replace(" ", ""))
        h_c = [cseg(x) for x in hyps]
        r_c = [cseg(x) for x in refs]
        res = {
            "BLEU": BLEU(tokenize="zh").corpus_score(hyps, [refs]).score,
            "chrF++": CHRF(word_order=2).corpus_score(h_c, [r_c]).score,
            "TER": TER().corpus_score(h_c, [r_c]).score,
        }
    else:
        res = {
            "BLEU": BLEU(tokenize="13a").corpus_score(hyps, [refs]).score,
            "chrF++": CHRF(word_order=2).corpus_score(hyps, [refs]).score,
            "TER": TER().corpus_score(hyps, [refs]).score,
        }
    if with_spbleu:
        res["spBLEU"] = BLEU(tokenize="flores200").corpus_score(hyps, [refs]).score
    return res


@torch.no_grad()
def corpus_perplexity(model, cfg, sp, src_texts, ref_texts, target_lang, device,
                      max_eval=None):
    import torch.nn.functional as F
    if max_eval:
        src_texts, ref_texts = src_texts[:max_eval], ref_texts[:max_eval]
    max_len = cfg.max_len if cfg.family == "transformer" else C.MAX_TOKENS_PER_SIDE
    total_ce, total_tok = 0.0, 0
    bs = 32
    for s in range(0, len(src_texts), bs):
        srcs = src_texts[s:s + bs]
        refs = ref_texts[s:s + bs]
        src_seqs = [encode_source(sp, t, target_lang, max_len) for t in srcs]
        tgt_seqs = [[C.BOS_ID] + sp.encode(t, out_type=int)[:max_len - 2] + [C.EOS_ID]
                    for t in refs]
        smax, tmax = max(map(len, src_seqs)), max(map(len, tgt_seqs))
        src = torch.full((len(srcs), smax), C.PAD_ID, dtype=torch.long, device=device)
        tgt = torch.full((len(srcs), tmax), C.PAD_ID, dtype=torch.long, device=device)
        src_len = torch.zeros(len(srcs), dtype=torch.long, device=device)
        for i, (a, b) in enumerate(zip(src_seqs, tgt_seqs)):
            src[i, :len(a)] = torch.tensor(a, device=device); src_len[i] = len(a)
            tgt[i, :len(b)] = torch.tensor(b, device=device)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and cfg.amp_safe)):
            logits = model(src, src_len, tgt[:, :-1])
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                             tgt[:, 1:].reshape(-1),
                             ignore_index=C.PAD_ID, reduction="sum")
        total_ce += ce.item()
        total_tok += (tgt[:, 1:] != C.PAD_ID).sum().item()
    import math
    return math.exp(min(total_ce / max(total_tok, 1), 20))


# ─────────────────────────────────────────────────────────────────────────────
# efficiency from metrics_log.csv
# ─────────────────────────────────────────────────────────────────────────────
def efficiency_from_log(arch, seed):
    path = os.path.join(C.RESULTS_DIR, f"metrics_log_seed{seed}.csv")
    if not os.path.exists(path):
        path = os.path.join(C.RESULTS_DIR, "metrics_log.csv")
    times, vram = [], []
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("arch") == arch and str(r.get("seed")) == str(seed):
                    try:
                        times.append(float(r["epoch_time_s"]))
                        vram.append(float(r["peak_vram_gb"]))
                    except (KeyError, ValueError):
                        pass
    return {
        "train_time_per_epoch_s": round(mean(times), 2) if times else None,
        "peak_vram_gb": round(max(vram), 3) if vram else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# paired bootstrap significance (self-contained; replaces unstable sacrebleu API)
# ─────────────────────────────────────────────────────────────────────────────
def paired_bootstrap_bleu(hyps_a, hyps_b, refs, output_lang,
                          n_samples=1000, seed=123):
    import random
    from sacrebleu.metrics import BLEU
    tok = "zh" if output_lang == "zh" else "13a"
    bleu = BLEU(tokenize=tok)

    def corpus(hyps, idx):
        return bleu.corpus_score([hyps[i] for i in idx],
                                 [[refs[i] for i in idx]]).score

    n = len(refs)
    full = list(range(n))
    obs = corpus(hyps_b, full) - corpus(hyps_a, full)
    rng = random.Random(seed)
    wins = 0
    diffs = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        d = corpus(hyps_b, idx) - corpus(hyps_a, idx)
        diffs.append(d)
        if d > 0:
            wins += 1
    p_value = 1.0 - wins / n_samples   # fraction where B did NOT beat A
    return {"observed_delta_BLEU": round(obs, 3),
            "mean_bootstrap_delta": round(mean(diffs), 3),
            "p_value": round(p_value, 4),
            "n_samples": n_samples}


# ─────────────────────────────────────────────────────────────────────────────
# driver
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_checkpoint(ckpt_path, sp, device, beam, benchmarks, max_eval, keep_hyps=False):
    model, cfg, arch = load_model(ckpt_path, sp, device)
    seed = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("seed", C.SEED)
    params_m = human_millions(count_params(model))
    results = {"arch": arch, "seed": seed, "params_M": params_m,
               "efficiency": efficiency_from_log(arch, seed), "scores": {}}

    for bench_name, bench in benchmarks.items():
        for direction, (src_texts, ref_texts) in bench.items():
            out_lang = "en" if direction == "zh-en" else "zh"
            hyps, tok_s = translate_corpus(model, cfg, sp, src_texts, out_lang,
                                           device, beam=beam, max_eval=max_eval)
            refs = ref_texts[:max_eval] if max_eval else ref_texts
            sc = score(hyps, refs, out_lang, with_spbleu=(bench_name == "flores+"))
            ppl = corpus_perplexity(model, cfg, sp, src_texts, ref_texts, out_lang,
                                    device, max_eval=max_eval)
            sc["perplexity"] = round(ppl, 3)
            sc["inference_tok_per_s"] = round(tok_s, 1)
            sc["BLEU_per_Mparams"] = round(sc["BLEU"] / params_m, 4) if params_m else None
            results["scores"][f"{bench_name}:{direction}"] = {
                k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items()}
            if keep_hyps:  # only the significance path needs hypotheses retained
                results["scores"][f"{bench_name}:{direction}"]["_hyps"] = hyps
            del hyps
    return results


def aggregate(per_seed_results):
    """mean ± std across seeds for every numeric score."""
    agg = {}
    keys = per_seed_results[0]["scores"].keys()
    for k in keys:
        agg[k] = {}
        metric_names = [m for m in per_seed_results[0]["scores"][k] if not m.startswith("_")]
        for m in metric_names:
            vals = [r["scores"][k][m] for r in per_seed_results
                    if isinstance(r["scores"][k].get(m), (int, float))]
            if vals:
                agg[k][m] = {"mean": round(mean(vals), 3),
                             "std": round(pstdev(vals), 3) if len(vals) > 1 else 0.0}
    return agg


def print_table(arch, agg, efficiency):
    print("\n" + "=" * 78)
    print(f"RESULTS — {arch}   (mean ± std over {len(C.EVAL_SEEDS)} seeds)")
    print(f"  train_time/epoch(s): {efficiency.get('train_time_per_epoch_s')}  "
          f"peak_VRAM(GB): {efficiency.get('peak_vram_gb')}")
    print("=" * 78)
    for bench_dir, metrics in agg.items():
        print(f"\n[{bench_dir}]")
        for m, v in metrics.items():
            print(f"    {m:<20} {v['mean']:>8.3f}  ± {v['std']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", help="one checkpoint per seed (same arch)")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--no-flores", action="store_true")
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap #sentences per direction (smoke/debug)")
    ap.add_argument("--significance", nargs=2, metavar=("ARCH4_CKPT", "ARCH5_CKPT"),
                    help="paired bootstrap Arch5 vs Arch4 on BLEU")
    args = ap.parse_args()
    seed_everything(C.SEED)
    device = get_device()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=C.TOKENIZER_PREFIX + ".model")

    benchmarks = {"wmt": load_wmt(C.DATA["wmt_testset"])}
    if not args.no_flores:
        try:
            benchmarks["flores+"] = load_flores()
        except Exception as e:  # noqa: BLE001
            print(f"[flores] skipped ({e})")

    if args.significance:
        r4 = evaluate_checkpoint(args.significance[0], sp, device, args.beam,
                                 benchmarks, args.max_eval, keep_hyps=True)
        r5 = evaluate_checkpoint(args.significance[1], sp, device, args.beam,
                                 benchmarks, args.max_eval, keep_hyps=True)
        print("\n# Paired bootstrap significance (Arch5 vs Arch4), BLEU")
        for key in r4["scores"]:
            out_lang = "en" if key.endswith("zh-en") else "zh"
            refs_bench = key.split(":")[0]
            direction = key.split(":")[1]
            refs = benchmarks[refs_bench][direction][1]
            refs = refs[:args.max_eval] if args.max_eval else refs
            sig = paired_bootstrap_bleu(r4["scores"][key]["_hyps"],
                                        r5["scores"][key]["_hyps"], refs, out_lang)
            print(f"  {key}: {sig}")
        return

    if not args.ckpts:
        ap.error("provide --ckpts or --significance")

    per_seed = [evaluate_checkpoint(c, sp, device, args.beam, benchmarks, args.max_eval)
                for c in args.ckpts]
    agg = aggregate(per_seed)
    print_table(per_seed[0]["arch"], agg, per_seed[0]["efficiency"])

    # write a machine-readable summary
    import json
    out = os.path.join(C.RESULTS_DIR, f"eval_{per_seed[0]['arch']}.json")
    clean = {bd: {m: v for m, v in mv.items()} for bd, mv in agg.items()}
    with open(out, "w") as f:
        json.dump({"arch": per_seed[0]["arch"], "aggregate": clean,
                   "efficiency": per_seed[0]["efficiency"]}, f, indent=2)
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
