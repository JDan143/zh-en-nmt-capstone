from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sentencepiece as spm  # noqa: E402
import config as C  # noqa: E402
from utils import get_device, count_params, human_millions  # noqa: E402
from models.common import greedy_decode, beam_search  # noqa: E402
from evaluate import load_wmt, load_flores, score, strip_decode  # noqa: E402

import sep_config as S  # noqa: E402


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    from config import ArchConfig
    cfg = ArchConfig(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    arch = ck["arch"]  # arch4_sep / arch5_sep
    mod = __import__(arch)
    model = mod.build(cfg, ck["src_vocab"], ck["tgt_vocab"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, ck


def encode_source_sep(sp_src, text, max_len):
    # NO language tag (separate tokenizers): just source subwords + EOS.
    ids = sp_src.encode(text, out_type=int)[: max_len - 2]
    return ids + [C.EOS_ID]


@torch.no_grad()
def translate_corpus_sep(model, cfg, sp_src, sp_tgt, src_texts, device, beam, batch_size=32,
                         max_eval=None):
    if max_eval:
        src_texts = src_texts[:max_eval]
    max_len = cfg.max_len
    hyps, t_total = [], 0.0
    for s in range(0, len(src_texts), batch_size):
        chunk = src_texts[s:s + batch_size]
        seqs = [encode_source_sep(sp_src, t, max_len) for t in chunk]
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
            hyps.append(strip_decode(sp_tgt, row))   # decode with TARGET tokenizer
    return hyps


@torch.no_grad()
def perplexity_sep(model, cfg, sp_src, sp_tgt, src_texts, ref_texts, device, max_eval=None):
    import torch.nn.functional as F
    if max_eval:
        src_texts, ref_texts = src_texts[:max_eval], ref_texts[:max_eval]
    max_len = cfg.max_len
    tot_ce, tot_tok = 0.0, 0
    for s in range(0, len(src_texts), 32):
        srcs = src_texts[s:s + 32]; refs = ref_texts[s:s + 32]
        seqs = [encode_source_sep(sp_src, t, max_len) for t in srcs]
        tgts = [[C.BOS_ID] + sp_tgt.encode(r, out_type=int)[: max_len - 2] + [C.EOS_ID]
                for r in refs]
        smax, tmax = max(len(x) for x in seqs), max(len(x) for x in tgts)
        src = torch.full((len(seqs), smax), C.PAD_ID, dtype=torch.long, device=device)
        tgt = torch.full((len(tgts), tmax), C.PAD_ID, dtype=torch.long, device=device)
        src_len = torch.zeros(len(seqs), dtype=torch.long, device=device)
        for i, x in enumerate(seqs):
            src[i, :len(x)] = torch.tensor(x, device=device); src_len[i] = len(x)
        for i, y in enumerate(tgts):
            tgt[i, :len(y)] = torch.tensor(y, device=device)
        logits = model(src, src_len, tgt[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                             ignore_index=C.PAD_ID, reduction="sum")
        tot_ce += ce.item(); tot_tok += (tgt[:, 1:] != C.PAD_ID).sum().item()
    import math
    return math.exp(min(tot_ce / max(tot_tok, 1), 20))


def main():
    ap = argparse.ArgumentParser(description="Evaluate one separate-tokenizer model.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--direction", required=True, choices=S.DIRECTIONS)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-eval", type=int, default=None)
    args = ap.parse_args()
    S.check_direction(args.direction)
    keep = "en-zh" if args.direction == "enzh" else "zh-en"
    out_lang = "zh" if args.direction == "enzh" else "en"

    device = get_device()
    model, cfg, ck = load_model(args.ckpt, device)
    src_lang, tgt_lang = ck["src_lang"], ck["tgt_lang"]
    sp_src = spm.SentencePieceProcessor(model_file=S.tokenizer_prefix(src_lang) + ".model")
    sp_tgt = spm.SentencePieceProcessor(model_file=S.tokenizer_prefix(tgt_lang) + ".model")
    params_m = human_millions(count_params(model))

    benchmarks = {"wmt": load_wmt(C.DATA["wmt_testset"])}
    try:
        benchmarks["flores+"] = load_flores()
    except Exception as e:  # noqa: BLE001
        print(f"[eval] FLORES+ unavailable ({e}); WMT only")

    out = {"arch": ck["arch"], "direction": args.direction, "seed": ck["seed"],
           "params_M": params_m, "src_vocab": ck["src_vocab"], "tgt_vocab": ck["tgt_vocab"],
           "beam": args.beam, "scores": {}}
    print(f"\n=== {ck['arch']} [{args.direction}] seed {ck['seed']} "
          f"({params_m}M, src {ck['src_vocab']//1000}k / tgt {ck['tgt_vocab']//1000}k, "
          f"beam {args.beam}) ===")
    for bench_name, bench in benchmarks.items():
        if keep not in bench:
            continue
        src_texts, ref_texts = bench[keep]
        hyps = translate_corpus_sep(model, cfg, sp_src, sp_tgt, src_texts, device,
                                    args.beam, max_eval=args.max_eval)
        refs = ref_texts[: args.max_eval] if args.max_eval else ref_texts
        sc = score(hyps, refs, out_lang, with_spbleu=(bench_name == "flores+"))
        sc["perplexity"] = round(perplexity_sep(model, cfg, sp_src, sp_tgt, src_texts,
                                                ref_texts, device, args.max_eval), 3)
        sc = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items()}
        out["scores"][f"{bench_name}:{keep}"] = sc
        line = "  ".join(f"{k}={v}" for k, v in sc.items()
                         if k in ("BLEU", "chrF++", "TER", "spBLEU", "perplexity"))
        print(f"  {bench_name}:{keep}: {line}")

    path = os.path.join(os.path.dirname(args.ckpt),
                        f"eval_{S.stem(ck['arch'], args.direction, ck['seed'])}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] wrote {path}")
    print(f"[compare] put this beside the JOINT-tokenizer uni result "
          f"(eval_{ck['arch'].replace('_sep','_transformer' if 'arch4' in ck['arch'] else '_improved_transformer')}...) "
          f"and the bidirectional {keep} BLEU to answer the experiment.")


if __name__ == "__main__":
    main()
