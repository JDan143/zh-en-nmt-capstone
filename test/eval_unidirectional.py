from __future__ import annotations
import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config as C  # noqa: E402
from utils import get_device  # noqa: E402
from evaluate import (evaluate_checkpoint, load_wmt, load_flores)  # noqa: E402
from train import load_tokenizer  # noqa: E402  (defined in train.py, reused)
import unidirectional as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Evaluate one unidirectional model.")
    ap.add_argument("--ckpt", required=True, help="the model's *_best.pt")
    ap.add_argument("--direction", required=True, choices=U.DIRECTIONS,
                    help="enzh or zhen — the model's own (trained) direction")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap test sentences (debug); default = full test set")
    args = ap.parse_args()
    U.check_direction(args.direction)
    keep = U.DIRECTION_TO_PAIR[args.direction]           # "en-zh" / "zh-en"

    device = get_device()
    sp = load_tokenizer()

    # Same benchmarks as the study evaluator.
    benchmarks = {"wmt": load_wmt(C.DATA["wmt_testset"])}
    try:
        benchmarks["flores+"] = load_flores()
    except Exception as e:  # noqa: BLE001
        print(f"[eval] FLORES+ unavailable ({e}); WMT only")

    res = evaluate_checkpoint(args.ckpt, sp, device, args.beam, benchmarks,
                              args.max_eval)

    kept = {k: v for k, v in res["scores"].items() if k.endswith(f":{keep}")}
    dropped = [k for k in res["scores"] if not k.endswith(f":{keep}")]
    out = {"arch": res["arch"], "direction": args.direction, "seed": res["seed"],
           "params_M": res["params_M"], "beam": args.beam,
           "scores": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                      for k, v in kept.items()}}

    print(f"\n=== {res['arch']} [{args.direction}] seed {res['seed']} "
          f"({res['params_M']}M, beam {args.beam}) ===")
    print(f"(reporting only :{keep} rows; dropped reverse-direction: {dropped})")
    for name, sc in out["scores"].items():
        line = "  ".join(f"{k}={v}" for k, v in sc.items()
                         if k in ("BLEU", "chrF++", "TER", "spBLEU", "perplexity"))
        print(f"  {name}: {line}")

    path = os.path.join(os.path.dirname(args.ckpt),
                        f"eval_{U.stem(res['arch'], args.direction, res['seed'])}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] wrote {path}")
    print(f"Results, compare {res['arch']} "
          f"seed-{res['seed']} {keep} BLEU for ablation.")


if __name__ == "__main__":
    main()
