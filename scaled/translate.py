"""
translate.py — interactive single-sentence demo (beam search).

    python translate.py --ckpt results/arch4_transformer_seed191_best.pt --direction zh-en
    python translate.py --ckpt <ckpt> --direction en-zh --text "Where is the hotel?"

If --text is omitted, reads lines from stdin until EOF. Direction selects which
target-language tag is prepended to the source ('zh-en' -> <en>, 'en-zh' -> <zh>).
"""
from __future__ import annotations
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from utils import seed_everything, get_device  # noqa: E402
from evaluate import load_model, encode_source, strip_decode  # noqa: E402
from models.common import beam_search  # noqa: E402


@torch.no_grad()
def translate_one(model, cfg, sp, text, target_lang, device, beam):
    max_len = cfg.max_len if cfg.family == "transformer" else C.MAX_TOKENS_PER_SIDE
    ids = encode_source(sp, text, target_lang, max_len)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_len = torch.tensor([len(ids)], dtype=torch.long, device=device)
    out = beam_search(model, src, src_len, max_len=max_len,
                      beam=beam, length_penalty=cfg.length_penalty)
    return strip_decode(sp, out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--direction", choices=["zh-en", "en-zh"], default="zh-en")
    ap.add_argument("--text", default=None)
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()

    seed_everything(C.SEED)
    device = get_device()
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=C.TOKENIZER_PREFIX + ".model")
    model, cfg, arch = load_model(args.ckpt, sp, device)
    target_lang = "en" if args.direction == "zh-en" else "zh"
    print(f"[translate] {arch}  {args.direction}  beam={args.beam}")

    if args.text is not None:
        print(translate_one(model, cfg, sp, args.text, target_lang, device, args.beam))
        return
    print("Enter text (Ctrl-D to quit):")
    for line in sys.stdin:
        line = line.strip()
        if line:
            print("  ->", translate_one(model, cfg, sp, line, target_lang, device, args.beam))


if __name__ == "__main__":
    main()
