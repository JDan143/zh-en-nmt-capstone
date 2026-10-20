"""
hospitality fine-tuning for scaled arch 5.

run from scaled/:
    python finetune.py register-rule ...       writes the selection rule, before any training
    python finetune.py train --method plain|mixed|ewc [--ewc-lambda N] [--seed S] [--no-convlab]
    python finetune.py score-dev               BLEU/chrF++ on dev for every finished run
    python finetune.py select                  applies the rule, writes selection.json
    python finetune.py test                    ft_test + WMT + FLORES+, winner and base only

method, lr, ewc lambda, mix ratio, interpolation alpha and which epoch to keep are all
picked on dev. ft_test, WMT and FLORES+ get scored once, by `test`, on the selected
model and the base.

paths come from env vars (the notebook sets them):
    SPEECHBRIDGE_ROOT   speechbridge_scaled/
    SPEECHBRIDGE_DATA   speechbridge_scaled/data   tokenizer, general train.tsv.gz + dev.tsv.gz
    SPEECHBRIDGE_FT     speechbridge_scaled/fine_tune_arch5
base checkpoint in speechbridge_scaled/checkpoints/ is only read, fine-tune saves under its own names.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ARCH = "arch5_improved_transformer"
FT_DIRNAME = "fine_tune_arch5"
BASE_SEED = 191          # the pretraining run we fine-tune from
SELECTION_SEED = 191     # only seed 191 runs compete in selection
REPLICATE_SEEDS = (1604, 4090)
EXPECTED_CONVLAB = 7412  # ConvLab-2 rows in train_meta.tsv: 3,821 CrossWOZ + 3,591 MultiWOZ


def _set_dir_defaults():
    # config.py reads these on import, so they get set first. checkpoints and results
    # go under fine_tune_arch5, tokenizer and general corpus stay in scaled data/
    root = os.environ.get("SPEECHBRIDGE_ROOT")
    if not root:
        return
    ft = os.environ.setdefault("SPEECHBRIDGE_FT", os.path.join(root, FT_DIRNAME))
    os.environ.setdefault("SPEECHBRIDGE_CKPT", os.path.join(ft, "checkpoints"))
    os.environ.setdefault("SPEECHBRIDGE_RESULTS", os.path.join(ft, "results"))


_set_dir_defaults()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config as C  # noqa: E402
from config import ArchConfig  # noqa: E402
from utils import (seed_everything, get_device, count_params, human_millions,  # noqa: E402
                   MetricsLogger, Timer, peak_vram_gb, reset_peak_vram)
from data.dataset import TranslationDataset, collate, MaxTokensBatchSampler  # noqa: E402
from models.common import build_criterion  # noqa: E402
from train import (run_epoch, atomic_save, acquire_lock, touch_lock,  # noqa: E402
                   release_lock, load_tokenizer, capture_rng, restore_rng)


# ---------------------------------------------------------------- paths ----

def ft_dirs() -> dict:
    root = os.environ.get("SPEECHBRIDGE_ROOT")
    if not root:
        raise SystemExit("SPEECHBRIDGE_ROOT isn't set, it should point at speechbridge_scaled/ "
                         "(the settings cell sets it)")
    ft = os.environ.get("SPEECHBRIDGE_FT", os.path.join(root, FT_DIRNAME))
    d = {"root": root, "ft": ft,
         "data": os.path.join(ft, "data"),
         "ckpt": os.path.join(ft, "checkpoints"),
         "results": os.path.join(ft, "results"),
         "general_train": os.path.join(C.DATA_DIR, "train.tsv.gz"),
         "general_dev": os.path.join(C.DATA_DIR, "dev.tsv.gz"),
         "base_ckpt": os.environ.get(
             "SPEECHBRIDGE_BASE_CKPT",
             os.path.join(root, "checkpoints", f"{ARCH}_seed{BASE_SEED}_best.pt"))}
    for k in ("data", "ckpt", "results"):
        os.makedirs(d[k], exist_ok=True)
    # the relay lock in train.py lives in C.CKPT_DIR, so that has to be our folder
    if os.path.abspath(C.CKPT_DIR) != os.path.abspath(d["ckpt"]):
        raise SystemExit(f"config.py picked up CKPT_DIR={C.CKPT_DIR}, expected {d['ckpt']}.\n"
                         f"config got imported before SPEECHBRIDGE_FT was set. restarting the "
                         f"runtime and running the settings cell first fixes it")
    return d


def lr_str(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def run_tag(method: str, lr: float, patience: int, mix: int = 0, ewc_lambda: float = 0,
            no_convlab: bool = False, alpha: float | None = None) -> str:
    """e.g. lr1e-5_p5_ewc50000, lr1e-5_p5_mix4_interp0.5, lr1e-5_p5_plain_noconvlab"""
    tag = f"lr{lr_str(lr)}_p{patience}"
    if method == "plain":
        tag += "_plain"
    elif method == "mixed":
        tag += f"_mix{mix}"
    elif method == "ewc":
        tag += f"_ewc{int(ewc_lambda)}"
    else:
        raise ValueError(f"unknown method {method}")
    if no_convlab:
        tag += "_noconvlab"
    if alpha is not None:
        tag += f"_interp{alpha:g}"
    return tag


def stem(seed, tag) -> str:
    return f"{ARCH}_ft_seed{seed}_{tag}"


def run_files(d, seed, tag) -> dict:
    s = stem(seed, tag)
    return {"best": os.path.join(d["ckpt"], s + "_best.pt"),
            "last": os.path.join(d["ckpt"], s + "_last.pt"),
            "metrics": os.path.join(d["results"], f"metrics_{s}.csv"),
            "done": os.path.join(d["results"], f"done_{s}.json")}


def safe_save(obj, path, d):
    # every write goes through here so nothing can land on the base checkpoint
    if os.path.abspath(path) == os.path.abspath(d["base_ckpt"]):
        raise SystemExit("refusing to write over the base checkpoint")
    if not os.path.abspath(path).startswith(os.path.abspath(d["ft"])):
        raise SystemExit(f"refusing to write outside {d['ft']}: {path}")
    atomic_save(obj, path)


def file_fingerprint(path) -> str:
    st = os.stat(path)
    return f"{st.st_size}-{int(st.st_mtime)}"


def content_sha16(path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if is_gzip(path) else open
    with opener(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- data ----

def is_gzip(path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def read_pairs(path) -> list:
    """zh<TAB>en per line, gzipped or not. stops on a bad line instead of skipping it,
    because the meta files line up with these by position."""
    opener = gzip.open if is_gzip(path) else open
    pairs = []
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise SystemExit(f"{os.path.basename(path)} line {i} isn't zh<TAB>en: {line[:80]!r}")
            pairs.append((parts[0], parts[1]))
    return pairs


def read_meta(path, n_expected: int) -> list:
    if not os.path.exists(path):
        raise SystemExit(f"can't find {path}, it goes in fine_tune_arch5/data/ with the "
                         f"ft_*.tsv.gz files")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != n_expected:
        raise SystemExit(f"{os.path.basename(path)} has {len(rows)} rows but the data file "
                         f"has {n_expected}. they're matched by position, so stopping")
    if [r.get("line") for r in rows] != [str(i) for i in range(1, n_expected + 1)]:
        raise SystemExit(f"{os.path.basename(path)}: the 'line' column isn't 1..{n_expected} in order")
    return rows


def natural_source_split(pairs, meta) -> dict:
    """EN->ZH only on English-original lines, ZH->EN only on Chinese-original lines.
    returns {direction: (sources, refs, row_indices)}"""
    out = {"en-zh": ([], [], []), "zh-en": ([], [], [])}
    for i, ((zh, en), m) in enumerate(zip(pairs, meta)):
        lang = m["source_lang"].strip()
        if lang == "en":
            out["en-zh"][0].append(en); out["en-zh"][1].append(zh); out["en-zh"][2].append(i)
        elif lang == "zh":
            out["zh-en"][0].append(zh); out["zh-en"][1].append(en); out["zh-en"][2].append(i)
        else:
            raise SystemExit(f"meta row {i + 1} has source_lang={lang!r}, expected en or zh")
    return out


def drop_convlab(pairs, d) -> list:
    meta = read_meta(os.path.join(d["data"], "train_meta.tsv"), len(pairs))
    is_cl = ["convlab" in m["source"].lower() or "crosswoz" in m["source"].lower() for m in meta]
    n_cl = sum(is_cl)
    counts = {}
    for m in meta:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    print(f"[ft] train_meta sources: {counts}")
    if n_cl != EXPECTED_CONVLAB:
        raise SystemExit(f"train_meta.tsv has {n_cl} ConvLab-2 rows, expected "
                         f"{EXPECTED_CONVLAB}. stopping instead of guessing which rows to drop, "
                         f"source counts are printed above")
    kept = [p for p, c in zip(pairs, is_cl) if not c]
    print(f"[ft] dropped {n_cl:,} ConvLab-2 rows, {len(kept):,} pairs left")
    return kept


class PairsDataset(TranslationDataset):
    """TranslationDataset built from pairs already in memory. it skips the on-disk
    lengths cache, which is keyed on a file path these sets don't have."""

    def __init__(self, pairs, sp, max_subword_len, directions=("zh-en", "en-zh")):
        self.sp = sp
        self.src_path = None
        self.max_len = max_subword_len
        self.tag = {"zh": sp.piece_to_id("<zh>"), "en": sp.piece_to_id("<en>")}
        self.items = []
        for zh, en in pairs:
            if "zh-en" in directions:
                self.items.append((self.tag["en"], zh, en, "zh-en"))
            if "en-zh" in directions:
                self.items.append((self.tag["zh"], en, zh, "en-zh"))
        self._lengths = None


def reservoir_sample(path, k, seed) -> list:
    """k random pairs from the 3.34M general file in one pass, without loading it all.
    same seed gives the same sample."""
    rng = random.Random(seed)
    keep, n = [], 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            n += 1
            if len(keep) < k:
                keep.append((parts[0], parts[1]))
            else:
                j = rng.randint(0, n - 1)
                if j < k:
                    keep[j] = (parts[0], parts[1])
    if len(keep) < k:
        print(f"[ft] general file only had {n:,} pairs, wanted {k:,}")
    return keep


def eval_loader(pairs, sp, cfg):
    return DataLoader(PairsDataset(pairs, sp, cfg.max_len), batch_size=64,
                      shuffle=False, collate_fn=collate)


# ---------------------------------------------------------------- model ----

def build_model(cfg, vocab_size):
    from models.arch5_improved_transformer import build
    return build(cfg, vocab_size)


def load_state(path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("arch") != ARCH:
        raise SystemExit(f"{os.path.basename(path)} is {ck.get('arch')}, not {ARCH}")
    return ck


def model_from_state(ck, state, device):
    cfg = ArchConfig(**ck["cfg"])
    model = build_model(cfg, ck["vocab_size"])
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def interpolate(base_state, ft_state, alpha):
    """weights = (1 - alpha) * base + alpha * fine-tuned  (Wortsman et al. 2022).
    alpha 0 is the base model, alpha 1 is the fine-tuned one."""
    if base_state.keys() != ft_state.keys():
        raise SystemExit("base and fine-tuned state dicts have different keys")
    out = {}
    for k, v in ft_state.items():
        b = base_state[k]
        out[k] = (1 - alpha) * b.float() + alpha * v.float() if torch.is_floating_point(v) else v
    return out


# ------------------------------------------------------------------ ewc ----

def estimate_fisher(model, pairs, sp, cfg, criterion, device, n_batches, seed):
    """diagonal fisher at the base weights: mean squared gradient of the training loss
    over some general batches. a big value means the weight matters for general
    translation, so ewc holds it closer. fp32 so small grads don't underflow.
    batches are max_tokens batches like in training. fixed batches of 128 padded to the
    longest sentence ran out of memory in the smoke test (fp32 logits alone can pass 4 GB)."""
    model.eval()  # no dropout, we want the base model's own gradients
    ds = PairsDataset(pairs, sp, cfg.max_len)
    loader = DataLoader(ds, batch_sampler=MaxTokensBatchSampler(ds, cfg.max_tokens, seed=seed),
                        collate_fn=collate)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    done = 0
    for src, src_len, tgt, _r, _d in loader:
        if done >= n_batches:
            break
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        model.zero_grad()
        logits = model(src, src_len, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        done += 1
    for n in fisher:
        fisher[n] /= max(done, 1)
    model.zero_grad()
    return fisher, done


class EWCLoss(nn.Module):
    """normal loss + lambda/2 * sum F * (w - w_base)^2 while training. done as a loss
    wrapper so train.run_epoch stays as it is. on dev (no grad) it's the plain loss."""

    def __init__(self, base_loss, model, fisher, anchor, lam):
        super().__init__()
        self.base_loss = base_loss
        self.lam = lam
        self.terms = [(p, fisher[n], anchor[n]) for n, p in model.named_parameters()
                      if n in fisher]
        self.model = model
        self.pen_sum = 0.0
        self.tok_sum = 0

    def forward(self, logits, target):
        loss = self.base_loss(logits, target)
        if self.model.training and torch.is_grad_enabled():
            pen = sum((f * (p - a) ** 2).sum() for p, f, a in self.terms)
            pen = 0.5 * self.lam * pen
            n_tok = int((target != C.PAD_ID).sum())
            self.pen_sum += float(pen.detach()) * n_tok
            self.tok_sum += n_tok
            loss = loss + pen
        return loss

    def epoch_penalty(self):
        v = self.pen_sum / max(self.tok_sum, 1)
        self.pen_sum, self.tok_sum = 0.0, 0
        return v


# -------------------------------------------------------------- scoring ----

def score_texts(hyps, refs, out_lang) -> dict:
    """the one scoring function for every set (dev, ft_test, WMT, FLORES+).
    same recipe as evaluate.py so base numbers line up with the earlier eval:
    BLEU tok=zh for Chinese output and 13a for English; chrF++ (word order 2), and for
    Chinese output chrF++ runs on text split into single characters."""
    from sacrebleu.metrics import BLEU, CHRF
    bleu = BLEU(tokenize="zh" if out_lang == "zh" else "13a")
    chrf = CHRF(word_order=2)
    if out_lang == "zh":
        cseg = lambda s: " ".join(s.replace(" ", ""))  # noqa: E731
        h_c, r_c = [cseg(x) for x in hyps], [cseg(x) for x in refs]
    else:
        h_c, r_c = hyps, refs
    b = bleu.corpus_score(hyps, [refs])
    c = chrf.corpus_score(h_c, [r_c])
    return {"BLEU": round(b.score, 2), "chrF++": round(c.score, 2),
            "bleu_sig": "BLEU|" + bleu.get_signature().format(),
            "chrf_sig": "chrF2++|" + chrf.get_signature().format()
                        + ("|pre:charsplit" if out_lang == "zh" else "")}


def _metric_stats(hyps, refs, out_lang):
    from sacrebleu.metrics import BLEU, CHRF
    bleu = BLEU(tokenize="zh" if out_lang == "zh" else "13a")
    chrf = CHRF(word_order=2)
    if out_lang == "zh":
        cseg = lambda s: " ".join(s.replace(" ", ""))  # noqa: E731
        h_c, r_c = [cseg(x) for x in hyps], [cseg(x) for x in refs]
    else:
        h_c, r_c = hyps, refs
    return {"BLEU": (bleu, np.array(bleu._extract_corpus_statistics(hyps, [refs]))),
            "chrF++": (chrf, np.array(chrf._extract_corpus_statistics(h_c, [r_c])))}


def paired_bootstrap(hyps_base, hyps_new, refs, out_lang, n_samples=1000, seed=12345):
    """paired bootstrap resampling (Koehn 2004). resamples sentences with replacement,
    scores both systems on the same sample. returns 95% intervals for each system and
    for the difference, plus p = share of samples where the new system isn't better.
    sums per-sentence sacrebleu statistics, which gives the exact corpus score for
    each sample without re-tokenizing. "delta" is the observed difference on the full
    set; that's the number to report, with delta_ci95 around it. p is one-sided."""
    a = _metric_stats(hyps_base, refs, out_lang)
    b = _metric_stats(hyps_new, refs, out_lang)
    n = len(refs)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_samples, n))
    out = {}
    for name in ("BLEU", "chrF++"):
        metric, sa = a[name]
        _, sb = b[name]
        va = np.array([metric._compute_score_from_stats(list(sa[r].sum(0))).score for r in idx])
        vb = np.array([metric._compute_score_from_stats(list(sb[r].sum(0))).score for r in idx])
        diff = vb - va
        observed = (metric._compute_score_from_stats(list(sb.sum(0))).score
                    - metric._compute_score_from_stats(list(sa.sum(0))).score)
        out[name] = {
            "delta": round(float(observed), 2),
            "base_ci95": [round(float(np.percentile(va, 2.5)), 2), round(float(np.percentile(va, 97.5)), 2)],
            "new_ci95": [round(float(np.percentile(vb, 2.5)), 2), round(float(np.percentile(vb, 97.5)), 2)],
            "delta_mean": round(float(diff.mean()), 2),
            "delta_ci95": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)],
            "p_not_better": round(float((diff <= 0).mean()), 4),
            "n_samples": n_samples}
    return out


def translate(model, cfg, sp, sources, out_lang, device, beam, limit=None):
    # evaluate.translate_corpus, so decoding is identical to the earlier base eval
    from evaluate import translate_corpus
    if limit:
        sources = sources[:limit]
    hyps, _ = translate_corpus(model, cfg, sp, sources, out_lang, device, beam=beam)
    return hyps


# -------------------------------------------------------- rule on file ----

def rule_path(d):
    return os.path.join(d["results"], "selection_rule.json")


def cmd_register_rule(args):
    d = ft_dirs()
    rule = {"pick": "highest mean hospitality dev chrF++ (mean of EN->ZH and ZH->EN, "
                    "natural-source lines only)",
            "subject_to": "mean general dev BLEU drop vs base <= max_gen_bleu_drop",
            "ties": "within tie_chrf of the best: take the smaller general BLEU drop",
            "max_gen_bleu_drop": args.max_gen_bleu_drop, "tie_chrf": args.tie,
            "budget": {"lr": args.lr, "patience": args.patience, "gen_limit": args.gen_limit},
            "candidates": {"mix": args.mix, "ewc_lambdas": args.ewc_lambdas,
                           "alphas": args.alphas},
            "beam": args.beam}
    p = rule_path(d)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            on_file = json.load(f)
        same = {k: on_file.get(k) for k in rule} == rule
        if same:
            print(f"[rule] matches the one on file from {on_file['registered_at']} "
                  f"(by {on_file['registered_by']})")
            return
        raise SystemExit(
            f"[rule] the selection rule on file ({p}) doesn't match these settings.\n"
            f"  on file: drop<={on_file.get('max_gen_bleu_drop')} tie={on_file.get('tie_chrf')} "
            f"budget={on_file.get('budget')} candidates={on_file.get('candidates')}\n"
            f"  now:     drop<={rule['max_gen_bleu_drop']} tie={rule['tie_chrf']} "
            f"budget={rule['budget']} candidates={rule['candidates']}\n"
            f"  the rule was fixed before training. usually this is a typo in the settings "
            f"cell. changing the rule on purpose means deleting that file and saying so in "
            f"the write-up")
    rule["registered_at"] = datetime.now().isoformat(timespec="seconds")
    rule["registered_by"] = args.owner
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rule, f, indent=2)
    print(f"[rule] saved to {p}, fixed from here on")


def load_rule(d):
    p = rule_path(d)
    if not os.path.exists(p):
        raise SystemExit("no selection_rule.json yet, the settings cell writes it "
                         "(finetune.py register-rule)")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- train ----

def cmd_train(args):
    d = ft_dirs()
    load_rule(d)  # stops here if there is no rule yet
    tag = run_tag(args.method, args.lr, args.patience, args.mix, args.ewc_lambda,
                  args.no_convlab)
    if args.expect_tag and args.expect_tag != tag:
        raise SystemExit(f"settings give tag {tag} but the notebook expected {args.expect_tag}")
    files = run_files(d, args.seed, tag)
    settings = {"method": args.method, "lr": args.lr, "patience": args.patience,
                "gen_limit": args.gen_limit, "mix": args.mix if args.method == "mixed" else 0,
                "ewc_lambda": args.ewc_lambda if args.method == "ewc" else 0,
                "fisher_batches": args.fisher_batches if args.method == "ewc" else 0,
                "no_convlab": args.no_convlab, "seed": args.seed}
    print(f"[ft] run {stem(args.seed, tag)}")

    if os.path.exists(files["done"]) and not args.fresh:
        with open(files["done"], encoding="utf-8") as f:
            info = json.load(f)
        print(f"[ft] already finished ({info['stopped_by']}, best epoch "
              f"{info['best_epoch']}), skipping. --fresh redoes it")
        return

    seed_everything(args.seed)
    device = get_device()
    base_ck = load_state(d["base_ckpt"])
    cfg = dataclasses.replace(ArchConfig(**base_ck["cfg"]), scheduler="none", lr=args.lr,
                              patience=args.patience, epochs=4 if C.SMOKE else C.EPOCH_CAP)
    sp = load_tokenizer()
    vocab_size = sp.get_piece_size()
    if vocab_size != base_ck["vocab_size"]:
        raise SystemExit(f"tokenizer has {vocab_size} pieces but the base model was trained "
                         f"with {base_ck['vocab_size']}. Wrong bpe_zh_en.model?")

    # data
    hosp = read_pairs(os.path.join(d["data"], "ft_train.tsv.gz"))
    if args.no_convlab:
        hosp = drop_convlab(hosp, d)
    train_pairs = list(hosp)
    if args.method == "mixed":
        k = args.mix * len(hosp)
        print(f"[ft] sampling {k:,} general pairs ({args.mix} per hospitality pair) "
              f"from {d['general_train']}, takes a minute or two...", flush=True)
        train_pairs += reservoir_sample(d["general_train"], k, args.seed)
    train_ds = PairsDataset(train_pairs, sp, cfg.max_len)
    sampler = MaxTokensBatchSampler(train_ds, cfg.max_tokens, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate)
    hosp_dev_loader = eval_loader(read_pairs(os.path.join(d["data"], "ft_dev.tsv.gz")), sp, cfg)
    gen_dev_loader = eval_loader(read_pairs(d["general_dev"]), sp, cfg)
    print(f"[ft] train {len(hosp):,} hospitality pairs + {len(train_pairs) - len(hosp):,} general "
          f"= {len(train_ds):,} examples (both directions)")

    model = build_model(cfg, vocab_size).to(device)
    print(f"[ft] {ARCH}, {human_millions(count_params(model))}M params, method {args.method}, "
          f"lr {args.lr:g} constant, patience {args.patience}, device {device}")
    criterion = build_criterion(vocab_size, cfg.label_smoothing)
    ce = nn.CrossEntropyLoss(ignore_index=C.PAD_ID)
    # same Adam settings as pretraining, constant lr, no warmup, no weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=tuple(cfg.betas),
                                 eps=cfg.eps)
    use_amp = device.type == "cuda" and cfg.amp_safe
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_batches = 30 if C.SMOKE else None

    lock = acquire_lock(f"{ARCH}_ft", f"{args.seed}_{tag}", args.owner, force=args.force)
    if args.fresh:
        # move the old run out of the way so a half-done redo never looks finished
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        for k in ("done", "metrics", "best"):
            if os.path.exists(files[k]):
                os.replace(files[k], f"{files[k]}.replaced-{ts}")
                print(f"[ft] --fresh: moved old {os.path.basename(files[k])} aside")
    try:
        _train_loop(args, d, files, settings, tag, base_ck, cfg, sp, vocab_size, model,
                    criterion, ce, optimizer, scaler, use_amp, train_loader, sampler,
                    hosp_dev_loader, gen_dev_loader, device, max_batches, lock)
    finally:
        release_lock(lock)


def _train_loop(args, d, files, settings, tag, base_ck, cfg, sp, vocab_size, model,
                criterion, ce, optimizer, scaler, use_amp, train_loader, sampler,
                hosp_dev_loader, gen_dev_loader, device, max_batches, lock):
    start_epoch, best, bad = 0, math.inf, 0
    best_epoch = None
    if os.path.exists(files["last"]) and not args.fresh:
        ck = torch.load(files["last"], map_location=device, weights_only=False)
        if ck.get("settings") != settings:
            raise SystemExit(f"[ft] {os.path.basename(files['last'])} was saved with different "
                             f"settings:\n  saved: {ck.get('settings')}\n  now:   {settings}\n"
                             f"  either the settings cell changed or this run needs --fresh to start over")
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        restore_rng(ck["rng"])
        start_epoch, best, bad = ck["epoch"] + 1, ck["best_hosp_ce"], ck["bad_epochs"]
        best_epoch = ck.get("best_epoch")
        base_hosp_ce, base_gen_ce = ck["base_hosp_ce"], ck["base_gen_ce"]
        print(f"[ft] picking up at epoch {start_epoch} (last session: {ck.get('owner', '?')}, "
              f"now: {args.owner}). best hospitality dev CE {best:.4f}, {bad}/{args.patience} bad")
    else:
        model.load_state_dict(base_ck["model_state"])
        print(f"[ft] starting from the base model, {os.path.basename(d['base_ckpt'])} "
              f"(pretraining epoch {base_ck.get('epoch')})")
        _, base_hosp_ce = run_epoch(model, hosp_dev_loader, criterion, ce, device, cfg,
                                    train=False, max_batches=max_batches, use_amp=use_amp)
        _, base_gen_ce = run_epoch(model, gen_dev_loader, criterion, ce, device, cfg,
                                   train=False, max_batches=max_batches, use_amp=use_amp)
    print(f"[ft] base model: hospitality dev CE {base_hosp_ce:.4f} "
          f"(ppl {math.exp(min(base_hosp_ce, 20)):.2f}), general dev CE {base_gen_ce:.4f} "
          f"(ppl {math.exp(min(base_gen_ce, 20)):.2f})")
    print(f"[ft] an epoch only counts as best if general dev CE <= "
          f"{base_gen_ce * (1 + args.gen_limit):.4f} (base + {args.gen_limit:.0%})")

    train_crit = criterion
    if args.method == "ewc":
        train_crit = _setup_ewc(args, d, base_ck, cfg, sp, criterion, model, device)

    logger = MetricsLogger(files["metrics"])
    stopped_by = None
    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        touch_lock(lock, args.owner)
        reset_peak_vram()
        with Timer() as t:
            tr_loss, _ = run_epoch(model, train_loader, train_crit, ce, device, cfg,
                                   optimizer, None, train=True, max_batches=max_batches,
                                   scaler=scaler, use_amp=use_amp)
        ewc_term = train_crit.epoch_penalty() if isinstance(train_crit, EWCLoss) else 0.0
        _, hosp_ce = run_epoch(model, hosp_dev_loader, criterion, ce, device, cfg,
                               train=False, max_batches=max_batches, use_amp=use_amp)
        _, gen_ce = run_epoch(model, gen_dev_loader, criterion, ce, device, cfg,
                              train=False, max_batches=max_batches, use_amp=use_amp)
        gen_change = gen_ce / base_gen_ce - 1.0
        gen_ok = gen_change <= args.gen_limit
        # over the general limit can't be a new best, and it counts toward patience,
        # so a run that keeps forgetting stops by itself
        improved = gen_ok and hosp_ce < best - 1e-4
        if improved:
            best, bad, best_epoch = hosp_ce, 0, epoch
        else:
            bad += 1

        logger.log({"tag": tag, "seed": args.seed, "epoch": epoch,
                    "train_loss": round(tr_loss, 4), "ewc_term": round(ewc_term, 4),
                    "hosp_dev_ce_base": round(base_hosp_ce, 4), "hosp_dev_ce": round(hosp_ce, 4),
                    "hosp_dev_ppl": round(math.exp(min(hosp_ce, 20)), 3),
                    "gen_dev_ce_base": round(base_gen_ce, 4), "gen_dev_ce": round(gen_ce, 4),
                    "gen_change_pct": round(100 * gen_change, 2), "gen_ok": gen_ok,
                    "saved": improved, "bad_epochs": bad, "lr": args.lr,
                    "epoch_time_s": round(t.elapsed, 1), "peak_vram_gb": peak_vram_gb(),
                    "owner": args.owner})
        note = "saved" if improved else f"{bad}/{args.patience}"
        if not gen_ok:
            note += f", general over the {args.gen_limit:.0%} limit"
        ewc_txt = f" ewc {ewc_term:.4f}" if args.method == "ewc" else ""
        print(f"[ft] epoch {epoch}: train {tr_loss:.4f}{ewc_txt} | hospitality dev CE "
              f"{hosp_ce:.4f} | general dev CE {gen_ce:.4f} ({100 * gen_change:+.2f}%) | "
              f"{t.elapsed:.0f}s | {note}", flush=True)

        meta = {"arch": ARCH, "cfg": cfg.to_dict(), "vocab_size": vocab_size,
                "seed": args.seed, "run_tag": tag, "settings": settings,
                "base_ckpt": os.path.basename(d["base_ckpt"]),
                "base_hosp_ce": base_hosp_ce, "base_gen_ce": base_gen_ce}
        if improved:
            safe_save({**meta, "model_state": model.state_dict(), "epoch": epoch,
                       "hosp_dev_ce": hosp_ce, "gen_dev_ce": gen_ce}, files["best"], d)
        finished = bad >= args.patience or epoch == cfg.epochs - 1
        if finished:
            stopped_by = "patience" if bad >= args.patience else "epoch cap"
        safe_save({**meta, "model_state": model.state_dict(),
                   "optimizer_state": optimizer.state_dict(),
                   "scaler_state": scaler.state_dict() if use_amp else None,
                   "rng": capture_rng(), "epoch": epoch, "best_hosp_ce": best,
                   "best_epoch": best_epoch, "bad_epochs": bad, "owner": args.owner,
                   "finished": finished}, files["last"], d)
        if finished:
            break

    if stopped_by is None:
        print("[ft] loop ended without finishing (smoke run?), not marking it done")
        return
    info = {"tag": tag, "seed": args.seed, "settings": settings, "stopped_by": stopped_by,
            "best_epoch": best_epoch, "best_hosp_dev_ce": None if math.isinf(best) else best,
            "base_hosp_ce": base_hosp_ce, "base_gen_ce": base_gen_ce,
            "saved_any": best_epoch is not None,
            "finished_at": datetime.now().isoformat(timespec="seconds"), "owner": args.owner}
    with open(files["done"], "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    if best_epoch is None:
        print(f"[ft] stopped ({stopped_by}) and nothing was saved: every epoch was more than "
              f"{args.gen_limit:.0%} worse than the base on general dev. out of selection, "
              f"but worth listing in the write-up as a run that failed the guard")
    else:
        print(f"[ft] done ({stopped_by}). best epoch {best_epoch}, hospitality dev CE "
              f"{best:.4f} vs base {base_hosp_ce:.4f}. checkpoint: {files['best']}")


def _setup_ewc(args, d, base_ck, cfg, sp, criterion, model, device):
    # anchor and fisher both come from the base weights, also on resume. anchoring to the
    # resumed weights would quietly turn this into a different method
    anchor = {k: v.to(device) for k, v in base_ck["model_state"].items()}
    fpath = os.path.join(d["ckpt"], f"fisher_{ARCH}_base_seed{args.seed}_tok{cfg.max_tokens}_b{args.fisher_batches}.pt")
    base_fp = file_fingerprint(d["base_ckpt"])
    fisher = None
    if os.path.exists(fpath):
        saved = torch.load(fpath, map_location="cpu", weights_only=False)
        if saved.get("base_fp") == base_fp:
            fisher = {k: v.to(device) for k, v in saved["fisher"].items()}
            print(f"[ewc] reusing fisher from {os.path.basename(fpath)}")
    if fisher is None:
        base_model = build_model(cfg, base_ck["vocab_size"]).to(device)
        base_model.load_state_dict(base_ck["model_state"])
        # ~128 pairs per batch is more than a max_tokens batch holds, so this is
        # enough for n batches. the loop stops once it has them.
        n_pairs = args.fisher_batches * 128
        print(f"[ewc] estimating fisher on {args.fisher_batches} general batches of about "
              f"{cfg.max_tokens} tokens (drawn from {n_pairs:,} pairs)...", flush=True)
        pairs = reservoir_sample(d["general_train"], n_pairs, args.seed)
        fisher, used = estimate_fisher(base_model, pairs, sp, cfg, criterion, device,
                                       args.fisher_batches, args.seed)
        if used < args.fisher_batches:
            print(f"[ewc] only got {used} batches out of {args.fisher_batches} (general sample too small)")
        del base_model
        safe_save({"fisher": {k: v.cpu() for k, v in fisher.items()}, "base_fp": base_fp,
                   "batches": used, "seed": args.seed}, fpath, d)
        print(f"[ewc] fisher done ({used} batches), cached at {os.path.basename(fpath)}")
    names = {n for n, _ in model.named_parameters()}
    anchor = {n: anchor[n] for n in names}
    return EWCLoss(criterion, model, fisher, anchor, args.ewc_lambda)


# ------------------------------------------------------------ score-dev ----

DEV_COLS = ["row_tag", "seed", "role", "method", "mix", "ewc_lambda", "alpha", "no_convlab",
            "lr", "patience", "gen_limit", "best_epoch",
            "hosp_enzh_bleu", "hosp_enzh_chrf", "hosp_zhen_bleu", "hosp_zhen_chrf",
            "hosp_bleu_mean", "hosp_chrf_mean",
            "gen_enzh_bleu", "gen_enzh_chrf", "gen_zhen_bleu", "gen_zhen_chrf",
            "gen_bleu_mean", "gen_bleu_drop", "n_hosp_enzh", "n_hosp_zhen", "n_gen",
            "beam", "ckpt", "ckpt_fp", "scored_at"]


def finished_runs(d) -> list:
    runs = []
    for p in sorted(glob.glob(os.path.join(d["results"], f"done_{ARCH}_ft_seed*.json"))):
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def dev_sets(d, limit=None):
    hosp = read_pairs(os.path.join(d["data"], "ft_dev.tsv.gz"))
    meta = read_meta(os.path.join(d["data"], "dev_meta.tsv"), len(hosp))
    nat = natural_source_split(hosp, meta)
    gen = read_pairs(d["general_dev"])
    sets = {"hosp_enzh": (nat["en-zh"][0], nat["en-zh"][1], "zh"),
            "hosp_zhen": (nat["zh-en"][0], nat["zh-en"][1], "en"),
            "gen_enzh": ([en for _, en in gen], [zh for zh, _ in gen], "zh"),
            "gen_zhen": ([zh for zh, _ in gen], [en for _, en in gen], "en")}
    if limit:
        sets = {k: (s[:limit], r[:limit], lang) for k, (s, r, lang) in sets.items()}
    return sets


def score_on_dev(model, cfg, sp, sets, device, beam):
    out, hyps_all, sigs = {}, {}, {}
    for name, (src, ref, lang) in sets.items():
        hyps = translate(model, cfg, sp, src, lang, device, beam)
        s = score_texts(hyps, ref, lang)
        out[f"{name}_bleu"], out[f"{name}_chrf"] = s["BLEU"], s["chrF++"]
        hyps_all[name] = hyps
        sigs[name] = {"bleu": s["bleu_sig"], "chrf": s["chrf_sig"]}
    out["n_hosp_enzh"], out["n_hosp_zhen"] = len(sets["hosp_enzh"][0]), len(sets["hosp_zhen"][0])
    out["n_gen"] = len(sets["gen_enzh"][0])
    out["hosp_bleu_mean"] = round((out["hosp_enzh_bleu"] + out["hosp_zhen_bleu"]) / 2, 2)
    out["hosp_chrf_mean"] = round((out["hosp_enzh_chrf"] + out["hosp_zhen_chrf"]) / 2, 2)
    out["gen_bleu_mean"] = round((out["gen_enzh_bleu"] + out["gen_zhen_bleu"]) / 2, 2)
    return out, hyps_all, sigs


def read_dev_scores(d) -> list:
    p = os.path.join(d["results"], "dev_scores.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_dev_scores(d, rows):
    p = os.path.join(d["results"], "dev_scores.csv")
    order = {"base": 0, "run": 1, "interp": 2, "ablation": 3, "ablation-interp": 4}
    rows = sorted(rows, key=lambda r: (order.get(r["role"], 9), r["row_tag"]))
    tmp = p + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DEV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in DEV_COLS})
    os.replace(tmp, p)


def cmd_score_dev(args):
    d = ft_dirs()
    rule = load_rule(d)
    beam = rule["beam"]
    alphas = rule["candidates"]["alphas"]
    device = get_device()
    sp = load_tokenizer()
    limit = args.limit or (24 if C.SMOKE else None)
    sets = dev_sets(d, limit)
    print(f"[dev] hospitality dev: EN->ZH {len(sets['hosp_enzh'][0])} English-original lines, "
          f"ZH->EN {len(sets['hosp_zhen'][0])} Chinese-original lines. general dev: "
          f"{len(sets['gen_enzh'][0])} lines each way. beam {beam}")
    hyp_dir = os.path.join(d["results"], "dev_hyps")
    os.makedirs(hyp_dir, exist_ok=True)
    rows = {(r["row_tag"], r["seed"]): r for r in read_dev_scores(d)}
    sig_path = os.path.join(d["results"], "dev_scores_signatures.json")

    base_ck = load_state(d["base_ckpt"])
    base_state = base_ck["model_state"]
    todo = [("base", str(BASE_SEED), "base", None, None, d["base_ckpt"])]
    for run in finished_runs(d):
        if not run["saved_any"]:
            print(f"[dev] skipping {run['tag']} seed {run['seed']}: it never saved a best epoch")
            continue
        seed = run["seed"]
        if seed != SELECTION_SEED:
            continue  # replicate seeds only get scored in the final test
        role = "ablation" if run["settings"]["no_convlab"] else "run"
        best = run_files(d, seed, run["tag"])["best"]
        todo.append((run["tag"], str(seed), role, run, None, best))
        st = run["settings"]
        for a in alphas:
            todo.append((run_tag(st["method"], st["lr"], st["patience"], st["mix"],
                                 st["ewc_lambda"], st["no_convlab"], a),
                         str(seed), "ablation-interp" if role == "ablation" else "interp",
                         run, a, best))

    sigs = {}
    for row_tag, seed, role, run, alpha, ckpt in todo:
        fp = file_fingerprint(ckpt) + ("" if alpha is None else f"-a{alpha}") + f"-b{beam}-l{limit}"
        old = rows.get((row_tag, seed))
        if old and old.get("ckpt_fp") == fp and not args.rescore:
            continue
        print(f"[dev] scoring {row_tag} (seed {seed})...", flush=True)
        if role == "base":
            state, ck = base_state, base_ck
        else:
            ck = load_state(ckpt)
            state = ck["model_state"] if alpha is None else interpolate(base_state, ck["model_state"], alpha)
        model, cfg = model_from_state(ck, state, device)
        scores, hyps, s = score_on_dev(model, cfg, sp, sets, device, beam)
        sigs = s
        del model
        row = {"row_tag": row_tag, "seed": seed, "role": role, "beam": beam,
               "ckpt": os.path.basename(ckpt), "ckpt_fp": fp,
               "scored_at": datetime.now().isoformat(timespec="seconds"), **scores}
        if run:
            st = run["settings"]
            row.update({"method": st["method"], "mix": st["mix"], "ewc_lambda": st["ewc_lambda"],
                        "no_convlab": st["no_convlab"], "lr": st["lr"], "patience": st["patience"],
                        "gen_limit": st["gen_limit"], "best_epoch": run["best_epoch"],
                        "alpha": "" if alpha is None else alpha})
        else:
            row.update({"method": "base", "alpha": ""})
        rows[(row_tag, seed)] = row
        with open(os.path.join(hyp_dir, f"{row_tag}_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(hyps, f, ensure_ascii=False)
        print(f"[dev]   hospitality chrF++ {scores['hosp_chrf_mean']:.2f} "
              f"(EN->ZH {scores['hosp_enzh_chrf']:.2f}, ZH->EN {scores['hosp_zhen_chrf']:.2f}) | "
              f"general BLEU {scores['gen_bleu_mean']:.2f}", flush=True)

    base_row = rows.get(("base", str(BASE_SEED)))
    for r in rows.values():
        r["gen_bleu_drop"] = round(float(base_row["gen_bleu_mean"]) - float(r["gen_bleu_mean"]), 2)
    write_dev_scores(d, list(rows.values()))
    if sigs:
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump(sigs, f, indent=2)
    print(f"[dev] {len(rows)} rows in {os.path.join(d['results'], 'dev_scores.csv')}")


# --------------------------------------------------------------- select ----

def apply_rule(rows, rule):
    """returns (winner_row, eligible_rows, reasons). base never wins: it's the reference."""
    drop_max, tie = rule["max_gen_bleu_drop"], rule["tie_chrf"]
    budget = rule["budget"]
    cands, notes = [], []
    for r in rows:
        if r["role"] not in ("run", "interp") or str(r["seed"]) != str(SELECTION_SEED):
            continue
        if (float(r["lr"]), int(r["patience"]), float(r["gen_limit"])) != \
                (float(budget["lr"]), int(budget["patience"]), float(budget["gen_limit"])):
            notes.append(f"{r['row_tag']}: not on the registered budget, left out")
            continue
        cands.append(r)
    eligible = [r for r in cands if float(r["gen_bleu_drop"]) <= drop_max]
    for r in cands:
        if r not in eligible:
            notes.append(f"{r['row_tag']}: general BLEU drop {float(r['gen_bleu_drop']):.2f} > {drop_max}")
    if not eligible:
        return None, eligible, notes
    top = max(float(r["hosp_chrf_mean"]) for r in eligible)
    close = [r for r in eligible if top - float(r["hosp_chrf_mean"]) <= tie]
    close.sort(key=lambda r: (float(r["gen_bleu_drop"]), -float(r["hosp_chrf_mean"]), r["row_tag"]))
    return close[0], eligible, notes


def _print_candidates(rows, base, winner, eligible, rule):
    """the same numbers the plot shows, as a table, best hospitality chrF++ first."""
    cands = [r for r in rows if r["role"] in ("run", "interp")
             and str(r["seed"]) == str(SELECTION_SEED)]
    cands.sort(key=lambda r: -float(r["hosp_chrf_mean"]))
    print(f"\n{'candidate':<38}{'hosp chrF++':>12}{'EN>ZH':>8}{'ZH>EN':>8}{'gen BLEU drop':>15}  note")
    print(f"{'base':<38}{float(base['hosp_chrf_mean']):>12.2f}{float(base['hosp_enzh_chrf']):>8.2f}"
          f"{float(base['hosp_zhen_chrf']):>8.2f}{0:>15.2f}")
    for r in cands:
        if winner is not None and r is winner:
            note = "<- selected"
        elif r not in eligible:
            note = "out (drop too big or off budget)"
        else:
            note = ""
        print(f"{r['row_tag']:<38}{float(r['hosp_chrf_mean']):>12.2f}{float(r['hosp_enzh_chrf']):>8.2f}"
              f"{float(r['hosp_zhen_chrf']):>8.2f}{float(r['gen_bleu_drop']):>15.2f}  {note}")
    print(f"(rule: highest hospitality chrF++ with general BLEU drop <= {rule['max_gen_bleu_drop']}; "
          f"within {rule['tie_chrf']} chrF++ counts as a tie, smaller drop wins)\n")


def cmd_select(args):
    d = ft_dirs()
    rule = load_rule(d)
    rows = read_dev_scores(d)
    base = next((r for r in rows if r["role"] == "base"), None)
    if base is None:
        raise SystemExit("dev_scores.csv has no base row yet, run score-dev first")
    winner, eligible, notes = apply_rule(rows, rule)
    for n in notes:
        print(f"[select] {n}")
    _print_candidates(rows, base, winner, eligible, rule)
    plot_path = os.path.join(d["results"], "selection_plot.png")
    plot_selection(rows, base, winner, rule, plot_path)
    sel_path = os.path.join(d["results"], "selection.json")
    if winner is None:
        raise SystemExit(f"[select] nothing selected, no candidate stays within "
                         f"{rule['max_gen_bleu_drop']} BLEU of the base on general dev. plot: {plot_path}")
    if float(winner["hosp_chrf_mean"]) <= float(base["hosp_chrf_mean"]):
        print("[select] the winner doesn't beat the base on hospitality dev chrF++")
    with open(os.path.join(d["results"], "dev_scores.csv"), "rb") as f:
        scores_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    alpha = winner["alpha"]
    sel = {"winner": winner["row_tag"], "seed": int(winner["seed"]),
           "parent_tag": run_tag(winner["method"], float(winner["lr"]), int(winner["patience"]),
                                 int(float(winner["mix"] or 0)), float(winner["ewc_lambda"] or 0)),
           "method": winner["method"], "mix": int(float(winner["mix"] or 0)),
           "ewc_lambda": float(winner["ewc_lambda"] or 0),
           "alpha": float(alpha) if alpha not in ("", None) else None,
           "lr": float(winner["lr"]), "patience": int(winner["patience"]),
           "gen_limit": float(winner["gen_limit"]),
           "dev": {k: winner[k] for k in ("hosp_chrf_mean", "hosp_bleu_mean", "gen_bleu_mean",
                                          "gen_bleu_drop")},
           "base_dev": {k: base[k] for k in ("hosp_chrf_mean", "hosp_bleu_mean", "gen_bleu_mean")},
           "n_candidates": len([r for r in rows if r["role"] in ("run", "interp")]),
           "n_eligible": len(eligible), "rule": {k: rule[k] for k in
                                                 ("max_gen_bleu_drop", "tie_chrf", "registered_at")},
           "dev_scores_sha16": scores_sha,
           "selected_at": datetime.now().isoformat(timespec="seconds")}
    if os.path.exists(sel_path):
        with open(sel_path, encoding="utf-8") as f:
            old = json.load(f)
        if old["winner"] != sel["winner"]:
            if os.path.exists(os.path.join(d["results"], "test_runs.jsonl")):
                raise SystemExit(
                    f"[select] the test was already scored on {old['winner']}, so selection.json "
                    f"stays as it is. switching now would mean picking with the test results "
                    f"in view")
            print(f"[select] replacing the earlier selection ({old['winner']}, "
                  f"{old['selected_at']}). the test hasn't been run, so that's fine")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=2)
    print(f"[select] winner: {sel['winner']}  hospitality dev chrF++ {winner['hosp_chrf_mean']} "
          f"(base {base['hosp_chrf_mean']}), general dev BLEU drop {winner['gen_bleu_drop']}")
    print(f"[select] wrote {sel_path} and {plot_path}")


METHOD_STYLE = {  # color slots 1-3 of the dataviz default palette, plus distinct markers
    "plain": ("#2a78d6", "o", "plain"),
    "mixed": ("#eb6834", "s", "mixed 1:4"),
    "ewc": ("#1baf7a", "^", "EWC"),
}


def plot_selection(rows, base, winner, rule, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=130)
    ink, muted = "#0b0b0b", "#52514e"
    bx, by = 0.0, float(base["hosp_chrf_mean"])
    drop_max = rule["max_gen_bleu_drop"]
    cand = [r for r in rows if r["role"] in ("run", "interp")]
    xs = [float(r["gen_bleu_drop"]) for r in cand] + [0.0]
    xmax = max(max(xs) + 0.4, drop_max + 0.6)
    ax.axvspan(drop_max, xmax, color="#ebeae6", zorder=0)
    ax.axvline(drop_max, color=muted, lw=1, ls="--", zorder=1)
    # each run: a thin path from the base through its interpolations to the run itself
    runs = [r for r in cand if r["role"] == "run"]
    seen = set()
    for r in runs:
        color, marker, label = METHOD_STYLE[r["method"]]
        path_rows = sorted([x for x in cand if x["role"] == "interp" and
                            x["row_tag"].startswith(r["row_tag"] + "_interp")],
                           key=lambda x: float(x["alpha"]))
        px = [bx] + [float(x["gen_bleu_drop"]) for x in path_rows] + [float(r["gen_bleu_drop"])]
        py = [by] + [float(x["hosp_chrf_mean"]) for x in path_rows] + [float(r["hosp_chrf_mean"])]
        ax.plot(px, py, color=color, lw=1, alpha=0.5, zorder=2)
        ax.scatter([float(x["gen_bleu_drop"]) for x in path_rows],
                   [float(x["hosp_chrf_mean"]) for x in path_rows], s=40, marker=marker,
                   facecolors="#fcfcfb", edgecolors=color, linewidths=1.5, zorder=3)
        ax.scatter([float(r["gen_bleu_drop"])], [float(r["hosp_chrf_mean"])], s=64, marker=marker,
                   color=color, edgecolors="#fcfcfb", linewidths=2, zorder=4,
                   label=label if r["method"] not in seen else None)
        seen.add(r["method"])
        short = r["row_tag"].split("_", 2)[-1]
        ax.annotate(short, (float(r["gen_bleu_drop"]), float(r["hosp_chrf_mean"])),
                    xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=muted)
    ax.scatter([bx], [by], s=80, marker="D", color=ink, zorder=5, label="base model")
    if winner is not None:
        wx, wy = float(winner["gen_bleu_drop"]), float(winner["hosp_chrf_mean"])
        ax.scatter([wx], [wy], s=260, facecolors="none", edgecolors=ink, linewidths=1.5, zorder=6)
        ax.annotate(f"selected: {winner['row_tag'].split('_', 2)[-1]}", (wx, wy), xytext=(10, -16),
                    textcoords="offset points", fontsize=8.5, color=ink)
    ax.scatter([], [], s=40, marker="o", facecolors="#fcfcfb", edgecolors=muted,
               label="interpolated with base (alpha 0.25 / 0.5 / 0.75)")
    ax.set_xlim(min(-0.3, min(xs) - 0.3), xmax)
    ax.set_xlabel("general dev BLEU drop vs base (lower is better)", color=ink)
    ax.set_ylabel("hospitality dev chrF++ (mean of both directions)", color=ink)
    ax.set_title(f"Selection on dev: max general drop {drop_max} BLEU (shaded = ruled out), "
                 f"ties within {rule['tie_chrf']} chrF++", fontsize=9.5, color=ink, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(muted); ax.spines["bottom"].set_color(muted)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(color="#e4e3df", lw=0.6, zorder=0)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)


# ----------------------------------------------------------------- test ----

def materialize(d, base_ck, seed, sel):
    """checkpoint for the winning recipe at this seed. interpolated winners get written
    out as a real .pt so the app can load them like any other checkpoint."""
    parent = run_files(d, seed, sel["parent_tag"])
    with open(parent["done"], encoding="utf-8") as f:
        info = json.load(f)
    if not info["saved_any"]:
        raise SystemExit(f"seed {seed} of {sel['parent_tag']} never saved a best epoch")
    if sel["alpha"] is None:
        return parent["best"]
    out = run_files(d, seed, sel["winner"])["best"]
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(parent["best"]):
        ck = load_state(parent["best"])
        ck["model_state"] = interpolate(base_ck["model_state"], ck["model_state"], sel["alpha"])
        ck["run_tag"] = sel["winner"]
        ck["interp_alpha"] = sel["alpha"]
        safe_save(ck, out, d)
        print(f"[test] wrote {os.path.basename(out)}")
    return out


def test_sets(d, limit=None):
    """ft_test natural-source halves, the optional pre-review references, WMT, FLORES+."""
    from evaluate import load_wmt, load_flores
    pairs = read_pairs(os.path.join(d["data"], "ft_test.tsv.gz"))
    meta = read_meta(os.path.join(d["data"], "test_meta.tsv"), len(pairs))
    nat = natural_source_split(pairs, meta)
    sets = {"ft_test:en-zh": (nat["en-zh"][0], nat["en-zh"][1], "zh"),
            "ft_test:zh-en": (nat["zh-en"][0], nat["zh-en"][1], "en")}
    prereview = {}
    pre_path = os.path.join(d["data"], "ft_test_prereview.tsv.gz")
    if os.path.exists(pre_path):
        pre = read_pairs(pre_path)
        if len(pre) != len(pairs):
            raise SystemExit(f"ft_test_prereview has {len(pre)} lines, ft_test has {len(pairs)}")
        for direction, (srcs, refs, rows) in nat.items():
            src_side, ref_side = (1, 0) if direction == "en-zh" else (0, 1)
            changed_src = sum(pre[i][src_side] != pairs[i][src_side] for i in rows)
            prereview[f"ft_test:{direction}"] = [pre[i][ref_side] for i in rows]
            print(f"[test] pre-review refs for {direction}: {len(rows)} lines, "
                  f"{changed_src} had their source edited in review (we translate the "
                  f"reviewed source)")
    else:
        print("[test] no ft_test_prereview.tsv.gz in data/, scoring reviewed references only")
    wmt = load_wmt(C.DATA["wmt_testset"])
    if set(wmt) != {"zh-en", "en-zh"}:
        raise SystemExit("couldn't load WMT for both directions, see the messages above")
    for lp, (s, r) in wmt.items():
        sets[f"wmt:{lp}"] = (s, r, lp.split("-")[1])
    flores = load_flores()
    for lp, (s, r) in flores.items():
        sets[f"flores+:{lp}"] = (s, r, lp.split("-")[1])
    if limit:
        sets = {k: (s[:limit], r[:limit], lang) for k, (s, r, lang) in sets.items()}
        prereview = {k: v[:limit] for k, v in prereview.items()}
    return sets, prereview


def cmd_test(args):
    d = ft_dirs()
    rule = load_rule(d)
    beam = rule["beam"]
    sel_path = os.path.join(d["results"], "selection.json")
    if not os.path.exists(sel_path):
        raise SystemExit("[test] no selection.json yet, run the selection cell first. the "
                         "test only scores a model that was picked on dev")
    with open(sel_path, encoding="utf-8") as f:
        sel = json.load(f)
    ledger = os.path.join(d["results"], "test_runs.jsonl")
    if os.path.exists(ledger):
        with open(ledger, encoding="utf-8") as f:
            past = [json.loads(x) for x in f if x.strip()]
        print("[test] the test set was scored before:")
        for p in past:
            print(f"    {p['ran_at']} by {p['owner']}, winner {p['winner']}")
        if not args.rerun:
            raise SystemExit("[test] stopping. every extra look at the test set weakens the "
                             "claim that nothing was tuned on it. after a crash or a missing "
                             "seed, --rerun goes ahead and both runs get reported")
        print("[test] --rerun given, going ahead. both runs stay in test_runs.jsonl")

    seeds = [int(s) for s in args.seeds.split(",")]
    if seeds[0] != sel["seed"]:
        raise SystemExit(f"the first seed has to be the selected one ({sel['seed']})")
    for s in seeds:
        done = run_files(d, s, sel["parent_tag"])["done"]
        if not os.path.exists(done):
            raise SystemExit(f"[test] seed {s} of {sel['parent_tag']} hasn't finished. it has to "
                             f"finish or come out of --seeds first, since testing now and "
                             f"again later would count as two looks")

    device = get_device()
    sp = load_tokenizer()
    base_ck = load_state(d["base_ckpt"])
    limit = args.limit or (24 if C.SMOKE else None)
    sets, prereview = test_sets(d, limit)
    print(f"[test] winner {sel['winner']}, seeds {seeds}, beam {beam}")
    for k, (s, _, _) in sets.items():
        print(f"    {k}: {len(s)} lines")

    systems = [("base", d["base_ckpt"])] + [(f"seed{s}", materialize(d, base_ck, s, sel))
                                           for s in seeds]
    hyps = {}
    hyp_dir = os.path.join(d["results"], "test_hyps")
    os.makedirs(hyp_dir, exist_ok=True)
    for name, ckpt in systems:
        ck = load_state(ckpt)
        model, cfg = model_from_state(ck, ck["model_state"], device)
        hyps[name] = {}
        for key, (src, _, lang) in sets.items():
            print(f"[test] {name}: translating {key}...", flush=True)
            hyps[name][key] = translate(model, cfg, sp, src, lang, device, beam)
        del model
        with open(os.path.join(hyp_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(hyps[name], f, ensure_ascii=False)

    ref_sets = {k: (v[1], v[2]) for k, v in sets.items()}
    for k, refs in prereview.items():
        ref_sets[k + ":prereview"] = (refs, sets[k][2])
    results = {"winner": sel["winner"], "seeds": seeds, "beam": beam, "sets": {}}
    final = f"seed{sel['seed']}"
    for key, (refs, lang) in ref_sets.items():
        hkey = key.replace(":prereview", "")
        entry = {"n": len(refs)}
        for name, _ in systems:
            entry[name] = score_texts(hyps[name][hkey], refs, lang)
        if len(seeds) > 1:
            for m in ("BLEU", "chrF++"):
                vals = [entry[f"seed{s}"][m] for s in seeds]
                entry[f"{m}_seeds_mean"] = round(float(np.mean(vals)), 2)
                entry[f"{m}_seeds_sd"] = round(float(np.std(vals, ddof=1)), 2)
        entry["bootstrap"] = paired_bootstrap(hyps["base"][hkey], hyps[final][hkey], refs, lang,
                                              n_samples=args.bootstrap)
        results["sets"][key] = entry

    # log the look before showing anything, so a crash after this still counts as one
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ran_at": datetime.now().isoformat(timespec="seconds"),
                            "owner": args.owner, "winner": sel["winner"], "seeds": seeds,
                            "selection_selected_at": sel["selected_at"],
                            "limit": limit}) + "\n")
    out_json = os.path.join(d["results"], "test_results.json")
    if os.path.exists(out_json):
        os.replace(out_json, out_json.replace(".json", f"_before_{int(time.time())}.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    _write_test_csv(results, seeds, os.path.join(d["results"], "test_results.csv"))
    _print_test_table(results, seeds, final)
    print(f"\n[test] written: {out_json}, test_results.csv, test_hyps/, and a line in test_runs.jsonl")


def _print_test_table(results, seeds, final):
    print("\n" + "=" * 100)
    print(f"final test, winner {results['winner']}, base vs {final}, paired bootstrap "
          f"(95% interval of the difference; p1 = share of resamples where the fine-tuned "
          f"model is not better, one-sided)")
    print("=" * 100)
    for key, e in results["sets"].items():
        b = e["bootstrap"]
        for m in ("BLEU", "chrF++"):
            seeds_txt = ""
            if f"{m}_seeds_mean" in e:
                seeds_txt = f"  | {len(seeds)} seeds {e[f'{m}_seeds_mean']:.2f} +/- {e[f'{m}_seeds_sd']:.2f}"
            print(f"{key:<28} n={e['n']:<5} {m:<6} base {e['base'][m]:6.2f}  final "
                  f"{e[final][m]:6.2f}  diff {b[m]['delta']:+6.2f} "
                  f"[{b[m]['delta_ci95'][0]:+.2f}, {b[m]['delta_ci95'][1]:+.2f}]  "
                  f"p1={b[m]['p_not_better']:.3f}{seeds_txt}")
    print("\nsacreBLEU signatures:")
    sigs = set()
    for e in results["sets"].values():
        sigs.add(e["base"]["bleu_sig"])
        sigs.add(e["base"]["chrf_sig"])
    for s in sorted(sigs):
        print("   ", s)


def _write_test_csv(results, seeds, path):
    cols = ["set", "n", "system", "BLEU", "chrF++", "BLEU_delta_vs_base", "BLEU_delta_ci_lo",
            "BLEU_delta_ci_hi", "BLEU_p_not_better", "chrF++_delta_vs_base", "chrF++_delta_ci_lo",
            "chrF++_delta_ci_hi", "chrF++_p_not_better", "bleu_sig", "chrf_sig"]
    final = f"seed{seeds[0]}"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key, e in results["sets"].items():
            for name in ["base"] + [f"seed{s}" for s in seeds]:
                row = {"set": key, "n": e["n"], "system": name, "BLEU": e[name]["BLEU"],
                       "chrF++": e[name]["chrF++"], "bleu_sig": e[name]["bleu_sig"],
                       "chrf_sig": e[name]["chrf_sig"]}
                if name == final:
                    for m in ("BLEU", "chrF++"):
                        bs = e["bootstrap"][m]
                        row.update({f"{m}_delta_vs_base": bs["delta"],
                                    f"{m}_delta_ci_lo": bs["delta_ci95"][0],
                                    f"{m}_delta_ci_hi": bs["delta_ci95"][1],
                                    f"{m}_p_not_better": bs["p_not_better"]})
                w.writerow(row)
            if "BLEU_seeds_mean" in e:
                w.writerow({"set": key, "n": e["n"], "system": f"mean of {len(seeds)} seeds",
                            "BLEU": f"{e['BLEU_seeds_mean']} +/- {e['BLEU_seeds_sd']}",
                            "chrF++": f"{e['chrF++_seeds_mean']} +/- {e['chrF++_seeds_sd']}"})


# ------------------------------------------------------------ preflight ----

def cmd_preflight(args):
    d = ft_dirs()
    problems = []
    need = {"base checkpoint": d["base_ckpt"],
            "tokenizer": C.TOKENIZER_PREFIX + ".model",
            "general train": d["general_train"], "general dev": d["general_dev"]}
    for name in ("ft_train.tsv.gz", "ft_dev.tsv.gz", "ft_test.tsv.gz", "dev_meta.tsv",
                 "test_meta.tsv"):
        need[name] = os.path.join(d["data"], name)
    for name, p in need.items():
        ok = os.path.exists(p)
        print(f"  {'ok     ' if ok else 'missing'} {name:<18} {p}")
        if not ok:
            problems.append(name)
    for opt in ("train_meta.tsv", "ft_test_prereview.tsv.gz"):
        p = os.path.join(d["data"], opt)
        print(f"  {'ok     ' if os.path.exists(p) else 'absent '} {opt:<18} (optional) {p}")
    if problems:
        raise SystemExit(f"missing: {', '.join(problems)}")
    counts = {}
    for name in ("ft_train", "ft_dev", "ft_test"):
        counts[name] = len(read_pairs(os.path.join(d["data"], name + ".tsv.gz")))
    counts["general_dev"] = len(read_pairs(d["general_dev"]))
    print(f"  pairs: {counts}")
    read_meta(os.path.join(d["data"], "dev_meta.tsv"), counts["ft_dev"])
    read_meta(os.path.join(d["data"], "test_meta.tsv"), counts["ft_test"])
    tm = os.path.join(d["data"], "train_meta.tsv")
    if os.path.exists(tm):
        read_meta(tm, counts["ft_train"])
    print("  meta files line up with their data files")
    sets = {n: read_pairs(os.path.join(d["data"], n + ".tsv.gz")) for n in ("ft_train", "ft_dev", "ft_test")}
    for a, b in (("ft_train", "ft_dev"), ("ft_train", "ft_test"), ("ft_dev", "ft_test")):
        zh = len({x[0] for x in sets[a]} & {x[0] for x in sets[b]})
        en = len({x[1] for x in sets[a]} & {x[1] for x in sets[b]})
        print(f"  shared sentences {a}/{b}: zh {zh}, en {en}")
        if zh or en:
            problems.append(f"{a}/{b} overlap")
    print(f"  general dev content sha16 {content_sha16(d['general_dev'])} "
          f"(data_manifest.json dev_sha256_16 should match)")
    ck = load_state(d["base_ckpt"])
    print(f"  base: {ck['arch']}, pretraining epoch {ck.get('epoch')}, vocab {ck['vocab_size']}")
    if problems:
        raise SystemExit(f"problems: {problems}")
    print("[preflight] all good")


# ------------------------------------------------------------------ cli ----

def main(argv=None):
    ap = argparse.ArgumentParser(description="hospitality fine-tuning for scaled arch 5")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--owner", default=os.environ.get("SPEECHBRIDGE_OWNER", "unknown"))

    p = sub.add_parser("preflight"); common(p)

    p = sub.add_parser("register-rule"); common(p)
    p.add_argument("--max-gen-bleu-drop", type=float, required=True)
    p.add_argument("--tie", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--patience", type=int, required=True)
    p.add_argument("--gen-limit", type=float, required=True)
    p.add_argument("--mix", type=int, required=True)
    p.add_argument("--ewc-lambdas", type=float, nargs="+", required=True)
    p.add_argument("--alphas", type=float, nargs="+", required=True)
    p.add_argument("--beam", type=int, required=True)

    p = sub.add_parser("train"); common(p)
    p.add_argument("--method", choices=["plain", "mixed", "ewc"], required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--patience", type=int, required=True)
    p.add_argument("--gen-limit", type=float, required=True,
                   help="relative: 0.05 means general dev CE may be at most 5%% above the base")
    p.add_argument("--mix", type=int, default=4, help="general pairs per hospitality pair (mixed)")
    p.add_argument("--ewc-lambda", type=float, default=0.0)
    p.add_argument("--fisher-batches", type=int, default=200)
    p.add_argument("--seed", type=int, default=SELECTION_SEED)
    p.add_argument("--no-convlab", action="store_true",
                   help="drop the ConvLab-2 rows, CrossWOZ and MultiWOZ (needs data/train_meta.tsv)")
    p.add_argument("--expect-tag", default=None, help="stop if the tag doesn't come out as this")
    p.add_argument("--force", action="store_true", help="take over someone else's stale lock")
    p.add_argument("--fresh", action="store_true", help="start this run over from the base")

    p = sub.add_parser("score-dev"); common(p)
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="only N lines per set (debugging)")

    p = sub.add_parser("select"); common(p)

    p = sub.add_parser("test"); common(p)
    p.add_argument("--seeds", default=",".join(str(s) for s in (SELECTION_SEED,) + REPLICATE_SEEDS))
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="only N lines per set (debugging)")

    args = ap.parse_args(argv)
    if args.cmd == "train" and args.method == "ewc" and args.ewc_lambda <= 0:
        ap.error("--method ewc needs --ewc-lambda")
    {"preflight": cmd_preflight, "register-rule": cmd_register_rule, "train": cmd_train,
     "score-dev": cmd_score_dev, "select": cmd_select, "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    main()
