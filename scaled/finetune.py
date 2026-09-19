from __future__ import annotations
import argparse
import importlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from utils import (seed_everything, get_device, count_params, human_millions,
                   MetricsLogger, Timer, reset_peak_vram, peak_vram_gb)
from data.dataset import TranslationDataset, collate, MaxTokensBatchSampler
from models.common import build_criterion


def estimate_fisher(model, loader, criterion, device, cfg, use_amp, max_batches):
    """Diagonal Fisher: average of squared gradients of the loss over general data.
    Larger value = that weight matters more to general translation, so protect it more.
    """
    import torch.nn as nn
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}
    n_batches = 0
    for i, (src, src_len, tgt, _r, _d) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        model.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(src, src_len, tgt_in, tf_ratio=1.0)
            V = logits.size(-1)
            loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        n_batches += 1
    for n in fisher:
        fisher[n] /= max(n_batches, 1)
    model.zero_grad()
    print(f"[ewc] fisher done ({n_batches} batches)")
    return fisher


def ewc_penalty(model, fisher, star_params):
    loss = 0.0
    for n, p in model.named_parameters():
        if n in fisher:
            loss = loss + (fisher[n] * (p - star_params[n]) ** 2).sum()
    return 0.5 * loss


def run_epoch_ewc(model, loader, criterion, ce_for_ppl, device, cfg, optimizer,
                  fisher, star_params, ewc_lambda, use_amp, scaler, max_batches=None):
    """A training epoch with the EWC penalty added to the task loss. Mirrors the study
    run_epoch (accumulation, clipping, AMP), plus the penalty term so backward sees it.
    """
    import torch.nn as nn  # noqa: F811
    model.train()
    total_loss, total_ce, total_tok = 0.0, 0.0, 0
    accum = cfg.accumulation_steps
    pending = 0
    optimizer.zero_grad()

    def step():
        if scaler is not None and use_amp:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        optimizer.zero_grad()

    for i, (src, src_len, tgt, _r, _d) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(src, src_len, tgt_in, tf_ratio=1.0)
            V = logits.size(-1)
            task_loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))
            loss = task_loss + ewc_lambda * ewc_penalty(model, fisher, star_params)
        with torch.no_grad():
            ce = ce_for_ppl(logits.float().reshape(-1, V), tgt_out.reshape(-1))
        scaled = scaler.scale(loss / accum) if (scaler is not None and use_amp) else (loss / accum)
        scaled.backward()
        pending += 1
        if pending == accum:
            step(); pending = 0
        n_tok = (tgt_out != C.PAD_ID).sum().item()
        total_loss += task_loss.item() * n_tok  # log the TASK loss, not task+penalty
        total_ce += ce.item() * n_tok
        total_tok += n_tok
    if pending > 0:
        step()
    return total_loss / max(total_tok, 1), total_ce / max(total_tok, 1)

import train
from train import (run_epoch, atomic_save, acquire_lock, touch_lock, release_lock,
                   load_tokenizer, capture_rng, restore_rng)

train.MODEL_MODULES["arch5_improved_transformer"] = "models.arch5_improved_transformer"

ARCH = "arch5_improved_transformer"

def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def run_tag(lr: float, patience: int, method: str = "mixed",
            general_mix: int = 0, ewc_lambda: float = 0.0) -> str:
    base = f"lr{lr_tag(lr)}_p{patience}"
    if method == "mixed" and general_mix > 0:
        return f"{base}_mix{general_mix}"
    if method == "ewc":
        return f"{base}_ewc{int(ewc_lambda)}"
    return base


def ft_root() -> str:
    return os.environ.get("SPEECHBRIDGE_FT",
                          os.path.join(os.path.dirname(C.DATA_DIR), "fine_tune_arch5"))


def ft_paths() -> dict:
    root = ft_root()
    paths = {"data": os.path.join(root, "data"),
             "ckpt": os.path.join(root, "checkpoints"),
             "results": os.path.join(root, "results")}
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def preflight(base_ckpt: str, paths: dict) -> None:
    print(f"[ft] config: {os.path.abspath(C.__file__)}")
    missing = []
    if not os.path.exists(base_ckpt):
        missing.append(f"no base ckpt at {base_ckpt}")
    for name in ("ft_train.tsv.gz", "ft_dev.tsv.gz"):
        if not os.path.exists(os.path.join(paths["data"], name)):
            missing.append(f"{name} not in {paths['data']}")
    tok = C.TOKENIZER_PREFIX + ".model"
    if not os.path.exists(tok):
        missing.append(f"no tokenizer at {tok}")
    if missing:
        raise SystemExit("[ft] can't start:\n  " + "\n  ".join(missing))
    if not os.path.exists(os.path.join(paths["data"], "ft_test.tsv.gz")):
        print("[ft] warning: no ft_test.tsv.gz, need it for eval later")


def reservoir_sample_general(path: str, k: int, seed: int) -> list:
    """Pull k random (zh, en) pairs from the big general train file without loading all of
    it into memory. Reservoir sampling reads the gzip stream once and keeps a uniform
    random k. Same seed gives the same sample every run, so results are reproducible.
    """
    import gzip
    import random
    rng = random.Random(seed)
    keep = []
    n = 0
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
    return keep


def make_ft_loaders(cfg, sp, paths, general_mix: int, general_path: str, seed: int):
    tr = os.path.join(paths["data"], "ft_train.tsv.gz")
    dv = os.path.join(paths["data"], "ft_dev.tsv.gz")
    train_ds = TranslationDataset(tr, sp, max_subword_len=cfg.max_len)
    dev_ds = TranslationDataset(dv, sp, max_subword_len=cfg.max_len)

    if general_mix > 0:
        n_hosp_pairs = len(train_ds.items) // 2
        n_general_pairs = general_mix * n_hosp_pairs
        if not os.path.exists(general_path):
            raise SystemExit(f"[ft] general train file not found: {general_path}")
        print(f"[ft] mix 1:{general_mix}, sampling {n_general_pairs:,} general pairs")
        gen_pairs = reservoir_sample_general(general_path, n_general_pairs, seed)
        gen_ds = TranslationDataset(tr, sp, max_subword_len=cfg.max_len)
        gen_ds.items = []
        for zh, en in gen_pairs:
            gen_ds.items.append((train_ds.tag["en"], zh, en, "zh-en"))
            gen_ds.items.append((train_ds.tag["zh"], en, zh, "en-zh"))
        train_ds.items = train_ds.items + gen_ds.items
        train_ds._lengths = None  # recompute lengths over the blended set
        print(f"[ft] train: {n_hosp_pairs:,} hosp + {len(gen_pairs):,} general "
              f"({len(train_ds.items):,} w/ both directions)")
    else:
        print(f"[ft] train {len(train_ds):,} / dev {len(dev_ds):,}")

    if cfg.batch_by_tokens:
        sampler = MaxTokensBatchSampler(train_ds, cfg.max_tokens, shuffle=True, seed=C.SEED)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  collate_fn=collate, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate)
    return train_loader, dev_loader


def main():
    ap = argparse.ArgumentParser(description="Hospitality fine-tuning for the scaled Arch 5 model.")
    ap.add_argument("--seed", type=int, default=getattr(C, "SEED", 191))
    ap.add_argument("--owner", default=os.environ.get("SPEECHBRIDGE_OWNER", "unknown"))
    ap.add_argument("--force", action="store_true",
                    help="take over a stale lock (the previous member has stopped)")
    ap.add_argument("--fresh", action="store_true",
                    help="restart from the base model, ignoring any fine-tune checkpoint")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="constant fine-tune LR. Small on purpose. Drop to 3e-5 if the "
                         "general WMT/FLORES scores regress; try 1e-4 if hospitality "
                         "barely moves after several epochs.")
    ap.add_argument("--epochs", type=int, default=100,
                    help="hard cap only; early stopping normally ends the run first.")
    ap.add_argument("--patience", type=int, default=5,
                    help="stop after this many epochs with no dev improvement.")
    ap.add_argument("--general-mix", type=int, default=4,
                    help="mixed fine-tuning ratio: general pairs blended in per hospitality "
                         "pair (Chu et al. 2017). 4 means 1 hospitality : 4 general. Set 0 "
                         "for hospitality-only training.")
    ap.add_argument("--general-file", default=None,
                    help="the general train file to sample from "
                         "(default: <scaled data>/train.tsv.gz)")
    ap.add_argument("--method", choices=["plain", "mixed", "ewc"], default="mixed",
                    help="plain = hospitality only; mixed = blend general data in "
                         "(Chu 2017); ewc = penalize moving weights important to the "
                         "general task (Kirkpatrick 2017 / Thompson 2019).")
    ap.add_argument("--ewc-lambda", type=float, default=5000.0,
                    help="EWC penalty strength. Higher protects general translation more "
                         "but adapts less. Typical range 1000 to 50000; tune it like LR.")
    ap.add_argument("--fisher-batches", type=int, default=200,
                    help="how many general batches to estimate the Fisher information from.")
    ap.add_argument("--base-ckpt", default=None,
                    help="the scaled Arch 5 model to start from "
                         "(default: checkpoints/arch5_improved_transformer_seed{seed}_best.pt)")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    paths = ft_paths()
    base_ckpt = args.base_ckpt or os.path.join(
        C.CKPT_DIR, f"{ARCH}_seed{args.seed}_best.pt")
    preflight(base_ckpt, paths)

    cfg = C.get_arch_config(ARCH)
    cfg.scheduler = "none"
    cfg.lr = args.lr
    cfg.epochs = args.epochs
    cfg.patience = args.patience
    cfg.warmup_steps = 0
    cfg.tf_decay_epochs = 0

    sp = load_tokenizer()
    vocab_size = sp.get_piece_size()
    general_path = args.general_file or os.path.join(C.DATA_DIR, "train.tsv.gz")

    effective_mix = args.general_mix if args.method == "mixed" else 0
    train_loader, dev_loader = make_ft_loaders(cfg, sp, paths, effective_mix,
                                               general_path, args.seed)
    train_sampler = getattr(train_loader, "batch_sampler", None)

    mod = importlib.import_module(train.MODEL_MODULES[ARCH])
    model = mod.build(cfg, vocab_size).to(device)
    print(f"[ft] arch5 {human_millions(count_params(model))}M params")

    criterion = build_criterion(vocab_size, cfg.label_smoothing)
    ce_for_ppl = nn.CrossEntropyLoss(ignore_index=C.PAD_ID)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = None

    use_amp = device.type == "cuda" and cfg.amp_safe
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.method == "mixed":
        method_desc = f"mixed 1:{args.general_mix}"
    elif args.method == "ewc":
        method_desc = f"ewc lambda={args.ewc_lambda:g}"
    else:
        method_desc = "plain"
    print(f"[ft] amp={use_amp} lr={cfg.lr} patience={cfg.patience} method={method_desc}")

    tag = run_tag(cfg.lr, cfg.patience, args.method, args.general_mix, args.ewc_lambda)
    os.environ["SPEECHBRIDGE_RESULTS"] = paths["results"]
    logger = MetricsLogger(os.path.join(
        paths["results"], f"metrics_log_ft_arch5_seed{args.seed}_{tag}.csv"))
    best_path = os.path.join(paths["ckpt"], f"{ARCH}_ft_seed{args.seed}_{tag}_best.pt")
    last_path = os.path.join(paths["ckpt"], f"{ARCH}_ft_seed{args.seed}_{tag}_last.pt")

    lock = acquire_lock(ARCH + "_ft_" + tag, args.seed, args.owner, force=args.force)

    start_epoch, best_dev, bad_epochs = 0, math.inf, 0
    if os.path.exists(last_path) and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        if ck.get("rng") is not None:
            restore_rng(ck["rng"])
        start_epoch = ck["epoch"] + 1
        best_dev = ck["best_dev"]
        bad_epochs = ck["bad_epochs"]
        print(f"[ft] resuming at epoch {start_epoch} (was {ck.get('owner', '?')}, "
              f"now {args.owner}), best dev {best_dev:.4f}, bad {bad_epochs}/{cfg.patience}")
    else:
        base = torch.load(base_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(base["model_state"])
        print(f"[ft] loaded base {os.path.basename(base_ckpt)} "
              f"(ep {base.get('epoch')}, dev {base.get('dev_loss')})")
        base_dev, base_ce = run_epoch(model, dev_loader, criterion, ce_for_ppl, device,
                                      cfg, train=False, scaler=scaler, use_amp=use_amp)
        print(f"[ft] baseline dev {base_dev:.4f} (ppl {math.exp(min(base_ce, 20)):.2f})")

    fisher, star_params = None, None
    if args.method == "ewc":
        star_params = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        gen_pairs = reservoir_sample_general(
            general_path, args.fisher_batches * getattr(cfg, "batch_size", 64), args.seed)
        fisher_ds = TranslationDataset(
            os.path.join(paths["data"], "ft_train.tsv.gz"), sp, max_subword_len=cfg.max_len)
        fisher_ds.items = []
        for zh, en in gen_pairs:
            fisher_ds.items.append((fisher_ds.tag["en"], zh, en, "zh-en"))
            fisher_ds.items.append((fisher_ds.tag["zh"], en, zh, "en-zh"))
        fisher_ds._lengths = None
        fisher_loader = DataLoader(fisher_ds, batch_size=getattr(cfg, "batch_size", 64),
                                   shuffle=False, collate_fn=collate)
        print(f"[ewc] computing fisher on {len(gen_pairs):,} general pairs")
        fisher = estimate_fisher(model, fisher_loader, criterion, device, cfg, use_amp,
                                 args.fisher_batches)

    for epoch in range(start_epoch, cfg.epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        touch_lock(lock, args.owner)
        reset_peak_vram()
        with Timer() as t:
            if args.method == "ewc":
                tr_loss, _ = run_epoch_ewc(model, train_loader, criterion, ce_for_ppl,
                                           device, cfg, optimizer, fisher, star_params,
                                           args.ewc_lambda, use_amp, scaler)
            else:
                tr_loss, _ = run_epoch(model, train_loader, criterion, ce_for_ppl, device,
                                       cfg, optimizer, scheduler, tf_ratio=1.0, train=True,
                                       scaler=scaler, use_amp=use_amp)
        dev_loss, dev_ce = run_epoch(model, dev_loader, criterion, ce_for_ppl, device,
                                     cfg, train=False, scaler=scaler, use_amp=use_amp)
        ppl = math.exp(min(dev_ce, 20))
        improved = dev_loss < best_dev - 1e-4
        if improved:
            best_dev, bad_epochs = dev_loss, 0
        else:
            bad_epochs += 1

        logger.log({"arch": ARCH + "_ft", "seed": args.seed, "epoch": epoch,
                    "train_loss": round(tr_loss, 4), "dev_loss": round(dev_loss, 4),
                    "dev_perplexity": round(ppl, 3), "epoch_time_s": round(t.elapsed, 2),
                    "lr": cfg.lr, "params_M": human_millions(count_params(model))})
        print(f"[ft] epoch {epoch}: train {tr_loss:.4f}  dev {dev_loss:.4f} "
              f"(ppl {ppl:.2f})  {'*' if improved else f'{bad_epochs}/{cfg.patience}'}")

        if improved:
            atomic_save({"model_state": model.state_dict(), "arch": ARCH,
                         "cfg": cfg.to_dict(), "vocab_size": vocab_size,
                         "seed": args.seed, "epoch": epoch, "dev_loss": dev_loss,
                         "finetuned": True, "ft_lr": cfg.lr, "ft_patience": cfg.patience,
                         "general_mix": args.general_mix, "method": args.method,
                         "ewc_lambda": args.ewc_lambda, "run_tag": tag,
                         "base_ckpt": os.path.basename(base_ckpt)}, best_path)
        atomic_save({"model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "scaler_state": scaler.state_dict() if use_amp else None,
                     "rng": capture_rng(), "arch": ARCH, "cfg": cfg.to_dict(),
                     "vocab_size": vocab_size, "seed": args.seed, "epoch": epoch,
                     "best_dev": best_dev, "bad_epochs": bad_epochs,
                     "owner": args.owner, "finetuned": True, "ft_lr": cfg.lr,
                     "ft_patience": cfg.patience, "general_mix": args.general_mix,
                     "method": args.method, "ewc_lambda": args.ewc_lambda,
                     "run_tag": tag}, last_path)

        if bad_epochs >= cfg.patience:
            print(f"[ft] early stop at epoch {epoch}")
            break

    release_lock(lock)
    print(f"[ft] done, best dev {best_dev:.4f} -> {best_path}")
    print("[ft] todo: eval on hosp test + rerun wmt/flores (if general drops >1pt, lower lr)")


if __name__ == "__main__":
    main()
