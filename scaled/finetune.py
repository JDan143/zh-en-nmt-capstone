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

import train
from train import (run_epoch, atomic_save, acquire_lock, touch_lock, release_lock,
                   load_tokenizer, capture_rng, restore_rng)

train.MODEL_MODULES["arch5_improved_transformer"] = "models.arch5_improved_transformer"

ARCH = "arch5_improved_transformer"


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
    print(f"[ft] config loaded from {os.path.abspath(C.__file__)}")
    missing = []
    if not os.path.exists(base_ckpt):
        missing.append(f"base model not found: {base_ckpt}")
    for name in ("ft_train.tsv.gz", "ft_dev.tsv.gz"):
        if not os.path.exists(os.path.join(paths["data"], name)):
            missing.append(f"missing {name} in {paths['data']}")
    tok = C.TOKENIZER_PREFIX + ".model"
    if not os.path.exists(tok):
        missing.append(f"tokenizer not found: {tok}")
    if missing:
        raise SystemExit("[ft] fix these first:\n  - " + "\n  - ".join(missing))
    if not os.path.exists(os.path.join(paths["data"], "ft_test.tsv.gz")):
        print("[ft] heads up: no ft_test.tsv.gz yet. Training will run, but you need a "
              "test set to measure the hospitality gain later.")


def make_ft_loaders(cfg, sp, paths):
    tr = os.path.join(paths["data"], "ft_train.tsv.gz")
    dv = os.path.join(paths["data"], "ft_dev.tsv.gz")
    train_ds = TranslationDataset(tr, sp, max_subword_len=cfg.max_len)
    dev_ds = TranslationDataset(dv, sp, max_subword_len=cfg.max_len)
    print(f"[ft] train pairs {len(train_ds):,}  dev pairs {len(dev_ds):,}")
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
    # Fine-tune overrides: constant low LR, no warmup, no teacher-forcing anneal.
    cfg.scheduler = "none"
    cfg.lr = args.lr
    cfg.epochs = args.epochs
    cfg.patience = args.patience
    cfg.warmup_steps = 0
    cfg.tf_decay_epochs = 0

    sp = load_tokenizer()
    vocab_size = sp.get_piece_size()
    train_loader, dev_loader = make_ft_loaders(cfg, sp, paths)
    train_sampler = getattr(train_loader, "batch_sampler", None)

    mod = importlib.import_module(train.MODEL_MODULES[ARCH])
    model = mod.build(cfg, vocab_size).to(device)
    print(f"[ft] Arch 5, {human_millions(count_params(model))}M params")

    criterion = build_criterion(vocab_size, cfg.label_smoothing)
    ce_for_ppl = nn.CrossEntropyLoss(ignore_index=C.PAD_ID)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = None

    use_amp = device.type == "cuda" and cfg.amp_safe
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[ft] AMP {'on' if use_amp else 'off'} | constant LR {cfg.lr} | patience {cfg.patience}")

    os.environ["SPEECHBRIDGE_RESULTS"] = paths["results"]
    logger = MetricsLogger(os.path.join(paths["results"], f"metrics_log_ft_arch5_seed{args.seed}.csv"))
    best_path = os.path.join(paths["ckpt"], f"{ARCH}_ft_seed{args.seed}_best.pt")
    last_path = os.path.join(paths["ckpt"], f"{ARCH}_ft_seed{args.seed}_last.pt")

    lock = acquire_lock(ARCH + "_ft", args.seed, args.owner, force=args.force)

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
        print("\n" + "=" * 60)
        print(f"  Picking up the fine-tune from {ck.get('owner', 'someone')}")
        print(f"  now training as: {args.owner}")
        print(f"  resuming at epoch {start_epoch}, best dev {best_dev:.4f} "
              f"(bad {bad_epochs}/{cfg.patience})")
        print("=" * 60 + "\n")
    else:
        base = torch.load(base_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(base["model_state"])
        print(f"[ft] started from the base model: {base_ckpt} "
              f"(pretrain epoch {base.get('epoch')}, dev {base.get('dev_loss')})")
        base_dev, base_ce = run_epoch(model, dev_loader, criterion, ce_for_ppl, device,
                                      cfg, train=False, scaler=scaler, use_amp=use_amp)
        print(f"[ft] hospitality dev before any adapting: loss {base_dev:.4f} "
              f"(ppl {math.exp(min(base_ce, 20)):.2f}). That is the number to beat.")

    for epoch in range(start_epoch, cfg.epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        touch_lock(lock, args.owner)
        reset_peak_vram()
        with Timer() as t:
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
              f"(ppl {ppl:.2f})  {'new best' if improved else f'no gain {bad_epochs}/{cfg.patience}'}")

        if improved:
            atomic_save({"model_state": model.state_dict(), "arch": ARCH,
                         "cfg": cfg.to_dict(), "vocab_size": vocab_size,
                         "seed": args.seed, "epoch": epoch, "dev_loss": dev_loss,
                         "finetuned": True,
                         "base_ckpt": os.path.basename(base_ckpt)}, best_path)
        atomic_save({"model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "scaler_state": scaler.state_dict() if use_amp else None,
                     "rng": capture_rng(), "arch": ARCH, "cfg": cfg.to_dict(),
                     "vocab_size": vocab_size, "seed": args.seed, "epoch": epoch,
                     "best_dev": best_dev, "bad_epochs": bad_epochs,
                     "owner": args.owner, "finetuned": True}, last_path)

        if bad_epochs >= cfg.patience:
            print(f"[ft] stopping. Dev loss has not improved for {cfg.patience} epochs.")
            break

    release_lock(lock)
    print(f"[ft] done. best dev {best_dev:.4f}, saved to {best_path}")
    print("[ft] next: evaluate this on the hospitality test set and re-run WMT/FLORES on "
          "it. You want the hospitality score up and the general score about the same. If "
          "WMT/FLORES dropped more than about a point, lower --lr and run again.")


if __name__ == "__main__":
    main()
