from __future__ import annotations
import argparse
import importlib
import math
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from utils import (seed_everything, get_device, count_params, human_millions,
                   MetricsLogger, Timer, reset_peak_vram, peak_vram_gb)
from data.dataset import TranslationDataset, collate, MaxTokensBatchSampler
from models.common import build_criterion, build_optimizer_scheduler
from train import (run_epoch, tf_ratio_for_epoch, capture_rng, restore_rng,
                   load_tokenizer, MODEL_MODULES)

import unidirectional as U


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    on_drive = "/drive/" in path or "/gdrive/" in path or "MyDrive" in path
    if not on_drive:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            torch.save(obj, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        return
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir="/tmp") as tf:
        local = tf.name
        torch.save(obj, tf); tf.flush(); os.fsync(tf.fileno())
    try:
        shutil.copyfile(local, path + ".part")
        if os.path.exists(path):
            os.remove(path)
        os.rename(path + ".part", path)
    finally:
        try:
            os.remove(local)
        except OSError:
            pass


def make_unidirectional_loaders(cfg, sp, pair):
    """Same as train.make_loaders, but the dataset is restricted to ONE direction.

    `pair` is the dataset direction string ("en-zh" or "zh-en"). The dataset yields each
    sentence pair ONCE in that direction (vs the bidirectional dataset which yields it in
    BOTH). Identical tokenization, tagging, batching, and max-length handling otherwise.
    """
    max_sub = cfg.max_len if cfg.family == "transformer" else C.MAX_TOKENS_PER_SIDE
    train_ds = TranslationDataset(os.path.join(C.DATA_DIR, "train.tsv.gz"), sp,
                                  max_subword_len=max_sub, directions=(pair,))
    dev_ds = TranslationDataset(os.path.join(C.DATA_DIR, "dev.tsv.gz"), sp,
                                max_subword_len=max_sub, directions=(pair,))
    if cfg.batch_by_tokens:
        sampler = MaxTokensBatchSampler(train_ds, cfg.max_tokens, shuffle=True, seed=C.SEED)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  collate_fn=collate, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate)
    return train_loader, dev_loader


def main():
    ap = argparse.ArgumentParser(description="Unidirectional ablation training (test/).")
    ap.add_argument("--arch", required=True, choices=U.ARCHS,
                    help="arch4_transformer or arch5_improved_transformer (this ablation only)")
    ap.add_argument("--direction", required=True, choices=U.DIRECTIONS,
                    help="enzh (English->Chinese) or zhen (Chinese->English)")
    ap.add_argument("--seed", type=int, default=191,
                    help="fixed at 191 to match the study's seed-191 bidirectional numbers")
    ap.add_argument("--owner", default=os.environ.get("SPEECHBRIDGE_OWNER", "unknown"),
                    help="recorded in the checkpoint + handoff banner (who trained)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing 'last' checkpoint and restart")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/fast end-to-end run (or set SPEECHBRIDGE_SMOKE=1)")
    args = ap.parse_args()
    U.check_arch(args.arch); U.check_direction(args.direction)
    if args.smoke:
        os.environ["SPEECHBRIDGE_SMOKE"] = "1"
        importlib.reload(C)

    pair = U.DIRECTION_TO_PAIR[args.direction]      # "en-zh" / "zh-en"
    paths = U.direction_paths(args.direction)
    seed_everything(args.seed)
    device = get_device()
    cfg = C.get_arch_config(args.arch)              # IDENTICAL to the study's cfg
    print(f"[uni] arch={args.arch} direction={args.direction} ({pair}) "
          f"seed={args.seed} device={device} smoke={C.SMOKE}")
    print(f"[uni] data={paths['data']}  ckpt={paths['ckpt']}")

    sp = load_tokenizer()                           # SHARED joint 32k tokenizer (reused)
    vocab_size = sp.get_piece_size()
    train_loader, dev_loader = make_unidirectional_loaders(cfg, sp, pair)
    train_sampler = getattr(train_loader, "batch_sampler", None)
    print(f"[uni] train examples={len(train_loader.dataset):,} "
          f"dev examples={len(dev_loader.dataset):,} (one direction only)")

    mod = importlib.import_module(MODEL_MODULES[args.arch])
    model = mod.build(cfg, vocab_size).to(device)
    print(f"[uni] params = {human_millions(count_params(model))} M")

    criterion = build_criterion(vocab_size, cfg.label_smoothing)
    ce_for_ppl = nn.CrossEntropyLoss(ignore_index=C.PAD_ID)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    use_amp = device.type == "cuda" and cfg.amp_safe
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[uni] AMP {'ON' if use_amp else ('off (cpu)' if device.type != 'cuda' else 'off (fp32 for this arch)')}")

    st = U.stem(args.arch, args.direction, args.seed)
    logger = MetricsLogger(os.path.join(paths["results"], f"metrics_log_{st}.csv"))
    best_path = os.path.join(paths["ckpt"], f"{st}_best.pt")
    last_path = os.path.join(paths["ckpt"], f"{st}_last.pt")
    max_batches = 30 if C.SMOKE else None

    start_epoch, best_dev, bad_epochs = 0, math.inf, 0
    if os.path.exists(last_path) and not args.fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if scheduler is not None and ck.get("scheduler_state") is not None:
            scheduler.load_state_dict(ck["scheduler_state"])
        if ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        restore_rng(ck["rng"])
        start_epoch = ck["epoch"] + 1
        best_dev = ck["best_dev"]
        bad_epochs = ck["bad_epochs"]
        prev = ck.get("owner", "?")
        print("\n" + "=" * 66)
        print(f"  RELAY HANDOFF — resuming {args.arch} {args.direction} seed {args.seed}")
        print(f"    previous session by : {prev}")
        print(f"    this session by     : {args.owner}")
        print(f"    resuming at epoch   : {start_epoch}  best dev {best_dev:.4f} "
              f"(bad {bad_epochs}/{cfg.patience})")
        print("=" * 66 + "\n")
    else:
        print(f"[uni] fresh start by {args.owner} (no checkpoint at {last_path})")

    for epoch in range(start_epoch, cfg.epochs):
        tf = tf_ratio_for_epoch(cfg, epoch)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        reset_peak_vram()
        with Timer() as t:
            tr_loss, _ = run_epoch(model, train_loader, criterion, ce_for_ppl, device,
                                   cfg, optimizer, scheduler, tf_ratio=tf, train=True,
                                   max_batches=max_batches, scaler=scaler, use_amp=use_amp)
        dev_loss, dev_ce = run_epoch(model, dev_loader, criterion, ce_for_ppl, device,
                                     cfg, train=False, max_batches=max_batches,
                                     scaler=scaler, use_amp=use_amp)
        dev_ppl = math.exp(min(dev_ce, 20))
        if scheduler is not None and cfg.scheduler == "plateau":
            scheduler.step(dev_loss)

        improved = dev_loss < best_dev - 1e-4
        if improved:
            best_dev, bad_epochs = dev_loss, 0
        else:
            bad_epochs += 1

        logger.log({
            "arch": args.arch, "direction": args.direction, "seed": args.seed,
            "epoch": epoch, "tf_ratio": round(tf, 3),
            "train_loss": round(tr_loss, 4), "dev_loss": round(dev_loss, 4),
            "dev_perplexity": round(dev_ppl, 3),
            "epoch_time_s": round(t.elapsed, 2), "peak_vram_gb": peak_vram_gb(),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "params_M": human_millions(count_params(model)),
        })
        print(f"[epoch {epoch}] train={tr_loss:.4f} dev={dev_loss:.4f} "
              f"ppl={dev_ppl:.2f} time={t.elapsed:.1f}s vram={peak_vram_gb()}GB"
              f"{'  *best' if improved else ''}")

        if improved:
            atomic_save({"model_state": model.state_dict(), "arch": args.arch,
                         "direction": args.direction, "cfg": cfg.to_dict(),
                         "vocab_size": vocab_size, "seed": args.seed,
                         "epoch": epoch, "dev_loss": dev_loss}, best_path)
        atomic_save({"model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict() if scheduler else None,
                     "scaler_state": scaler.state_dict() if use_amp else None,
                     "rng": capture_rng(), "epoch": epoch,
                     "best_dev": best_dev, "bad_epochs": bad_epochs,
                     "arch": args.arch, "direction": args.direction,
                     "cfg": cfg.to_dict(), "vocab_size": vocab_size,
                     "seed": args.seed, "owner": args.owner}, last_path)

        if bad_epochs >= cfg.patience:
            print(f"[early-stop] no dev improvement for {cfg.patience} epochs")
            break

    print(f"[done] {args.arch} {args.direction} best dev_loss={best_dev:.4f}  best={best_path}")


if __name__ == "__main__":
    main()
