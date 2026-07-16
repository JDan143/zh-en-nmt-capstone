"""
train.py — one training harness for all five architectures, RESUMABLE.

    python train.py --arch arch1_gru
    python train.py --arch arch4_transformer --seed 1604
    SPEECHBRIDGE_SMOKE=1 python train.py --arch arch1_gru   # fast end-to-end check

Resumability (important on free Colab/Kaggle, where sessions end at ~12h):
  * After every epoch a FULL "last" checkpoint is written atomically:
    model + optimizer + scheduler + epoch + best_dev + bad_epochs + RNG states.
  * On startup the harness auto-resumes from that last checkpoint if present
    (disable with --fresh). Resume granularity is one epoch: if a session dies
    mid-epoch you re-run the same cell and it continues from the last COMPLETED
    epoch. Both checkpoints live in CKPT_DIR (point this at Drive to persist).

Everything else is identical across architectures (shared tokenizer, bidirectional
data, token/sentence batching, teacher-forcing annealing for RNNs, grad
accumulation + clip, Noam/plateau scheduling, dev-loss early stopping).
"""
from __future__ import annotations
import argparse
import importlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from utils import (seed_everything, get_device, count_params, human_millions,  # noqa: E402
                   MetricsLogger, Timer, peak_vram_gb, reset_peak_vram)
from data.dataset import TranslationDataset, collate, MaxTokensBatchSampler  # noqa: E402
from models.common import build_criterion, build_optimizer_scheduler  # noqa: E402

MODEL_MODULES = {
    "arch4_transformer": "models.arch4_transformer",
}


def load_tokenizer():
    import sentencepiece as spm
    model_path = C.TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} missing. Run data/train_tokenizer.py first.")
    return spm.SentencePieceProcessor(model_file=model_path)


def make_loaders(cfg, sp):
    max_sub = cfg.max_len if cfg.family == "transformer" else C.MAX_TOKENS_PER_SIDE
    train_ds = TranslationDataset(os.path.join(C.DATA_DIR, "train.tsv.gz"), sp,
                                  max_subword_len=max_sub)
    dev_ds = TranslationDataset(os.path.join(C.DATA_DIR, "dev.tsv.gz"), sp,
                                max_subword_len=max_sub)
    if cfg.batch_by_tokens:
        sampler = MaxTokensBatchSampler(train_ds, cfg.max_tokens, shuffle=True, seed=C.SEED)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  collate_fn=collate, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate)
    return train_loader, dev_loader


def tf_ratio_for_epoch(cfg, epoch: int) -> float:
    if not cfg.teacher_forcing:
        return 1.0
    horizon = max(cfg.tf_decay_epochs, 1)
    frac = min(epoch / horizon, 1.0)
    return cfg.tf_start + (cfg.tf_end - cfg.tf_start) * frac


# ─────────────────────────── RNG capture/restore (resume) ───────────────────
def capture_rng():
    s = {"python": random.getstate(),
         "numpy": np.random.get_state(),
         "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and s.get("cuda") is not None:
        torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in s["cuda"]])


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


# ───────────────────────── multi-member relay lock ──────────────────────────
def _lock_path(arch, seed):
    return os.path.join(C.CKPT_DIR, f"{arch}_seed{seed}.lock")


def acquire_lock(arch, seed, owner, force=False, stale_hours=6.0):
    p = _lock_path(arch, seed)
    if os.path.exists(p) and not force:
        try:
            info = json.load(open(p, encoding="utf-8"))
            age_h = (time.time() - info.get("ts", 0)) / 3600.0
            if age_h < stale_hours:
                raise SystemExit(
                    f"\n[LOCKED] '{info.get('owner')}' started this run {age_h:.1f}h ago.\n"
                    f"  Two members training {arch} seed {seed} at once WILL corrupt the\n"
                    f"  checkpoints. Coordinate first. If they have definitely stopped,\n"
                    f"  re-run with --force.\n")
            print(f"[lock] stale lock from '{info.get('owner')}' ({age_h:.1f}h) — taking over")
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            print("[lock] unreadable lock file; taking over")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"owner": owner, "ts": time.time(),
                   "started": datetime.now().isoformat(timespec="seconds")}, f)
    return p


def touch_lock(p, owner):
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"owner": owner, "ts": time.time(),
                       "started": datetime.now().isoformat(timespec="seconds")}, f)
    except Exception:  # noqa: BLE001
        pass


def release_lock(p):
    try:
        os.remove(p)
    except OSError:
        pass


def run_epoch(model, loader, criterion, ce_for_ppl, device, cfg,
              optimizer=None, scheduler=None, tf_ratio=1.0, train=True,
              max_batches=None, scaler=None, use_amp=False):
    model.train(train)
    total_loss, total_ce, total_tok = 0.0, 0.0, 0
    accum = cfg.accumulation_steps if train else 1
    pending = 0  # micro-batches accumulated since the last optimizer step
    if train:
        optimizer.zero_grad()

    def optimizer_step():
        # unscale → clip → step → update (AMP-aware); also steps Noam per update
        if scaler is not None and use_amp:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        if scheduler is not None and cfg.scheduler == "noam":
            scheduler.step()
        optimizer.zero_grad()

    for i, (src, src_len, tgt, _refs, _dirs) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(src, src_len, tgt_in, tf_ratio=tf_ratio)
                V = logits.size(-1)
                loss = criterion(logits.reshape(-1, V), tgt_out.reshape(-1))
            with torch.no_grad():
                ce = ce_for_ppl(logits.float().reshape(-1, V), tgt_out.reshape(-1))
        if train:
            scaled = scaler.scale(loss / accum) if (scaler is not None and use_amp) else (loss / accum)
            scaled.backward()
            pending += 1
            if pending == accum:
                optimizer_step()
                pending = 0
        n_tok = (tgt_out != C.PAD_ID).sum().item()
        total_loss += loss.item() * n_tok
        total_ce += ce.item() * n_tok
        total_tok += n_tok
    # flush a partial accumulation group (batch count not divisible by accum, or an
    # early break) so its gradients update the model instead of being discarded.
    if train and pending > 0:
        optimizer_step()
    return total_loss / max(total_tok, 1), total_ce / max(total_tok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(MODEL_MODULES))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/fast end-to-end run (or set SPEECHBRIDGE_SMOKE=1)")
    ap.add_argument("--owner", default=os.environ.get("SPEECHBRIDGE_OWNER", "unknown"),
                    help="who is running this session (recorded in the lock + checkpoint)")
    ap.add_argument("--force", action="store_true",
                    help="take over an existing relay lock (only if the previous member "
                         "has definitely stopped)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing 'last' checkpoint and restart")
    args = ap.parse_args()
    if args.smoke:
        os.environ["SPEECHBRIDGE_SMOKE"] = "1"
        importlib.reload(C)

    seed_everything(args.seed)
    device = get_device()
    cfg = C.get_arch_config(args.arch)
    print(f"[train] arch={args.arch} seed={args.seed} device={device} smoke={C.SMOKE}")

    sp = load_tokenizer()
    vocab_size = sp.get_piece_size()
    train_loader, dev_loader = make_loaders(cfg, sp)
    train_sampler = getattr(train_loader, "batch_sampler", None)  # MaxTokensBatchSampler or None

    mod = importlib.import_module(MODEL_MODULES[args.arch])
    model = mod.build(cfg, vocab_size).to(device)
    print(f"[train] params = {human_millions(count_params(model))} M")

    criterion = build_criterion(vocab_size, cfg.label_smoothing)
    ce_for_ppl = nn.CrossEntropyLoss(ignore_index=C.PAD_ID)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    # Mixed precision (AMP): ~2x speed and ~half VRAM on a T4. Active only on CUDA
    # AND when the architecture is fp16-safe (ConvS2S is not — it overflows to NaN,
    # so it trains in fp32). On CPU this is a transparent no-op.
    use_amp = device.type == "cuda" and cfg.amp_safe
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[train] AMP {'ON' if use_amp else ('off (cpu)' if device.type != 'cuda' else 'off (fp32 for this arch)')}")

    # Per-seed log file so multiple runs (e.g. teammates splitting seeds for the
    # same arch into a shared Drive folder) never write the same CSV concurrently.
    logger = MetricsLogger(os.path.join(C.RESULTS_DIR, f"metrics_log_seed{args.seed}.csv"))
    best_path = os.path.join(C.CKPT_DIR, f"{args.arch}_seed{args.seed}_best.pt")
    last_path = os.path.join(C.CKPT_DIR, f"{args.arch}_seed{args.seed}_last.pt")
    max_batches = 30 if C.SMOKE else None

    lock = acquire_lock(args.arch, args.seed, args.owner, force=args.force)

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
        print("\n" + "=" * 68)
        print(f"  RELAY HANDOFF — resuming '{args.arch}' seed {args.seed}")
        print(f"    previous session by : {prev}")
        print(f"    this session by     : {args.owner}")
        print(f"    resuming at epoch   : {start_epoch}")
        print(f"    best dev so far     : {best_dev:.4f}  (bad epochs {bad_epochs}/{cfg.patience})")
        print("=" * 68 + "\n")
    else:
        print(f"[train] fresh start by {args.owner} (no checkpoint at {last_path})")

    for epoch in range(start_epoch, cfg.epochs):
        tf = tf_ratio_for_epoch(cfg, epoch)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)   # reshuffle token batches each epoch
        touch_lock(lock, args.owner)         # heartbeat so the lock isn't seen as stale
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
            "arch": args.arch, "seed": args.seed, "epoch": epoch,
            "tf_ratio": round(tf, 3),
            "train_loss": round(tr_loss, 4), "dev_loss": round(dev_loss, 4),
            "dev_perplexity": round(dev_ppl, 3),
            "epoch_time_s": round(t.elapsed, 2),
            "peak_vram_gb": peak_vram_gb(),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "params_M": human_millions(count_params(model)),
        })
        print(f"[epoch {epoch}] train={tr_loss:.4f} dev={dev_loss:.4f} "
              f"ppl={dev_ppl:.2f} time={t.elapsed:.1f}s vram={peak_vram_gb()}GB"
              f"{'  *best' if improved else ''}")

        # best checkpoint (for evaluation)
        if improved:
            atomic_save({"model_state": model.state_dict(), "arch": args.arch,
                         "cfg": cfg.to_dict(), "vocab_size": vocab_size,
                         "seed": args.seed, "epoch": epoch, "dev_loss": dev_loss},
                        best_path)
        # last checkpoint (for resuming) — full state, every epoch
        atomic_save({"model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict() if scheduler else None,
                     "scaler_state": scaler.state_dict() if use_amp else None,
                     "rng": capture_rng(), "epoch": epoch,
                     "best_dev": best_dev, "bad_epochs": bad_epochs,
                     "arch": args.arch, "cfg": cfg.to_dict(),
                     "vocab_size": vocab_size, "seed": args.seed,
                     "owner": args.owner}, last_path)

        if bad_epochs >= cfg.patience:
            print(f"[early-stop] no dev improvement for {cfg.patience} epochs")
            break

    release_lock(lock)
    print(f"[done] best dev_loss={best_dev:.4f}  best={best_path}")
    print("[relay] lock released — another member can now resume this run.")


if __name__ == "__main__":
    main()
