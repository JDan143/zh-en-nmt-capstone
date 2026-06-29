"""
utils.py — small shared helpers used across the harness.
Kept dependency-light (torch is imported lazily where needed).
"""
from __future__ import annotations
import csv
import os
import random
import time
from typing import Any

import numpy as np


def seed_everything(seed: int = 123) -> None:
    """Seed python, numpy and torch; make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_millions(n: int) -> float:
    return round(n / 1e6, 3)


class MetricsLogger:
    """Append-only CSV logger (one row per epoch). Header written once."""

    def __init__(self, path: str):
        self.path = path
        self._fieldnames: list[str] | None = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        write_header = not os.path.exists(self.path)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self._fieldnames})


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0


def peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    except Exception:
        pass
    return 0.0


def reset_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
