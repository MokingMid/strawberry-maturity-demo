"""训练公共工具：随机种子、早停、目录统计。"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.0, mode: str = "max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.num_bad = 0
        self.stopped = False

    def step(self, value: float) -> bool:
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.stopped = True
        return self.stopped


def count_image_folder(root: str | Path) -> dict[str, int]:
    root = Path(root)
    if not root.is_dir():
        return {}
    counts: dict[str, int] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            counts[child.name] = sum(
                1 for p in child.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            )
    return counts


def class_balance_warnings(counts: dict[str, int], split: str = "train") -> list[str]:
    warnings: list[str] = []
    total = sum(counts.values())
    if total == 0:
        return [f"{split} 没有样本"]
    for name, count in counts.items():
        if count == 0:
            warnings.append(f"{split}/{name} 没有样本")
        elif count < 3:
            warnings.append(f"{split}/{name} 只有 {count} 张，指标不可靠")
    return warnings


def summarize_counts(counts: Counter | dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(dict(counts).items()))
