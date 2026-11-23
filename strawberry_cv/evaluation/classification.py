"""三分类评测：accuracy、macro F1、per-class P/R/F1、混淆矩阵。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("需要 scikit-learn：pip install scikit-learn") from exc

from ..constants import CLASS_NAMES


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = CLASS_NAMES,
) -> dict:
    labels = list(labels)
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")
    if not y_true:
        return {
            "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0,
            "labels": labels, "per_class": {},
            "confusion": [[0] * len(labels) for _ in labels], "support": 0,
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(
            np.average(f1, weights=support) if np.sum(support) else 0.0
        ),
        "labels": labels,
        "per_class": per_class,
        "confusion": cm.astype(int).tolist(),
        "support": int(len(y_true)),
    }


def top_confusions(cm: list[list[int]], labels: Sequence[str], k: int = 3) -> list[dict]:
    pairs = []
    for i, true_name in enumerate(labels):
        for j, pred_name in enumerate(labels):
            if i != j and cm[i][j] > 0:
                pairs.append({"true": true_name, "pred": pred_name, "count": int(cm[i][j])})
    return sorted(pairs, key=lambda x: x["count"], reverse=True)[:k]


def save_confusion_png(
    cm: list[list[int]],
    labels: Sequence[str],
    path: str | Path,
    title: str = "Confusion matrix",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(array, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(j, i, str(array[i, j]), ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
