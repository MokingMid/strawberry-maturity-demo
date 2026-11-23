"""对比多个分类器 checkpoint 在两个数据集 test 集上的表现。

用途：demo 要在"网络特写域"和"温室大棚域"之间都能工作，
需要知道哪个 checkpoint 的跨域表现最好，而不是只看单域指标。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "handoff_2026-11-22"))

from strawberry_cv.constants import CLASS_NAMES  # noqa: E402
from strawberry_cv.models.classifier import Classifier  # noqa: E402

PKG = Path(r"E:\临时\handoff_2026-11-22")

CHECKPOINTS = [
    ("v1 原分类器", PKG / "work/runs/classifier/resnet18_3cls/best.pt"),
    ("归档集分类器", PKG / "work_archive/runs/classifier/resnet18_3cls_archive/best.pt"),
    ("合并分类器", PKG / "work_merged/runs/classifier/resnet18_3cls_merged/best.pt"),
]

EVAL_SETS = [
    ("v1 test", PKG / "work/roi_classification/test"),
    ("归档 test", PKG / "work_archive/roi_classification/test"),
    ("合并 test", PKG / "work_merged/roi_classification/test"),
]


def evaluate(ckpt: Path, data_dir: Path) -> dict | None:
    if not ckpt.exists() or not data_dir.is_dir():
        return None
    clf = Classifier.from_checkpoint(ckpt, device="cpu")
    return clf.evaluate(data_dir, batch=32, workers=0)


def main() -> None:
    rows = []
    for name, ckpt in CHECKPOINTS:
        if not ckpt.exists():
            print(f"[skip] {name}: {ckpt.name} 不存在")
            continue
        for set_name, data_dir in EVAL_SETS:
            metrics = evaluate(ckpt, data_dir)
            if metrics is None:
                continue
            per = metrics.get("per_class", {})
            rows.append({
                "model": name,
                "eval_set": set_name,
                "acc": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "half_f1": per.get("half", {}).get("f1", 0.0),
                "half_support": per.get("half", {}).get("support", 0),
                "n": metrics.get("count", 0),
            })
            print(f"{name:<14} {set_name:<10} acc={metrics['accuracy']:.3f} "
                  f"macroF1={metrics['macro_f1']:.3f} "
                  f"halfF1={per.get('half', {}).get('f1', 0.0):.3f} "
                  f"(n={metrics.get('count', 0)})")

    out = Path(r"E:\临时\demo_out\classifier_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
