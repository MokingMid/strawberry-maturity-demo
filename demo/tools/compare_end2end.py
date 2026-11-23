"""端到端对比：多个分类器 × 多个数据集，走完整流水线（检测+分类）。

这是 demo 实际面对的场景（真实检测框 + 分类），比纯分类评测更贴近用户视角。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "handoff_2026-11-22"))

from demo_core import MaturityDemo, list_images, load_gt_boxes, read_image  # noqa: E402

ROOT = Path(r"E:\临时")
PKG = ROOT / "handoff_2026-11-22"

MODELS = [
    ("v1 原分类器", PKG / "work/runs/classifier/resnet18_3cls/best.pt"),
    ("归档集分类器", PKG / "work_archive/runs/classifier/resnet18_3cls_archive/best.pt"),
    ("合并分类器", PKG / "work_merged/runs/classifier/resnet18_3cls_merged/best.pt"),
]

DATASETS = [
    ("v1", ROOT / "datasets/strawberry_v1/detection/images",
     ROOT / "datasets/strawberry_v1/detection/labels"),
    ("归档", ROOT / "datasets/strawberry_archive_v1/detection/images",
     ROOT / "datasets/strawberry_archive_v1/detection/labels"),
]

CLASSES = ("ripe", "half", "unripe")
DETECTOR = PKG / "work/runs/detector/strawberry_1cls_lowlr/weights/best.pt"
N_IMAGES = 30


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def run(model_name: str, ckpt: Path, ds_name: str, images_dir: Path, labels_dir: Path):
    demo = MaturityDemo(detector_weights=DETECTOR, classifier_checkpoint=ckpt,
                         conf=0.55, device="cpu")
    confusion: Counter = Counter()
    det_tp = det_fp = det_fn = 0
    latencies = []
    images = list_images(images_dir)[:N_IMAGES]
    for path in images:
        lp = labels_dir / f"{path.stem}.json"
        if not lp.exists():
            continue
        image = read_image(path)
        gt = load_gt_boxes(lp)
        report = demo.analyze(image, source=path.name)
        latencies.append(report.latency_ms)
        matched = set()
        for fruit in report.fruits:
            best, best_iou = None, 0.3
            for i, g in enumerate(gt):
                if i in matched:
                    continue
                v = iou(fruit.box, g["box"])
                if v > best_iou:
                    best, best_iou = i, v
            if best is None:
                det_fp += 1
                continue
            matched.add(best)
            det_tp += 1
            confusion[(gt[best]["name"], fruit.state)] += 1
        det_fn += len(gt) - len(matched)

    total = sum(confusion.values())
    correct = sum(confusion[(c, c)] for c in CLASSES)
    p = det_tp / max(1, det_tp + det_fp)
    r = det_tp / max(1, det_tp + det_fn)
    print(f"{model_name:<14} {ds_name:<6} 状态正确 {correct/max(1,total)*100:5.1f}% "
          f"({correct}/{total}) | 检测 P {p:.3f} R {r:.3f} "
          f"(TP{det_tp} FP{det_fp} FN{det_fn}) | {np.mean(latencies):.0f}ms/图")
    return {
        "model": model_name, "dataset": ds_name, "images": len(images),
        "state_acc": correct / max(1, total), "state_correct": correct,
        "state_total": total,
        "det_precision": p, "det_recall": r,
        "det_tp": det_tp, "det_fp": det_fp, "det_fn": det_fn,
        "confusion": {f"{t}->{pr}": n for (t, pr), n in confusion.items()},
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def main() -> None:
    rows = []
    for model_name, ckpt in MODELS:
        if not ckpt.exists():
            print(f"[skip] {model_name}")
            continue
        for ds_name, images_dir, labels_dir in DATASETS:
            rows.append(run(model_name, ckpt, ds_name, images_dir, labels_dir))
    out = Path(r"E:\临时\demo_out\end2end_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
