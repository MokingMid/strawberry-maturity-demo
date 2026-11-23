"""验证 demo 的 GT↔预测配对正确性。

构造一张有 3 个框的图，其中一个框太小会被 crop_roi 跳过；
若配对按顺序 zip，跳过之后会发生错位；用 gt_state 配对则不会。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_core import MaturityDemo  # noqa: E402

demo = MaturityDemo(detector_weights=None, classifier_checkpoint=None)

img = np.full((300, 400, 3), 30, dtype=np.uint8)
gt = [
    {"box": (10, 10, 120, 120), "score": 1.0, "name": "ripe"},
    {"box": (200, 200, 208, 208), "score": 1.0, "name": "half"},   # 太小，会被跳过
    {"box": (250, 30, 360, 140), "score": 1.0, "name": "unripe"},
]

report = demo.analyze(img, gt_boxes=gt)
print(f"GT 框数 = {len(gt)}，产出 fruit 数 = {len(report.fruits)}（小框被跳过 → 少 1）")
for f in report.fruits:
    print(f"  box={tuple(int(v) for v in f.box)}  gt_state={f.gt_state}")
assert len(report.fruits) == 2, report.fruits
assert report.fruits[0].gt_state == "ripe"
assert report.fruits[1].gt_state == "unripe", report.fruits[1]
print("OK：gt_state 配对正确，跨过被跳过的框也不会错位")
