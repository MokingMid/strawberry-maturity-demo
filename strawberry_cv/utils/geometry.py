"""框几何工具：统一使用 xyxy 绝对坐标（像素，x2/y2 为不含边界）。"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Box = tuple[float, float, float, float]


def xyxy_to_cxcywh(box: Box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Box:
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), float(width))
    y1 = min(max(0.0, y1), float(height))
    x2 = min(max(0.0, x2), float(width))
    y2 = min(max(0.0, y2), float(height))
    return x1, y1, x2, y2


def expand_box(box: Box, ratio: float) -> Box:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * ratio, h * ratio
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def square_box(box: Box, width: int | None = None, height: int | None = None) -> Box:
    """把框补成正方形（以中心为准），可选地裁到图像内。"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1)
    box = cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0
    if width is not None and height is not None:
        box = clip_box(box, width, height)
    return box


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(boxes_a: Sequence[Box], boxes_b: Sequence[Box]) -> list[list[float]]:
    return [[iou(a, b) for b in boxes_b] for a in boxes_a]


def center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def line_side(point: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """点在有向线段的哪一侧：叉积符号。"""
    px, py = point
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def greedy_match(
    pred_boxes: Sequence[Box],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Box],
    iou_thr: float = 0.5,
    pred_classes: Sequence[int] | None = None,
    gt_classes: Sequence[int] | None = None,
    class_aware: bool = False,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """按分数从高到低贪心匹配。

    返回 (matches, unmatched_pred_indices, unmatched_gt_indices)。
    matches 的元素是 (pred_idx, gt_idx, iou)。
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: float(pred_scores[i]), reverse=True)
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    unmatched_pred: list[int] = []

    for pi in order:
        best_gi, best_iou = -1, 0.0
        for gi, gt in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            if class_aware and pred_classes is not None and gt_classes is not None:
                if int(pred_classes[pi]) != int(gt_classes[gi]):
                    continue
            value = iou(pred_boxes[pi], gt)
            if value > best_iou:
                best_iou, best_gi = value, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            used_gt.add(best_gi)
            matches.append((pi, best_gi, best_iou))
        else:
            unmatched_pred.append(pi)

    unmatched_gt = [gi for gi in range(len(gt_boxes)) if gi not in used_gt]
    return matches, unmatched_pred, unmatched_gt


def intersects_any(box: Box, others: Iterable[Box], thr: float = 0.1) -> bool:
    return any(iou(box, other) > thr for other in others)
