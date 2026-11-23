"""检测评测：P/R/F1、AP、匹配框混淆矩阵。

不依赖 Ultralytics，可用合成预测/GT 做单元测试；训练后的官方指标仍以
Ultralytics val() 的 mAP 为准，这里提供统一的、可解释的补充口径。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..utils.geometry import Box, greedy_match


def detection_metrics(
    pred_boxes: list[Box],
    pred_scores: list[float],
    gt_boxes: list[Box],
    iou_thr: float = 0.5,
    pred_classes: list[int] | None = None,
    gt_classes: list[int] | None = None,
    class_aware: bool = False,
    class_names: tuple[str, ...] = (),
) -> dict:
    matches, unmatched_pred, unmatched_gt = greedy_match(
        pred_boxes, pred_scores, gt_boxes, iou_thr,
        pred_classes=pred_classes, gt_classes=gt_classes, class_aware=class_aware,
    )
    tp, fp, fn = len(matches), len(unmatched_pred), len(unmatched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_class = {}
    if class_aware and pred_classes is not None and gt_classes is not None:
        ids = sorted(set(gt_classes) | set(pred_classes))
        for cid in ids:
            g = [i for i, c in enumerate(gt_classes) if c == cid]
            p = [i for i, c in enumerate(pred_classes) if c == cid]
            m, up, ug = greedy_match(
                [pred_boxes[i] for i in p], [pred_scores[i] for i in p],
                [gt_boxes[i] for i in g], iou_thr, class_aware=True,
                pred_classes=[pred_classes[i] for i in p],
                gt_classes=[gt_classes[i] for i in g],
            )
            ctp, cfp, cfn = len(m), len(up), len(ug)
            cp = ctp / (ctp + cfp) if ctp + cfp else 0.0
            cr = ctp / (ctp + cfn) if ctp + cfn else 0.0
            name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            per_class[name] = {
                "tp": ctp, "fp": cfp, "fn": cfn,
                "precision": cp, "recall": cr,
                "f1": 2 * cp * cr / (cp + cr) if cp + cr else 0.0,
            }

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "iou_thr": iou_thr, "class_aware": class_aware,
        "per_class": per_class,
        "matches": [(int(pi), int(gi), float(v)) for pi, gi, v in matches],
    }


def average_precision(
    pred_boxes: list[Box],
    pred_scores: list[float],
    gt_boxes: list[Box],
    iou_thr: float = 0.5,
    pred_classes: list[int] | None = None,
    gt_classes: list[int] | None = None,
    class_id: int | None = None,
) -> float:
    """单类 AP（all-point 插值）。class_id=None 时把所有框视为同一类。"""
    if class_id is not None:
        if pred_classes is None or gt_classes is None:
            raise ValueError("指定 class_id 时必须提供 pred_classes / gt_classes")
        pidx = [i for i, c in enumerate(pred_classes) if c == class_id]
        gidx = [i for i, c in enumerate(gt_classes) if c == class_id]
    else:
        pidx = list(range(len(pred_boxes)))
        gidx = list(range(len(gt_boxes)))
    if not gidx:
        return 0.0

    order = sorted(pidx, key=lambda i: float(pred_scores[i]), reverse=True)
    used: set[int] = set()
    tp = []
    fp = []
    for pi in order:
        best_gi, best_iou = -1, 0.0
        for gi in gidx:
            if gi in used:
                continue
            from ..utils.geometry import iou

            value = iou(pred_boxes[pi], gt_boxes[gi])
            if value > best_iou:
                best_iou, best_gi = value, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            used.add(best_gi)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if not tp:
        return 0.0
    tp_cum = np.cumsum(tp).astype(float)
    fp_cum = np.cumsum(fp).astype(float)
    recall = tp_cum / len(gidx)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # precision envelope
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_average_precision(
    pred_boxes: list[Box],
    pred_scores: list[float],
    gt_boxes: list[Box],
    pred_classes: list[int],
    gt_classes: list[int],
    class_ids: list[int],
    iou_thr: float = 0.5,
) -> dict:
    aps = {
        cid: average_precision(
            pred_boxes, pred_scores, gt_boxes, iou_thr,
            pred_classes, gt_classes, class_id=cid,
        )
        for cid in class_ids
    }
    return {"ap": aps, "map": float(np.mean(list(aps.values()))) if aps else 0.0}


def confusion_matrix_matched(
    pred_boxes: list[Box],
    pred_scores: list[float],
    pred_classes: list[int],
    gt_boxes: list[Box],
    gt_classes: list[int],
    num_classes: int = 3,
    iou_thr: float = 0.5,
) -> dict:
    """只统计匹配上的框：行=GT，列=预测。"""
    matches, unmatched_pred, unmatched_gt = greedy_match(
        pred_boxes, pred_scores, gt_boxes, iou_thr,
        pred_classes=pred_classes, gt_classes=gt_classes, class_aware=False,
    )
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pi, gi, _ in matches:
        matrix[int(gt_classes[gi]), int(pred_classes[pi])] += 1
    return {
        "matrix": matrix.tolist(),
        "matched": len(matches),
        "unmatched_pred": len(unmatched_pred),
        "unmatched_gt": len(unmatched_gt),
    }


def summarize_counts(counts: dict) -> str:
    labels = Counter()
    for key, value in counts.items():
        labels[key] = value
    return ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
