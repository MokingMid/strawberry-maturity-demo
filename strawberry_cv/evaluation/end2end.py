"""端到端评测：检测匹配 + 状态准确率 + 计数误差。"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..constants import CLASS_NAMES
from ..utils.geometry import Box, greedy_match


def end2end_metrics(
    preds_by_image: dict[str, list[dict]],
    gts_by_image: dict[str, list[dict]],
    iou_thr: float = 0.5,
    labels: tuple[str, ...] = CLASS_NAMES,
) -> dict:
    """preds/gts 的元素：{"box": xyxy, "state": ripe/half/unripe, "score"?: float}。"""
    total_tp = total_fp = total_fn = 0
    state_correct = 0
    state_total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    gt_counts = Counter()
    pred_counts = Counter()
    per_image = []

    images = sorted(set(preds_by_image) | set(gts_by_image))
    for image in images:
        preds = preds_by_image.get(image, [])
        gts = gts_by_image.get(image, [])
        pred_boxes = [p["box"] for p in preds]
        pred_scores = [float(p.get("score", 1.0)) for p in preds]
        gt_boxes = [g["box"] for g in gts]
        matches, unmatched_pred, unmatched_gt = greedy_match(
            pred_boxes, pred_scores, gt_boxes, iou_thr
        )
        total_tp += len(matches)
        total_fp += len(unmatched_pred)
        total_fn += len(unmatched_gt)

        correct = 0
        for pi, gi, _ in matches:
            true_state = gts[gi]["state"]
            pred_state = preds[pi]["state"]
            state_total += 1
            per_class_total[true_state] += 1
            if pred_state == true_state:
                state_correct += 1
                per_class_correct[true_state] += 1
                correct += 1

        for g in gts:
            gt_counts[g["state"]] += 1
        for p in preds:
            pred_counts[p["state"]] += 1

        per_image.append(
            {
                "image": image,
                "gt": len(gts),
                "pred": len(preds),
                "tp": len(matches),
                "fp": len(unmatched_pred),
                "fn": len(unmatched_gt),
                "state_correct": correct,
                "state_acc": correct / len(matches) if matches else 0.0,
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    state_acc = state_correct / state_total if state_total else 0.0
    per_class_acc = {
        name: (per_class_correct[name] / per_class_total[name]
               if per_class_total[name] else 0.0)
        for name in labels
    }
    count_error = {
        name: int(pred_counts.get(name, 0) - gt_counts.get(name, 0)) for name in labels
    }
    return {
        "images": len(images),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": 2 * precision * recall / (precision + recall)
                         if precision + recall else 0.0,
        "state_accuracy": state_acc,
        "per_class_state_accuracy": per_class_acc,
        "per_class_gt": {name: int(gt_counts.get(name, 0)) for name in labels},
        "per_class_pred": {name: int(pred_counts.get(name, 0)) for name in labels},
        "count_error": count_error,
        "total_count_error": int(sum(pred_counts.values()) - sum(gt_counts.values())),
        "end2end_correct_rate": state_correct / max(1, sum(gt_counts.values())),
        "per_image": per_image,
        "iou_thr": iou_thr,
    }


def summarize_end2end(metrics: dict) -> str:
    p = metrics["per_class_state_accuracy"]
    return (
        f"images={metrics['images']} "
        f"P={metrics['detection_precision']:.3f} R={metrics['detection_recall']:.3f} "
        f"state_acc={metrics['state_accuracy']:.3f} "
        f"ripe={p.get('ripe', 0):.2f} half={p.get('half', 0):.2f} "
        f"unripe={p.get('unripe', 0):.2f} "
        f"count_err={metrics['total_count_error']}"
    )
