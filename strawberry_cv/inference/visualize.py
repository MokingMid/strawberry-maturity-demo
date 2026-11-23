"""检测/分类结果可视化（英文标签，避免中文字体问题）。"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from ..utils.geometry import Box

COLORS = {
    "ripe": (0, 200, 0),      # BGR green
    "half": (0, 165, 255),    # orange
    "unripe": (255, 120, 0),  # blue
    "unknown": (200, 200, 200),
}


def draw_box(
    frame: np.ndarray,
    box: Box,
    label: str,
    conf: float | None = None,
    track_id: int | None = None,
    color: tuple[int, int, int] | None = None,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    color = color or COLORS.get(label, COLORS["unknown"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    parts = [label]
    if conf is not None:
        parts.append(f"{conf:.2f}")
    if track_id is not None:
        parts.append(f"#{track_id}")
    text = " ".join(parts)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        frame, text, (x1 + 3, max(th + 1, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_counts(frame: np.ndarray, counts: Counter, origin: tuple[int, int] = (12, 28)) -> None:
    x, y = origin
    cv2.putText(frame, "counts:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 24
    for name in ("ripe", "half", "unripe"):
        value = int(counts.get(name, 0))
        cv2.putText(
            frame, f"{name}: {value}", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[name], 2,
        )
        y += 24
    cv2.putText(
        frame, f"total: {int(sum(counts.values()))}", (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


def draw_line(frame: np.ndarray, p1: tuple[float, float], p2: tuple[float, float]) -> None:
    cv2.line(
        frame,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        (0, 255, 255), 2,
    )


def annotate(
    frame: np.ndarray,
    fruits: list[dict],
    counts: Counter | None = None,
    line: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> np.ndarray:
    out = frame.copy()
    for fruit in fruits:
        draw_box(
            out, fruit["box"], fruit.get("state", "unknown"),
            conf=fruit.get("state_conf"), track_id=fruit.get("track_id"),
        )
    if line:
        draw_line(out, line[0], line[1])
    if counts is not None:
        draw_counts(out, counts)
    return out
