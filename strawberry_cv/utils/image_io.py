"""中文路径安全的图片读写。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """cv2.imread 在 Windows 中文路径下可能返回 None，这里走 imdecode。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"图片编码失败：{path}")
    buf.tofile(str(path))


def image_size(path: str | Path) -> tuple[int, int]:
    """返回 (width, height)。"""
    image = imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"无法读取图片：{path}")
    h, w = image.shape[:2]
    return int(w), int(h)


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
