"""传统特征：HSV 颜色矩、颜色直方图、GLCM 纹理、LBP 纹理。

不依赖 scikit-image；用 numpy/OpenCV 实现，方便助教复现。
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ---------------- 颜色 ----------------
def color_moments(hsv: np.ndarray) -> np.ndarray:
    values = []
    hsv = hsv.astype(np.float32)
    for c in range(3):
        channel = hsv[:, :, c].ravel()
        mean = float(channel.mean())
        std = float(channel.std())
        if std > 1e-6:
            skew = float(np.mean(((channel - mean) / std) ** 3))
        else:
            skew = 0.0
        values.extend([mean, std, skew])
    return np.asarray(values, dtype=np.float32)


def color_histogram(hsv: np.ndarray, bins: int = 32) -> np.ndarray:
    h_bins = max(4, bins // 2)
    s_bins = max(4, bins // 4)
    v_bins = max(4, bins // 4)
    h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).ravel()
    s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).ravel()
    hist = np.concatenate([h, s, v]).astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist


# ---------------- GLCM ----------------
def _quantize(gray: np.ndarray, levels: int) -> np.ndarray:
    q = (gray.astype(np.float32) / 256.0 * levels).astype(np.int32)
    return np.clip(q, 0, levels - 1)


def _glcm(q: np.ndarray, dy: int, dx: int, levels: int) -> np.ndarray | None:
    h, w = q.shape
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)
    if y1 <= y0 or x1 <= x0:
        return None
    a = q[y0:y1, x0:x1]
    b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    idx = a.ravel() * levels + b.ravel()
    mat = np.bincount(idx, minlength=levels * levels).reshape(levels, levels)
    mat = mat.astype(np.float64)
    mat = mat + mat.T
    total = mat.sum()
    return mat / total if total > 0 else None


def glcm_features(
    gray: np.ndarray,
    levels: int = 32,
    distances: tuple[int, ...] = (1, 2, 4),
    angles: tuple[int, ...] = (0, 45, 90, 135),
) -> tuple[np.ndarray, list[str]]:
    q = _quantize(gray, levels)
    features: list[float] = []
    names: list[str] = []
    for d in distances:
        for angle in angles:
            if angle == 0:
                dy, dx = 0, d
            elif angle == 45:
                dy, dx = -d, d
            elif angle == 90:
                dy, dx = -d, 0
            elif angle == 135:
                dy, dx = -d, -d
            else:
                continue
            mat = _glcm(q, dy, dx, levels)
            if mat is None:
                values = [0.0] * 6
            else:
                i, j = np.indices(mat.shape)
                mu_i = float((i * mat).sum())
                mu_j = float((j * mat).sum())
                sigma_i = math.sqrt(max(0.0, float(((i - mu_i) ** 2 * mat).sum())))
                sigma_j = math.sqrt(max(0.0, float(((j - mu_j) ** 2 * mat).sum())))
                contrast = float(((i - j) ** 2 * mat).sum())
                dissimilarity = float((np.abs(i - j) * mat).sum())
                homogeneity = float((mat / (1.0 + (i - j) ** 2)).sum())
                energy = float((mat ** 2).sum())
                correlation = (
                    float(((i - mu_i) * (j - mu_j) * mat).sum()
                          / (sigma_i * sigma_j + 1e-9))
                )
                values = [contrast, dissimilarity, homogeneity, energy, correlation,
                          float(mat.sum())]
            for key, value in zip(
                ("contrast", "dissimilarity", "homogeneity", "energy",
                 "correlation", "mass"), values
            ):
                features.append(float(value))
                names.append(f"glcm_d{d}_a{angle}_{key}")
    return np.asarray(features, dtype=np.float32), names


# ---------------- LBP ----------------
def lbp_histogram(
    gray: np.ndarray,
    points: int = 8,
    radius: float = 1.0,
) -> np.ndarray:
    grayf = gray.astype(np.float32)
    h, w = grayf.shape
    code = np.zeros((h, w), dtype=np.uint32)
    valid = np.ones((h, w), dtype=bool)

    for p in range(points):
        angle = 2.0 * math.pi * p / points
        dy = -radius * math.sin(angle)
        dx = radius * math.cos(angle)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sy = yy + dy
        sx = xx + dx
        y0 = np.floor(sy).astype(np.int32)
        x0 = np.floor(sx).astype(np.int32)
        y1, x1 = y0 + 1, x0 + 1
        inside = (y0 >= 0) & (x0 >= 0) & (y1 < h) & (x1 < w)
        valid &= inside

        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        wy = (sy - y0c).astype(np.float32)
        wx = (sx - x0c).astype(np.float32)
        sample = (
            (1 - wy) * (1 - wx) * grayf[y0c, x0c]
            + (1 - wy) * wx * grayf[y0c, x1c]
            + wy * (1 - wx) * grayf[y1c, x0c]
            + wy * wx * grayf[y1c, x1c]
        )
        code |= ((sample >= grayf).astype(np.uint32) << p)

    hist = np.bincount(code[valid].ravel(), minlength=1 << points).astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist


# ---------------- 汇总 ----------------
def feature_names(traditional: dict) -> list[str]:
    names = [f"hsv_c{c}_{stat}" for c in range(3) for stat in ("mean", "std", "skew")]
    h_bins = max(4, int(traditional.get("color_bins", 32)) // 2)
    s_bins = max(4, int(traditional.get("color_bins", 32)) // 4)
    v_bins = max(4, int(traditional.get("color_bins", 32)) // 4)
    names += [f"hist_h{i}" for i in range(h_bins)]
    names += [f"hist_s{i}" for i in range(s_bins)]
    names += [f"hist_v{i}" for i in range(v_bins)]

    distances = tuple(traditional.get("glcm_distances", (1, 2, 4)))
    angles = tuple(traditional.get("glcm_angles", (0, 45, 90, 135)))
    for d in distances:
        for a in angles:
            names += [
                f"glcm_d{d}_a{a}_{k}"
                for k in ("contrast", "dissimilarity", "homogeneity",
                          "energy", "correlation", "mass")
            ]
    points = int(traditional.get("lbp_points", 8))
    names += [f"lbp_{i}" for i in range(1 << points)]
    return names


def extract_features(image_bgr: np.ndarray, traditional: dict) -> tuple[np.ndarray, list[str]]:
    size = int(traditional.get("feature_size", 128))
    image = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    moments = color_moments(hsv)
    hist = color_histogram(hsv, int(traditional.get("color_bins", 32)))
    glcm, glcm_names = glcm_features(
        gray,
        levels=int(traditional.get("glcm_levels", 32)),
        distances=tuple(traditional.get("glcm_distances", (1, 2, 4))),
        angles=tuple(traditional.get("glcm_angles", (0, 45, 90, 135))),
    )
    lbp = lbp_histogram(
        gray,
        points=int(traditional.get("lbp_points", 8)),
        radius=float(traditional.get("lbp_radius", 1.0)),
    )
    names = feature_names(traditional)
    vector = np.concatenate([moments, hist, glcm, lbp]).astype(np.float32)
    if len(vector) != len(names):
        raise RuntimeError(f"特征维度 {len(vector)} 与名称数 {len(names)} 不一致")
    return vector, names
