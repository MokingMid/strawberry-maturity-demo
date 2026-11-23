"""传统基线：HSV 颜色矩/直方图 + GLCM + LBP → RBF-SVM。

用场景组做交叉验证，避免同图/翻转副本跨折泄漏。
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("需要 scikit-learn：pip install scikit-learn") from exc

from ..constants import CLASS_NAMES, SPLIT_NAMES
from ..evaluation.classification import classification_metrics
from ..utils.image_io import imread
from ..utils.io import get_logger, save_json
from .features import extract_features

LOGGER = get_logger("traditional")


@dataclass
class RoiSample:
    path: Path
    label: str
    image: str
    group: str
    split: str


def _scene_map(scene_manifest: str | Path | None) -> dict[str, str]:
    if not scene_manifest:
        return {}
    path = Path(scene_manifest)
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {row["image"]: row["scene_group"] for row in rows if "image" in row}


def load_roi_samples(
    roi_root: str | Path,
    split: str,
    scene_manifest: str | Path | None = None,
) -> list[RoiSample]:
    roi_root = Path(roi_root)
    scene_map = _scene_map(scene_manifest)
    manifest = roi_root / "roi_manifest.csv"
    samples: list[RoiSample] = []
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                image = row["image"]
                samples.append(
                    RoiSample(
                        path=roi_root / row["crop_path"],
                        label=row["class"],
                        image=image,
                        group=scene_map.get(image, image),
                        split=split,
                    )
                )
        return samples

    split_dir = roi_root / split
    if not split_dir.is_dir():
        return []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in CLASS_NAMES:
            continue
        for path in sorted(class_dir.glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            image = path.stem.split("__b")[0]
            samples.append(
                RoiSample(path, class_dir.name, image,
                          scene_map.get(image, image), split)
            )
    return samples


def build_feature_matrix(
    samples: list[RoiSample],
    traditional: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vectors, labels, groups, names = [], [], [], []
    for i, sample in enumerate(samples):
        image = imread(sample.path)
        if image is None:
            LOGGER.warning("无法读取 ROI：%s", sample.path)
            continue
        vector, feature_names = extract_features(image, traditional)
        vectors.append(vector)
        labels.append(sample.label)
        groups.append(sample.group)
        names = feature_names
        if (i + 1) % 50 == 0:
            LOGGER.info("features %d/%d", i + 1, len(samples))
    if not vectors:
        return np.zeros((0, 0), np.float32), np.array([]), np.array([]), names
    return (
        np.stack(vectors).astype(np.float32),
        np.asarray(labels),
        np.asarray(groups),
        names,
    )


def _folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    n_splits = max(2, int(n_splits))
    dummy = np.zeros((len(y), 1))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(dummy, y, groups))
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(dummy, y, groups))


def fit_svm_grid(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_values: list[float],
    gamma_values: list,
    cv_folds: int = 3,
    seed: int = 42,
    labels: tuple[str, ...] = CLASS_NAMES,
) -> tuple[Pipeline, dict, list[dict]]:
    if len(X) == 0:
        raise ValueError("训练特征为空")
    counts = Counter(y)
    min_count = min(counts.values()) if counts else 0
    n_splits = min(int(cv_folds), max(2, min_count), len(set(groups)))
    folds = _folds(y, groups, n_splits, seed)
    results: list[dict] = []
    best = {"score": -1.0, "params": None}
    for C in C_values:
        for gamma in gamma_values:
            scores = []
            for train_idx, val_idx in folds:
                if len(set(y[train_idx])) < 2:
                    continue
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", C=float(C), gamma=gamma,
                                class_weight="balanced")),
                ])
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_idx])
                scores.append(f1_score(y[val_idx], pred, labels=list(labels),
                                       average="macro", zero_division=0))
            score = float(np.mean(scores)) if scores else 0.0
            row = {"C": C, "gamma": gamma, "macro_f1": score, "folds": len(scores)}
            results.append(row)
            if score > best["score"]:
                best = {"score": score, "params": {"C": C, "gamma": gamma}}
    if best["params"] is None:
        best["params"] = {"C": C_values[0], "gamma": gamma_values[0]}
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=float(best["params"]["C"]),
                    gamma=best["params"]["gamma"], class_weight="balanced")),
    ])
    final.fit(X, y)
    return final, best, results


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    if len(X) == 0:
        return classification_metrics([], [], labels=CLASS_NAMES)
    pred = model.predict(X)
    return classification_metrics(list(y), list(pred), labels=CLASS_NAMES)


def run_traditional(
    roi_root: str | Path,
    out_dir: str | Path,
    traditional: dict,
    scene_manifest: str | Path | None = None,
    seed: int = 42,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict] = {}
    feature_names: list[str] = []

    for split in SPLIT_NAMES:
        samples = load_roi_samples(roi_root, split, scene_manifest)
        X, y, groups, names = build_feature_matrix(samples, traditional)
        if names:
            feature_names = names
        LOGGER.info("%s: samples=%d features=%s", split, len(X), X.shape[1] if len(X) else 0)
        splits[split] = {"X": X, "y": y, "groups": groups, "samples": samples}

    train = splits["train"]
    if len(train["X"]) == 0:
        raise ValueError("train ROI 为空，请先 export-roi")

    model, best, cv_results = fit_svm_grid(
        train["X"], train["y"], train["groups"],
        C_values=[float(x) for x in traditional.get("svm_c", [0.1, 1, 10])],
        gamma_values=list(traditional.get("svm_gamma", ["scale", 0.001, 0.01])),
        cv_folds=int(traditional.get("cv_folds", 3)),
        seed=seed,
        labels=CLASS_NAMES,
    )

    metrics = {"best_cv": best, "splits": {}}
    for split in SPLIT_NAMES:
        X, y = splits[split]["X"], splits[split]["y"]
        m = evaluate_model(model, X, y)
        metrics["splits"][split] = m
        LOGGER.info("%s macroF1=%.3f acc=%.3f",
                    split, m["macro_f1"], m["accuracy"])

    import joblib

    joblib.dump(
        {"model": model, "feature_names": feature_names, "traditional": traditional,
         "class_names": list(CLASS_NAMES)},
        out_dir / "svm_model.joblib",
    )
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "feature_config.json",
              {"feature_names": feature_names, "traditional": traditional})
    with (out_dir / "cv_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(cv_results[0].keys()))
        writer.writeheader()
        writer.writerows(cv_results)
    return metrics
