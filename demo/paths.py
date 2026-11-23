"""仓库内的路径解析。

同一个项目在不同布局下都能跑：
- GitHub 仓库（推荐）：  `<repo>/strawberry_cv/`、`<repo>/models/`、`<repo>/samples/`
- 本地工作区：          `<repo>/handoff_2026-11-22/strawberry_cv/`、`<repo>/handoff_2026-11-22/work_merged/...`
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- strawberry_cv 包


def find_package_dir() -> Path:
    """返回包含 `strawberry_cv` 包的那一层目录（也就是项目根）。"""
    candidates = [
        REPO_ROOT,
        REPO_ROOT / "handoff_2026-11-22",
    ]
    for cand in candidates:
        if (cand / "strawberry_cv" / "__init__.py").exists():
            return cand
    return REPO_ROOT


PACKAGE_DIR = find_package_dir()


def ensure_importable() -> Path:
    """把项目根加入 sys.path，供 `import strawberry_cv` 使用。"""
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))
    return PACKAGE_DIR


# ---------------------------------------------------------------- 模型权重


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def find_weights(kind: str) -> Path | None:
    """按类型查找权重：'detector' / 'classifier'。

    优先级：仓库的 `models/` → 本地工作区的训练产物目录。
    """
    if kind == "detector":
        return _first_existing(
            REPO_ROOT / "models" / "detector_merged.pt",
            PACKAGE_DIR / "work_merged/runs/detector/merged_1cls_lowlr/weights/best.pt",
            PACKAGE_DIR / "work/runs/detector/strawberry_1cls_lowlr/weights/best.pt",
        )
    if kind == "classifier":
        return _first_existing(
            REPO_ROOT / "models" / "classifier_merged.pt",
            PACKAGE_DIR / "work_merged/runs/classifier/resnet18_3cls_merged/best.pt",
            PACKAGE_DIR / "work_archive/runs/classifier/resnet18_3cls_archive/best.pt",
            PACKAGE_DIR / "work/runs/classifier/resnet18_3cls/best.pt",
        )
    raise ValueError(f"未知权重类型：{kind}")


# ---------------------------------------------------------------- 数据集


def find_dataset_dir(name: str) -> Path | None:
    """查找数据集目录：优先完整的 `datasets/`，回退到仓库自带的 `samples/`。

    name ∈ {'v1', 'archive'}
    """
    mapping = {
        "v1": ("strawberry_v1", "v1"),
        "archive": ("strawberry_archive_v1", "archive"),
    }
    if name not in mapping:
        raise ValueError(f"未知数据集：{name}")
    full, sample = mapping[name]
    for cand in (REPO_ROOT / "datasets" / full, REPO_ROOT / "samples" / sample):
        if (cand / "detection" / "images").is_dir():
            return cand
    return None


def images_dir(name: str) -> Path | None:
    base = find_dataset_dir(name)
    return base / "detection" / "images" if base else None


def labels_dir(name: str) -> Path | None:
    base = find_dataset_dir(name)
    return base / "detection" / "labels" if base else None
