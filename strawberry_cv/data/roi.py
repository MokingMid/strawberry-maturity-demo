"""用 GT 框裁剪单果 ROI，构建三分类数据集。

注意：只使用标注框，不使用 traditional_cv 的算法裁剪。
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import CLASS_NAMES, SPLIT_NAMES
from ..utils.geometry import box_area, expand_box, iou, square_box
from ..utils.image_io import imread, imwrite
from .labels import Box, StrawberryDataset
from .splits import SplitResult


@dataclass
class RoiCrop:
    crop: np.ndarray | None
    crop_box: tuple[int, int, int, int]
    pad_used: float
    neighbor_overlap: bool
    touches_border: bool
    skip_reason: str | None = None


def crop_roi(
    image: np.ndarray,
    box: Box,
    pad_ratio: float = 0.12,
    square: bool = True,
    min_size: int = 32,
    neighbor_boxes: list[Box] | None = None,
    max_neighbor_iou: float = 0.1,
    pad_steps: int = 6,
) -> RoiCrop:
    """自适应外扩裁剪：外扩带入邻果时逐步减小 padding。"""
    height, width = image.shape[:2]
    neighbors = [b for b in (neighbor_boxes or []) if b.index != box.index]
    neighbor_xyxy = [b.xyxy for b in neighbors]
    pad_values = [pad_ratio * (1 - i / max(1, pad_steps)) for i in range(pad_steps + 1)]
    chosen_pad = 0.0
    chosen_box = box.xyxy
    overlap_flag = False

    for pad in pad_values:
        candidate = expand_box(box.xyxy, pad)
        if square:
            candidate = square_box(candidate)
        candidate = (
            max(0.0, candidate[0]),
            max(0.0, candidate[1]),
            min(float(width), candidate[2]),
            min(float(height), candidate[3]),
        )
        overlap = max((iou(candidate, other) for other in neighbor_xyxy), default=0.0)
        chosen_pad, chosen_box, overlap_flag = pad, candidate, overlap > max_neighbor_iou
        if not overlap_flag:
            break

    x1, y1, x2, y2 = (int(round(v)) for v in chosen_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return RoiCrop(None, (x1, y1, x2, y2), 0.0, overlap_flag,
                       box.touches_border, "empty")

    crop = image[y1:y2, x1:x2].copy()
    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        return RoiCrop(None, (x1, y1, x2, y2), chosen_pad, overlap_flag,
                       box.touches_border, "too_small")
    return RoiCrop(crop, (x1, y1, x2, y2), chosen_pad, overlap_flag,
                   box.touches_border, None)


def export_roi_dataset(
    dataset: StrawberryDataset,
    split_result: SplitResult,
    out_dir: str | Path,
    pad_ratio: float = 0.12,
    square: bool = True,
    min_size: int = 32,
    skip_border: bool = True,
    max_neighbor_iou: float = 0.1,
    pad_steps: int = 6,
    force: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise FileExistsError(f"输出目录非空：{out_dir}（加 force=True 覆盖）")
        shutil.rmtree(out_dir)

    for split in SPLIT_NAMES:
        for name in CLASS_NAMES:
            (out_dir / split / name).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped: Counter[str] = Counter()
    class_counts: dict[str, Counter] = {split: Counter() for split in SPLIT_NAMES}
    warnings: list[str] = []

    for split in SPLIT_NAMES:
        stems = [s for s in dataset.stems if split_result.assignments.get(s) == split]
        for stem in stems:
            image = imread(dataset.image_path(stem))
            if image is None:
                warnings.append(f"无法读取图片：{stem}")
                continue
            boxes = dataset.load(stem)
            for box in boxes:
                if skip_border and box.touches_border:
                    skipped["border"] += 1
                    continue
                roi = crop_roi(
                    image, box,
                    pad_ratio=pad_ratio, square=square, min_size=min_size,
                    neighbor_boxes=boxes, max_neighbor_iou=max_neighbor_iou,
                    pad_steps=pad_steps,
                )
                if roi.crop is None:
                    skipped[roi.skip_reason or "unknown"] += 1
                    continue
                filename = f"{stem}__b{box.index:02d}.jpg"
                rel_path = Path(split) / box.name / filename
                imwrite(out_dir / rel_path, roi.crop)
                class_counts[split][box.name] += 1
                rows.append(
                    {
                        "split": split,
                        "image": stem,
                        "box_index": box.index,
                        "class": box.name,
                        "crop_path": rel_path.as_posix(),
                        "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
                        "crop_x1": roi.crop_box[0], "crop_y1": roi.crop_box[1],
                        "crop_x2": roi.crop_box[2], "crop_y2": roi.crop_box[3],
                        "crop_w": roi.crop.shape[1], "crop_h": roi.crop.shape[0],
                        "pad_used": round(roi.pad_used, 4),
                        "neighbor_overlap": int(roi.neighbor_overlap),
                        "touches_border": int(roi.touches_border),
                    }
                )

    columns = [
        "split", "image", "box_index", "class", "crop_path",
        "x1", "y1", "x2", "y2", "crop_x1", "crop_y1", "crop_x2", "crop_y2",
        "crop_w", "crop_h", "pad_used", "neighbor_overlap", "touches_border",
    ]
    with (out_dir / "roi_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    for split in SPLIT_NAMES:
        for name in CLASS_NAMES:
            if class_counts[split].get(name, 0) == 0:
                warnings.append(f"{split}/{name} 没有样本")

    neighbor_overlap_crops = sum(1 for row in rows if row["neighbor_overlap"])
    if neighbor_overlap_crops:
        warnings.append(
            f"{neighbor_overlap_crops} 张 ROI 在缩小 padding 后仍与邻框重叠，"
            "已写入清单的 neighbor_overlap 列，训练前可人工抽查/过滤"
        )

    report = {
        "out_dir": str(out_dir),
        "counts": {split: {name: class_counts[split].get(name, 0) for name in CLASS_NAMES}
                   for split in SPLIT_NAMES},
        "total_crops": len(rows),
        "neighbor_overlap_crops": neighbor_overlap_crops,
        "skipped": dict(skipped),
        "warnings": warnings,
        "settings": {
            "pad_ratio": pad_ratio, "square": square, "min_size": min_size,
            "skip_border": skip_border, "max_neighbor_iou": max_neighbor_iou,
        },
    }
    (out_dir / "roi_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
