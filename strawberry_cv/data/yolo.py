"""把项目 JSON 标注导出为 Ultralytics YOLO 检测格式。

两种编码：
- 1cls：所有框 class 0，names = [strawberry]（主线检测器）
- 3cls：框保留 ripe/half/unripe，names = [ripe, half, unripe]（对照实验）
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from ..constants import CLASS_NAMES, DETECTOR_CLASS_NAME
from .labels import Box, StrawberryDataset
from .splits import SplitResult


class ExportError(ValueError):
    pass


def _class_config(class_mode: str) -> tuple[int, list[str]]:
    if class_mode == "1cls":
        return 1, [DETECTOR_CLASS_NAME]
    if class_mode == "3cls":
        return 3, list(CLASS_NAMES)
    raise ExportError("class_mode 只支持 '1cls' 或 '3cls'")


def _yolo_line(box: Box, class_mode: str) -> str:
    cls = 0 if class_mode == "1cls" else box.class_id
    cx, cy, w, h = box.normalized_cxcywh
    if not all(0.0 <= value <= 1.0 for value in (cx, cy, w, h)):
        raise ExportError(f"{box.image}: 归一化坐标越界 {cx,cy,w,h}")
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _write_data_yaml(
    path: Path,
    dataset_dir: Path,
    names: list[str],
    class_mode: str,
) -> None:
    lines = [
        "# 由 strawberry_cv 自动生成",
        f"# class_mode: {class_mode}",
        f"path: {dataset_dir.as_posix()}",
        "train: splits/train.txt",
        "val: splits/val.txt",
        "test: splits/test.txt",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_yolo(
    dataset: StrawberryDataset,
    out_dir: str | Path,
    class_mode: str = "1cls",
    split_result: SplitResult | None = None,
    copy_images: bool = True,
    force: bool = False,
    verify: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    nc, names = _class_config(class_mode)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise ExportError(f"输出目录非空：{out_dir}（加 force=True 覆盖）")
        shutil.rmtree(out_dir)
    images_out = out_dir / "images" / "all"
    labels_out = out_dir / "labels" / "all"
    splits_out = out_dir / "splits"
    for path in (images_out, labels_out, splits_out):
        path.mkdir(parents=True, exist_ok=True)

    per_class = Counter()
    total_boxes = 0
    warnings: list[str] = []

    for stem in dataset.stems:
        image_path = dataset.image_path(stem)
        boxes = dataset.load(stem)
        target_image = images_out / image_path.name
        if copy_images:
            shutil.copy2(image_path, target_image)
        else:
            raise ExportError("copy_images=False 暂未实现（Windows 符号链接权限不稳定）")

        lines = []
        for box in boxes:
            lines.append(_yolo_line(box, class_mode))
            per_class[box.name] += 1
            total_boxes += 1
        (labels_out / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    # 划分文件
    if split_result is None:
        warnings.append("没有提供划分，train/val/test 都指向全部图片；仅适合冒烟测试")
        split_names = {"train": dataset.stems, "val": dataset.stems, "test": dataset.stems}
    else:
        split_names = {
            split: [stem for stem in dataset.stems if split_result.assignments.get(stem) == split]
            for split in ("train", "val", "test")
        }
    for split, stems in split_names.items():
        # Ultralytics 对 .txt 里的相对路径不会自动拼 data.yaml 的 path，
        # 这里直接写绝对路径，避免从项目根目录找 images/all/... 而报错。
        paths = [
            str((images_out / dataset.image_path(stem).name).resolve())
            for stem in stems
        ]
        (splits_out / f"{split}.txt").write_text(
            "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8"
        )

    _write_data_yaml(out_dir / "data.yaml", out_dir, names, class_mode)

    report = {
        "class_mode": class_mode,
        "nc": nc,
        "names": names,
        "images": len(dataset.stems),
        "boxes": total_boxes,
        "per_class": {name: per_class.get(name, 0) for name in CLASS_NAMES},
        "splits": {k: len(v) for k, v in split_names.items()},
        "out_dir": str(out_dir),
        "warnings": warnings,
    }

    if verify:
        issues = verify_export(dataset, out_dir, class_mode, split_result)
        report["verify_issues"] = issues
        if issues:
            warnings.append(f"校验发现 {len(issues)} 个问题，见 verify_issues")

    (out_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def verify_export(
    dataset: StrawberryDataset,
    out_dir: str | Path,
    class_mode: str,
    split_result: SplitResult | None = None,
) -> list[str]:
    """回读 YOLO txt，检查行数、字段、归一化范围与划分覆盖。"""
    out_dir = Path(out_dir)
    labels_dir = out_dir / "labels" / "all"
    images_dir = out_dir / "images" / "all"
    issues: list[str] = []

    for stem in dataset.stems:
        path = labels_dir / f"{stem}.txt"
        if not path.exists():
            issues.append(f"缺少 YOLO 标签：{path}")
            continue
        boxes = dataset.load(stem)
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != len(boxes):
            issues.append(f"{stem}: 行数 {len(lines)} != 框数 {len(boxes)}")
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                issues.append(f"{stem}: 第 {i} 行不是 5 列：{line!r}")
                continue
            try:
                cls = int(parts[0])
                values = [float(x) for x in parts[1:]]
            except ValueError:
                issues.append(f"{stem}: 第 {i} 行含非数字：{line!r}")
                continue
            if cls < 0 or cls >= (1 if class_mode == "1cls" else 3):
                issues.append(f"{stem}: 第 {i} 行类别越界：{cls}")
            if any(v < 0.0 or v > 1.0 for v in values):
                issues.append(f"{stem}: 第 {i} 行归一化越界：{values}")
            if values[2] <= 0 or values[3] <= 0:
                issues.append(f"{stem}: 第 {i} 行宽高非正：{values}")
        if not (images_dir / dataset.image_path(stem).name).exists():
            issues.append(f"{stem}: 图片没有复制到导出目录")

    if split_result is not None:
        seen: set[str] = set()
        for split in ("train", "val", "test"):
            path = out_dir / "splits" / f"{split}.txt"
            if not path.exists():
                issues.append(f"缺少划分文件：{path}")
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                stem = Path(line).stem
                if stem not in dataset.stems:
                    issues.append(f"{split}: 未知图片 {line}")
                if stem in seen:
                    issues.append(f"{split}: 图片重复出现在多个 split：{line}")
                seen.add(stem)
        if seen != set(dataset.stems):
            missing = sorted(set(dataset.stems) - seen)
            issues.append(f"划分未覆盖全部图片：{missing[:10]}")
    return issues


def inspect_yolo_dir(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    labels = list((out_dir / "labels" / "all").glob("*.txt"))
    images = list((out_dir / "images" / "all").glob("*"))
    counts = Counter()
    for path in labels:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return {
        "out_dir": str(out_dir),
        "images": len(images),
        "labels": len(labels),
        "class_counts": dict(sorted(counts.items())),
    }
