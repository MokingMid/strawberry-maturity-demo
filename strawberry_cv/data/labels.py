"""读取并校验项目自带的逐框 JSON 标注。

JSON 结构：
{
  "labels": [
    {"name": "ripe", "x1": 0, "y1": 6, "x2": 269, "y2": 280,
     "size": {"height": 525, "width": 700}}
  ]
}
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..constants import CLASS_NAMES, CLASS_TO_ID, IMAGE_EXTS
from ..utils.image_io import image_size


class LabelError(ValueError):
    """标注不合法。"""


@dataclass(frozen=True)
class Box:
    image: str
    index: int
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int

    @property
    def class_id(self) -> int:
        return CLASS_TO_ID[self.name]

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def normalized_cxcywh(self) -> tuple[float, float, float, float]:
        if self.width <= 0 or self.height <= 0:
            raise LabelError(f"{self.image}: 图片尺寸非法")
        cx = (self.x1 + self.x2) / 2.0 / self.width
        cy = (self.y1 + self.y2) / 2.0 / self.height
        w = (self.x2 - self.x1) / self.width
        h = (self.y2 - self.y1) / self.height
        return cx, cy, w, h

    @property
    def touches_border(self) -> bool:
        # 1 像素容差：贴边/被截断的果实在分类裁剪时通常不完整
        return (
            self.x1 <= 1
            or self.y1 <= 1
            or self.x2 >= self.width - 1
            or self.y2 >= self.height - 1
        )


class StrawberryDataset:
    """检测数据集的只读封装。"""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        class_names: tuple[str, ...] = CLASS_NAMES,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = tuple(class_names)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"图片目录不存在：{self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"标签目录不存在：{self.labels_dir}")
        self._cache: dict[str, list[Box]] = {}

    # ---------- 基本列表 ----------
    @property
    def stems(self) -> list[str]:
        return sorted(
            p.stem for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

    @property
    def image_paths(self) -> list[Path]:
        return [self.image_path(stem) for stem in self.stems]

    def image_path(self, stem: str) -> Path:
        for ext in IMAGE_EXTS:
            path = self.images_dir / f"{stem}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"找不到图片：{stem}")

    def label_path(self, stem: str) -> Path:
        return self.labels_dir / f"{stem}.json"

    # ---------- 读取 ----------
    def load(self, stem: str) -> list[Box]:
        if stem in self._cache:
            return self._cache[stem]
        path = self.label_path(stem)
        if not path.exists():
            raise LabelError(f"缺少标签：{path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LabelError(f"JSON 解析失败：{path}: {exc}") from exc

        labels = raw.get("labels")
        if not isinstance(labels, list):
            raise LabelError(f"{path}: 缺少 labels 数组")

        boxes: list[Box] = []
        for i, item in enumerate(labels):
            name = item.get("name")
            if name not in self.class_names:
                raise LabelError(f"{path}: 第 {i} 个框类别非法：{name!r}")
            try:
                x1, y1, x2, y2 = (int(item[k]) for k in ("x1", "y1", "x2", "y2"))
                size = item["size"]
                width, height = int(size["width"]), int(size["height"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LabelError(f"{path}: 第 {i} 个框字段缺失或类型错误") from exc
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise LabelError(
                    f"{path}: 第 {i} 个框越界/退化：({x1},{y1},{x2},{y2}) vs {width}x{height}"
                )
            boxes.append(Box(stem, i, name, x1, y1, x2, y2, width, height))
        self._cache[stem] = boxes
        return boxes

    def iter_boxes(self) -> Iterator[Box]:
        for stem in self.stems:
            yield from self.load(stem)

    # ---------- 校验 ----------
    def validate_all(self, verify_image_size: bool = True) -> list[str]:
        issues: list[str] = []
        stems = self.stems
        label_stems = {p.stem for p in self.labels_dir.glob("*.json")}
        for missing in sorted(set(stems) - label_stems):
            issues.append(f"缺少标签：{missing}")
        for orphan in sorted(label_stems - set(stems)):
            issues.append(f"孤儿标签（无图片）：{orphan}")
        for stem in stems:
            try:
                boxes = self.load(stem)
            except LabelError as exc:
                issues.append(str(exc))
                continue
            if verify_image_size and boxes:
                try:
                    width, height = image_size(self.image_path(stem))
                except IOError as exc:
                    issues.append(str(exc))
                    continue
                for box in boxes:
                    if (box.width, box.height) != (width, height):
                        issues.append(
                            f"{stem}: 标签 size {box.width}x{box.height} "
                            f"与真实图片 {width}x{height} 不一致"
                        )
                        break
        return issues

    # ---------- 统计 ----------
    def stats(self) -> dict:
        per_class: Counter[str] = Counter()
        per_class_images: Counter[str] = Counter()
        per_image: dict[str, int] = {}
        border = 0
        mixed: dict[str, dict[str, int]] = {}
        sizes: set[tuple[int, int]] = set()

        for stem in self.stems:
            boxes = self.load(stem)
            counts = Counter(box.name for box in boxes)
            per_class.update(counts)
            for name in counts:
                per_class_images[name] += 1
            per_image[stem] = len(boxes)
            border += sum(1 for box in boxes if box.touches_border)
            if len(counts) > 1:
                mixed[stem] = dict(counts)
            if boxes:
                sizes.add((boxes[0].width, boxes[0].height))

        return {
            "images": len(self.stems),
            "boxes": sum(per_class.values()),
            "per_class": {name: per_class.get(name, 0) for name in self.class_names},
            "per_class_images": {
                name: per_class_images.get(name, 0) for name in self.class_names
            },
            "per_image_counts": per_image,
            "boxes_per_image": dict(sorted(Counter(per_image.values()).items())),
            "border_boxes": border,
            "mixed_images": mixed,
            "image_sizes": sorted(sizes),
            "num_image_sizes": len(sizes),
        }

    def class_counts_by_image(self) -> dict[str, Counter[str]]:
        return {
            stem: Counter(box.name for box in self.load(stem)) for stem in self.stems
        }


def load_dataset_from_config(config) -> StrawberryDataset:
    return StrawberryDataset(
        config.path("paths.images"),
        config.path("paths.labels"),
        class_names=config.class_names(),
    )
