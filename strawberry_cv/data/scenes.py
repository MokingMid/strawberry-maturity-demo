"""场景分组清单（防止翻转/缩放重复图跨 train/val/test 泄漏）。"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


class SceneError(ValueError):
    pass


class SceneIndex:
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows
        self.image_to_group: dict[str, str] = {}
        self.image_to_relation: dict[str, str] = {}
        self.image_to_duplicate_of: dict[str, str] = {}
        self.group_to_images: dict[str, list[str]] = {}
        for row in rows:
            image = row["image"]
            group = row["scene_group"]
            if image in self.image_to_group:
                raise SceneError(f"清单里图片重复：{image}")
            self.image_to_group[image] = group
            self.image_to_relation[image] = row.get("relation", "none")
            self.image_to_duplicate_of[image] = row.get("duplicate_of", "")
            self.group_to_images.setdefault(group, []).append(image)

    @classmethod
    def from_csv(cls, path: str | Path) -> "SceneIndex":
        with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SceneError(f"场景清单为空：{path}")
        for key in ("image", "scene_group"):
            if key not in rows[0]:
                raise SceneError(f"场景清单缺少列：{key}")
        return cls(rows)

    @property
    def groups(self) -> list[str]:
        return sorted(self.group_to_images)

    def group_of(self, image: str) -> str:
        if image not in self.image_to_group:
            raise SceneError(f"场景清单里没有图片：{image}")
        return self.image_to_group[image]

    def validate(self, images: list[str]) -> list[str]:
        issues: list[str] = []
        known = set(self.image_to_group)
        for image in sorted(set(images) - known):
            issues.append(f"场景清单缺少图片：{image}")
        for image in sorted(known - set(images)):
            issues.append(f"场景清单多余图片：{image}")
        sizes = Counter(len(v) for v in self.group_to_images.values())
        if any(size > 2 for size in sizes):
            issues.append(f"存在超过 2 张图的场景组：{dict(sizes)}")
        return issues

    def stats(self) -> dict:
        sizes = Counter(len(v) for v in self.group_to_images.values())
        return {
            "groups": len(self.group_to_images),
            "images": len(self.image_to_group),
            "group_size_distribution": dict(sorted(sizes.items())),
            "duplicate_pairs": {
                group: images
                for group, images in self.group_to_images.items()
                if len(images) > 1
            },
        }


def load_scenes_from_config(config) -> SceneIndex:
    return SceneIndex.from_csv(config.path("paths.scene_manifest"))
