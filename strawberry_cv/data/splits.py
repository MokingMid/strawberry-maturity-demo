"""按场景组做 train/val/test 划分。

关键约束：
- 同一 scene_group 的图片只能进同一个 split（7 对翻转/缩放重复图）；
- 尽量让 half 这类稀有类别在三个 split 都出现；
- 固定随机种子，结果可复现。
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..constants import CLASS_NAMES, SPLIT_NAMES
from .labels import StrawberryDataset
from .scenes import SceneIndex


class SplitError(ValueError):
    pass


@dataclass
class SplitResult:
    assignments: dict[str, str]
    group_assignments: dict[str, str]
    class_counts: dict[str, Counter] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)
    image_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "assignments": self.assignments,
            "group_assignments": self.group_assignments,
            "class_counts": {k: dict(v) for k, v in self.class_counts.items()},
            "group_counts": self.group_counts,
            "image_counts": self.image_counts,
            "warnings": self.warnings,
        }


def _largest_remainder(ratios: list[float], total: int, minimum: int = 1) -> list[int]:
    if total < 3 * minimum:
        raise SplitError(f"场景组太少（{total}），无法保证每个 split 至少 {minimum} 组")
    raw = [float(r) * total for r in ratios]
    base = [int(x) for x in raw]
    remainder = total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - base[i], -i), reverse=True)
    for i in order[:remainder]:
        base[i] += 1

    # 保证最小值，从最大的 split 里挪
    while any(x < minimum for x in base):
        donor = max(range(len(base)), key=lambda i: (base[i], -i))
        receiver = min(range(len(base)), key=lambda i: (base[i], i))
        if base[donor] <= minimum:
            break
        base[donor] -= 1
        base[receiver] += 1
    return base


def make_group_split(
    dataset: StrawberryDataset,
    scenes: SceneIndex,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    rare_ratio: float = 0.5,
) -> SplitResult:
    images = dataset.stems
    issues = scenes.validate(images)
    if issues:
        raise SplitError("场景清单与数据集不一致：\n" + "\n".join(issues[:10]))

    class_by_image = dataset.class_counts_by_image()
    total_class = Counter()
    for counts in class_by_image.values():
        total_class.update(counts)
    total_boxes = sum(total_class.values())

    # 场景组 -> 图片 / 类别计数
    group_images: dict[str, list[str]] = {
        group: sorted(ims) for group, ims in scenes.group_to_images.items()
    }
    group_classes: dict[str, Counter] = {}
    for group, ims in group_images.items():
        counter = Counter()
        for image in ims:
            counter.update(class_by_image[image])
        group_classes[group] = counter

    groups = sorted(group_images)
    targets = _largest_remainder(list(ratios), len(groups), minimum=1)
    target_groups = dict(zip(SPLIT_NAMES, targets))
    if sum(targets) != len(groups):
        raise SplitError("划分数不等于场景组数")

    rng = random.Random(seed)
    rng.shuffle(groups)

    # 稀有类别：half（总量 11）会落到这里
    # 比最大类别少一半以上的类别视为稀有（真实数据里只有 half 会命中）
    max_class_count = max(total_class.values()) if total_class else 0
    rare_classes = {
        name
        for name, count in total_class.items()
        if max_class_count > 0 and count <= rare_ratio * max_class_count
    }
    rare_groups = [
        g for g in groups if any(group_classes[g].get(c, 0) > 0 for c in rare_classes)
    ]
    ordinary_groups = [g for g in groups if g not in set(rare_groups)]

    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    current_class = {name: Counter() for name in SPLIT_NAMES}
    capacity = {name: target_groups[name] for name in SPLIT_NAMES}

    def place(group: str, split: str) -> None:
        assigned[split].append(group)
        capacity[split] -= 1
        current_class[split].update(group_classes[group])

    # 1) 稀有组先分，尽量每个 split 都有
    rare_groups.sort(
        key=lambda g: (-sum(group_classes[g].get(c, 0) for c in rare_classes), g)
    )
    for group in rare_groups:
        candidates = [s for s in SPLIT_NAMES if capacity[s] > 0]
        if not candidates:
            candidates = list(SPLIT_NAMES)
        best = min(
            candidates,
            key=lambda s: (
                sum(current_class[s].get(c, 0) for c in rare_classes),
                -capacity[s],
                SPLIT_NAMES.index(s),
            ),
        )
        place(group, best)

    # 2) 其余组按类别缺口贪心分配
    for group in ordinary_groups:
        candidates = [s for s in SPLIT_NAMES if capacity[s] > 0]
        if not candidates:
            candidates = list(SPLIT_NAMES)
        counts = group_classes[group]

        def deficit(split: str) -> float:
            score = 0.0
            for name in CLASS_NAMES:
                target = ratios[SPLIT_NAMES.index(split)] * total_class.get(name, 0)
                score += target - current_class[split].get(name, 0)
            # 让“这一组包含的类别”权重更高
            for name in CLASS_NAMES:
                if counts.get(name, 0):
                    target = ratios[SPLIT_NAMES.index(split)] * total_class.get(name, 0)
                    score += 0.5 * (target - current_class[split].get(name, 0))
            return score

        best = max(
            candidates,
            key=lambda s: (deficit(s), capacity[s], -SPLIT_NAMES.index(s)),
        )
        place(group, best)

    group_assignments = {
        group: split for split in SPLIT_NAMES for group in assigned[split]
    }
    assignments = {
        image: group_assignments[group]
        for group, ims in group_images.items()
        for image in ims
    }

    result = SplitResult(assignments=assignments, group_assignments=group_assignments)
    for split in SPLIT_NAMES:
        result.group_counts[split] = len(assigned[split])
        result.image_counts[split] = sum(len(group_images[g]) for g in assigned[split])
        counter = Counter()
        for group in assigned[split]:
            counter.update(group_classes[group])
        result.class_counts[split] = counter

    for split in SPLIT_NAMES:
        for name in CLASS_NAMES:
            if result.class_counts[split].get(name, 0) == 0:
                result.warnings.append(f"{split} 缺少类别 {name}")
    if result.class_counts["train"].get("half", 0) < 3:
        result.warnings.append("训练集 half 框少于 3 个，分类器基本学不到 half")
    return result


def load_split_assignments(path: str | Path) -> dict[str, str]:
    """从 split_manifest.csv 读取 image -> split。"""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "image" not in rows[0] or "split" not in rows[0]:
        raise SplitError(f"划分清单格式不正确：{path}")
    return {row["image"]: row["split"] for row in rows}


def write_split(
    result: SplitResult,
    dataset: StrawberryDataset,
    scenes: SceneIndex,
    out_dir: str | Path,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for stem in dataset.stems:
        counts = Counter(box.name for box in dataset.load(stem))
        split = result.assignments[stem]
        rows.append(
            {
                "image": stem,
                "image_file": dataset.image_path(stem).name,
                "scene_group": scenes.group_of(stem),
                "split": split,
                "boxes": sum(counts.values()),
                **{name: counts.get(name, 0) for name in CLASS_NAMES},
            }
        )
    rows.sort(key=lambda r: (SPLIT_NAMES.index(r["split"]), r["image"]))

    with (out_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for split in SPLIT_NAMES:
        names = [r["image_file"] for r in rows if r["split"] == split]
        (out_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    report = result.to_dict()
    (out_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
