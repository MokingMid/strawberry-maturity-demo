"""测试用合成数据集（不依赖真实数据）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from strawberry_cv.utils.image_io import imwrite

COLORS = {
    "ripe": (20, 30, 220),
    "half": (20, 150, 240),
    "unripe": (40, 180, 60),
}


def make_dataset(root: str | Path, specs: list[dict], duplicate_groups: dict[str, str] | None = None):
    """specs: [{"stem","width","height","boxes":[(name,x1,y1,x2,y2),...]}]"""
    root = Path(root)
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    duplicate_groups = duplicate_groups or {}

    for spec in specs:
        w, h = int(spec["width"]), int(spec["height"])
        image = np.full((h, w, 3), 45, dtype=np.uint8)
        labels = []
        for name, x1, y1, x2, y2 in spec["boxes"]:
            color = COLORS[name]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            labels.append(
                {
                    "name": name,
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "size": {"height": h, "width": w},
                }
            )
        imwrite(images_dir / f"{spec['stem']}.jpg", image)
        (labels_dir / f"{spec['stem']}.json").write_text(
            json.dumps({"labels": labels}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manifest = root / "scene_groups.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "scene_group", "relation", "duplicate_of",
                         "width", "height", "boxes", "ripe", "half", "unripe", "split"])
        for spec in specs:
            stem = spec["stem"]
            group = duplicate_groups.get(stem, f"G_{stem}")
            relation = "duplicate" if stem in duplicate_groups else "none"
            duplicate_of = ""
            if stem in duplicate_groups:
                duplicate_of = next(
                    (s["stem"] for s in specs
                     if duplicate_groups.get(s["stem"]) == group and s["stem"] != stem),
                    "",
                )
            counts = {"ripe": 0, "half": 0, "unripe": 0}
            for name, *_ in spec["boxes"]:
                counts[name] += 1
            writer.writerow([
                stem, group, relation, duplicate_of,
                spec["width"], spec["height"], len(spec["boxes"]),
                counts["ripe"], counts["half"], counts["unripe"], "",
            ])
    return images_dir, labels_dir, manifest


def standard_specs() -> list[dict]:
    """13 张图 / 12 个场景组 / 3 个 half 组。"""
    specs = [
        {"stem": "i01", "width": 200, "height": 150,
         "boxes": [("ripe", 20, 20, 80, 100)]},
        {"stem": "i02", "width": 200, "height": 150,
         "boxes": [("ripe", 110, 20, 170, 100)]},
        {"stem": "i03", "width": 200, "height": 150,
         "boxes": [("half", 20, 20, 90, 110)]},
        {"stem": "i04", "width": 200, "height": 150,
         "boxes": [("half", 100, 30, 180, 120)]},
        {"stem": "i05", "width": 200, "height": 150,
         "boxes": [("half", 30, 20, 100, 100), ("ripe", 110, 30, 180, 130)]},
        {"stem": "i06", "width": 200, "height": 150,
         "boxes": [("ripe", 20, 20, 90, 110)]},
        {"stem": "i07", "width": 200, "height": 150,
         "boxes": [("ripe", 100, 20, 180, 110)]},
        {"stem": "i08", "width": 200, "height": 150,
         "boxes": [("ripe", 20, 20, 90, 100), ("ripe", 100, 20, 180, 120)]},
        {"stem": "i09", "width": 200, "height": 150,
         "boxes": [("unripe", 20, 20, 90, 110)]},
        {"stem": "i10", "width": 200, "height": 150,
         "boxes": [("unripe", 100, 20, 180, 110)]},
        {"stem": "i11", "width": 200, "height": 150,
         "boxes": [("unripe", 20, 20, 90, 100), ("unripe", 100, 20, 180, 120)]},
        {"stem": "i12", "width": 200, "height": 150,
         "boxes": [("unripe", 20, 20, 90, 110)]},
        {"stem": "i13", "width": 200, "height": 150,
         "boxes": [("ripe", 20, 20, 90, 110)]},
    ]
    return specs
