"""构建合并的 YOLO 检测数据集（v1 + 归档集）。

目的：现有检测器只在 64 张网络特写上训练过，在温室大棚域召回只有 0.55。
合并两个域重训一个检测器，demo 才能在两类图上都可用。

做法：
- 复用两套已导出的 YOLO 目录（图片是字节级副本，不重复占用空间）
- 合并 train/val/test 清单，两套数据各自的 scene_group 级划分原样保留
- 生成统一的 data.yaml
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PKG = Path(r"E:\临时\handoff_2026-11-22")

SOURCES = [
    ("v1", PKG / "work/yolo_detection_1cls", PKG / "work/splits/split_manifest.csv"),
    ("ar", PKG / "work_archive/yolo_detection_1cls",
     PKG / "work_archive/splits/split_manifest.csv"),
]
OUT = PKG / "work_merged/yolo_detection_1cls"
SPLITS = ("train", "val", "test")


def main() -> None:
    import csv

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images" / "all").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "all").mkdir(parents=True, exist_ok=True)

    manifest_lines: dict[str, list[str]] = {s: [] for s in SPLITS}
    label_rows: list[dict] = []
    copied = 0

    for tag, root, split_csv in SOURCES:
        if not root.is_dir():
            raise SystemExit(f"缺少 YOLO 目录：{root}")
        with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            stem = row["image"]
            split = row["split"]
            src_img = root / "images" / "all" / f"{stem}.jpg"
            src_lbl = root / "labels" / "all" / f"{stem}.txt"
            if not src_img.exists():
                continue
            dst_img = OUT / "images" / "all" / f"{stem}.jpg"
            dst_lbl = OUT / "labels" / "all" / f"{stem}.txt"
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                copied += 1
            manifest_lines[split].append(f"./images/all/{stem}.jpg")
            label_rows.append({
                "image": stem, "source": tag, "split": split,
                "scene_group": row.get("scene_group", ""),
                "boxes": row.get("boxes", ""),
            })

    for split, lines in manifest_lines.items():
        (OUT / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "data.yaml").write_text(
        "# 合并检测数据集：strawberry_v1（网络特写） + strawberry_archive_v1（温室大棚）\n"
        "# train/val/test 清单沿用两套数据各自的 scene_group 级划分\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "nc: 1\n"
        "names:\n"
        "  0: strawberry\n",
        encoding="utf-8",
    )

    with (OUT / "merged_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "source", "split",
                                               "scene_group", "boxes"])
        writer.writeheader()
        writer.writerows(label_rows)

    report = {
        "out_dir": str(OUT),
        "copied_images": copied,
        "splits": {s: len(manifest_lines[s]) for s in SPLITS},
        "per_source": {
            tag: sum(1 for r in label_rows if r["source"] == tag)
            for tag, _root, _csv in SOURCES
        },
    }
    (OUT / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
