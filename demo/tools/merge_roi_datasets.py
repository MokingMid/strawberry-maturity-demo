"""把两套数据集的 ROI 裁剪合并成一个三分类训练集。

动机：
- v1（网络单果特写）只有 11 个 half 框 → 分类器学不到半熟，推理时永远不输出「半熟」
- archive（温室大棚）有 247 个 half 框，但域与 v1 不同
- 合并后同时覆盖两种拍摄域，half 样本量提升 22 倍

划分沿用各自数据集的 scene_group 级划分（不重新划分，避免泄漏）。
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

PKG = Path(r"E:\临时\handoff_2026-11-22")
SOURCES = [
    ("v1", PKG / "work/roi_classification"),
    ("ar", PKG / "work_archive/roi_classification"),
]
OUT = PKG / "work_merged/roi_classification"
SPLITS = ("train", "val", "test")
CLASSES = ("ripe", "half", "unripe")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    manifest_rows: list[dict] = []

    for tag, root in SOURCES:
        if not root.is_dir():
            raise SystemExit(f"缺少 ROI 目录：{root}")
        manifest = root / "roi_manifest.csv"
        for split in SPLITS:
            for cls in CLASSES:
                src_dir = root / split / cls
                dst_dir = OUT / split / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                if not src_dir.is_dir():
                    continue
                for path in sorted(src_dir.glob("*.jpg")):
                    shutil.copy2(path, dst_dir / f"{tag}_{path.name}")
                    counts[split][cls] += 1

        if manifest.exists():
            import csv

            with manifest.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    row["source_dataset"] = tag
                    manifest_rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    import csv

    # 两个来源的 manifest 列可能不同，取并集
    columns = ["source_dataset"]
    seen = set()
    for row in manifest_rows:
        for key in row:
            if key not in seen and key != "source_dataset":
                seen.add(key)
                columns.append(key)

    with (OUT / "roi_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "out_dir": str(OUT),
        "sources": {tag: str(root) for tag, root in SOURCES},
        "counts": {s: {c: counts[s][c] for c in CLASSES} for s in SPLITS},
        "totals": {s: sum(counts[s].values()) for s in SPLITS},
        "total_crops": sum(sum(counts[s].values()) for s in SPLITS),
    }
    (OUT / "roi_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
