"""生成 Demo 成果画廊：把若干张标注结果拼成一张对比图。

用于汇报/答辩时一眼看到 demo 在两类场景下的表现。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "handoff_2026-11-22"))

from demo_core import MaturityDemo, list_images, read_image, save_image  # noqa: E402

ROOT = Path(r"E:\临时")
PKG = ROOT / "handoff_2026-11-22"

SAMPLES = [
    ("strawberry_v1", "网络单果特写"),
    ("strawberry_archive_v1", "温室大棚场景"),
]


def pick_images(dataset: str, count: int) -> list[Path]:
    folder = ROOT / "datasets" / dataset / "detection" / "images"
    images = list_images(folder)
    if not images:
        return []
    step = max(1, len(images) // count)
    return images[::step][:count]


def label_strip(text: str, width: int, height: int = 34) -> np.ndarray:
    """生成一条带文字的标题条（用 PIL 渲染中文）。"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    font = None
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        try:
            font = ImageFont.truetype(path, 19)
            break
        except Exception:
            continue
    if font is not None:
        draw.text((10, 6), text, font=font, fill=(235, 235, 235))
    return np.asarray(img)[:, :, ::-1].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 demo 成果画廊")
    ap.add_argument("--out", default=str(ROOT / "demo_out" / "gallery.jpg"))
    ap.add_argument("--per-dataset", type=int, default=4)
    ap.add_argument("--cell-width", type=int, default=420)
    args = ap.parse_args()

    merged = PKG / "work_merged/runs/classifier/resnet18_3cls_merged/best.pt"
    detector = PKG / "work_merged/runs/detector/merged_1cls_lowlr/weights/best.pt"
    if not detector.exists():
        detector = PKG / "work/runs/detector/strawberry_1cls_lowlr/weights/best.pt"
    if not merged.exists():
        merged = PKG / "work/runs/classifier/resnet18_3cls/best.pt"

    demo = MaturityDemo(detector_weights=detector, classifier_checkpoint=merged,
                         conf=0.55, device="cpu")

    tiles = []
    for dataset, title in SAMPLES:
        for path in pick_images(dataset, args.per_dataset):
            image = read_image(path)
            if image is None:
                continue
            report = demo.analyze(image, source=path.name)
            canvas = demo.render(image, report)

            h, w = canvas.shape[:2]
            scale = args.cell_width / w
            resized = np.asarray(
                __import__("cv2").resize(canvas, (args.cell_width, int(h * scale)))
            )
            strip = label_strip(
                f"{title} · {path.stem} · {report.total} 颗 · 成熟度 {report.maturity*100:.0f}%",
                args.cell_width,
            )
            tiles.append(np.vstack([strip, resized]))

    if not tiles:
        print("没有可用的样例图")
        return

    cell_h = max(t.shape[0] for t in tiles)
    padded = [
        np.vstack([t, np.zeros((cell_h - t.shape[0], t.shape[1], 3), np.uint8)])
        for t in tiles
    ]
    rows = [np.hstack(padded[i:i + 2]) for i in range(0, len(padded), 2)]
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)])
            for r in rows]
    height = max(r.shape[0] for r in rows)
    rows = [np.vstack([r, np.zeros((height - r.shape[0], r.shape[1], 3), np.uint8)])
            for r in rows]

    gallery = np.vstack(rows)
    save_image(args.out, gallery)
    print(f"已生成 {args.out}（{len(tiles)} 张样例，{gallery.shape[1]}x{gallery.shape[0]}）")


if __name__ == "__main__":
    main()
