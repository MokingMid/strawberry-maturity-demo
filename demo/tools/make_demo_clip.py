"""用数据集图片合成一段短视频，用于演示 demo 的视频模式与目标追踪。

温室连拍风格的平移序列：每帧取一张图，模拟摄像机缓慢移动。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "handoff_2026-11-22"))

from demo_core import UnicodeVideoWriter  # noqa: E402
from strawberry_cv.utils.image_io import imread  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="合成演示视频")
    ap.add_argument("--dataset", default="archive", choices=["archive", "v1"])
    ap.add_argument("--out", default=r"E:\临时\demo_out\demo_clip.mp4")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--pan", type=float, default=3.0, help="每帧平移像素")
    args = ap.parse_args()

    root = Path(r"E:\临时\datasets")
    src = (root / "strawberry_archive_v1/detection/images" if args.dataset == "archive"
           else root / "strawberry_v1/detection/images")
    images = sorted(src.glob("*.jpg"))
    if not images:
        raise SystemExit(f"没有找到图片：{src}")

    size = (640, 640)
    out = Path(args.out)
    written = 0
    with UnicodeVideoWriter(out, args.fps, size) as writer:
        for i in range(args.frames):
            path = images[i % len(images)]
            frame = imread(path)
            if frame is None:
                continue
            frame = cv2.resize(frame, size)
            # 轻微平移，模拟手持晃动
            dx = int(round(args.pan * (i % 12)))
            dy = int(round(args.pan * 0.5 * ((i + 3) % 9)))
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, matrix, size, borderMode=cv2.BORDER_REPLICATE)
            writer.write(frame)
            written += 1

    print(f"已生成 {out}（{written} 帧 / {args.fps} fps / {size[0]}x{size[1]}）")


if __name__ == "__main__":
    import numpy as np

    main()
