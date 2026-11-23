"""草莓成熟度检测 Demo —— 核心模块。

两阶段推理：
  1) YOLO 检测器找出画面里每一颗草莓
  2) ResNet18 对每颗草莓的 ROI 裁剪做成熟度三分类（ripe / half / unripe）

本模块只负责"模型 + 推理 + 渲染"，命令行交互在 strawberry_demo.py。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 让 demo 目录可以直接运行：把算法包根目录加入 sys.path
try:
    from paths import PACKAGE_DIR as _PKG_ROOT  # noqa: E402
except ImportError:  # pragma: no cover - 直接以脚本方式运行时
    _PKG_ROOT = Path(__file__).resolve().parents[1] / "handoff_2026-11-22"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from strawberry_cv.constants import CLASS_NAMES  # noqa: E402
from strawberry_cv.data.roi import crop_roi  # noqa: E402
from strawberry_cv.utils.image_io import imread, imwrite  # noqa: E402

# ---------------------------------------------------------------- 配色与文案

# BGR。ripe 用红、half 用橙、unripe 用绿，和「红=熟」的自然认知一致。
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "ripe": (48, 48, 220),      # BGR 偏红
    "half": (0, 165, 255),      # BGR 橙
    "unripe": (96, 200, 96),    # BGR 绿
    "unknown": (180, 180, 180),
}
STATE_CN: dict[str, str] = {
    "ripe": "成熟",
    "half": "半熟",
    "unripe": "未熟",
    "unknown": "未知",
}
# 成熟度权重：成熟 1.0，半熟 0.5，未熟 0
STATE_WEIGHT: dict[str, float] = {"ripe": 1.0, "half": 0.5, "unripe": 0.0}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------- 中文文字渲染
#
# OpenCV 的 cv2.putText 只能画 Hershey 字体，中文会变成 "???"，
# 所以所有中文文字改用 PIL 渲染（系统自带微软雅黑/黑体）。

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",     # 微软雅黑 Bold
    r"C:\Windows\Fonts\simhei.ttf",     # 黑体
    r"C:\Windows\Fonts\simsun.ttc",     # 宋体
    r"C:\Windows\Fonts\Deng.ttf",       # 等线
    r"C:\Windows\Fonts\arial.ttf",      # 纯 ASCII 兜底
)
_FONT_CACHE: dict[int, object] = {}
_FONT_PATH: str | None = None


def _load_font(size: int):
    """按字号缓存字体对象；找不到任何字体时返回 None。"""
    global _FONT_PATH
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
        except Exception:
            continue
        _FONT_CACHE[size] = font
        if _FONT_PATH is None:
            _FONT_PATH = path
        return font
    _FONT_CACHE[size] = None
    return None


def has_cjk_font() -> bool:
    """是否找到能显示中文的字体。"""
    global _FONT_PATH
    if _FONT_PATH is not None:
        return _FONT_PATH.lower().endswith((".ttc", ".ttf")) and "arial" not in _FONT_PATH.lower()
    for path in _FONT_CANDIDATES:
        if "arial" in path.lower():
            continue
        from PIL import ImageFont

        try:
            ImageFont.truetype(path, 12)
        except Exception:
            continue
        _FONT_PATH = path
        return True
    return False


class TextRenderer:
    """把 BGR numpy 图转成 PIL 图来画中文，最后再转回 BGR。"""

    def __init__(self, canvas: np.ndarray):
        from PIL import Image, ImageDraw

        self._canvas = canvas
        self._rgb = canvas[:, :, ::-1].copy()
        self._pil = Image.fromarray(self._rgb)
        self._draw = ImageDraw.Draw(self._pil)

    def font(self, size: int):
        return _load_font(max(8, int(size)))

    def size(self, text: str, font_size: int) -> tuple[int, int]:
        """返回文字的像素宽高。"""
        font = self.font(font_size)
        if font is None:
            return (len(text) * font_size // 2, font_size)
        try:
            left, top, right, bottom = font.getbbox(text)
        except Exception:
            return (len(text) * font_size // 2, font_size)
        return (right - left, bottom - top)

    def text(self, xy: tuple[int, int], text: str, font_size: int, color_bgr) -> None:
        font = self.font(font_size)
        if font is None:
            return
        self._draw.text(xy, text, font=font,
                        fill=(int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])))

    def rect(self, box, fill=None, outline=None, width: int = 1) -> None:
        if fill is not None:
            self._draw.rectangle(box, fill=(int(fill[2]), int(fill[1]), int(fill[0])))
        if outline is not None:
            self._draw.rectangle(box, outline=(int(outline[2]), int(outline[1]), int(outline[0])),
                                 width=int(width))

    def result(self) -> np.ndarray:
        return np.asarray(self._pil)[:, :, ::-1].copy()


# ---------------------------------------------------------------- 数据结构


@dataclass
class Fruit:
    """一颗草莓的检测 + 分类结果。"""

    box: tuple[float, float, float, float]
    det_conf: float
    state: str
    state_conf: float
    probs: dict[str, float] = field(default_factory=dict)
    track_id: int | None = None
    gt_state: str | None = None   # 仅自检模式：对应的 GT 类别

    @property
    def cn(self) -> str:
        return STATE_CN.get(self.state, self.state)


@dataclass
class FrameReport:
    """一帧（一张图）的完整分析结果。"""

    fruits: list[Fruit]
    counts: Counter
    latency_ms: float
    frame_index: int = 0
    source: str = ""

    @property
    def total(self) -> int:
        return len(self.fruits)

    @property
    def maturity(self) -> float:
        """成熟度指数 0~1：成熟记 1，半熟记 0.5，未熟记 0。"""
        if not self.fruits:
            return 0.0
        acc = sum(STATE_WEIGHT.get(f.state, 0.0) for f in self.fruits)
        return acc / len(self.fruits)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "frame_index": self.frame_index,
            "total": self.total,
            "counts": {name: int(self.counts.get(name, 0)) for name in CLASS_NAMES},
            "maturity": round(self.maturity, 4),
            "latency_ms": round(self.latency_ms, 2),
            "fruits": [
                {
                    "box": [round(v, 1) for v in f.box],
                    "det_conf": round(f.det_conf, 4),
                    "state": f.state,
                    "state_cn": f.cn,
                    "state_conf": round(f.state_conf, 4),
                    "probs": {k: round(v, 4) for k, v in f.probs.items()},
                    "track_id": f.track_id,
                }
                for f in self.fruits
            ],
        }


# ---------------------------------------------------------------- 追踪


class SimpleTracker:
    """轻量 IoU 追踪：给同一颗草莓在视频里稳定分配 ID。

    不追求工业级精度（换 ByteTrack / BoT-SORT 需另装依赖），只保证
    "同一颗草莓前后帧 ID 不乱跳"，够 demo 用了。
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: list[dict] = []
        self._next_id = 1

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, fruits: list[Fruit]) -> None:
        for fruit in fruits:
            best_id, best_iou = None, 0.0
            for track in self._tracks:
                value = self._iou(fruit.box, track["box"])
                if value > best_iou:
                    best_iou, best_id = value, track["id"]
            if best_id is not None and best_iou >= self.iou_threshold:
                fruit.track_id = best_id
                for track in self._tracks:
                    if track["id"] == best_id:
                        track["box"] = fruit.box
                        track["age"] = 0
                        track["state"] = fruit.state
                        break
            else:
                new_id = self._next_id
                self._next_id += 1
                fruit.track_id = new_id
                self._tracks.append(
                    {"id": new_id, "box": fruit.box, "age": 0, "state": fruit.state}
                )
        for track in self._tracks:
            if track["id"] not in {f.track_id for f in fruits}:
                track["age"] += 1
        self._tracks = [t for t in self._tracks if t["age"] <= self.max_age]


# ---------------------------------------------------------------- Demo 主体


class MaturityDemo:
    """成熟度检测 Demo。

    检测器与分类器都做成可选：缺权重时自动降级
    （只有分类器 → 只对整图分类；都没有 → 只做数据自检），
    保证在没有权重的机器上也能把程序跑起来看到界面。
    """

    def __init__(
        self,
        detector_weights: str | Path | None = None,
        classifier_checkpoint: str | Path | None = None,
        conf: float = 0.55,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "auto",
        pad_ratio: float = 0.12,
        min_size: int = 32,
        max_neighbor_iou: float = 0.1,
    ):
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        self.pad_ratio = float(pad_ratio)
        self.min_size = int(min_size)
        self.max_neighbor_iou = float(max_neighbor_iou)

        self.detector = None
        self.classifier = None
        self.tracker = SimpleTracker()

        if detector_weights:
            self.detector = self._load_detector(Path(detector_weights))
        if classifier_checkpoint:
            self.classifier = self._load_classifier(Path(classifier_checkpoint))

    # ---------- 模型加载 ----------

    def _load_detector(self, path: Path):
        from strawberry_cv.models.detector import Detector

        if not path.exists():
            raise FileNotFoundError(f"检测权重不存在：{path}")
        print(f"[demo] 加载检测器：{path.name}")
        return Detector(
            weights=str(path), device=self.device, conf=self.conf,
            iou=self.iou, imgsz=self.imgsz,
        )

    def _load_classifier(self, path: Path):
        from strawberry_cv.models.classifier import Classifier

        if not path.exists():
            raise FileNotFoundError(f"分类权重不存在：{path}")
        print(f"[demo] 加载分类器：{path.name}")
        return Classifier.from_checkpoint(path, device=self.device)

    # ---------- 推理 ----------

    def analyze(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        source: str = "",
        gt_boxes: list[dict] | None = None,
        reset_tracker: bool = True,
    ) -> FrameReport:
        """分析一帧，返回每颗草莓的状态与整帧统计。"""
        if frame is None or frame.size == 0:
            raise ValueError("空帧")
        started = time.perf_counter()

        if gt_boxes is not None:
            # 数据集自检模式：直接用 GT 框，绕过检测器
            dets = [(tuple(b["box"]), float(b.get("score", 1.0))) for b in gt_boxes]
            gt_names = [b.get("name") or None for b in gt_boxes]
        elif self.detector is not None:
            dets = [(tuple(d.box), float(d.score)) for d in self.detector.predict(frame)]
            gt_names = [None] * len(dets)
        else:
            dets = []
            gt_names = []

        # 构造伪 Box 供 crop_roi 使用（它需要 index / width / height 做邻果判断）
        from strawberry_cv.data.labels import Box

        h, w = frame.shape[:2]
        pseudo = [
            Box("", i, "", int(round(b[0])), int(round(b[1])),
                int(round(b[2])), int(round(b[3])), w, h)
            for i, (b, _) in enumerate(dets)
        ]

        fruits: list[Fruit] = []
        for i, ((box, score), pb) in enumerate(zip(dets, pseudo)):
            roi = crop_roi(
                frame, pb,
                pad_ratio=self.pad_ratio, square=True, min_size=self.min_size,
                neighbor_boxes=pseudo, max_neighbor_iou=self.max_neighbor_iou,
            )
            if roi.crop is None:
                continue
            gt_name = gt_names[i] if i < len(gt_names) else None
            if self.classifier is None:
                fruits.append(Fruit(box, score, "unknown", 0.0, {}, gt_state=gt_name))
                continue
            state, probs = self.classifier.predict(roi.crop)
            full = {name: float(probs.get(name, 0.0)) for name in CLASS_NAMES}
            fruits.append(
                Fruit(box, score, state, full.get(state, 0.0), full, gt_state=gt_name)
            )

        if reset_tracker:
            self.tracker = SimpleTracker()
        if fruits:
            self.tracker.update(fruits)

        counts = Counter(f.state for f in fruits)
        return FrameReport(
            fruits=fruits, counts=counts,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            frame_index=frame_index, source=source,
        )

    # ---------- 渲染 ----------

    def render(self, frame: np.ndarray, report: FrameReport, show_panel: bool = True) -> np.ndarray:
        """在帧上画框、标签和成熟度面板（中文用 PIL 渲染）。"""
        import cv2

        canvas = frame.copy()
        scale = max(0.45, min(1.6, max(canvas.shape[:2]) / 900.0))
        thickness = max(1, int(round(scale * 2)))

        # --- 阶段 1：检测框（cv2）---
        for fruit in report.fruits:
            x1, y1, x2, y2 = (int(round(v)) for v in fruit.box)
            color = STATE_COLORS.get(fruit.state, STATE_COLORS["unknown"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # --- 阶段 2：面板底板（cv2，半透明叠加）---
        panel = self._panel_geometry(canvas, report) if show_panel else None
        if panel is not None:
            x0, y0, pw, ph = panel
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (24, 24, 24), -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
            cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + ph), (90, 90, 90), thickness)

        # --- 阶段 3：所有文字（PIL，支持中文）---
        tr = TextRenderer(canvas)
        font_size = max(11, int(round(15 * scale)))

        for fruit in report.fruits:
            x1, y1, x2, y2 = (int(round(v)) for v in fruit.box)
            color = STATE_COLORS.get(fruit.state, STATE_COLORS["unknown"])
            label = f"{fruit.cn} {fruit.state_conf:.2f}"
            if fruit.track_id is not None:
                label = f"#{fruit.track_id} {label}"
            tw, th = tr.size(label, font_size)
            ty = max(th + 4, y1)
            # 标签落在面板区域内就不画，避免遮住面板统计
            if panel is not None and not (
                x1 + tw < panel[0] or x1 > panel[0] + panel[2]
                or ty - th - 4 < panel[1] or ty > panel[1] + panel[3]
            ):
                continue
            tr.rect((x1, ty - th - 4, x1 + tw + 6, ty + 3), fill=color)
            tr.text((x1 + 3, ty - th - 2), label, font_size, (255, 255, 255))

        if panel is not None:
            self._draw_panel_text(tr, panel, report, scale)

        return tr.result()

    def _panel_geometry(self, canvas: np.ndarray, report: FrameReport):
        h, w = canvas.shape[:2]
        scale = max(0.45, min(1.6, max(h, w) / 900.0))
        pw = int(262 * scale)
        ph = int((158 + 30 * len(CLASS_NAMES)) * scale)
        x0 = w - pw - int(12 * scale)
        y0 = int(12 * scale)
        if x0 < 0:
            return None
        ph = min(ph, h - y0 - 6)
        return x0, y0, pw, ph

    def _draw_panel_text(self, tr: TextRenderer, panel, report: FrameReport, scale: float) -> None:
        x0, y0, pw, ph = panel
        fs = max(11, int(round(15 * scale)))
        small = max(10, int(round(13 * scale)))
        pad = int(10 * scale)

        y = y0 + pad + fs
        tr.text((x0 + pad, y), "草莓成熟度检测 Demo", fs, (255, 255, 255))
        y += int(26 * scale)

        total = max(1, report.total)
        bar_x = x0 + pad + int(72 * scale)
        bar_w = max(20, int(92 * scale))
        bar_h = max(6, int(9 * scale))
        for name in CLASS_NAMES:
            count = int(report.counts.get(name, 0))
            ratio = count / total
            color = STATE_COLORS[name]
            tr.text((x0 + pad, y - bar_h), STATE_CN[name], small, color)
            tr.rect((bar_x, y - bar_h, bar_x + bar_w, y), fill=(60, 60, 60))
            if ratio > 0:
                tr.rect((bar_x, y - bar_h, bar_x + int(bar_w * ratio), y), fill=color)
            tr.text((bar_x + bar_w + int(6 * scale), y - bar_h), f"{count} {ratio * 100:.0f}%",
                    small, (230, 230, 230))
            y += int(22 * scale)

        y += int(4 * scale)
        tr.text((x0 + pad, y), f"总计 {report.total} 颗", fs, (255, 255, 255))
        y += int(24 * scale)
        mat = report.maturity
        color = (60, 220, 60) if mat >= 0.6 else ((0, 180, 255) if mat >= 0.3 else (60, 200, 255))
        tr.text((x0 + pad, y), f"成熟度 {mat * 100:.1f}%", fs, color)
        y += int(22 * scale)
        tr.text((x0 + pad, y), f"{report.latency_ms:.0f} ms", small, (180, 180, 180))

    # ---------- 导出 ----------

    @staticmethod
    def write_csv(path: str | Path, reports: list[FrameReport]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "source", "frame", "track_id", "x1", "y1", "x2", "y2",
            "det_conf", "state", "state_cn", "state_conf",
            "conf_ripe", "conf_half", "conf_unripe",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for report in reports:
                for fruit in report.fruits:
                    writer.writerow({
                        "source": report.source,
                        "frame": report.frame_index,
                        "track_id": fruit.track_id if fruit.track_id is not None else "",
                        "x1": round(fruit.box[0], 1), "y1": round(fruit.box[1], 1),
                        "x2": round(fruit.box[2], 1), "y2": round(fruit.box[3], 1),
                        "det_conf": round(fruit.det_conf, 4),
                        "state": fruit.state, "state_cn": fruit.cn,
                        "state_conf": round(fruit.state_conf, 4),
                        "conf_ripe": round(fruit.probs.get("ripe", 0.0), 4),
                        "conf_half": round(fruit.probs.get("half", 0.0), 4),
                        "conf_unripe": round(fruit.probs.get("unripe", 0.0), 4),
                    })

    @staticmethod
    def write_json(path: str | Path, reports: list[FrameReport]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        total = Counter()
        for report in reports:
            total.update(report.counts)
        payload = {
            "frames": len(reports),
            "total_fruits": int(sum(total.values())),
            "counts": {name: int(total.get(name, 0)) for name in CLASS_NAMES},
            "maturity": round(
                sum(STATE_WEIGHT.get(f.state, 0.0) for r in reports for f in r.fruits)
                / max(1, sum(total.values())), 4),
            "reports": [r.to_dict() for r in reports],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 便捷函数


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_gt_boxes(label_json: str | Path) -> list[dict]:
    """读数据集的 GT 标注（自检模式用）。"""
    data = json.loads(Path(label_json).read_text(encoding="utf-8"))
    return [
        {
            "box": (b["x1"], b["y1"], b["x2"], b["y2"]),
            "score": 1.0,
            "name": b.get("name", ""),
        }
        for b in data.get("labels", [])
    ]


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(path, image)


def read_image(path: str | Path) -> np.ndarray | None:
    return imread(path)


def open_video(path: str | Path):
    """中文路径安全的视频打开。

    cv2.VideoCapture 在 Windows 中文路径下会打不开（返回 isOpened()==False），
    遇到非 ASCII 路径时先复制到临时目录（纯 ASCII）再打开。
    返回 (capture, cleanup)；cleanup 需要在用完后调用来删临时文件。
    """
    import shutil
    import tempfile

    import cv2

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    def _is_ascii(p: Path) -> bool:
        try:
            str(p).encode("ascii")
        except UnicodeEncodeError:
            return False
        return True

    tmp_path = None
    target = path
    if not _is_ascii(path.resolve()):
        fd, tmp = tempfile.mkstemp(suffix=path.suffix or ".mp4")
        os.close(fd)
        tmp_path = Path(tmp)
        shutil.copy2(path, tmp_path)
        target = tmp_path

    cap = cv2.VideoCapture(str(target))
    if not cap.isOpened() and tmp_path is not None:
        cap.release()
        cap = cv2.VideoCapture(str(path))
        tmp_path = None

    def cleanup() -> None:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return cap, cleanup


class UnicodeVideoWriter:
    """中文路径安全的视频写出器。

    cv2.VideoWriter 在 Windows 中文路径下会静默失败（写出 0 字节），
    所以先写到系统临时目录（纯 ASCII 路径），关闭后再搬过去。
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int],
                 codec: str = "mp4v"):
        import shutil
        import tempfile

        import cv2

        self._final = Path(path)
        self._final.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        self._tmp = Path(tmp)
        self._writer = cv2.VideoWriter(
            str(self._tmp), cv2.VideoWriter_fourcc(*codec), float(fps), size
        )
        self._shutil = shutil

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> Path:
        self._writer.release()
        if self._final.exists():
            self._final.unlink()
        self._shutil.move(str(self._tmp), str(self._final))
        return self._final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except Exception:
            pass
        return False
