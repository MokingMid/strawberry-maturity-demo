"""两阶段推理：1 类检测 → ROI 裁剪 → 3 类分类 → 跟踪/计数 → 可视化/CSV。

检测器和分类器通过鸭子类型注入：
- detector.predict(frame) -> [Detection 或 {"box","score"}]
- classifier.predict(roi)  -> (state, probs_dict)
这样在没有权重/没有 torch 的环境里，也能用桩对象测通整条管线。
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import CLASS_NAMES, DETECTION_CSV_COLUMNS
from ..data.labels import Box
from ..data.roi import crop_roi
from ..errors import PipelineError
from ..utils.image_io import imread, imwrite
from ..utils.io import get_logger, save_json
from .tracker import IoUTracker, LineCounter, Track
from .visualize import annotate

LOGGER = get_logger("pipeline")


@dataclass
class FruitResult:
    box: tuple[float, float, float, float]
    det_score: float
    state: str
    state_conf: float
    probs: dict[str, float] = field(default_factory=dict)
    track_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "box": self.box, "det_score": self.det_score,
            "state": self.state, "state_conf": self.state_conf,
            "probs": self.probs, "track_id": self.track_id,
        }


@dataclass
class FrameResult:
    frame_index: int
    fruits: list[FruitResult]
    counts: Counter
    annotated: np.ndarray
    latency_ms: float = 0.0
    tracks: list[Track] = field(default_factory=list)


def _as_box(item: dict, frame_shape: tuple[int, ...], index: int) -> Box:
    x1, y1, x2, y2 = (int(round(v)) for v in item["box"])
    h, w = frame_shape[:2]
    return Box("", index, "", x1, y1, x2, y2, w, h)


def _as_det(item) -> tuple[tuple, float]:
    if hasattr(item, "box"):
        return tuple(item.box), float(getattr(item, "score", 1.0))
    if isinstance(item, dict):
        return tuple(item["box"]), float(item.get("score", 1.0))
    raise PipelineError(f"无法识别的检测结果：{type(item)}")


class TwoStagePipeline:
    def __init__(
        self,
        detector=None,
        classifier=None,
        pad_ratio: float = 0.12,
        square: bool = True,
        min_size: int = 32,
        max_neighbor_iou: float = 0.1,
        tracker: IoUTracker | None = None,
        counter: LineCounter | None = None,
    ):
        self.detector = detector
        self.classifier = classifier
        self.pad_ratio = float(pad_ratio)
        self.square = bool(square)
        self.min_size = int(min_size)
        self.max_neighbor_iou = float(max_neighbor_iou)
        self.tracker = tracker
        self.counter = counter

    def _classify(self, crop) -> tuple[str, float, dict[str, float]]:
        if self.classifier is None:
            return "unknown", 0.0, {}
        output = self.classifier.predict(crop)
        if isinstance(output, tuple) and len(output) == 2:
            state, probs = output
        elif isinstance(output, dict):
            probs = output
            state = max(probs, key=probs.get)
        else:
            raise PipelineError("classifier.predict 必须返回 (state, probs) 或 probs 字典")
        probs = {name: float(probs.get(name, 0.0)) for name in CLASS_NAMES}
        conf = float(probs.get(state, 0.0))
        return state, conf, probs

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        gt_fruits: list[dict] | None = None,
    ) -> FrameResult:
        started = time.perf_counter()
        if frame is None or frame.size == 0:
            raise PipelineError("空帧")
        if gt_fruits is not None:
            det_items = [{"box": item["box"], "score": 1.0} for item in gt_fruits]
        else:
            if self.detector is None:
                raise PipelineError("没有检测器时，必须传入 gt_fruits（GT 框冒烟模式）")
            det_items = list(self.detector.predict(frame))

        boxes_and_scores = [_as_det(item) for item in det_items]
        pseudo_boxes = [
            _as_box({"box": box}, frame.shape, i)
            for i, (box, _) in enumerate(boxes_and_scores)
        ]

        fruits: list[FruitResult] = []
        for i, ((box, score), pseudo) in enumerate(zip(boxes_and_scores, pseudo_boxes)):
            roi = crop_roi(
                frame, pseudo,
                pad_ratio=self.pad_ratio, square=self.square, min_size=self.min_size,
                neighbor_boxes=pseudo_boxes, max_neighbor_iou=self.max_neighbor_iou,
            )
            if roi.crop is None:
                continue
            state, conf, probs = self._classify(roi.crop)
            fruits.append(FruitResult(box, score, state, conf, probs))

        tracks: list[Track] = []
        counts: Counter = Counter()
        if self.tracker is not None:
            detections = [
                {"box": f.box, "state": f.state, "state_conf": f.state_conf}
                for f in fruits
            ]
            tracks = self.tracker.update(detections)
            for fruit in fruits:
                best_id, best_iou = None, -1.0
                for track in tracks:
                    from ..utils.geometry import iou

                    value = iou(fruit.box, track.box)
                    if value > best_iou:
                        best_iou, best_id = value, track.track_id
                if best_iou >= self.tracker.iou_threshold:
                    fruit.track_id = best_id
            if self.counter is not None:
                counts = self.counter.update(tracks, frame_index)
            else:
                for track in tracks:
                    counts[track.state] += 1
        else:
            for fruit in fruits:
                counts[fruit.state] += 1

        line = (self.counter.p1, self.counter.p2) if self.counter is not None else None
        annotated = annotate(
            frame,
            [{"box": f.box, "state": f.state, "state_conf": f.state_conf,
              "track_id": f.track_id} for f in fruits],
            counts=counts,
            line=line,
        )
        return FrameResult(
            frame_index=frame_index,
            fruits=fruits,
            counts=counts,
            annotated=annotated,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            tracks=tracks,
        )

    # ---------- 图片/文件夹 ----------
    def run_image(
        self,
        image_path: str | Path,
        out_dir: str | Path,
        gt_fruits: list[dict] | None = None,
    ) -> dict:
        image = imread(image_path)
        if image is None:
            raise PipelineError(f"无法读取图片：{image_path}")
        result = self.process_frame(image, 0, gt_fruits=gt_fruits)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(image_path).stem}_annotated.jpg"
        imwrite(out_path, result.annotated)
        summary = {
            "image": str(image_path),
            "output": str(out_path),
            "fruits": [f.to_dict() for f in result.fruits],
            "counts": dict(result.counts),
            "latency_ms": result.latency_ms,
        }
        save_json(out_dir / f"{Path(image_path).stem}_result.json", summary)
        return summary

    def run_folder(
        self,
        images_dir: str | Path,
        out_dir: str | Path,
        gt_provider=None,
    ) -> dict:
        images_dir = Path(images_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        summaries = []
        for index, path in enumerate(sorted(images_dir.iterdir())):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            gt = gt_provider(path.stem) if gt_provider else None
            summary = self.run_image(path, out_dir, gt_fruits=gt)
            summaries.append(summary)
            for fruit in summary["fruits"]:
                rows.append(self._csv_row(index, path.stem, fruit))
        self._write_csv(out_dir / "predictions.csv", rows)
        report = {
            "images": len(summaries),
            "counts": dict(sum((Counter(s["counts"]) for s in summaries), Counter())),
            "avg_latency_ms": float(np.mean([s["latency_ms"] for s in summaries]))
            if summaries else 0.0,
            "summaries": summaries,
        }
        save_json(out_dir / "summary.json", report)
        return report

    # ---------- 视频 ----------
    def run_video(
        self,
        video_path: str | Path,
        out_dir: str | Path,
        max_frames: int | None = None,
    ) -> dict:
        import cv2

        video_path = Path(video_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise PipelineError(f"无法打开视频：{video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(out_dir / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height),
        )
        rows = []
        counts = Counter()
        latencies = []
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.process_frame(frame, frame_index)
                writer.write(result.annotated)
                latencies.append(result.latency_ms)
                for fruit in result.fruits:
                    rows.append(self._csv_row(frame_index, f"frame_{frame_index:06d}", fruit))
                if self.counter is None:
                    counts.update(result.counts)
                frame_index += 1
                if max_frames and frame_index >= max_frames:
                    break
        finally:
            cap.release()
            writer.release()
        if self.counter is not None:
            counts = Counter(self.counter.counts)
        self._write_csv(out_dir / "predictions.csv", rows)
        report = {
            "video": str(video_path),
            "frames": frame_index,
            "fps": fps,
            "counts": dict(counts),
            "counts_mode": "line" if self.counter is not None else "per_frame",
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "output_video": str(out_dir / "annotated.mp4"),
        }
        save_json(out_dir / "summary.json", report)
        return report

    # ---------- 辅助 ----------
    @staticmethod
    def _csv_row(frame_index: int, image: str, fruit: dict) -> dict:
        probs = fruit.get("probs", {})
        row = {
            "frame": frame_index,
            "track_id": fruit.get("track_id", ""),
            "x1": round(fruit["box"][0], 1), "y1": round(fruit["box"][1], 1),
            "x2": round(fruit["box"][2], 1), "y2": round(fruit["box"][3], 1),
            "det_conf": round(float(fruit.get("det_score", 0.0)), 4),
            "state": fruit.get("state", ""),
            "state_conf": round(float(fruit.get("state_conf", 0.0)), 4),
        }
        for name in CLASS_NAMES:
            row[f"class_conf_{name}"] = round(float(probs.get(name, 0.0)), 4)
        return row

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=DETECTION_CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in DETECTION_CSV_COLUMNS})
