"""1 类检测器封装（Ultralytics YOLO）。

设计成薄封装：训练/验证/推理都走 Ultralytics，不自己写检测框架。
缺少 ultralytics 时只在真正调用时报错，保证数据模块可以独立运行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DependencyError, PipelineError


def require_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise DependencyError(
            "训练/推理检测器需要 ultralytics：\n"
            "  pip install ultralytics\n"
            "建议使用 Python 3.10–3.12 的虚拟环境（当前 Python 3.14 可能没有对应 torch 轮子）。"
        ) from exc
    return YOLO


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    cls: int = 0

    def to_dict(self) -> dict:
        return {"box": self.box, "score": self.score, "cls": self.cls}


class Detector:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        max_det: int = 100,
    ):
        self.weights = weights
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            YOLO = require_ultralytics()
            self._model = YOLO(self.weights)
        return self._model

    def _device(self) -> str:
        from ..config import resolve_device

        return resolve_device(self.device)

    # ---------- 训练 ----------
    def train(
        self,
        data_yaml: str | Path,
        epochs: int | None = None,
        batch: int | None = None,
        imgsz: int | None = None,
        patience: int | None = None,
        project: str | Path | None = None,
        name: str | None = None,
        workers: int = 4,
        augment: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict:
        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise PipelineError(
                f"data.yaml 不存在：{data_yaml}\n"
                "先运行：python -m strawberry_cv.cli export-yolo --class-mode 1cls"
            )
        YOLO = require_ultralytics()
        model = YOLO(self.weights)
        train_args = {
            "data": str(data_yaml),
            "epochs": int(epochs if epochs is not None else 60),
            "batch": int(batch if batch is not None else 16),
            "imgsz": int(imgsz if imgsz is not None else self.imgsz),
            "patience": int(patience if patience is not None else 20),
            "device": self._device(),
            "workers": int(workers),
            "project": str(project) if project else None,
            "name": name or "strawberry_1cls",
            "exist_ok": True,
            "verbose": True,
        }
        if project is None:
            train_args.pop("project")
        if augment:
            train_args.update(augment)
        if extra:
            train_args.update(extra)
        results = model.train(**train_args)
        save_dir = str(getattr(model.trainer, "save_dir", "")) if model.trainer else ""
        metrics = {}
        try:
            metrics = {
                "map": float(results.box.map),
                "map50": float(results.box.map50),
                "map75": float(results.box.map75),
            }
        except Exception:
            pass
        return {
            "save_dir": save_dir,
            "weights": str(Path(save_dir) / "weights" / "best.pt") if save_dir else "",
            "metrics": metrics,
        }

    # ---------- 推理 ----------
    def predict(
        self,
        image: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> list[Detection]:
        if image is None or not isinstance(image, np.ndarray):
            raise PipelineError("Detector.predict 需要 BGR numpy 图像")
        results = self.model.predict(
            source=image,
            conf=float(conf if conf is not None else self.conf),
            iou=float(iou if iou is not None else self.iou),
            imgsz=int(imgsz if imgsz is not None else self.imgsz),
            max_det=self.max_det,
            device=self._device(),
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, classes):
            detections.append(
                Detection((float(x1), float(y1), float(x2), float(y2)),
                          float(score), int(cls))
            )
        return detections

    # ---------- 验证 ----------
    def val(
        self,
        data_yaml: str | Path,
        split: str = "val",
        project: str | Path | None = None,
        name: str | None = None,
    ) -> dict:
        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise PipelineError(f"data.yaml 不存在：{data_yaml}")
        kwargs = {
            "data": str(data_yaml),
            "split": split,
            "device": self._device(),
            "verbose": False,
        }
        if project is not None:
            kwargs["project"] = str(project)
        if name:
            kwargs["name"] = name
        results = self.model.val(**kwargs)
        out = {"split": split}
        try:
            out.update(
                {
                    "map": float(results.box.map),
                    "map50": float(results.box.map50),
                    "map75": float(results.box.map75),
                    "maps": [float(x) for x in results.box.maps],
                    "speed": dict(results.speed),
                }
            )
        except Exception as exc:
            out["error"] = str(exc)
        return out

    def export(self, fmt: str = "onnx") -> str:
        path = self.model.export(format=fmt)
        return str(path)

    @classmethod
    def from_checkpoint(cls, weights: str | Path, **kwargs) -> "Detector":
        return cls(weights=str(weights), **kwargs)
