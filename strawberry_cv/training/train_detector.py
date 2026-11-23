"""检测器训练入口（1 类草莓）。"""

from __future__ import annotations

from pathlib import Path

from ..config import Config, ensure_dirs
from ..errors import PipelineError
from ..models.detector import Detector
from ..utils.io import get_logger, save_json

LOGGER = get_logger("train_detector")


def run(
    config: Config,
    data_yaml: str | Path | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    name: str | None = None,
) -> dict:
    data_yaml = Path(data_yaml) if data_yaml else config.derived("yolo_1cls") / "data.yaml"
    if not data_yaml.exists():
        raise PipelineError(
            f"data.yaml 不存在：{data_yaml}\n"
            "先运行：python -m strawberry_cv.cli export-yolo --class-mode 1cls"
        )
    runs_dir = config.derived("runs") / "detector"
    ensure_dirs([runs_dir])
    run_name = name or config.get("detector.run_name", "strawberry_1cls")

    detector = Detector(
        weights=config.get("detector.weights", "yolo11n.pt"),
        device=device or config.get("detector.device", "auto"),
        conf=float(config.get("detector.conf", 0.25)),
        iou=float(config.get("detector.iou", 0.5)),
        imgsz=int(imgsz or config.get("detector.imgsz", 640)),
        max_det=int(config.get("detector.max_det", 100)),
    )
    LOGGER.info("detector: %s | data: %s", detector.weights, data_yaml)
    result = detector.train(
        data_yaml,
        epochs=epochs if epochs is not None else config.get("detector.epochs", 60),
        batch=batch if batch is not None else config.get("detector.batch", 16),
        imgsz=imgsz or config.get("detector.imgsz", 640),
        patience=config.get("detector.patience", 20),
        project=runs_dir,
        name=run_name,
        workers=int(config.get("detector.workers", 4)),
        augment=config.get("augment", {}),
        extra=config.get("detector.train_args", {}),
    )
    save_json(runs_dir / run_name / "train_report.json", result)
    LOGGER.info("done: %s", result.get("save_dir"))
    return result
