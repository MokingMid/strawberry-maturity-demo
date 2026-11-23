"""三分类器训练入口（GT 框 ROI）。"""

from __future__ import annotations

from pathlib import Path

from ..config import Config, ensure_dirs
from ..errors import PipelineError
from ..models.classifier import Classifier
from ..training.common import class_balance_warnings, count_image_folder
from ..utils.io import get_logger, save_json

LOGGER = get_logger("train_classifier")


def run(
    config: Config,
    roi_dir: str | Path | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    name: str | None = None,
) -> dict:
    roi_root = Path(roi_dir) if roi_dir else config.derived("roi")
    train_dir = roi_root / "train"
    val_dir = roi_root / "val"
    if not train_dir.is_dir():
        raise PipelineError(
            f"ROI 训练目录不存在：{train_dir}\n"
            "先运行：python -m strawberry_cv.cli export-roi（需要先 make-split）"
        )
    if not val_dir.is_dir():
        raise PipelineError(f"ROI 验证目录不存在：{val_dir}")

    train_counts = count_image_folder(train_dir)
    val_counts = count_image_folder(val_dir)
    for warning in class_balance_warnings(train_counts, "train") + \
            class_balance_warnings(val_counts, "val"):
        LOGGER.warning(warning)

    runs_dir = config.derived("runs") / "classifier"
    ensure_dirs([runs_dir])
    run_name = name or config.get("classifier.run_name", "resnet18_3cls")
    out_dir = runs_dir / run_name

    classifier = Classifier(
        arch=config.get("classifier.arch", "resnet18"),
        num_classes=int(config.get("classifier.num_classes", 3)),
        imgsz=int(config.get("classifier.imgsz", 224)),
        device=device or config.get("classifier.device", "auto"),
        class_names=config.class_names(),
    )
    LOGGER.info("classifier: %s | train: %s", classifier.arch, train_dir)
    result = classifier.train(
        train_dir, val_dir, out_dir,
        epochs=int(epochs if epochs is not None else config.get("classifier.epochs", 30)),
        batch=int(batch if batch is not None else config.get("classifier.batch", 32)),
        lr=float(config.get("classifier.lr", 3e-4)),
        weight_decay=float(config.get("classifier.weight_decay", 1e-4)),
        label_smoothing=float(config.get("classifier.label_smoothing", 0.05)),
        patience=int(config.get("classifier.patience", 8)),
        workers=int(config.get("classifier.workers", 4)),
        class_weights=config.get("classifier.class_weights", "balanced"),
        pretrained=bool(config.get("classifier.pretrained", True)),
        seed=int(config.get("project.seed", 42)),
    )
    result["train_counts"] = train_counts
    result["val_counts"] = val_counts
    save_json(out_dir / "train_report.json", result)
    LOGGER.info("done: %s", out_dir)
    return result
