"""三状态分类器（ResNet18 / ResNet34 / ResNet50 迁移学习）。

输入是 GT 框裁剪出的单果 ROI；输出 ripe / half / unripe。
训练与推理都显式处理 ImageFolder 的字母序类别映射，避免类别错位。
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import CLASS_NAMES
from ..errors import DependencyError, PipelineError
from ..utils.io import get_logger, save_json

LOGGER = get_logger("classifier")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def require_torch():
    try:
        import torch
        import torchvision
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on env
        raise DependencyError(
            "训练/推理分类器需要 torch + torchvision：\n"
            "  pip install torch torchvision\n"
            "建议使用 Python 3.10–3.12 的虚拟环境。"
        ) from exc
    return torch, torchvision, nn, DataLoader, datasets, models, transforms, Image


def _resolve_torch_device(value: str | int | None = None) -> str:
    torch, *_ = require_torch()
    if value in (None, "", "auto"):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    text = str(value)
    if text.isdigit():
        return f"cuda:{text}"
    return text


class Classifier:
    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 3,
        imgsz: int = 224,
        device: str = "auto",
        class_names: tuple[str, ...] = CLASS_NAMES,
    ):
        self.arch = arch
        self.num_classes = int(num_classes)
        self.imgsz = int(imgsz)
        self.device_name = device
        self.class_names = tuple(class_names)
        self.model = None
        self.folder_classes: list[str] = []
        self._device = None
        self._transform = None

    # ---------- 构建 ----------
    def build(self, pretrained: bool = True) -> None:
        torch, _, nn, _, _, models, _, _ = require_torch()
        builders = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
            "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
            "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
        }
        if self.arch not in builders:
            raise ValueError(f"不支持的 arch：{self.arch}，可选 {sorted(builders)}")
        builder, weights_enum = builders[self.arch]
        weights = weights_enum if pretrained else None
        try:
            model = builder(weights=weights)
        except Exception as exc:
            LOGGER.warning("预训练权重加载失败（%s），改为从零训练", exc)
            model = builder(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        self._device = torch.device(_resolve_torch_device(self.device_name))
        self.model = model.to(self._device)
        self._transform = self._val_transform()

    def _train_transform(self):
        _, _, _, _, _, _, transforms, _ = require_torch()
        return transforms.Compose([
            transforms.RandomResizedCrop(self.imgsz, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            # 颜色是成熟度线索，hue 只做很小的扰动
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _val_transform(self):
        _, _, _, _, _, _, transforms, _ = require_torch()
        size = int(round(self.imgsz * 256 / 224))
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(self.imgsz),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # ---------- 训练 ----------
    def train(
        self,
        train_dir: str | Path,
        val_dir: str | Path,
        out_dir: str | Path,
        epochs: int = 30,
        batch: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        patience: int = 8,
        workers: int = 4,
        class_weights: str = "balanced",
        pretrained: bool = True,
        seed: int = 42,
    ) -> dict:
        torch, _, nn, DataLoader, datasets, _, transforms, _ = require_torch()
        from ..evaluation.classification import classification_metrics, save_confusion_png
        from ..training.common import EarlyStopping

        train_dir, val_dir = Path(train_dir), Path(val_dir)
        if not train_dir.is_dir():
            raise PipelineError(f"训练目录不存在：{train_dir}")
        if not val_dir.is_dir():
            raise PipelineError(f"验证目录不存在：{val_dir}")
        if os.name == "nt" and workers > 0:
            LOGGER.warning("Windows 下 DataLoader workers 设为 0，避免多进程问题")
            workers = 0

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_ds = datasets.ImageFolder(train_dir, transform=self._train_transform())
        val_ds = datasets.ImageFolder(val_dir, transform=self._val_transform())
        self.folder_classes = list(train_ds.classes)
        if set(self.folder_classes) != set(self.class_names):
            raise PipelineError(
                f"训练目录类别 {self.folder_classes} 与期望 {self.class_names} 不一致"
            )
        if list(val_ds.classes) != self.folder_classes:
            LOGGER.warning("验证集类别顺序与训练集不同：%s vs %s",
                           val_ds.classes, self.folder_classes)

        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=True, num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch, shuffle=False, num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )

        if self.model is None:
            self.build(pretrained=pretrained)

        counts = np.array([len(list((train_dir / c).glob("*"))) for c in self.folder_classes],
                          dtype=float)
        if np.any(counts == 0):
            LOGGER.warning("训练集存在空类别：%s",
                           {c: int(n) for c, n in zip(self.folder_classes, counts)})
        if class_weights == "balanced":
            total = counts.sum()
            weights = np.where(counts > 0, total / (len(counts) * np.maximum(counts, 1)), 0.0)
        else:
            weights = None
        weight_tensor = (
            torch.tensor(weights, dtype=torch.float32, device=self._device)
            if weights is not None else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stopper = EarlyStopping(patience=patience, mode="max")
        history: list[dict[str, Any]] = []
        best_f1 = -1.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            seen = 0
            for images, targets in train_loader:
                images = images.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * images.size(0)
                seen += images.size(0)
            train_loss = running / max(1, seen)

            metrics = self._evaluate_loader(val_loader, val_ds.classes)
            val_loss = metrics["loss"]
            scheduler.step()
            row = {
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_acc": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            LOGGER.info(
                "epoch %02d | train_loss %.4f | val_loss %.4f | acc %.3f | macroF1 %.3f",
                epoch, train_loss, val_loss, metrics["accuracy"], metrics["macro_f1"],
            )

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                self.save(out_dir / "best.pt")
                save_json(out_dir / "best_metrics.json", metrics)
                save_confusion_png(
                    metrics["confusion"], metrics["labels"],
                    out_dir / "best_confusion.png", title=f"epoch {epoch}",
                )

            if stopper.step(metrics["macro_f1"]):
                LOGGER.info("early stopping at epoch %d", epoch)
                break

        save_json(out_dir / "history.json", {"history": history,
                                             "folder_classes": self.folder_classes})
        return {
            "out_dir": str(out_dir),
            "best_macro_f1": best_f1,
            "best_checkpoint": str(out_dir / "best.pt"),
            "folder_classes": self.folder_classes,
            "history": history,
        }

    def _evaluate_loader(self, loader, target_classes: list[str] | None = None) -> dict:
        torch, *_ = require_torch()
        from ..evaluation.classification import classification_metrics

        if self.model is None:
            raise PipelineError("模型尚未构建/加载")
        if not self.folder_classes:
            raise PipelineError("缺少类别映射，请先训练或加载 checkpoint")
        # 目标索引属于当前 ImageFolder，预测索引属于训练时的 folder_classes，
        # 两者在验证集缺类时可能不同，必须分别映射。
        target_names = list(target_classes or self.folder_classes)
        pred_names = list(self.folder_classes)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()
        y_true, y_pred = [], []
        loss_sum, seen = 0.0, 0
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits = self.model(images)
                loss_sum += float(criterion(logits, targets).item()) * images.size(0)
                seen += images.size(0)
                preds = logits.argmax(dim=1).detach().cpu().tolist()
                labels = targets.detach().cpu().tolist()
                y_true.extend(target_names[i] for i in labels)
                y_pred.extend(pred_names[i] for i in preds)
        metrics = classification_metrics(y_true, y_pred, labels=list(self.class_names))
        metrics["loss"] = loss_sum / max(1, seen)
        metrics["count"] = seen
        return metrics

    def evaluate(self, data_dir: str | Path, batch: int = 32, workers: int = 4) -> dict:
        torch, _, _, DataLoader, datasets, _, _, _ = require_torch()
        if self.model is None:
            raise PipelineError("模型尚未构建/加载")
        if os.name == "nt" and workers > 0:
            workers = 0
        ds = datasets.ImageFolder(Path(data_dir), transform=self._val_transform())
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
        return self._evaluate_loader(loader, ds.classes)

    # ---------- 推理 ----------
    def predict(self, image_bgr: np.ndarray) -> tuple[str, dict[str, float]]:
        torch, _, _, _, _, _, _, Image = require_torch()
        if self.model is None:
            raise PipelineError("模型尚未构建/加载")
        if not self.folder_classes:
            raise PipelineError("分类器缺少类别映射，请先训练或加载 checkpoint")
        if image_bgr is None or image_bgr.size == 0:
            raise PipelineError("Classifier.predict 收到空图")
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self._device)
        self.model.eval()
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].detach().cpu().tolist()
        mapping = {
            name: float(prob) for name, prob in zip(self.folder_classes, probs)
        }
        # 补齐缺失类别
        full = {name: mapping.get(name, 0.0) for name in self.class_names}
        best = max(full, key=full.get)
        return best, full

    # ---------- 保存/加载 ----------
    def save(self, path: str | Path) -> None:
        torch, *_ = require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": self.arch,
                "num_classes": self.num_classes,
                "imgsz": self.imgsz,
                "class_names": list(self.class_names),
                "folder_classes": list(self.folder_classes),
                "state_dict": self.model.state_dict(),
                "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        torch, _, nn, _, _, models, _, _ = require_torch()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.0
            ckpt = torch.load(path, map_location="cpu")
        self.arch = ckpt.get("arch", self.arch)
        self.num_classes = int(ckpt.get("num_classes", self.num_classes))
        self.imgsz = int(ckpt.get("imgsz", self.imgsz))
        self.class_names = tuple(ckpt.get("class_names", self.class_names))
        self.folder_classes = list(ckpt.get("folder_classes", []))
        self._device = torch.device(_resolve_torch_device(self.device_name))
        self.model = self._build_architecture(models, nn, pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self._device)
        self._transform = self._val_transform()

    def _build_architecture(self, models, nn, pretrained: bool):
        builders = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
        }
        return builders[self.arch](weights=None)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto", **kwargs) -> "Classifier":
        instance = cls(device=device, **kwargs)
        instance.load(path)
        return instance
