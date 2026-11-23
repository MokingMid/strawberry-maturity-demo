"""把 PyTorch 模型导出为浏览器可用的 ONNX（检测器 fp32 + 分类器 int8）。

浏览器端 Demo（site/demo/）用的是这两个文件，导出过程如下：

1) 检测器：Ultralytics 自带导出，输出 [1, 5, 8400]（cx,cy,w,h + 1 个 sigmoid 类别分数）
   —— NMS 不在图里，由 JS 侧实现，与 ultralytics 的 non_max_suppression 对齐。
2) 分类器：torch.onnx.export（legacy TorchScript exporter，ORT Web 兼容性最好）
   —— 输入 [1,3,224,224]，输出 logits。
3) 分类器 int8：onnxruntime.quantization 静态量化（MinMax 校准），
   体积 42.6MB → 10.8MB，test 集精度 0.8770 → 0.8730。

用法：
    python demo/tools/export_onnx.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MODELS = REPO / "models"
CLASSES = ["ripe", "half", "unripe"]
# ImageFolder 按字母序：模型输出索引 0/1/2 → half/ripe/unripe
FOLDER_ORDER = ["half", "ripe", "unripe"]


# ---------------------------------------------------------------- 公共


def imread_safe(path) -> np.ndarray | None:
    """cv2.imread 在 Windows 中文路径下会返回 None，走 imdecode。"""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def classifier_input(crop: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """对齐 torchvision: Resize(256 短边) → CenterCrop(224) → ToTensor → Normalize。"""
    import cv2

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    size = int(round(imgsz * 256 / 224))
    h, w = crop.shape[:2]
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_h, new_w = size, int(size * w / h)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = (new_h - imgsz) // 2, (new_w - imgsz) // 2
    cropped = resized[top:top + imgsz, left:left + imgsz]
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = rgb.transpose(2, 0, 1)
    return ((tensor - mean[:, None, None]) / std[:, None, None])[None]


# ---------------------------------------------------------------- 检测器


def export_detector() -> Path:
    from ultralytics import YOLO

    src = MODELS / "detector_merged.pt"
    out = MODELS / "detector_merged.onnx"
    print("=== 导出检测器 ONNX ===", flush=True)
    YOLO(str(src)).export(format="onnx", imgsz=640, opset=17, simplify=False,
                           dynamic=False, nms=False, device="cpu")
    return out


# ---------------------------------------------------------------- 分类器


def export_classifier_fp32() -> Path:
    from strawberry_cv.models.classifier import Classifier

    src = MODELS / "classifier_merged.pt"
    out = MODELS / "classifier_merged.onnx"
    print("=== 导出分类器 ONNX (fp32) ===", flush=True)
    clf = Classifier.from_checkpoint(src, device="cpu")
    clf.model.eval()
    torch.onnx.export(
        clf.model, torch.zeros(1, 3, 224, 224), str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    return out


def _roi_root() -> Path:
    """分类器训练用的 ROI 目录（做量化校准与评测）。"""
    for cand in (REPO / "work_merged/roi_classification",
                 REPO / "handoff_2026-11-22/work_merged/roi_classification"):
        if cand.is_dir():
            return cand
    raise SystemExit("找不到 ROI 目录（用于量化校准），请先跑 export-roi")


def collect_calibration(n: int = 200) -> np.ndarray:
    root = _roi_root()
    files = []
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            files += sorted((root / split / cls).glob("*.jpg"))[:40]
    data = []
    for f in files[:n]:
        img = imread_safe(f)
        if img is not None:
            data.append(classifier_input(img)[0])
    return np.stack(data).astype(np.float32)


def quantize_int8(fp32_path: Path, out_path: Path, calib: np.ndarray) -> Path:
    """静态量化。onnx 的文件 IO 在中文路径下会出问题，先搬到 ASCII 临时目录。"""
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    tmp = Path(tempfile.mkdtemp(prefix="onnxq_"))
    try:
        src = tmp / "model.onnx"
        shutil.copy2(fp32_path, src)
        prep = tmp / "model_prep.onnx"
        quant_pre_process(str(src), str(prep), skip_symbolic_shape=False)

        class Reader(CalibrationDataReader):
            def __init__(self, arr):
                self.arr = arr
                self.i = 0

            def get_next(self):
                if self.i >= len(self.arr):
                    return None
                item = {"input": self.arr[self.i: self.i + 1]}
                self.i += 1
                return item

        out = tmp / "model_int8.onnx"
        quantize_static(
            str(prep), str(out), Reader(calib),
            quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
            weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
        shutil.copy2(out, out_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_path


def evaluate_onnx(path: Path) -> tuple[float, dict]:
    import onnxruntime as ort

    root = _roi_root()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    per: dict[str, list[int]] = {}
    total = correct = 0
    for cls in CLASSES:
        for f in sorted((root / "test" / cls).glob("*.jpg")):
            img = imread_safe(f)
            if img is None:
                continue
            logits = sess.run(None, {"input": classifier_input(img).astype(np.float32)})[0][0]
            pred = FOLDER_ORDER[int(np.argmax(logits))]
            total += 1
            correct += int(pred == cls)
            per.setdefault(cls, [0, 0])
            per[cls][0] += int(pred == cls)
            per[cls][1] += 1
    return correct / max(1, total), {k: (v[0] / v[1], v[1]) for k, v in per.items()}


# ---------------------------------------------------------------- main


def main() -> int:
    det = export_detector()
    print(f"  {det.name}  {det.stat().st_size / 1024 / 1024:.1f} MB\n", flush=True)

    fp32 = export_classifier_fp32()
    print(f"  {fp32.name}  {fp32.stat().st_size / 1024 / 1024:.1f} MB", flush=True)

    print("\n=== 量化校准 ===", flush=True)
    calib = collect_calibration(200)
    int8 = quantize_int8(fp32, MODELS / "classifier_merged_int8.onnx", calib)
    print(f"  {int8.name}  {int8.stat().st_size / 1024 / 1024:.1f} MB\n", flush=True)

    print("=== test 集准确率 ===", flush=True)
    acc32, per32 = evaluate_onnx(fp32)
    print(f"  fp32 : acc {acc32:.4f}  {per32}", flush=True)
    acc8, per8 = evaluate_onnx(int8)
    print(f"  int8 : acc {acc8:.4f}  {per8}", flush=True)
    print(f"  量化损失: {acc32 - acc8:+.4f}", flush=True)

    # fp32 版本体积大，服务端用 .pt 就够，这里可以删掉
    fp32.unlink(missing_ok=True)
    print("\n已删除 fp32 分类器 ONNX（服务端用 .pt 即可）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
