"""草莓成熟度检测 —— 可视化网站后端。

基于 demo/demo_core.py 的两阶段推理（YOLO 检测 → ResNet18 成熟度分类），
把结果通过网页展示：上传图片 / 浏览数据集样例 / 批量处理 / 模型指标对比。

启动：
    python demo/web/app.py --port 8000
    然后浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

_HERE = Path(__file__).resolve().parent
_DEMO = _HERE.parent
_ROOT = _DEMO.parent
sys.path.insert(0, str(_DEMO))

from paths import (  # noqa: E402
    PACKAGE_DIR as _PKG_ROOT,
    find_weights,
    images_dir,
    labels_dir,
)
from demo_core import (  # noqa: E402
    CLASS_NAMES,
    MaturityDemo,
    list_images,
    load_gt_boxes,
    read_image,
    save_image,
)

PKG = _PKG_ROOT
UPLOAD_DIR = _HERE / "uploads"
RESULT_DIR = _HERE / "results"

DATASETS = {}
for _key, _label in (("v1", "网络单果特写"), ("archive", "温室大棚场景")):
    _imgs = images_dir(_key)
    _lbls = labels_dir(_key)
    if _imgs is None:
        continue
    DATASETS[_key] = {
        "label": _label,
        "dir": _imgs,
        "labels": _lbls,
    }

MODEL_FILES = {
    "detector": {
        "merged": find_weights("detector"),
        "v1": PKG / "work/runs/detector/strawberry_1cls_lowlr/weights/best.pt",
    },
    "classifier": {
        "merged": find_weights("classifier"),
        "archive": PKG / "work_archive/runs/classifier/resnet18_3cls_archive/best.pt",
        "v1": PKG / "work/runs/classifier/resnet18_3cls/best.pt",
    },
}

app = Flask(__name__, template_folder=str(_HERE / "templates"), static_folder=str(_HERE / "static"))
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB

# 模型只加载一次，全站共用
_demo: MaturityDemo | None = None
_demo_key: tuple | None = None
_lock = threading.Lock()


def get_demo(detector_key: str = "merged", classifier_key: str = "merged",
             conf: float = 0.55, iou: float = 0.5) -> MaturityDemo:
    """按需加载/切换模型。切换型号时会重建 Demo（重新载入权重）。"""
    global _demo, _demo_key
    with _lock:
        key = (detector_key, classifier_key, round(conf, 3), round(iou, 3))
        if _demo is not None and _demo_key == key:
            return _demo
        det_path = MODEL_FILES["detector"].get(detector_key)
        cls_path = MODEL_FILES["classifier"].get(classifier_key)
        if det_path is None or not det_path.exists():
            det_path = MODEL_FILES["detector"]["v1"]
        if cls_path is None or not cls_path.exists():
            cls_path = MODEL_FILES["classifier"]["v1"]
        _demo = MaturityDemo(
            detector_weights=det_path, classifier_checkpoint=cls_path,
            conf=conf, iou=iou, device="cpu",
        )
        _demo_key = key
        return _demo


# ---------------------------------------------------------------- 页面


@app.route("/")
def index():
    return render_template("index.html", datasets=DATASETS, models=MODEL_FILES)


def _gt_for(spec: dict, stem: str) -> list[dict]:
    """读取某张图的 GT 标注；没有标注目录时返回空。"""
    labels = spec.get("labels")
    if labels is None:
        return []
    path = labels / f"{stem}.json"
    if not path.exists():
        return []
    return load_gt_boxes(path)


# ---------------------------------------------------------------- API


@app.route("/api/samples")
def api_samples():
    """列出某个数据集的图片（分页）。"""
    key = request.args.get("dataset", "v1")
    page = max(1, int(request.args.get("page", 1)))
    size = min(120, max(1, int(request.args.get("size", 24))))
    spec = DATASETS.get(key)
    if spec is None:
        abort(404)
    images = list_images(spec["dir"])
    total = len(images)
    start = (page - 1) * size
    chunk = images[start:start + size]
    items = [
        {
            "name": p.name,
            "url": f"/api/sample/{key}/{p.name}",
            "thumb": f"/api/sample/{key}/{p.name}?thumb=1",
        }
        for p in chunk
    ]
    return jsonify({"total": total, "page": page, "size": size, "items": items})


@app.route("/api/sample/<dataset>/<filename>")
def api_sample(dataset: str, filename: str):
    """对数据集里的某张图跑 demo。thumb=1 时只返回原图（用于缩略图）。"""
    spec = DATASETS.get(dataset)
    if spec is None:
        abort(404)
    path = spec["dir"] / filename
    if not path.exists() or path.parent != spec["dir"]:
        abort(404)

    if request.args.get("thumb"):
        return send_file(path)

    demo = _current_demo()
    image = read_image(path)
    if image is None:
        abort(500)
    report = demo.analyze(image, source=filename)
    canvas = demo.render(image, report)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{dataset}_{path.stem}_demo.jpg"
    save_image(RESULT_DIR / out_name, canvas)

    gt = None
    gt_list = _gt_for(spec, path.stem)
    if gt_list:
        gt = [{"name": g["name"], "box": list(g["box"])} for g in gt_list]

    return jsonify({
        "image_url": f"/results/{out_name}",
        "source": filename,
        "gt": gt,
        **report.to_dict(),
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """上传图片跑 demo。"""
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "没有收到文件"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename).name or "upload.jpg"
    suffix = Path(name).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return jsonify({"error": f"不支持的格式：{suffix}"}), 400
    safe = f"upload_{abs(hash(name)) % 10**8}{suffix}"
    path = UPLOAD_DIR / safe
    file.save(path)

    demo = _current_demo()
    image = read_image(path)
    if image is None:
        return jsonify({"error": "图片解码失败"}), 400
    report = demo.analyze(image, source=name)
    canvas = demo.render(image, report)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"upload_{safe}"
    save_image(RESULT_DIR / out_name, canvas)
    return jsonify({
        "image_url": f"/results/{out_name}",
        "source": name,
        "gt": None,
        **report.to_dict(),
    })


@app.route("/api/batch")
def api_batch():
    """对某个数据集的一批图跑 demo，返回汇总 + 每张的结果链接。"""
    key = request.args.get("dataset", "v1")
    count = min(60, max(1, int(request.args.get("count", 12))))
    offset = max(0, int(request.args.get("offset", 0)))
    spec = DATASETS.get(key)
    if spec is None:
        abort(404)

    demo = _current_demo()
    images = list_images(spec["dir"])[offset:offset + count]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    items = []
    total_counts: Counter = Counter()
    state_correct = 0
    state_total = 0
    det_tp = det_fp = det_fn = 0
    for path in images:
        image = read_image(path)
        if image is None:
            continue
        report = demo.analyze(image, source=path.name)
        canvas = demo.render(image, report)
        out_name = f"{key}_{path.stem}_demo.jpg"
        save_image(RESULT_DIR / out_name, canvas)
        total_counts.update(report.counts)

        label_path = spec["labels"] / f"{path.stem}.json" if spec.get("labels") else None
        gt = []
        if label_path is not None and label_path.exists():
            gt = load_gt_boxes(label_path)

        # 按 IoU 匹配预测框与 GT 框（不能按顺序 zip：检测顺序与标注顺序不一致）
        matched = set()
        for fruit in report.fruits:
            best, best_iou = None, 0.3
            for i, g in enumerate(gt):
                if i in matched:
                    continue
                value = _iou(fruit.box, g["box"])
                if value > best_iou:
                    best, best_iou = i, value
            if best is None:
                det_fp += 1
                continue
            matched.add(best)
            det_tp += 1
            state_total += 1
            if gt[best]["name"] == fruit.state:
                state_correct += 1
        det_fn += len(gt) - len(matched)

        items.append({
            "name": path.name,
            "image_url": f"/results/{out_name}",
            "total": report.total,
            "maturity": round(report.maturity, 4),
            "counts": {n: int(report.counts.get(n, 0)) for n in CLASS_NAMES},
            "latency_ms": round(report.latency_ms, 1),
        })

    det_p = det_tp / max(1, det_tp + det_fp)
    det_r = det_tp / max(1, det_tp + det_fn)
    return jsonify({
        "dataset": key,
        "items": items,
        "counts": {n: int(total_counts.get(n, 0)) for n in CLASS_NAMES},
        "total_fruits": int(sum(total_counts.values())),
        "state_acc": (state_correct / state_total) if state_total else None,
        "state_total": state_total,
        "det_precision": det_p,
        "det_recall": det_r,
        "det_tp": det_tp, "det_fp": det_fp, "det_fn": det_fn,
    })


@app.route("/api/stats")
def api_stats():
    """模型指标对比（读取已固化的评测 JSON，不重新计算）。"""
    out = {}
    candidates = [
        _ROOT / "reports",
        _DEMO.parent / "demo_out",
        _ROOT / "demo_out",
    ]
    for name, filename in (
        ("classifier", "classifier_comparison.json"),
        ("end2end", "end2end_comparison.json"),
    ):
        for folder in candidates:
            path = folder / filename
            if path.exists():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
                break
    out["class_names"] = list(CLASS_NAMES)
    out["datasets"] = {k: v["label"] for k, v in DATASETS.items()}
    return jsonify(out)


@app.route("/api/gallery")
def api_gallery():
    for candidate in (
        _ROOT / "site" / "assets" / "gallery.jpg",
        _ROOT / "demo_out" / "gallery.jpg",
        _DEMO.parent / "demo_out" / "gallery.jpg",
    ):
        if candidate.exists():
            return send_file(candidate)
    abort(404)


@app.route("/results/<path:name>")
def results(name: str):
    path = (RESULT_DIR / name).resolve()
    if not str(path).startswith(str(RESULT_DIR.resolve())) or not path.exists():
        abort(404)
    return send_file(path)


def _current_demo() -> MaturityDemo:
    return get_demo(
        detector_key=request.args.get("detector", "merged"),
        classifier_key=request.args.get("classifier", "merged"),
        conf=float(request.args.get("conf", 0.55)),
        iou=float(request.args.get("iou", 0.5)),
    )


def _iou(a, b) -> float:
    """两个 xyxy 框的交并比。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@app.errorhandler(500)
def on_error(e):
    return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


def main() -> int:
    ap = argparse.ArgumentParser(description="草莓成熟度检测可视化网站")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("  草莓成熟度检测 Demo —— 可视化网站")
    print("=" * 60)
    print(f"  地址: http://{args.host}:{args.port}")
    print("  首次请求会加载模型（CPU 约 10~20 秒）")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
