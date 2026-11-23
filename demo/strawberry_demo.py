"""草莓成熟度检测 Demo —— 命令行入口。

用法示例
--------
# 单张图片
python demo/strawberry_demo.py --image D:\\demo.jpg

# 整个文件夹
python demo/strawberry_demo.py --folder datasets/strawberry_v1/detection/images --out demo_out

# 视频
python demo/strawberry_demo.py --video clip.mp4 --out demo_out

# 摄像头实时
python demo/strawberry_demo.py --camera 0 --gui

# 用数据集做自检（GT 框 + 分类器，输出混淆矩阵与分类准确率）
python demo/strawberry_demo.py --self-test --dataset archive

不指定任何输入时，会列出用法并退出。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from paths import PACKAGE_DIR as _PKG_ROOT  # noqa: E402

if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from demo_core import (  # noqa: E402
    CLASS_NAMES,
    FrameReport,
    MaturityDemo,
    UnicodeVideoWriter,
    list_images,
    load_gt_boxes,
    open_video,
    read_image,
    save_image,
)

# ---------------------------------------------------------------- 默认权重

PKG = _PKG_ROOT
DEFAULT_DETECTOR = PKG / "work/runs/detector/strawberry_1cls_lowlr/weights/best.pt"
MERGED_DETECTOR = PKG / "work_merged/runs/detector/merged_1cls_lowlr/weights/best.pt"
DEFAULT_CLASSIFIER = PKG / "work/runs/classifier/resnet18_3cls/best.pt"
ARCHIVE_CLASSIFIER = PKG / "work_archive/runs/classifier/resnet18_3cls_archive/best.pt"
MERGED_CLASSIFIER = PKG / "work_merged/runs/classifier/resnet18_3cls_merged/best.pt"

# 仓库布局下权重在 models/ 下（GitHub clone 后直接可用）
_REPO_MODELS = _ROOT / "models"
if (_REPO_MODELS / "detector_merged.pt").exists():
    MERGED_DETECTOR = _REPO_MODELS / "detector_merged.pt"
if (_REPO_MODELS / "classifier_merged.pt").exists():
    MERGED_CLASSIFIER = _REPO_MODELS / "classifier_merged.pt"

DATASETS = {}
for _key, _full, _sample in (("v1", "strawberry_v1", "v1"),
                             ("archive", "strawberry_archive_v1", "archive")):
    _imgs = _ROOT / "datasets" / _full / "detection" / "images"
    _lbls = _ROOT / "datasets" / _full / "detection" / "labels"
    if not _imgs.is_dir():
        _imgs = _ROOT / "samples" / _sample / "detection" / "images"
        _lbls = _ROOT / "samples" / _sample / "detection" / "labels"
    DATASETS[_key] = {
        "name": _full if _imgs.parent.parent.name == "datasets" else _sample,
        "images": _imgs,
        "labels": _lbls,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="草莓成熟度检测 Demo（检测 + 三状态分类）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_argument_group("输入源（四选一）")
    src.add_argument("--image", help="单张图片路径")
    src.add_argument("--folder", help="图片文件夹")
    src.add_argument("--video", help="视频文件路径")
    src.add_argument("--camera", type=int, help="摄像头编号，如 0")
    src.add_argument("--self-test", action="store_true",
                     help="用数据集自检：GT 框 + 分类器，统计分类准确率")

    m = p.add_argument_group("模型")
    m.add_argument("--detector", default=None,
                   help="检测权重 .pt（不指定则自动选：合并检测器优先，其次 v1 检测器）")
    m.add_argument("--classifier", default=None,
                   help="分类权重 .pt（不指定则自动选：合并分类器优先，其次归档集、v1）")
    m.add_argument("--no-detector", action="store_true", help="不加载检测器")
    m.add_argument("--no-classifier", action="store_true", help="不加载分类器")
    m.add_argument("--conf", type=float, default=0.55, help="检测置信度阈值")
    m.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    m.add_argument("--imgsz", type=int, default=640, help="检测输入尺寸")
    m.add_argument("--device", default="auto", help="auto / cpu / 0")

    o = p.add_argument_group("输出")
    o.add_argument("--out", default="demo_out", help="输出目录")
    o.add_argument("--gui", action="store_true", help="弹出实时预览窗口")
    o.add_argument("--no-save", action="store_true", help="不保存标注图/CSV")
    o.add_argument("--max-frames", type=int, default=None, help="视频最多处理帧数")
    o.add_argument("--limit", type=int, default=None, help="文件夹模式最多处理张数")

    s = p.add_argument_group("自检")
    s.add_argument("--dataset", choices=sorted(DATASETS), default="v1", help="自检用数据集")
    s.add_argument("--self-test-n", type=int, default=40, help="自检图片数量上限")
    s.add_argument("--with-detector", action="store_true",
                   help="自检时走完整流水线（检测+分类），衡量 demo 实际效果")
    return p


# ---------------------------------------------------------------- 各模式


def _print_report(report: FrameReport, prefix: str = "") -> None:
    counts = "  ".join(f"{name}={int(report.counts.get(name, 0))}" for name in CLASS_NAMES)
    print(f"{prefix}检测 {report.total:3d} 颗 | {counts} | 成熟度 {report.maturity * 100:5.1f}% "
          f"| {report.latency_ms:6.1f} ms")


def run_image(demo: MaturityDemo, args) -> int:
    image = read_image(args.image)
    if image is None:
        print(f"[demo] 读图失败：{args.image}", file=sys.stderr)
        return 1
    report = demo.analyze(image, source=Path(args.image).name)
    _print_report(report)
    canvas = demo.render(image, report)
    if not args.no_save:
        out = Path(args.out)
        save_image(out / f"{Path(args.image).stem}_demo.jpg", canvas)
        MaturityDemo.write_json(out / "result.json", [report])
        print(f"[demo] 已保存到 {out.resolve()}")
    if args.gui:
        _show(canvas, "草莓成熟度检测 Demo")
    return 0


def run_folder(demo: MaturityDemo, args) -> int:
    images = list_images(args.folder)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"[demo] 文件夹里没有图片：{args.folder}", file=sys.stderr)
        return 1

    reports: list[FrameReport] = []
    out = Path(args.out)
    for i, path in enumerate(images, 1):
        image = read_image(path)
        if image is None:
            continue
        report = demo.analyze(image, frame_index=i, source=path.name)
        reports.append(report)
        canvas = demo.render(image, report)
        if not args.no_save:
            save_image(out / "annotated" / f"{path.stem}_demo.jpg", canvas)
        print(f"[{i:>4}/{len(images)}] {path.name:<28} {report.total:3d} 颗 | "
              f"成熟度 {report.maturity * 100:5.1f}% | {report.latency_ms:6.1f} ms")
        if args.gui:
            _show(canvas, path.name)

    total = Counter()
    for r in reports:
        total.update(r.counts)
    print("\n===== 汇总 =====")
    for name in CLASS_NAMES:
        print(f"  {name:<8} {int(total.get(name, 0))}")
    print(f"  合计     {sum(total.values())}")
    if not args.no_save:
        MaturityDemo.write_csv(out / "predictions.csv", reports)
        MaturityDemo.write_json(out / "summary.json", reports)
        print(f"[demo] 已保存到 {out.resolve()}")
    return 0


def run_video(demo: MaturityDemo, args) -> int:
    import cv2

    cap, cleanup = open_video(args.video)
    if not cap.isOpened():
        cleanup()
        print(f"[demo] 打开视频失败：{args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(args.out)
    writer = None
    if not args.no_save:
        out.mkdir(parents=True, exist_ok=True)
        writer = UnicodeVideoWriter(out / "annotated.mp4", fps, (width, height))

    reports: list[FrameReport] = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            report = demo.analyze(frame, frame_index=index, source=Path(args.video).name,
                                  reset_tracker=False)
            reports.append(report)
            canvas = demo.render(frame, report)
            if writer is not None:
                writer.write(canvas)
            if index % 10 == 0:
                print(f"[frame {index:>5}] {report.total:3d} 颗 | 成熟度 "
                      f"{report.maturity * 100:5.1f}% | {report.latency_ms:6.1f} ms")
            if args.gui:
                _show(canvas, f"frame {index}")
            index += 1
            if args.max_frames and index >= args.max_frames:
                break
    finally:
        cap.release()
        cleanup()
        if writer is not None:
            writer.close()

    print(f"\n[demo] 处理 {index} 帧")
    if not args.no_save:
        MaturityDemo.write_csv(out / "predictions.csv", reports)
        MaturityDemo.write_json(out / "summary.json", reports)
        print(f"[demo] 已保存到 {out.resolve()}")
    return 0


def run_camera(demo: MaturityDemo, args) -> int:
    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[demo] 打开摄像头 {args.camera} 失败", file=sys.stderr)
        return 1
    print("[demo] 摄像头已打开，按 q 退出")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            report = demo.analyze(frame, frame_index=index, source="camera",
                                  reset_tracker=False)
            canvas = demo.render(frame, report)
            cv2.imshow("草莓成熟度检测 Demo", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


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


def run_self_test(demo: MaturityDemo, args) -> int:
    """用数据集自检。

    默认用 GT 框 + 分类器，衡量「分类器单独」的能力；
    加 --with-detector 则走完整流水线（检测 + 分类），衡量 demo 的实际效果。
    """
    if demo.classifier is None:
        print("[demo] 自检需要分类器（--classifier 指定）", file=sys.stderr)
        return 1
    spec = DATASETS[args.dataset]
    images = list_images(spec["images"])
    if args.self_test_n:
        images = images[: args.self_test_n]

    use_detector = bool(args.with_detector and demo.detector is not None)
    if args.with_detector and demo.detector is None:
        print("[demo] 未加载检测器，退回 GT 框模式")

    confusion: dict[tuple[str, str], int] = Counter()
    total = 0
    latency: list[float] = []
    det_tp = det_fp = det_fn = 0
    state_tp = state_fn = 0

    for path in images:
        label_path = spec["labels"] / f"{path.stem}.json"
        if not label_path.exists():
            continue
        image = read_image(path)
        if image is None:
            continue
        gt = load_gt_boxes(label_path)
        report = demo.analyze(image, source=path.name,
                              gt_boxes=None if use_detector else gt)
        latency.append(report.latency_ms)

        if use_detector:
            # 端到端：按 IoU 匹配预测框与 GT 框
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
                truth = gt[best]["name"]
                confusion[(truth, fruit.state)] += 1
                total += 1
                if truth == fruit.state:
                    state_tp += 1
                else:
                    state_fn += 1
            det_fn += len(gt) - len(matched)
        else:
            # 用 gt_state 逐果配对（不能按顺序 zip：ROI 被跳过时两边长度会不一致）
            for fruit in report.fruits:
                truth = fruit.gt_state or "unknown"
                confusion[(truth, fruit.state)] += 1
                total += 1

    mode = "端到端（检测+分类）" if use_detector else "仅分类（GT 框）"
    print(f"\n===== 自检：{spec['name']} · {mode} ({len(images)} 图 / {total} 颗) =====")
    header = "GT\\Pred".ljust(10) + "".join(n.rjust(9) for n in CLASS_NAMES) + "   合计"
    print(header)
    for truth in CLASS_NAMES:
        row = truth.ljust(10)
        for pred in CLASS_NAMES:
            row += str(confusion[(truth, pred)]).rjust(9)
        row += str(sum(confusion[(truth, p)] for p in CLASS_NAMES)).rjust(8)
        print(row)
    print("合计".ljust(10) + "".join(
        str(sum(confusion[(t, p)] for t in CLASS_NAMES)).rjust(9) for p in CLASS_NAMES
    ) + str(total).rjust(8))

    correct = sum(confusion[(c, c)] for c in CLASS_NAMES)
    print(f"\n  成熟度判断正确率: {correct / total * 100:.2f}%  ({correct}/{total})")
    for name in CLASS_NAMES:
        tp = confusion[(name, name)]
        support = sum(confusion[(name, p)] for p in CLASS_NAMES)
        predicted = sum(confusion[(t, name)] for t in CLASS_NAMES)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"  {name:<8} P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}  支持 {support}")
    if use_detector:
        p = det_tp / max(1, det_tp + det_fp)
        r = det_tp / max(1, det_tp + det_fn)
        print(f"\n  检测: TP {det_tp}  FP {det_fp}  FN {det_fn}  "
              f"P {p:.3f}  R {r:.3f}  F1 {2 * p * r / max(1e-9, p + r):.3f}")
    if latency:
        print(f"  平均耗时: {np.mean(latency):.1f} ms/图")
    return 0


def _show(canvas: np.ndarray, title: str) -> None:
    import cv2

    cv2.imshow(title, canvas)
    cv2.waitKey(0)


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    has_source = any([args.image, args.folder, args.video,
                      args.camera is not None, args.self_test])
    if not has_source:
        build_parser().print_help()
        return 0

    # 检测器：显式指定就用指定的；没指定则自动挑（合并检测器覆盖两个域）
    if args.no_detector:
        detector_weights = None
    elif args.detector:
        detector_weights = args.detector
        print(f"[demo] 使用指定检测器：{Path(detector_weights).name}")
    elif MERGED_DETECTOR.exists():
        detector_weights = str(MERGED_DETECTOR)
        print(f"[demo] 自动选择合并检测器：{MERGED_DETECTOR.name}"
              "（v1 + 归档集联合训练，两个域都覆盖）")
    else:
        detector_weights = str(DEFAULT_DETECTOR)
        print(f"[demo] 自动选择 v1 检测器：{DEFAULT_DETECTOR.name}"
              "（仅在 64 张网络图上训练，温室域召回偏低）")

    # 分类器：显式指定就用指定的；没指定才自动挑（合并模型 > 归档集模型 > v1 模型）
    if args.no_classifier:
        classifier_ckpt = None
    elif args.classifier:
        classifier_ckpt = args.classifier
        print(f"[demo] 使用指定分类器：{Path(classifier_ckpt).name}")
    elif MERGED_CLASSIFIER.exists():
        classifier_ckpt = str(MERGED_CLASSIFIER)
        print(f"[demo] 自动选择合并分类器：{MERGED_CLASSIFIER.name}"
              "（v1 + 归档集联合训练，兼顾两种拍摄域）")
    elif ARCHIVE_CLASSIFIER.exists():
        classifier_ckpt = str(ARCHIVE_CLASSIFIER)
        print(f"[demo] 自动选择归档集分类器：{ARCHIVE_CLASSIFIER.name}"
              "（half 样本量是 v1 的 22 倍）")
    else:
        classifier_ckpt = str(DEFAULT_CLASSIFIER)
        print(f"[demo] 自动选择 v1 分类器：{DEFAULT_CLASSIFIER.name}"
              "（half 识别能力有限）")

    demo = MaturityDemo(
        detector_weights=detector_weights,
        classifier_checkpoint=classifier_ckpt,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
    )
    if demo.detector is None:
        print("[demo] 未加载检测器：只输出整图分类或 GT 框结果")
    if demo.classifier is None:
        print("[demo] 未加载分类器：只输出检测框")

    if args.self_test:
        return run_self_test(demo, args)
    if args.image:
        return run_image(demo, args)
    if args.folder:
        return run_folder(demo, args)
    if args.video:
        return run_video(demo, args)
    if args.camera is not None:
        return run_camera(demo, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
