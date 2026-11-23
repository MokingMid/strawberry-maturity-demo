"""命令行入口。

常用：
  python -m strawberry_cv.cli check
  python -m strawberry_cv.cli stats
  python -m strawberry_cv.cli export-yolo --class-mode 1cls
  python -m strawberry_cv.cli make-split
  python -m strawberry_cv.cli export-roi
  python -m strawberry_cv.cli train-detector
  python -m strawberry_cv.cli train-classifier
  python -m strawberry_cv.cli infer-image --image xxx.jpg --weights best.pt --classifier best.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from .config import Config, ConfigError
from .constants import CLASS_NAMES
from .errors import DependencyError, PipelineError
from .utils.io import format_table, get_logger, save_json

LOGGER = get_logger("cli")


# ---------------- 通用 ----------------
def _config(args) -> Config:
    return Config.load(args.config)


def _load_data(config: Config):
    from .data.labels import load_dataset_from_config
    from .data.scenes import load_scenes_from_config

    return load_dataset_from_config(config), load_scenes_from_config(config)


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _packages() -> dict[str, bool]:
    names = ["numpy", "cv2", "sklearn", "matplotlib", "PIL", "scipy",
             "yaml", "torch", "torchvision", "ultralytics"]
    return {name: importlib.util.find_spec(name) is not None for name in names}


# ---------------- check / stats ----------------
def cmd_check(args) -> int:
    config = _config(args)
    print("=" * 66)
    print("strawberry_cv 环境检查")
    print("=" * 66)
    print(f"python      : {platform.python_version()} ({platform.system()})")
    print(f"project root: {config.root}")
    print(f"work dir    : {config.path('paths.work_dir')}")

    packages = _packages()
    rows = [[name, "OK" if ok else "缺少"] for name, ok in packages.items()]
    print("\n依赖：")
    print(format_table(rows, ["package", "status"]))

    print("\n配置问题：")
    issues = config.validate(require_data=True)
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  - 无")

    try:
        dataset, scenes = _load_data(config)
        stats = dataset.stats()
        print("\n数据：")
        print(f"  images={stats['images']} boxes={stats['boxes']} "
              f"classes={stats['per_class']}")
        print(f"  scenes={scenes.stats()['groups']} "
              f"groups={scenes.stats()['group_size_distribution']}")
        print(f"  border_boxes={stats['border_boxes']} "
              f"mixed_images={len(stats['mixed_images'])}")
        if stats["per_class"].get("half", 0) < 10:
            print("  [警告] half 样本少于 10 个框，训练/评估都不稳")
    except Exception as exc:
        print(f"\n数据读取失败：{exc}")

    print("\n现在可运行：check / stats / export-yolo / make-split / export-roi / prepare(默认 dry-run)")
    if packages.get("torch") and packages.get("ultralytics"):
        print("训练/推理：train-detector / train-classifier / evaluate-* / infer-*")
    else:
        print("训练/推理需要先安装 torch、torchvision、ultralytics（建议 Python 3.10–3.12 虚拟环境）")
    return 0


def cmd_stats(args) -> int:
    config = _config(args)
    dataset, scenes = _load_data(config)
    stats = dataset.stats()
    stats["scenes"] = scenes.stats()
    stats["warnings"] = []
    if stats["per_class"].get("half", 0) < 10:
        stats["warnings"].append("half 少于 10 个框")
    if stats["border_boxes"]:
        stats["warnings"].append(f"{stats['border_boxes']} 个贴边框")
    _print_json(stats)
    return 0


# ---------------- 数据准备 ----------------
def cmd_export_yolo(args) -> int:
    config = _config(args)
    from .data.splits import load_split_assignments
    from .data.yolo import export_yolo

    dataset, _ = _load_data(config)
    split_result = None
    if args.split_manifest:
        split_result = SimpleNamespace(assignments=load_split_assignments(args.split_manifest))
    out_dir = Path(args.out) if args.out else config.derived(
        "yolo_1cls" if args.class_mode == "1cls" else "yolo_3cls"
    )
    report = export_yolo(
        dataset, out_dir, class_mode=args.class_mode,
        split_result=split_result, force=args.force, verify=not args.no_verify,
    )
    _print_json(report)
    return 0


def cmd_make_split(args) -> int:
    config = _config(args)
    from .data.splits import make_group_split, write_split

    dataset, scenes = _load_data(config)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    result = make_group_split(dataset, scenes, ratios=ratios, seed=args.seed)
    out_dir = Path(args.out) if args.out else config.derived("splits")
    report = write_split(result, dataset, scenes, out_dir)
    print(f"written: {out_dir}")
    _print_json(report)
    return 0


def cmd_export_roi(args) -> int:
    config = _config(args)
    from .data.roi import export_roi_dataset
    from .data.splits import load_split_assignments

    dataset, _ = _load_data(config)
    split_path = (
        Path(args.split_manifest) if args.split_manifest
        else config.derived("splits") / "split_manifest.csv"
    )
    if not split_path.exists():
        raise PipelineError(
            f"划分清单不存在：{split_path}\n先运行 make-split"
        )
    split_result = SimpleNamespace(assignments=load_split_assignments(split_path))
    roi_cfg = config.get("roi", {})
    out_dir = Path(args.out) if args.out else config.derived("roi")
    report = export_roi_dataset(
        dataset, split_result, out_dir,
        pad_ratio=float(roi_cfg.get("pad_ratio", 0.12)),
        square=bool(roi_cfg.get("square", True)),
        min_size=int(roi_cfg.get("min_size", 32)),
        skip_border=bool(roi_cfg.get("skip_border", True)),
        max_neighbor_iou=float(roi_cfg.get("max_neighbor_iou", 0.1)),
        pad_steps=int(roi_cfg.get("pad_steps", 6)),
        force=args.force,
    )
    _print_json(report)
    return 0


def cmd_prepare(args) -> int:
    """一条龙：导出 YOLO → 划分 → ROI。默认 dry-run。"""
    config = _config(args)
    dataset, scenes = _load_data(config)
    stats = dataset.stats()
    print("prepare 计划：")
    print(f"  数据：{stats['images']} 图 / {stats['boxes']} 框 / {stats['per_class']}")
    print(f"  yolo 1cls -> {config.derived('yolo_1cls')}")
    if args.with_3cls:
        print(f"  yolo 3cls -> {config.derived('yolo_3cls')}")
    print(f"  splits    -> {config.derived('splits')}")
    print(f"  roi       -> {config.derived('roi')}")
    if not args.execute:
        print("\n[dry-run] 未写入任何文件；确认后加 --execute")
        return 0

    from .data.roi import export_roi_dataset
    from .data.splits import make_group_split, write_split
    from .data.yolo import export_yolo

    out = config.derived("splits")
    result = make_group_split(
        dataset, scenes,
        ratios=tuple(float(x) for x in args.ratios.split(",")),
        seed=args.seed,
    )
    write_split(result, dataset, scenes, out)
    print(f"splits written: {out}")

    yolo1 = config.derived("yolo_1cls")
    report1 = export_yolo(dataset, yolo1, "1cls", result, force=args.force)
    print(f"yolo 1cls: {report1['images']} images / {report1['boxes']} boxes -> {yolo1}")
    if args.with_3cls:
        yolo3 = config.derived("yolo_3cls")
        report3 = export_yolo(dataset, yolo3, "3cls", result, force=args.force)
        print(f"yolo 3cls: {report3['images']} images / {report3['boxes']} boxes -> {yolo3}")

    roi_cfg = config.get("roi", {})
    roi_report = export_roi_dataset(
        dataset, result, config.derived("roi"),
        pad_ratio=float(roi_cfg.get("pad_ratio", 0.12)),
        square=bool(roi_cfg.get("square", True)),
        min_size=int(roi_cfg.get("min_size", 32)),
        skip_border=bool(roi_cfg.get("skip_border", True)),
        max_neighbor_iou=float(roi_cfg.get("max_neighbor_iou", 0.1)),
        pad_steps=int(roi_cfg.get("pad_steps", 6)),
        force=args.force,
    )
    print(f"roi: {roi_report['total_crops']} crops -> {roi_report['out_dir']}")
    if roi_report["warnings"]:
        print("warnings:")
        for warning in roi_report["warnings"]:
            print("  -", warning)
    return 0


# ---------------- 训练 ----------------
def cmd_train_detector(args) -> int:
    config = _config(args)
    from .training.train_detector import run

    result = run(
        config, data_yaml=args.data, epochs=args.epochs, batch=args.batch,
        imgsz=args.imgsz, device=args.device, name=args.name,
    )
    _print_json(result)
    return 0


def cmd_train_classifier(args) -> int:
    config = _config(args)
    from .training.train_classifier import run

    result = run(
        config, roi_dir=args.roi_dir, epochs=args.epochs, batch=args.batch,
        device=args.device, name=args.name,
    )
    _print_json(result)
    return 0


# ---------------- 评测 ----------------
def cmd_eval_detector(args) -> int:
    config = _config(args)
    from .models.detector import Detector

    data = Path(args.data) if args.data else config.derived("yolo_1cls") / "data.yaml"
    detector = Detector(
        weights=args.weights or config.get("detector.weights", "yolo11n.pt"),
        device=args.device or config.get("detector.device", "auto"),
        imgsz=int(config.get("detector.imgsz", 640)),
    )
    result = detector.val(
        data,
        split=args.split,
        project=config.derived("runs") / "detector_eval",
        name=args.split,
    )
    _print_json(result)
    return 0


def cmd_eval_classifier(args) -> int:
    config = _config(args)
    from .evaluation.classification import save_confusion_png
    from .models.classifier import Classifier

    classifier = Classifier.from_checkpoint(args.checkpoint, device=args.device or "auto")
    data_dir = Path(args.data) if args.data else config.derived("roi") / "val"
    metrics = classifier.evaluate(data_dir, batch=args.batch)
    out_dir = Path(args.out) if args.out else config.derived("reports") / "classifier_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", metrics)
    save_confusion_png(metrics["confusion"], metrics["labels"],
                       out_dir / "confusion.png", title="classifier")
    _print_json(metrics)
    return 0


def cmd_eval_end2end(args) -> int:
    """在指定 split 上跑完整两阶段，计算检测匹配 + 状态准确率 + 计数误差。"""
    config = _config(args)
    from .data.labels import load_dataset_from_config
    from .data.splits import load_split_assignments
    from .evaluation.end2end import end2end_metrics, summarize_end2end
    from .utils.image_io import imread

    dataset = load_dataset_from_config(config)
    split_path = config.derived("splits") / "split_manifest.csv"
    if not split_path.exists():
        raise PipelineError(f"划分清单不存在：{split_path}\n先运行 make-split")
    split_map = load_split_assignments(split_path)
    stems = [stem for stem in dataset.stems if split_map.get(stem) == args.split]
    if not stems:
        raise PipelineError(f"split={args.split} 没有图片")

    pipeline = _build_pipeline(args, config)
    preds_by_image, gts_by_image = {}, {}
    for stem in stems:
        image = imread(dataset.image_path(stem))
        if image is None:
            LOGGER.warning("无法读取：%s", stem)
            continue
        result = pipeline.process_frame(image)
        preds_by_image[stem] = [
            {"box": f.box, "state": f.state, "score": f.det_score}
            for f in result.fruits
        ]
        gts_by_image[stem] = [
            {"box": box.xyxy, "state": box.name} for box in dataset.load(stem)
        ]

    metrics = end2end_metrics(preds_by_image, gts_by_image, iou_thr=args.iou)
    out_dir = Path(args.out) if args.out else config.derived("reports") / f"end2end_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", metrics)
    print(summarize_end2end(metrics))
    print(f"written: {out_dir / 'metrics.json'}")
    return 0


# ---------------- 推理 ----------------
def _parse_line(text: str | None):
    if not text:
        return None
    values = [float(x) for x in text.split(",")]
    if len(values) != 4:
        raise ValueError("--line 格式：x1,y1,x2,y2")
    return (values[0], values[1]), (values[2], values[3])


def _build_pipeline(args, config: Config, video: bool = False):
    from .inference.pipeline import TwoStagePipeline
    from .inference.tracker import IoUTracker, LineCounter
    from .models.classifier import Classifier
    from .models.detector import Detector

    detector = None
    if args.weights:
        conf = getattr(args, "conf", None)
        detector = Detector(
            weights=args.weights,
            device=args.device or config.get("inference.device", "auto"),
            conf=float(conf if conf is not None else config.get("detector.conf", 0.55)),
            iou=float(config.get("detector.iou", 0.5)),
            imgsz=int(config.get("detector.imgsz", 640)),
        )
    classifier = None
    if args.classifier:
        classifier = Classifier.from_checkpoint(args.classifier,
                                                device=args.device or "auto")
    tracker = None
    counter = None
    if video:
        tracking = config.get("tracking", {})
        tracker = IoUTracker(
            iou_threshold=float(tracking.get("iou_threshold", 0.3)),
            max_age=int(tracking.get("max_age", 30)),
        )
        line = _parse_line(args.line)
        if line:
            counter = LineCounter(
                line[0], line[1],
                min_hits=int(tracking.get("min_hits", 3)),
                debounce=int(tracking.get("debounce", 3)),
            )
    roi_cfg = config.get("roi", {})
    return TwoStagePipeline(
        detector=detector, classifier=classifier,
        pad_ratio=float(roi_cfg.get("pad_ratio", 0.12)),
        square=bool(roi_cfg.get("square", True)),
        min_size=int(roi_cfg.get("min_size", 32)),
        max_neighbor_iou=float(roi_cfg.get("max_neighbor_iou", 0.1)),
        tracker=tracker, counter=counter,
    )


def _gt_provider_from_dataset(config: Config):
    from .data.labels import load_dataset_from_config

    dataset = load_dataset_from_config(config)

    def provider(stem: str):
        return [
            {"box": box.xyxy, "state": box.name}
            for box in dataset.load(stem)
        ]

    return dataset, provider


def cmd_infer_image(args) -> int:
    config = _config(args)
    pipeline = _build_pipeline(args, config)
    gt = None
    if args.use_gt_labels:
        _, provider = _gt_provider_from_dataset(config)
        stem = Path(args.image).stem
        gt = provider(stem)
        if not args.classifier:
            LOGGER.warning("GT 模式但未提供 --classifier，状态会是 unknown")
    result = pipeline.run_image(args.image, args.out, gt_fruits=gt)
    _print_json(result)
    return 0


def cmd_infer_folder(args) -> int:
    config = _config(args)
    pipeline = _build_pipeline(args, config)
    provider = None
    if args.use_gt_labels:
        _, provider = _gt_provider_from_dataset(config)
    result = pipeline.run_folder(args.images, args.out, gt_provider=provider)
    _print_json({k: v for k, v in result.items() if k != "summaries"})
    return 0


def cmd_infer_video(args) -> int:
    config = _config(args)
    pipeline = _build_pipeline(args, config, video=True)
    result = pipeline.run_video(args.video, args.out, max_frames=args.max_frames)
    _print_json(result)
    return 0


# ---------------- 传统基线 ----------------
def cmd_traditional(args) -> int:
    config = _config(args)
    from .baselines.svm import run_traditional

    roi_root = Path(args.roi_dir) if args.roi_dir else config.derived("roi")
    out_dir = Path(args.out) if args.out else config.derived("runs") / "traditional"
    scene_manifest = args.scene_manifest or config.path("paths.scene_manifest")
    metrics = run_traditional(
        roi_root, out_dir, config.get("traditional", {}),
        scene_manifest=scene_manifest, seed=int(config.get("project.seed", 42)),
    )
    _print_json(metrics)
    return 0


# ---------------- argparse ----------------
def _add_config(parser):
    parser.add_argument("--config", default="configs/base.json", help="配置文件")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strawberry_cv",
        description="草莓成熟度两阶段算法（1 类检测 + 3 类分类）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="环境与数据检查")
    _add_config(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("stats", help="输出数据集统计")
    _add_config(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export-yolo", help="导出 YOLO 检测数据")
    _add_config(p)
    p.add_argument("--class-mode", choices=["1cls", "3cls"], default="1cls")
    p.add_argument("--out", default=None)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(func=cmd_export_yolo)

    p = sub.add_parser("make-split", help="按场景组划分 train/val/test")
    _add_config(p)
    p.add_argument("--ratios", default="0.7,0.15,0.15")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_make_split)

    p = sub.add_parser("export-roi", help="用 GT 框裁剪单果 ROI")
    _add_config(p)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_export_roi)

    p = sub.add_parser("prepare", help="一条龙数据准备（默认 dry-run）")
    _add_config(p)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--with-3cls", action="store_true")
    p.add_argument("--ratios", default="0.7,0.15,0.15")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train-detector", help="训练 1 类检测器")
    _add_config(p)
    p.add_argument("--data", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_train_detector)

    p = sub.add_parser("train-classifier", help="训练 3 类分类器")
    _add_config(p)
    p.add_argument("--roi-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default=None)
    p.set_defaults(func=cmd_train_classifier)

    p = sub.add_parser("evaluate-detector", help="验证检测器")
    _add_config(p)
    p.add_argument("--weights", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_eval_detector)

    p = sub.add_parser("evaluate-classifier", help="验证分类器")
    _add_config(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_eval_classifier)

    p = sub.add_parser("evaluate-end2end", help="两阶段端到端评测")
    _add_config(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--classifier", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=None,
                   help="检测置信度；默认用 configs/base.json 的 detector.conf")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_eval_end2end)

    p = sub.add_parser("infer-image", help="单图两阶段推理")
    _add_config(p)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--classifier", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--use-gt-labels", action="store_true")
    p.set_defaults(func=cmd_infer_image)

    p = sub.add_parser("infer-folder", help="文件夹两阶段推理")
    _add_config(p)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--classifier", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--use-gt-labels", action="store_true")
    p.set_defaults(func=cmd_infer_folder)

    p = sub.add_parser("infer-video", help="视频两阶段推理 + 跟踪/越线计数")
    _add_config(p)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--classifier", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--line", default=None, help="x1,y1,x2,y2")
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_infer_video)

    p = sub.add_parser("traditional", help="传统特征 + SVM 基线")
    _add_config(p)
    p.add_argument("--roi-dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--scene-manifest", default=None)
    p.set_defaults(func=cmd_traditional)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ConfigError, DependencyError, PipelineError, FileNotFoundError,
            FileExistsError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
