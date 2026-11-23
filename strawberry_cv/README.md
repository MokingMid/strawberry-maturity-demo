> **交接包路径说明**：本文件是 `handoff_2026-11-22` 的副本。原项目根目录对应本包根目录；原工作目录对应本包 `work/`；原虚拟环境不随包提供，请按文档自建。本包配置已改为相对路径（`configs/base.json` 的 `work_dir=work`）。

# strawberry_cv —— 算法代码骨架

主线：**1 类检测（strawberry）+ 3 类分类（ripe / half / unripe）**，两阶段。
对照：3 类检测（单阶段消融）、传统 HSV+GLCM+LBP+SVM。

> 交接说明、已验证/未验证事项、下一步分工见 `docs/算法交接文档.md`。

## 当前能跑什么

> 训练/推理请用虚拟环境：`python`。数据准备/传统基线也可以用项目根目录的 `python`。

```bash
# 环境与数据检查（只读）
python -m strawberry_cv.cli check
python -m strawberry_cv.cli stats

# 数据准备（会写文件；prepare 默认 dry-run）
python -m strawberry_cv.cli make-split
python -m strawberry_cv.cli export-yolo --class-mode 1cls
python -m strawberry_cv.cli export-roi
python -m strawberry_cv.cli prepare --execute

# 传统基线（当前环境可跑）
python -m strawberry_cv.cli traditional

# 训练/推理（用 venv 的 python）
python -m strawberry_cv.cli train-detector
python -m strawberry_cv.cli train-classifier
python -m strawberry_cv.cli evaluate-detector --weights <best.pt>
python -m strawberry_cv.cli evaluate-classifier --checkpoint <best.pt>
python -m strawberry_cv.cli evaluate-end2end --weights <det.pt> --classifier <cls.pt> --split test
python -m strawberry_cv.cli infer-image --image <img> --weights <det.pt> --classifier <cls.pt>
python -m strawberry_cv.cli infer-video --video <mp4> --weights <det.pt> --classifier <cls.pt> --line 640,0,640,720

# 单元测试（31 个，不需要 torch）
python -m unittest discover -s tests -v
```

## 当前权重与结果（2026-11-22）

```text
检测器：work/runs/detector/strawberry_1cls_lowlr/weights/best.pt
分类器：work/runs/classifier/resnet18_3cls/best.pt
传统 SVM：work/runs/traditional/svm_model.joblib
```

| 实验 | test 结果 |
|---|---|
| 1 类检测 | mAP50 .952 / mAP50-95 .722（9 图/26 框） |
| 3 类分类 | acc .960 / macro F1 .653（half 0/1） |
| 传统 SVM | acc .920 / macro F1 .639（half 0/1） |
| 两阶段端到端（conf .55） | P .920 / R .885 / state .870 / count_err -1 |

**half 不可用**：train half 6 张来自同一场景，test 只有 1 张 ROI，F1=0。test 很小，指标不能当泛化结论。

## 目录

| 路径 | 作用 |
|---|---|
| `configs/base.json` | 全部配置；工作目录默认 `work`（ASCII，避开中文路径问题） |
| `data/labels.py` | 读取/校验项目 JSON 标注 |
| `data/scenes.py` | 场景组清单（防重复图跨集） |
| `data/splits.py` | 按场景组划分 train/val/test |
| `data/yolo.py` | JSON → YOLO 1cls/3cls，含回读校验 |
| `data/roi.py` | GT 框裁剪单果 ROI（自适应 padding、贴边跳过） |
| `models/detector.py` | Ultralytics YOLO 薄封装（1 类） |
| `models/classifier.py` | ResNet18/34/50 三分类迁移学习 |
| `training/` | 训练入口与早停/种子 |
| `evaluation/` | 检测 P/R/AP、分类指标、端到端指标 |
| `inference/` | 两阶段管线、IoU 跟踪、越线计数、可视化 |
| `baselines/` | HSV 颜色矩/直方图 + GLCM + LBP + SVM |
| `tests/` | 合成数据单元测试，不依赖真实数据 |

## 重要约定

- 类别 id：`0=ripe`、`1=half`、`2=unripe`；1 类检测时所有框 `class 0`。
- 工作产物写到 `work`，不改动 `../datasets/strawberry_v1`。
- 单果分类只用 GT 框裁剪，不用 `traditional_cv/output/单果区域`。
- 划分必须按 `scene_groups.csv` 的 `scene_group`，同组不跨集。
- 没有安装 torch/ultralytics 时，`train-*`/`infer-*` 会给出明确的安装提示；`check`/`stats`/数据模块/传统基线不受影响。
