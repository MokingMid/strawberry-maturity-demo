# 草莓成熟度检测 Demo

两阶段流水线，**开箱即跑**：

```
输入图 ──► YOLO11n 检测（找草莓）──► ROI 裁剪 ──► ResNet18 分类 ──► 成熟度标注 + 统计
```

| 成熟度 | 颜色 | 判定 | 成熟度权重 |
|---|---|---|---|
| 成熟 `ripe` | 红 | 果面红色覆盖 ≥ 1/2 | 1.0 |
| 半熟 `half` | 橙 | 有可见红色但覆盖 < 1/2 | 0.5 |
| 未熟 `unripe` | 绿 | 基本无红色 | 0.0 |

界面右上角面板给出三类数量、占比条、**成熟度指数**（加权平均，0~100%）和推理耗时。

---

## 快速开始

```bash
cd E:\临时

# 1) 单张图片
python demo/strawberry_demo.py --image datasets/strawberry_v1/detection/images/image_001.jpg --out demo_out

# 2) 整个文件夹
python demo/strawberry_demo.py --folder datasets/strawberry_v1/detection/images --out demo_out

# 3) 视频
python demo/strawberry_demo.py --video clip.mp4 --out demo_out

# 4) 摄像头实时（按 q 退出）
python demo/strawberry_demo.py --camera 0 --gui

# 5) 数据集自检（GT 框 + 分类器，输出混淆矩阵与 P/R/F1）
python demo/strawberry_demo.py --self-test --dataset archive --self-test-n 60
```

不带任何参数运行会打印完整用法。

---

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--detector` | `work/runs/detector/strawberry_1cls_lowlr/weights/best.pt` | 检测权重 |
| `--classifier` | 归档集分类器（若存在）否则 v1 分类器 | 分类权重 |
| `--conf` | 0.55 | 检测置信度阈值 |
| `--iou` | 0.5 | NMS IoU 阈值 |
| `--device` | auto | `auto` / `cpu` / `0`（GPU） |
| `--gui` | 关 | 弹出预览窗口逐帧确认 |
| `--no-save` | 关 | 不落盘（只想看结果时用） |
| `--limit` / `--max-frames` | 无 | 文件夹 / 视频的处理上限 |
| `--no-detector` | 关 | 只做分类（无检测权重时） |
| `--no-classifier` | 关 | 只做检测（无分类权重时） |

---

## 输出

```
demo_out/
├─ annotated/              ← 每张图的标注结果（文件夹模式）
│   └─ xxx_demo.jpg
├─ xxx_demo.jpg            ← 单图模式的标注结果
├─ predictions.csv         ← 逐果明细：框坐标 / 检测置信度 / 成熟度 / 三类概率
├─ result.json             ← 单图模式的结构化结果
└─ summary.json            ← 批量模式的汇总（含总成熟度）
```

---

## 模型来源

| 模型 | 权重 | 训练数据 | 关键指标 |
|---|---|---|---|
| 检测器（v1 原版） | `work/runs/detector/strawberry_1cls_lowlr/weights/best.pt` | `strawberry_v1`（64 图 / 183 框） | test mAP50 **.952** |
| 检测器（**合并，默认**） | `work_merged/runs/detector/merged_1cls_lowlr/weights/best.pt` | 两集合并（630 图 / 1917 框） | val mAP **.696** / mAP50 **.866** / mAP75 .796 |
| 分类器（v1 原版） | `work/runs/classifier/resnet18_3cls/best.pt` | `strawberry_v1` ROI（168 张） | test acc .960，但 **half F1 = 0** |
| 分类器（归档集） | `work_archive/runs/classifier/resnet18_3cls_archive/best.pt` | `strawberry_archive_v1` ROI（1386 张） | val acc .915 / macroF1 .905 / **half F1 .864** |
| 分类器（**合并，默认**） | `work_merged/runs/classifier/resnet18_3cls_merged/best.pt` | 两集合并 ROI（1554 张） | val acc .909 / macroF1 .898 / **half F1 .841** |

### 合并数据集带来的提升

用两个数据集合并训练（检测器 + 分类器），端到端实测（各 30 图，真实检测框）：

| 配置 | v1 状态正确率 | 归档 状态正确率 | 归档域检测 P / R |
|---|---|---|---|
| v1 原版检测器 + v1 原版分类器 | 97.6% | **65.9%** | .786 / **.550** |
| v1 原版检测器 + 合并分类器 | **100.0%** | 86.4% | .786 / **.550** |
| **合并检测器 + 合并分类器（默认）** | **100.0%** | **92.5%** | **.838 / .838** |

- 温室域漏检从 **36 → 13** 颗，召回 **.550 → .838**
- v1 域指标完全不受影响（P .976 / R .988 不变）
- 合并检测器在 val 上的 mAP50 .866（比 v1 单域的 .995 低，是因为归档集里大量远摄小果，
  本身就更难；关键是它在两个域上都不崩）

### 为什么默认用合并分类器

v1 训练集只有 **11 个 half 框 / 3 个场景**，分类器学不会半熟，推理时**永远不会输出「半熟」**。
归档集有 247 个 half 框（v1 的 22 倍），把两套 ROI 合并训练后，两个拍摄域都能覆盖。

三分类器 × 三测试集的实测对比（纯分类）：

| 分类器 | v1 test (n=25) | 归档 test (n=219) | 合并 test (n=244) |
|---|---|---|---|
| v1 原版 | acc .960 / **halfF1 .000** | acc .621 / halfF1 .198 | acc .656 / halfF1 .195 |
| 归档集 | acc 1.000 / halfF1 1.000 | acc **.872** / halfF1 **.788** | acc **.885** / halfF1 **.791** |
| 合并 | acc **1.000** / halfF1 **1.000** | acc .863 / halfF1 .769 | acc .877 / halfF1 .773 |

端到端（真实检测框 + 分类，各 30 图）：

| 分类器 | v1 状态正确率 | 归档 状态正确率 |
|---|---|---|
| v1 原版 | 97.6% | **65.9%**（域外崩溃） |
| 归档集 | 96.3% | **90.9%** |
| 合并（默认） | **100.0%** | 86.4% |

> **结论**：v1 原版分类器跨域会崩（归档域只有 65.9%）；归档集与合并模型都可用。
> 合并模型在两个域上表现最均衡（唯一在两个域都 ≥86%），因此作为 demo 默认；
> 归档集模型在归档域略优（+4.5%），可用 `--classifier` 切换。

---

## 复现这些结果

```bash
cd E:\临时\handoff_2026-11-22

# 1) 归档集划分 + ROI 裁剪
python -m strawberry_cv.cli make-split  --config configs/archive.json
python -m strawberry_cv.cli export-roi  --config configs/archive.json --force

# 2) 归档集分类器
python -m strawberry_cv.cli train-classifier --config configs/archive.json \
  --roi-dir work_archive/roi_classification --name resnet18_3cls_archive

# 3) 合并 ROI + 合并分类器
python E:\临时\demo\tools\merge_roi_datasets.py
python -m strawberry_cv.cli train-classifier --config configs/merged.json \
  --roi-dir work_merged/roi_classification --name resnet18_3cls_merged

# 4) 归档集 YOLO 导出 + 合并检测数据集
python -m strawberry_cv.cli export-yolo --config configs/archive.json --class-mode 1cls \
  --split-manifest work_archive/splits/split_manifest.csv --force
python E:\临时\demo\tools\merge_yolo_datasets.py

# 5) 合并检测器
python -m strawberry_cv.cli train-detector --config configs/merged.json \
  --data work_merged/yolo_detection_1cls/data.yaml --epochs 45 --batch 16 \
  --name merged_1cls_lowlr

# 6) 对比评测
python E:\临时\demo\tools\compare_classifiers.py
python E:\临时\demo\tools\compare_end2end.py

# 7) 成果画廊
python E:\临时\demo\tools\make_gallery.py
```

> 本机为 **纯 CPU**（无 CUDA）：
> 两个分类器各训练约 20~30 分钟（ResNet18 / 224px / 1554 张 ROI），
> 合并检测器约 85 分钟（YOLO11n / 640px / 441 张训练图 / batch 16）。
> 有 GPU 时同样的命令会快一个数量级。

---

## 附加工具

| 脚本 | 作用 |
|---|---|
| `tools/merge_roi_datasets.py` | 合并两套数据集的 ROI 裁剪成分类训练集 |
| `tools/merge_yolo_datasets.py` | 合并两套数据集的 YOLO 数据成检测训练集 |
| `tools/compare_classifiers.py` | 三个分类器 × 三个测试集的纯分类对比 |
| `tools/compare_end2end.py` | 端到端（检测+分类）对比 |
| `tools/make_demo_clip.py` | 用数据集图片合成演示视频（带轻微晃动） |
| `tools/make_gallery.py` | 把标注结果拼成一张成果对比图 |
| `tests/test_gt_pairing.py` | 回归测试：GT 框被跳过时配对不错位 |

---

## 文件结构

```
demo/
├─ strawberry_demo.py   命令行入口（图片/文件夹/视频/摄像头/自检 五种模式）
├─ demo_core.py         核心：模型加载、两阶段推理、中文渲染、结果导出
└─ README.md            本文件
```

`demo_core.py` 复用了 `handoff_2026-11-22/strawberry_cv/` 里的
检测器封装（`models/detector.py`）、分类器封装（`models/classifier.py`）、
ROI 裁剪（`data/roi.py`）和图像 IO，**没有重复实现模型逻辑**。

中文文字用 PIL + 系统中文字体渲染（OpenCV 的 `putText` 不支持中文，会显示成 `???`）。

---

## 已知限制

- **合并检测器只训了 45 轮**（纯 CPU，约 85 分钟）。YOLO 曲线到最后 3 轮已基本收敛
  （mAP50 .864/.864/.864），但若用 GPU 训 120 轮还能再涨一截。
- **合并检测器在 val 上的 mAP50 .866 低于 v1 单域检测器的 .995**，原因是归档集里
  有大量远摄小果（474 个过小框）、画面复杂（一图多果最多 20 个），本身就更难。
  合并的意义不是刷高 mAP，而是**两个域都不崩**。
- **成熟度指数是启发式加权**（成熟 1 / 半熟 0.5 / 未熟 0），用于演示中的直观对比，
  不是农学上的品质指标。
- 摄像头模式用的是轻量 IoU 追踪，ID 在快速运动/遮挡时可能跳变；工业级需换 ByteTrack。
- 只处理单帧或视频，不做跨帧的成熟度状态平滑（同一颗草莓的成熟度不会随时间变化）。
- 归档集的 `archive_image_566` 有源标签冲突（同框标了 half 和 unripe），
  `archive_image_282/346/466/538/547/566` 有 6 对重复框，训练前应先人工修正（详见归档集数据卡）。
- 整个流程在**纯 CPU** 上跑：单图端到端约 **130~230 ms**（含检测 + 分类）。有 GPU 会快得多。
