<div align="center">

# 🍓 草莓成熟度检测 Demo

**YOLO 检测 + ResNet18 三状态分类**（成熟 / 半熟 / 未熟）

[![CI](https://github.com/MokingMid/strawberry-maturity-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/MokingMid/strawberry-maturity-demo/actions/workflows/ci.yml)
[![Pages](https://github.com/MokingMid/strawberry-maturity-demo/actions/workflows/pages.yml/badge.svg)](https://github.com/MokingMid/strawberry-maturity-demo/actions/workflows/pages.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[在线展示](https://MokingMid.github.io/strawberry-maturity-demo/) ·
[快速开始](#-快速开始) · [实测指标](#-实测指标) · [仓库结构](#-仓库结构)

</div>

---

## 这是什么

一个**可以直接跑**的草莓成熟度检测系统：

```
输入图 ──► YOLO11n 检测（找草莓）──► ROI 裁剪 ──► ResNet18 分类 ──► 成熟度标注 + 统计
```

三种打开方式：

| 方式 | 需要什么 | 适合谁 |
|---|---|---|
| **🌐 浏览器端 Demo** | 只要能打开网址 | 任何人 —— 模型跑在访客浏览器里，图片不外传 |
| 🐍 命令行 Demo | Python 环境 | 本地批量处理、视频、摄像头 |
| 🌐 服务端网站 | 部署一台服务器 | 需要批量评测、指标对比、GPU 加速 |

```
成熟 = 红色覆盖 ≥ 1/2      ┐
半熟 = 有红色但 < 1/2      ├─ 按「果面可见红色区域」估计，不按检测框面积
未熟 = 基本无红色          ┘
```

---

## 🚀 快速开始

### 方式一：Docker（推荐，跨平台）

```bash
git clone https://github.com/MokingMid/strawberry-maturity-demo.git
cd strawberry-maturity-demo
docker build -t strawberry-demo .
docker run -p 8000:8000 strawberry-demo
# → http://localhost:8000
```

### 方式二：本地 Python

```bash
git clone https://github.com/MokingMid/strawberry-maturity-demo.git
cd strawberry-maturity-demo

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python app.py                      # 网站 → http://127.0.0.1:8000
```

### 方式三：命令行 Demo

```bash
# 单张图片
python demo/strawberry_demo.py --image samples/v1/detection/images/image_001.jpg --out demo_out

# 批量
python demo/strawberry_demo.py --folder samples/v1/detection/images --out demo_out

# 视频（带目标追踪 ID）
python demo/strawberry_demo.py --video clip.mp4 --out demo_out

# 摄像头实时（按 q 退出）
python demo/strawberry_demo.py --camera 0 --gui

# 数据集自检：输出混淆矩阵与 P/R/F1
python demo/strawberry_demo.py --self-test --dataset archive --with-detector
```

### 方式四：部署到云端

仓库内置 `Dockerfile` 与 `render.yaml`。在仓库页点 **Deploy to Render**，
或在 [render.com/deploy](https://render.com/deploy) 选这个仓库 → Apply，
几分钟后得到公网地址。

> 免费版实例空闲 15 分钟会休眠，首次访问约需 1 分钟冷启动。

---

## 🌐 可视化网站

### A. 浏览器端 Demo（零部署，推荐给访客）

**<https://mokingmid.github.io/strawberry-maturity-demo/demo/>**

模型转成 ONNX 后用 **ONNX Runtime Web（WASM）** 在访客浏览器里跑，**图片不会上传到任何服务器**：

- 检测器 YOLO11n → ONNX fp32（10.1 MB）
- 分类器 ResNet18 → ONNX **int8 静态量化**（42.6 MB → 10.8 MB，精度只损失 **0.4%**）
- LetterBox、YOLO 解码、NMS、ROI 裁剪、分类预处理**逐项对齐 Python 端**
  （自实现的 bilinear 与 cv2.INTER_LINEAR 同口径）

实测成熟度判定 **17/17** 与 PyTorch 完全相同；单图推理约 **0.6~1.0 秒**（CPU/WASM）。
首次访问需下载约 32 MB（模型 + WASM 运行时），之后浏览器缓存。

### B. 服务端网站（Flask，支持 GPU 与批量）

四个页面：

| 页面 | 功能 |
|---|---|
| ▶ 单张分析 | 上传图片 / 从数据集挑一张 → 标注图 + 统计卡片 + 逐果明细表 |
| 🖼 数据集浏览 | 分页浏览数据集图片，点缩略图即时分析 |
| 📊 批量评测 | 三类计数、**与 GT 一致率**、检测 P/R、结果缩略图墙 |
| 📈 模型指标 | 分类器 × 测试集对比表、端到端对比、成果画廊 |

可实时调整检测阈值 conf、NMS IoU，切换检测器与分类器型号，下载标注结果图。

API（`/api/samples`、`/api/upload`、`/api/batch`、`/api/stats`）详见
[`demo/web/README.md`](demo/web/README.md)。

---

## 📊 实测指标

端到端（真实检测框 + 分类，各 30 图）：

| 配置 | 网络特写域 | 温室大棚域 | 温室域检测 P / R |
|---|---|---|---|
| v1 原版检测器 + v1 原版分类器 | 97.6% | **65.9%** | .786 / **.550** |
| v1 原版检测器 + 合并分类器 | 100.0% | 86.4% | .786 / .550 |
| **合并检测器 + 合并分类器**（默认） | **100.0%** | **92.5%** | **.838 / .838** |

分类器对比（纯分类，GT 框）：

| 分类器 | 训练数据 | v1 test | 归档 test | half F1 |
|---|---|---|---|---|
| v1 原版 | 168 张 ROI（仅 11 个 half） | acc .960 | acc .621 | **.000** |
| 归档集 | 1386 张 ROI（247 个 half） | acc 1.000 | acc .872 | .788 |
| **合并** | 1554 张 ROI | acc 1.000 | acc .863 | .773 |

### ⚠️ 最关键的一课：half 类

`half`（半熟）是三类里最难的，也是最初整个项目最致命的问题：

- v1 数据集只有 **11 个 half 框、3 个场景** → 分类器根本学不会 → **推理时永远不会输出「半熟」**，F1 = 0
- 归档集有 **247 个 half 框** → 重训后 F1 直接拉到 **.79~.86**

> 这不是调参能解决的，是**数据量**问题。合并两套数据集训练后，两个拍摄域都不再崩溃。

另一个必须注意的坑：**划分必须按场景组，不能按图片随机抽**。
`manifests/scene_groups.csv` 里的 `scene_group` 记录了哪些图是同一场景/连拍，
随机划分会让近似图片同时进入 train/val/test，指标虚高。

---

## 📁 两套数据集

| 数据集 | 场景 | 图片 | 标注框 | ripe / half / unripe |
|---|---|---|---|---|
| `strawberry_v1` | 网络单果特写 | 64 | 183 | 76 / **11** / 96 |
| `strawberry_archive_v1` | 温室大棚连拍 | 566 | 1734 | 520 / **247** / 967 |

标注格式（两套一致）：

```json
{"labels": [{"name": "ripe", "x1": 0, "y1": 443, "x2": 79, "y2": 489,
             "size": {"height": 640, "width": 640}}]}
```

> 完整数据集（41 MB）因体积与第三方来源不随仓库分发，
> 仓库内附 `samples/`（20 张图 + 标注），网站开箱即可浏览。
> 需要完整数据集请看 [`docs/数据集说明.md`](docs/) 或联系作者。

---

## 📂 仓库结构

```
strawberry-maturity-demo/
├─ app.py                    WSGI 入口（Render / gunicorn）
├─ Dockerfile / render.yaml  部署
├─ models/
│  ├─ detector_merged.pt           合并检测器 PyTorch（5.2 MB，服务端用）
│  ├─ classifier_merged.pt         合并分类器 PyTorch（42.7 MB，服务端用）
│  ├─ detector_merged.onnx         检测器 ONNX fp32（10.1 MB，浏览器端用）
│  └─ classifier_merged_int8.onnx  分类器 ONNX int8（10.8 MB，浏览器端用）
├─ strawberry_cv/            算法代码（CLI / 数据 / 模型 / 训练 / 评测 / 推理）
├─ demo/
│  ├─ strawberry_demo.py     命令行 Demo
│  ├─ demo_core.py           两阶段推理 + 中文渲染 + 导出
│  ├─ paths.py               跨布局路径解析
│  ├─ web/                   Flask 可视化网站（服务端版）
│  ├─ tools/                 ONNX 导出 / 数据合并 / 模型对比 / 画廊生成
│  └─ tests/                 回归测试
├─ site/                     GitHub Pages 站点
│  ├─ index.html             项目主页（静态展示）
│  ├─ demo/                  🌐 浏览器端推理 Demo（ONNX Runtime Web）
│  └─ ort/                   onnxruntime-web 运行时（本地托管，不依赖 CDN）
├─ samples/                  20 张样例图 + 标注
├─ traditional_cv/           传统 CV 基线（HSV + GLCM + LBP + SVM）
├─ configs/base.json
├─ docs/                     交接文档 / 数据审计 / 开题材料
├─ reports/                  固化的评测指标 JSON（网站指标页直接读）
└─ tests/                    31 个单元测试
```

---

## 🧪 测试与复现

```bash
python -m unittest discover -s tests -v        # 31 个单元测试
python -m strawberry_cv.cli check              # 数据自检
python demo/tools/compare_classifiers.py       # 分类器对比表
python demo/tools/compare_end2end.py           # 端到端对比表
python demo/tools/make_gallery.py              # 生成成果画廊
```

CI 会在每次 push 自动跑测试和冒烟检查（`/.github/workflows/ci.yml`）。

---

## 🛠 踩过的坑（都在代码里规避了）

| 问题 | 处理 |
|---|---|
| OpenCV `imread`/`VideoCapture` 在中文路径下失效 | 走 `imdecode`；视频先复制到 ASCII 临时路径 |
| OpenCV `putText` 不支持中文（显示 `???`） | 标注文字改用 PIL + 系统中文字体渲染 |
| GT 与预测按顺序 `zip` 会错位 | 改用 IoU 匹配（实测同一批数据 58.8% vs 100%） |
| `scene_group=none` 的图片会被折叠成一个大组 | 未分组图片各分配独立组 id，避免整组塌进同一 split |
| `tests/` 缺 `__init__.py` 导致测试命令跑不通 | 已补，31 个测试全部通过 |

---

## 📄 License

[MIT](LICENSE)

数据集与第三方资源（Roboflow 导出的温室数据集、网络收集图片）版权归原作者所有，
本仓库仅用于课程设计的学习与实验。
