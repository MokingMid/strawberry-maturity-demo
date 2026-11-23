# 草莓成熟度检测 —— 可视化网站

基于 `demo/demo_core.py` 的两阶段推理，包装成一个 Flask 单页应用。

## 启动

```bash
cd E:\临时
python demo/web/app.py --port 8000
```

浏览器打开 **http://127.0.0.1:8000**

> 首次请求会加载模型（CPU 约 10~20 秒），之后单图约 200~300ms。
> 换检测器/分类器型号会重新载入权重（约 10 秒）。
>
> 局域网访问：`python demo/web/app.py --host 0.0.0.0 --port 8000`，
> 然后用 `http://<本机IP>:8000` 访问（需放行防火墙）。

依赖：Flask（已随 ultralytics 安装）。无需额外前端构建，纯 HTML/CSS/JS。

---

## 四个页面

| Tab | 功能 |
|---|---|
| **▶ 单张分析** | 上传图片 或 从数据集里挑一张 → 实时推理 → 标注图 + 统计卡片 + 逐果明细表 |
| **🖼 数据集浏览** | 分页浏览两套数据集的 630 张图，点任意缩略图即时分析并展开详情 |
| **📊 批量评测** | 对某个数据集批量跑，给出三类计数、**与 GT 的状态一致率**、检测 P/R 和结果画廊 |
| **📈 模型指标** | 三个分类器 × 三个测试集的对比表、端到端对比表、成果画廊 |

## 可调参数

- **检测阈值 conf**（0.10~0.90，默认 0.55）—— 越低召回越高、误检越多
- **NMS IoU**（0.10~0.90，默认 0.50）
- **检测器**：合并检测器（推荐，两个域都覆盖）/ v1 检测器（仅网络特写）
- **分类器**：合并分类器（推荐）/ 归档集分类器 / v1 分类器（half F1=0）

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 主页面 |
| GET | `/api/samples?dataset=v1\|archive&page=1&size=24` | 列出数据集图片（分页） |
| GET | `/api/sample/<dataset>/<文件名>?conf=&iou=&detector=&classifier=` | 分析数据集里的某张图 |
| POST | `/api/upload`（multipart `file`） | 上传图片分析 |
| GET | `/api/batch?dataset=&count=&offset=` | 批量分析 + 与 GT 对比 |
| GET | `/api/stats` | 模型指标对比（读已生成的评测 JSON） |
| GET | `/api/gallery` | 成果画廊图片 |
| GET | `/results/<文件名>` | 取回标注结果图 |

返回的 JSON 结构：

```json
{
  "image_url": "/results/archive_archive_image_001_demo.jpg",
  "source": "archive_image_001.jpg",
  "total": 4,
  "counts": {"ripe": 1, "half": 0, "unripe": 3},
  "maturity": 0.25,
  "latency_ms": 251.8,
  "fruits": [
    {"box": [0, 370, 58, 451], "det_conf": 0.95, "state": "unripe",
     "state_cn": "未熟", "state_conf": 0.876, "probs": {"ripe": 0.068, "half": 0.055, "unripe": 0.876}}
  ]
}
```

---

## 文件结构

```
demo/web/
├─ app.py                  Flask 应用（后端 + API）
├─ templates/index.html    单页界面
├─ static/style.css        样式（深色主题，无外部 CDN 依赖）
├─ static/app.js           前端逻辑
├─ uploads/                上传的图片
├─ results/                生成的标注结果图
└─ server.log / server.err 启动日志（用 Start-Process 后台启动时产生）
```

---

## 实现要点

- **中文路径**：OpenCV 的 `imread`/`VideoCapture` 在 `E:\临时` 下会失效，
  所以网站里统一用 `demo_core` 的 `imread`（走 imdecode）和 `open_video`（临时 ASCII 路径中转）。
- **中文文字**：OpenCV 的 `putText` 不支持中文（会显示 `???`），
  标注图用 PIL + 微软雅黑渲染。
- **模型只加载一次**：按 (检测器, 分类器, conf, iou) 四元组缓存，切换型号才重建。
  推理用锁保护，避免并发请求同时改模型。
- **GT 配对按 IoU 匹配**，不是按顺序 zip —— 检测框顺序与标注顺序不一致，
  按顺序 zip 会让统计结果完全失真（实测同一批数据：58.8% vs 100%）。
- **本机是纯 CPU**，Flask 用的是开发服务器，只适合本地演示；
  要正式部署请换 waitress / gunicorn + 真正的 WSGI。

## 已知限制

- 视频推理在 CLI 里更方便（`python demo/strawberry_demo.py --video ...`），网站没有做视频上传
  （浏览器端要传 640×640 帧流，实时性和复杂度都不划算）。
- 摄像头实时预览也没放进网站，理由同上；用 CLI 的 `--camera 0 --gui`。
- 批量评测一次最多 60 张（`count` 参数上限），更多请分批跑。
