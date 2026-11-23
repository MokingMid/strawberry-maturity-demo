"""项目常量：类别、路径约定、指标列名。"""

from __future__ import annotations

# 三状态分类（主线分类器）
CLASS_NAMES = ("ripe", "half", "unripe")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}

# 1 类检测：所有框都归为“草莓”
DETECTOR_CLASS_NAME = "strawberry"
DETECTOR_CLASS_ID = 0

# 3 类检测对照实验直接复用分类类别
DETECTOR_CLASS_NAMES_3 = CLASS_NAMES

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 检测结果 / 轨迹日志列
DETECTION_CSV_COLUMNS = (
    "frame", "track_id", "x1", "y1", "x2", "y2",
    "det_conf", "state", "state_conf", "class_conf_ripe",
    "class_conf_half", "class_conf_unripe",
)

# 评测输出
SPLIT_NAMES = ("train", "val", "test")
