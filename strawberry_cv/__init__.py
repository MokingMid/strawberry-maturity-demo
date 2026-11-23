"""草莓成熟度检测与分拣 —— 两阶段算法包。

主线：1 类检测（strawberry）+ 3 类分类（ripe / half / unripe）。
数据与训练产物默认写到 ASCII 工作目录，避免中文路径在 Ultralytics 下的兼容问题。
"""

__version__ = "0.1.0"
