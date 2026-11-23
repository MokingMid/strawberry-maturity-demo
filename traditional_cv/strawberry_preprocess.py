# -*- coding: utf-8 -*-
"""
草莓成熟度检测 —— 预处理与单果分离脚本（最终版）
=================================================
流程（对应课程要求 6 步）：
  1. 读取图像，统一 resize 到 640x480
  2. 降噪：高斯滤波 vs 中值滤波，输出对比图，选定中值滤波
  3. RGB -> HSV，调试“红色 / 青色(黄绿)”阈值，生成二值掩码
  4. 形态学：开运算(去小噪点) + 闭运算(填空洞)，对比 3x3 / 5x5 核；
     再用“大核开运算”切掉细长突起（红花、细茎、细叶），留下粗壮的草莓主体
  5. findContours 查找连通域，按面积过滤，分离单个草莓
  6. 保存全套中间图（原图/滤波/HSV通道/二值掩码/开运算/闭运算/切后/轮廓标记）

检测策略（经过数据对比选定）：
  - 红果掩码（深红 + 淡粉）检测成熟/半熟，黄绿掩码检测未熟，各自用大核开运算“切”；
  - 用大核开运算“切”而不是“扔”，把粘连草莓切开、把细突起磨掉；
  - 最终报“成熟 / 半熟 / 未熟”三类；未熟绿果与绿叶颜色重叠，召回有限，作为已知局限说明。

依赖：opencv-python, numpy, matplotlib
说明：中文路径用 imdecode/imencode 读写；阈值集中在“参数配置区”。
"""

import os
from pathlib import Path
import sys
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 参数配置区
# ============================================================
BASE_DIR = str(Path(__file__).resolve().parents[1] / "data" / "strawberry_v1")
SOURCE_DIR = os.path.join(BASE_DIR, "detection", "images")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")

TARGET_SIZE = (640, 480)

# ---- 降噪 ----
FILTER_METHOD = "median"          # 选定方案：中值滤波
GAUSS_KSIZE = 5
MEDIAN_KSIZE = 5

# ---- HSV 阈值（H∈[0,180], S∈[0,255], V∈[0,255]）----
# 红果掩码 = 深红(红熟) + 淡粉(半熟)
DEEP_RED_HSV = [
    (0, 90, 60, 22, 255, 255),
    (165, 90, 60, 180, 255, 255),
]
PINK_HSV = [
    (0, 30, 150, 25, 90, 255),
    (165, 30, 150, 180, 90, 255),
]
# 青色/黄绿（未熟绿果），用于报告“红色/青色阈值”对比，不参与最终检测
GREEN_HSV = [
    (18, 45, 150, 60, 255, 255),
]

# ---- 形态学 ----
OPEN_KSIZE = 3                    # 开运算核（去小噪点）
CLOSE_KSIZE = 5                   # 闭运算核（填内部空洞）
CUT_RADIUS = 44                   # 红果“切”半径（实际核=2*R+1=89，数据对比选定）
GREEN_CUT_RADIUS = 0              # 黄绿果不切割（只做开运算+闭运算清洗，最大化召回绿果）

# ---- 轮廓 ----
MIN_AREA = 400                    # 最小面积，过滤微小杂轮廓

# ---- 熟度判定 ----
RIPENESS_RIPE = 0.5               # 深红占比 >= 0.5 判“成熟”
RIPENESS_GREEN = 0.5              # 绿占比 >= 0.5 判“未熟”，否则“半熟”

# ---- 控制量 ----
MAX_IMAGES = None
COMPARE_LIMIT = 3                 # 单独输出“滤波/形态学对比”大图的前 N 张

# ============================================================
# 工具函数
# ============================================================
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"编码图像失败: {path}")
    buf.tofile(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_chinese_font():
    from matplotlib import font_manager as fm
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return "default"


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gray_to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# ============================================================
# 步骤函数
# ============================================================
def step_resize(img, size=TARGET_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def step_denoise(img):
    gauss = cv2.GaussianBlur(img, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    median = cv2.medianBlur(img, MEDIAN_KSIZE)
    return {"gaussian": gauss, "median": median}, median if FILTER_METHOD == "median" else gauss


def step_to_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return hsv, (h, s, v)


def build_hsv_mask(hsv, ranges):
    mask = None
    for (loH, loS, loV, hiH, hiS, hiV) in ranges:
        part = cv2.inRange(hsv, (loH, loS, loV), (hiH, hiS, hiV))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    return mask


def step_color_masks(hsv):
    """深红、淡粉、黄绿(青)三个掩码。"""
    deep = build_hsv_mask(hsv, DEEP_RED_HSV)
    pink = build_hsv_mask(hsv, PINK_HSV)
    green = build_hsv_mask(hsv, GREEN_HSV)
    return deep, pink, green


def open_mask(mask, ksize):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def close_mask(mask, ksize):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def cut_pipeline(mask, cut_radius):
    """开运算去噪 -> 闭运算填孔 -> 大核开运算切细突起。返回 (opened, closed, cut)。"""
    opened = open_mask(mask, OPEN_KSIZE)
    closed = close_mask(opened, CLOSE_KSIZE)
    cut = open_mask(closed, 2 * cut_radius + 1)   # 半径 R -> 核 2R+1
    return opened, closed, cut


def build_berry(cnt, deep_mask, pink_mask, green_mask):
    """由轮廓构建草莓信息（面积/框/质心/红绿占比/熟度）。"""
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

    region = np.zeros(deep_mask.shape, np.uint8)
    cv2.drawContours(region, [cnt], -1, 255, -1)
    region_px = int(region.sum() // 255)
    if region_px == 0:
        return None
    deep_ratio = int(cv2.bitwise_and(region, deep_mask).sum() // 255) / region_px
    pink_ratio = int(cv2.bitwise_and(region, pink_mask).sum() // 255) / region_px
    green_ratio = int(cv2.bitwise_and(region, green_mask).sum() // 255) / region_px

    if green_ratio >= RIPENESS_GREEN:
        ripeness = "unripe"
    elif deep_ratio >= RIPENESS_RIPE:
        ripeness = "ripe"
    else:
        ripeness = "half"

    return {
        "contour": cnt, "area": int(area), "bbox": (x, y, w, h), "center": (cx, cy),
        "deep_ratio": deep_ratio, "pink_ratio": pink_ratio, "green_ratio": green_ratio,
        "ripeness": ripeness,
    }


def detect_berries(mask, deep_mask, pink_mask, green_mask):
    """取掩码连通域，按面积过滤。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    berries = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        b = build_berry(cnt, deep_mask, pink_mask, green_mask)
        if b is not None:
            berries.append(b)
    berries.sort(key=lambda b: b["area"], reverse=True)
    return berries


def draw_result(img, berries):
    marked = img.copy()
    colors = {"ripe": (0, 255, 0), "half": (0, 200, 255), "unripe": (255, 0, 0)}
    for b in berries:
        color = colors.get(b["ripeness"], (255, 255, 255))
        cv2.drawContours(marked, [b["contour"]], -1, color, 2)
        x, y, w, h = b["bbox"]
        cv2.rectangle(marked, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.circle(marked, b["center"], 4, (0, 0, 255), -1)
        cv2.putText(marked, b["ripeness"], (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return marked


# ============================================================
# 保存与绘图
# ============================================================
def save_stage_images(out_img_dir, name, images):
    for stage, img in images.items():
        imwrite_unicode(os.path.join(out_img_dir, f"{name}_{stage}.png"), img)


def save_pipeline_figure(out_fig_dir, name, panels):
    n = len(panels)
    cols = 4 if n <= 12 else 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.2))
    axes = np.array(axes).ravel()
    for ax, (title, img) in zip(axes, panels):
        if img is None:
            ax.axis("off")
            continue
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"草莓预处理流水线 —— {name}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_fig_dir, f"{name}_pipeline.png"), dpi=110)
    plt.close(fig)


def save_filter_compare(out_fig_dir, name, orig, gauss, median):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (t, img) in zip(axes, [("原图", orig), ("高斯滤波", gauss), ("中值滤波", median)]):
        ax.imshow(bgr_to_rgb(img))
        ax.set_title(t, fontsize=13)
        ax.axis("off")
    fig.suptitle(f"降噪方案对比（选定：{FILTER_METHOD}） —— {name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out_fig_dir, f"{name}_denoise_compare.png"), dpi=110)
    plt.close(fig)


def save_morph_compare(out_fig_dir, name, mask, open_3, open_5, close_3, close_5):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = [
        ("二值掩码", mask), ("开运算 3x3", open_3), ("开运算 5x5", open_5),
        ("闭运算 3x3", close_3), ("闭运算 5x5", close_5),
    ]
    for ax, (t, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=12)
        ax.axis("off")
    fig.suptitle(f"形态学核尺寸对比（选定：开 {OPEN_KSIZE}x{OPEN_KSIZE} / 闭 {CLOSE_KSIZE}x{CLOSE_KSIZE}） —— {name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out_fig_dir, f"{name}_morphology_compare.png"), dpi=110)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def process_one(img_path, out, writer, counters):
    name = os.path.splitext(os.path.basename(img_path))[0]
    orig_full = imread_unicode(img_path)
    if orig_full is None:
        print(f"  [警告] 无法读取 {img_path}")
        return 0, 0, 0

    img = step_resize(orig_full)
    filter_results, filtered = step_denoise(img)
    hsv, (h, s, v) = step_to_hsv(filtered)
    deep, pink, green = step_color_masks(hsv)
    red = cv2.bitwise_or(deep, pink)          # 红果掩码（深红 + 淡粉）

    # 步骤4：开运算去噪 -> 闭运算填孔 -> 大核开运算“切”掉细突起（红、绿各自切）
    red_opened, red_closed, red_cut = cut_pipeline(red, CUT_RADIUS)
    green_opened, green_closed, green_cut = cut_pipeline(green, GREEN_CUT_RADIUS)
    combined = cv2.bitwise_or(red_cut, green_cut)

    # 步骤5：轮廓检测 + 面积过滤
    berries = detect_berries(combined, deep, pink, green)
    marked = draw_result(img, berries)

    n_ripe = sum(1 for b in berries if b["ripeness"] == "ripe")
    n_half = sum(1 for b in berries if b["ripeness"] == "half")
    n_unripe = sum(1 for b in berries if b["ripeness"] == "unripe")
    counters["ripe"] += n_ripe
    counters["half"] += n_half
    counters["unripe"] += n_unripe
    counters["total"] += len(berries)

    # ---- 保存中间过程图 ----
    stage_images = {
        "01_原图": img,
        "02_高斯滤波": filter_results["gaussian"],
        "03_中值滤波": filter_results["median"],
        "04_H通道": gray_to_bgr(h),
        "05_S通道": gray_to_bgr(s),
        "06_V通道": gray_to_bgr(v),
        "07_红果掩码": gray_to_bgr(red),
        "08_黄绿掩码(青)": gray_to_bgr(green),
        "09_开运算(红)": gray_to_bgr(red_opened),
        "10_闭运算(红)": gray_to_bgr(red_closed),
        "11_切后掩码(红)": gray_to_bgr(red_cut),
        "12_黄绿掩码(清洗)": gray_to_bgr(green_cut),
        "13_合并掩码": gray_to_bgr(combined),
        "14_轮廓标记": marked,
    }
    save_stage_images(out["stage"], name, stage_images)

    # ---- 保存单果区域 ----
    berry_dir = out["berry"]
    for i, b in enumerate(berries):
        bx, by, bw, bh = b["bbox"]
        crop = img[max(0, by - 5):by + bh + 5, max(0, bx - 5):bx + bw + 5]
        imwrite_unicode(os.path.join(berry_dir, f"{name}_berry{i:02d}.png"), crop)

    # ---- CSV ----
    for i, b in enumerate(berries):
        bx, by, bw, bh = b["bbox"]
        cx, cy = b["center"]
        writer.writerow([
            name, i, b["area"], bx, by, bw, bh, cx, cy,
            round(b["deep_ratio"], 3), round(b["pink_ratio"], 3), round(b["green_ratio"], 3),
            {"ripe": "成熟", "half": "半熟", "unripe": "未熟"}[b["ripeness"]],
        ])

    # ---- 流水线拼图 ----
    save_pipeline_figure(out["fig"], name, [
        ("1 原图", bgr_to_rgb(img)),
        ("2 高斯滤波", bgr_to_rgb(filter_results["gaussian"])),
        ("3 中值滤波", bgr_to_rgb(filter_results["median"])),
        ("4 H通道", h),
        ("5 S通道", s),
        ("6 V通道", v),
        ("7 红果掩码", red),
        ("8 黄绿掩码(青)", green),
        ("9 开运算(红)", red_opened),
        ("10 闭运算(红)", red_closed),
        ("11 切后掩码(红)", red_cut),
        ("12 黄绿掩码(清洗)", green_cut),
        ("13 合并掩码", combined),
        ("14 轮廓标记", bgr_to_rgb(marked)),
    ])

    return n_ripe, n_half, n_unripe, len(berries)


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"[错误] 输入目录不存在: {SOURCE_DIR}")
        sys.exit(1)

    setup_chinese_font()

    out = {
        "stage": os.path.join(OUTPUT_DIR, "中间过程图"),
        "fig": os.path.join(OUTPUT_DIR, "对比图"),
        "berry": os.path.join(OUTPUT_DIR, "单果区域"),
    }
    for p in out.values():
        ensure_dir(p)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = sorted(f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(exts))
    if MAX_IMAGES is not None:
        img_files = img_files[:MAX_IMAGES]

    print(f"共发现 {len(img_files)} 张图像，开始处理...")

    csv_path = os.path.join(OUTPUT_DIR, "轮廓信息.csv")
    header = ["图像", "轮廓序号", "面积(px)", "x", "y", "宽", "高", "质心x", "质心y",
              "深红占比", "淡粉占比", "绿占比", "熟度"]
    counters = {"ripe": 0, "half": 0, "unripe": 0, "total": 0}
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, fname in enumerate(img_files):
            img_path = os.path.join(SOURCE_DIR, fname)
            n_ripe, n_half, n_unripe, n = process_one(img_path, out, writer, counters)
            print(f"  [{idx + 1}/{len(img_files)}] {fname}: 成熟 {n_ripe} 颗, 半熟 {n_half} 颗, 未熟 {n_unripe} 颗")

            if idx < COMPARE_LIMIT:
                name = os.path.splitext(fname)[0]
                orig_full = imread_unicode(img_path)
                img = step_resize(orig_full)
                filter_results, _ = step_denoise(img)
                deep_c, pink_c, _ = step_color_masks(step_to_hsv(filter_results[FILTER_METHOD])[0])
                red_for_cmp = cv2.bitwise_or(deep_c, pink_c)
                save_filter_compare(out["fig"], name, img,
                                    filter_results["gaussian"], filter_results["median"])
                save_morph_compare(out["fig"], name, red_for_cmp,
                                   open_mask(red_for_cmp, 3), open_mask(red_for_cmp, 5),
                                   close_mask(red_for_cmp, 3), close_mask(red_for_cmp, 5))

    print(f"\n处理完成！")
    print(f"  检测到草莓共 {counters['total']} 颗：成熟 {counters['ripe']} 颗, 半熟 {counters['half']} 颗, 未熟 {counters['unripe']} 颗")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"  - 中间过程图: {out['stage']}")
    print(f"  - 对比图:     {out['fig']}")
    print(f"  - 单果区域:   {out['berry']}")
    print(f"  - 轮廓信息:   {csv_path}")


if __name__ == "__main__":
    main()
