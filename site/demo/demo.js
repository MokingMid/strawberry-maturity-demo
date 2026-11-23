/* 草莓成熟度检测 —— 浏览器端两阶段推理
 *
 * 与 Python 端 strawberry_cv / demo_core 的行为逐项对齐：
 *   1) LetterBox 预处理（640×640，灰边 114）
 *   2) YOLO 输出解码（cx,cy,w,h → xyxy）+ 阈值过滤 + IoU NMS
 *   3) 坐标还原到原图
 *   4) ROI 裁剪（外扩 12% → 补正方形 → 邻果避让）
 *   5) 分类器预处理（短边缩到 256 → 中心裁 224 → ImageNet 归一化）
 *
 * 缩放用自己实现的 bilinear（与 cv2.INTER_LINEAR 同口径：不做抗混叠），
 * 以保证浏览器端与 Python 端结果一致。
 */

'use strict';

const CFG = {
  imgsz: 640,
  conf: 0.55,
  iou: 0.5,
  maxDet: 100,
  padRatio: 0.12,
  minSize: 32,
  maxNeighborIou: 0.1,
  padSteps: 6,
  clsSize: 224,
  clsShort: 256,
};

/* 模型输出索引 → 类别。ImageFolder 按字母序：0=half, 1=ripe, 2=unripe */
const FOLDER_ORDER = ['half', 'ripe', 'unripe'];
/* 逻辑顺序（对外展示） */
const LOGICAL_ORDER = ['ripe', 'half', 'unripe'];

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

/* ============================================================ 图像工具 */

/** 双线性缩放（对齐 cv2.INTER_LINEAR，不做抗混叠）。RGBA → RGBA */
function resizeBilinear(src, sw, sh, dw, dh) {
  const out = new Uint8ClampedArray(dw * dh * 4);
  if (sw === dw && sh === dh) {
    out.set(src.subarray(0, dw * dh * 4));
    return out;
  }
  const scaleX = sw / dw;
  const scaleY = sh / dh;
  for (let y = 0; y < dh; y++) {
    let srcY = (y + 0.5) * scaleY - 0.5;
    let y0 = Math.floor(srcY);
    let wy = srcY - y0;
    if (y0 < 0) { y0 = 0; } else if (y0 > sh - 1) { y0 = sh - 1; }
    let y1 = y0 + 1;
    if (y1 > sh - 1) { y1 = sh - 1; }
    for (let x = 0; x < dw; x++) {
      let srcX = (x + 0.5) * scaleX - 0.5;
      let x0 = Math.floor(srcX);
      let wx = srcX - x0;
      if (x0 < 0) { x0 = 0; } else if (x0 > sw - 1) { x0 = sw - 1; }
      let x1 = x0 + 1;
      if (x1 > sw - 1) { x1 = sw - 1; }
      const i00 = (y0 * sw + x0) * 4;
      const i01 = (y0 * sw + x1) * 4;
      const i10 = (y1 * sw + x0) * 4;
      const i11 = (y1 * sw + x1) * 4;
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 4; c++) {
        const top = src[i00 + c] * (1 - wx) + src[i01 + c] * wx;
        const bot = src[i10 + c] * (1 - wx) + src[i11 + c] * wx;
        out[o + c] = top * (1 - wy) + bot * wy;
      }
    }
  }
  return out;
}

/** 裁剪一块 RGBA */
function cropRGBA(src, sw, x, y, w, h) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let row = 0; row < h; row++) {
    const s = ((y + row) * sw + x) * 4;
    out.set(src.subarray(s, s + w * 4), row * w * 4);
  }
  return out;
}

/** LetterBox：缩放到 size×size 居中，四周补灰 114 */
function letterbox(src, sw, sh, size) {
  const r = Math.min(size / sh, size / sw);
  const newW = Math.round(sw * r);
  const newH = Math.round(sh * r);
  const resized = resizeBilinear(src, sw, sh, newW, newH);
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    out[i * 4] = 114; out[i * 4 + 1] = 114; out[i * 4 + 2] = 114; out[i * 4 + 3] = 255;
  }
  // 与 ultralytics 一致：left = round(dw/2 - 0.1)，top = round(dh/2 - 0.1)
  const dw = (size - newW) / 2;
  const dh = (size - newH) / 2;
  const left = Math.round(dw - 0.1);
  const top = Math.round(dh - 0.1);
  for (let row = 0; row < newH; row++) {
    const dst = ((top + row) * size + left) * 4;
    out.set(resized.subarray(row * newW * 4, (row + 1) * newW * 4), dst);
  }
  return { data: out, r, padLeft: left, padTop: top };
}

/** RGBA → CHW float32，RGB，/255 */
function toTensorCHW(rgba, w, h) {
  const out = new Float32Array(3 * w * h);
  const plane = w * h;
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    out[i] = rgba[p] / 255;               // R
    out[plane + i] = rgba[p + 1] / 255;   // G
    out[2 * plane + i] = rgba[p + 2] / 255; // B
  }
  return out;
}

/** 分类器输入：短边缩到 256 → 中心裁 224 → ImageNet 归一化 */
function classifierInput(rgba, w, h) {
  const size = CFG.clsSize;
  const short = CFG.clsShort;
  let newW, newH;
  if (w <= h) {
    newW = short; newH = Math.trunc(short * h / w);
  } else {
    newH = short; newW = Math.trunc(short * w / h);
  }
  const resized = resizeBilinear(rgba, w, h, newW, newH);
  const top = Math.floor((newH - size) / 2);
  const left = Math.floor((newW - size) / 2);
  const cropped = cropRGBA(resized, newW, left, top, size, size);

  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    out[i] = (cropped[p] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    out[plane + i] = (cropped[p + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    out[2 * plane + i] = (cropped[p + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return out;
}

/* ============================================================ 解码 + NMS */

function iouBox(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

/** 贪心 NMS（按分数降序），返回保留项在入参数组中的下标 */
function nms(boxes, scores, iouThr, maxDet) {
  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep = [];
  while (order.length > 0 && keep.length < maxDet) {
    const i = order.shift();
    keep.push(i);
    const rest = [];
    for (const j of order) {
      if (iouBox(boxes[i], boxes[j]) <= iouThr) rest.push(j);
    }
    order.length = 0; order.push(...rest);
  }
  return keep;
}

/**
 * out: [1, 4+nc, N] 的 Tensor → 原图坐标的检测框。
 * Tensor.data 是扁平数组，按 dims 重塑：data[c * N + i]
 */
function decodeDetections(tensor, confThr, iouThr, r, padLeft, padTop, maxDet) {
  const dims = tensor.dims;              // [1, C, N]
  const C = dims[dims.length - 2];
  const N = dims[dims.length - 1];
  const data = tensor.data;
  const at = (c, i) => data[c * N + i];

  const cands = [];
  for (let i = 0; i < N; i++) {
    const score = at(4, i);
    if (score <= confThr) continue;
    const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
    cands.push({
      box: [(cx - w / 2 - padLeft) / r, (cy - h / 2 - padTop) / r,
            (cx + w / 2 - padLeft) / r, (cy + h / 2 - padTop) / r],
      score: score,
    });
  }
  if (!cands.length) return [];
  const boxes = cands.map((c) => c.box);
  const scores = cands.map((c) => c.score);
  const kept = nms(boxes, scores, iouThr, maxDet);
  return kept.map((i) => ({ box: boxes[i], score: scores[i] }));
}

/* ============================================================ ROI 裁剪 */

function expandBox(box, padRatio) {
  const [x1, y1, x2, y2] = box;
  const w = x2 - x1, h = y2 - y1;
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  const hw = w / 2 + padRatio * w;
  const hh = h / 2 + padRatio * h;
  return [cx - hw, cy - hh, cx + hw, cy + hh];
}

function squareBox(box) {
  const [x1, y1, x2, y2] = box;
  const side = Math.max(x2 - x1, y2 - y1);
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2];
}

/** 与 strawberry_cv.data.roi.crop_roi 同规则；返回 {data,w,h} 或 null */
function cropRoi(rgba, W, H, box, neighbors) {
  let chosen = box.slice();
  for (let i = 0; i <= CFG.padSteps; i++) {
    const pad = CFG.padRatio * (1 - i / Math.max(1, CFG.padSteps));
    let cand = expandBox(box, pad);
    cand = squareBox(cand);
    cand = [Math.max(0, cand[0]), Math.max(0, cand[1]),
            Math.min(W, cand[2]), Math.min(H, cand[3])];
    let overlap = 0;
    for (const nb of neighbors) overlap = Math.max(overlap, iouBox(cand, nb));
    chosen = cand;
    if (overlap <= CFG.maxNeighborIou) break;
  }
  const x1 = Math.max(0, Math.round(chosen[0]));
  const y1 = Math.max(0, Math.round(chosen[1]));
  const x2 = Math.min(W, Math.round(chosen[2]));
  const y2 = Math.min(H, Math.round(chosen[3]));
  if (x2 <= x1 || y2 <= y1) return null;
  if (y2 - y1 < CFG.minSize || x2 - x1 < CFG.minSize) return null;
  return { data: cropRGBA(rgba, W, x1, y1, x2 - x1, y2 - y1), w: x2 - x1, h: y2 - y1 };
}

/* ============================================================ 主流程 */

/** ORT Web 不同版本 run() 的返回值形态不同：数组（旧）或按输出名索引的对象（新）。 */
function firstOutput(res) {
  if (!res) return null;
  if (Array.isArray(res)) return res[0];
  const keys = Object.keys(res);
  return keys.length ? res[keys[0]] : null;
}

/** ImageBitmap → RGBA Uint8ClampedArray */
function bitmapToRGBA(bitmap) {
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return { data: ctx.getImageData(0, 0, bitmap.width, bitmap.height).data,
           w: bitmap.width, h: bitmap.height };
}

/**
 * 完整分析。
 * @param {ImageBitmap} bitmap
 * @param {object} sessions {det, cls}
 * @returns {Promise<object>} report
 */
async function analyze(bitmap, sessions, opts) {
  const o = Object.assign({}, CFG, opts || {});
  const t0 = performance.now();
  const { data: rgba, w, h } = bitmapToRGBA(bitmap);

  // --- 检测 ---
  const lb = letterbox(rgba, w, h, o.imgsz);
  const detInput = toTensorCHW(lb.data, o.imgsz, o.imgsz);
  const detOut = await sessions.det.run({
    images: new ort.Tensor('float32', detInput, [1, 3, o.imgsz, o.imgsz]),
  });
  const dets = decodeDetections(firstOutput(detOut), o.conf, o.iou, lb.r, lb.padLeft, lb.padTop, o.maxDet);

  // --- 分类 ---
  const boxes = dets.map((d) => d.box);
  const fruits = [];
  for (let i = 0; i < dets.length; i++) {
    const neighbors = boxes.filter((_, j) => j !== i);
    const roi = cropRoi(rgba, w, h, dets[i].box, neighbors);
    if (!roi) continue;
    const clsInput = classifierInput(roi.data, roi.w, roi.h);
    const clsOut = await sessions.cls.run({
      input: new ort.Tensor('float32', clsInput, [1, 3, o.clsSize, o.clsSize]),
    });
    const logits = firstOutput(clsOut).data;
    let best = 0;
    for (let k = 1; k < logits.length; k++) if (logits[k] > logits[best]) best = k;
    // softmax
    let mx = -Infinity;
    for (const v of logits) if (v > mx) mx = v;
    let sum = 0;
    const probs = [];
    for (const v of logits) { const e = Math.exp(v - mx); probs.push(e); sum += e; }
    for (let k = 0; k < probs.length; k++) probs[k] /= sum;

    const probsByClass = {};
    FOLDER_ORDER.forEach((name, idx) => { probsByClass[name] = probs[idx]; });
    fruits.push({
      box: dets[i].box,
      detConf: dets[i].score,
      state: FOLDER_ORDER[best],
      stateCn: { ripe: '成熟', half: '半熟', unripe: '未熟' }[FOLDER_ORDER[best]],
      stateConf: probs[best],
      probs: probsByClass,
    });
  }

  const counts = { ripe: 0, half: 0, unripe: 0 };
  let maturity = 0;
  for (const f of fruits) {
    counts[f.state] = (counts[f.state] || 0) + 1;
    maturity += f.state === 'ripe' ? 1 : (f.state === 'half' ? 0.5 : 0);
  }
  if (fruits.length) maturity /= fruits.length;

  return {
    fruits: fruits,
    counts: counts,
    total: fruits.length,
    maturity: maturity,
    latencyMs: performance.now() - t0,
    width: w,
    height: h,
  };
}

/** 在 canvas 上画标注结果 */
function renderResult(canvas, bitmap, report, opts) {
  const o = Object.assign({}, CFG, opts || {});
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);

  const colors = { ripe: '#ef4444', half: '#f59e0b', unripe: '#22c55e' };
  const scale = Math.max(0.6, Math.min(2.0, Math.max(bitmap.width, bitmap.height) / 800));
  const fontPx = Math.round(13 * scale);

  ctx.font = `${fontPx}px "Microsoft YaHei", "PingFang SC", sans-serif`;
  ctx.textBaseline = 'top';
  ctx.lineWidth = Math.max(1.5, 2 * scale);

  for (const f of report.fruits) {
    const [x1, y1, x2, y2] = f.box;
    ctx.strokeStyle = colors[f.state] || '#94a3b8';
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${f.stateCn} ${(f.stateConf * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width;
    const ty = Math.max(y1, 0);
    ctx.fillStyle = colors[f.state] || '#94a3b8';
    ctx.fillRect(x1, ty - fontPx - 4, tw + 8, fontPx + 6);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x1 + 4, ty - fontPx - 2);
  }

  drawPanel(ctx, canvas, report, scale);
  return canvas;
}

function drawPanel(ctx, canvas, report, scale) {
  const W = canvas.width, H = canvas.height;
  const pw = Math.round(220 * scale);
  const ph = Math.round(168 * scale);
  const x0 = W - pw - Math.round(12 * scale);
  const y0 = Math.round(12 * scale);
  if (x0 < 0) return;

  ctx.save();
  ctx.fillStyle = 'rgba(20,22,26,0.78)';
  ctx.strokeStyle = 'rgba(150,150,150,0.6)';
  ctx.lineWidth = 1;
  roundRect(ctx, x0, y0, pw, ph, 8 * scale);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  const pad = Math.round(10 * scale);
  const fs = Math.round(13 * scale);
  ctx.font = `${fs}px "Microsoft YaHei", "PingFang SC", sans-serif`;
  ctx.textBaseline = 'top';

  let y = y0 + pad;
  ctx.fillStyle = '#e6edf3';
  ctx.fillText('草莓成熟度检测 Demo', x0 + pad, y);
  y += fs + 8 * scale;

  const colors = { ripe: '#ef4444', half: '#f59e0b', unripe: '#22c55e' };
  const total = Math.max(1, report.total);
  const barX = x0 + pad + Math.round(54 * scale);
  const barW = Math.round(96 * scale);
  const barH = Math.round(9 * scale);

  for (const name of LOGICAL_ORDER) {
    const count = report.counts[name] || 0;
    const ratio = count / total;
    ctx.fillStyle = colors[name];
    ctx.fillText({ ripe: '成熟', half: '半熟', unripe: '未熟' }[name], x0 + pad, y);
    ctx.fillStyle = 'rgba(90,90,90,0.9)';
    ctx.fillRect(barX, y, barW, barH);
    ctx.fillStyle = colors[name];
    ctx.fillRect(barX, y, barW * ratio, barH);
    ctx.fillStyle = '#c9d1d9';
    ctx.fillText(`${count}  ${(ratio * 100).toFixed(0)}%`, barX + barW + 6 * scale, y);
    y += barH + 7 * scale;
  }

  y += 4 * scale;
  ctx.fillStyle = '#e6edf3';
  ctx.fillText(`总计 ${report.total} 颗`, x0 + pad, y);
  y += fs + 6 * scale;
  ctx.fillStyle = report.maturity >= 0.6 ? '#4ade80' : (report.maturity >= 0.3 ? '#fbbf24' : '#60a5fa');
  ctx.font = `bold ${Math.round(fs * 1.05)}px "Microsoft YaHei", sans-serif`;
  ctx.fillText(`成熟度 ${(report.maturity * 100).toFixed(1)}%`, x0 + pad, y);
  y += fs + 4 * scale;
  ctx.font = `${Math.round(fs * 0.85)}px "Microsoft YaHei", sans-serif`;
  ctx.fillStyle = '#9aa4b0';
  ctx.fillText(`${report.latencyMs.toFixed(0)} ms`, x0 + pad, y);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
