/* 浏览器端 Demo 的界面逻辑
 *
 * 模型加载是这个页面最脆弱的环节（大文件 + 不稳定网络），所以：
 *   1) 自己 fetch 拿字节，全程汇报进度（ORT 自带的加载没有进度）
 *   2) 超时 + 指数退避重试
 *   3) 模型存进 Cache Storage，第二次访问直接命中，瞬间加载
 *   4) 任何一步失败都给出明确的错误信息和「重试」按钮，而不是干等
 */

'use strict';

const $ = (s) => document.querySelector(s);

const SAMPLES = [
  { name: '网络特写 · image_001', url: '../samples/v1/detection/images/image_001.jpg' },
  { name: '网络特写 · image_005', url: '../samples/v1/detection/images/image_005.jpg' },
  { name: '网络特写 · image_007', url: '../samples/v1/detection/images/image_007.jpg' },
  { name: '温室 · archive_001', url: '../samples/archive/detection/images/archive_image_001.jpg' },
  { name: '温室 · archive_005', url: '../samples/archive/detection/images/archive_image_005.jpg' },
  { name: '温室 · archive_007', url: '../samples/archive/detection/images/archive_image_007.jpg' },
];

const sessions = { det: null, cls: null };
let currentBitmap = null;
let currentCanvas = null;
let loading = false;

/* ---------------------------------------------------------------- 状态 */

function setStatus(text, kind) {
  const el = $('#status');
  el.textContent = text;
  el.className = 'status' + (kind ? ' ' + kind : '');
}

function setProgress(pct, label) {
  const el = $('#progress');
  if (pct === null) { el.classList.add('hidden'); return; }
  el.classList.remove('hidden');
  $('#bar').style.width = Math.max(0, Math.min(100, pct)) + '%';
  if (label) $('#progressLabel').textContent = label;
}

function showRetry(label) {
  const btn = $('#retry');
  if (!btn) return;
  if (label) btn.textContent = label;
  btn.classList.remove('hidden');
}

function hideRetry() {
  const btn = $('#retry');
  if (btn) btn.classList.add('hidden');
}

/* ---------------------------------------------------------------- 下载 */

/** 流式 fetch：汇报进度，带超时 */
async function fetchWithProgress(url, onProgress, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error('下载超时')), timeoutMs || 180000);
  try {
    const resp = await fetch(url, { signal: ctrl.signal, cache: 'force-cache' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const total = parseInt(resp.headers.get('content-length') || '0', 10);

    if (!resp.body || !resp.body.getReader) {
      // 老浏览器没有流式接口，退化成一次性读
      const buf = await resp.arrayBuffer();
      onProgress(buf.byteLength, total || buf.byteLength);
      return buf;
    }
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received, total);
    }
    // 连接中途断开时 received 会小于 content-length，要显式识别出来
    if (total && received < total) {
      throw new Error(`下载不完整（${(received / 1048576).toFixed(1)}/${(total / 1048576).toFixed(1)} MB，网络中断）`);
    }
    const out = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out.buffer;
  } finally {
    clearTimeout(timer);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** 把底层错误翻译成人能看懂的话 */
function friendlyError(err) {
  const msg = (err && err.message) || String(err);
  if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
    return '网络请求失败（连不上或被中断）';
  }
  if (msg.includes('下载超时')) return '下载超时';
  if (msg.includes('下载不完整')) return msg;
  if (msg.includes('abort')) return '请求被取消';
  return msg;
}

/** 带重试的下载 */
async function downloadWithRetry(url, onProgress, label, attempts) {
  let lastErr = null;
  for (let i = 1; i <= (attempts || 3); i++) {
    try {
      return await fetchWithProgress(url, onProgress, 180000);
    } catch (err) {
      lastErr = err;
      if (i < (attempts || 3)) {
        setStatus(`${label}下载失败（${friendlyError(err)}），${i} 秒后重试…`);
        await sleep(1200 * i);
      }
    }
  }
  throw lastErr;
}

/** 模型字节：优先 Cache Storage，其次网络 */
async function loadModelBytes(url, onProgress, label) {
  const cache = await caches.open('strawberry-models-v1');
  const hit = await cache.match(url);
  if (hit) {
    onProgress(1, 1);
    return await hit.arrayBuffer();
  }
  const buf = await downloadWithRetry(url, onProgress, label);
  try {
    await cache.put(url, new Response(buf, {
      headers: { 'Content-Type': 'application/octet-stream' },
    }));
  } catch (e) {
    console.warn('写入 Cache Storage 失败（不影响本次使用）', e);
  }
  return buf;
}

/* ---------------------------------------------------------------- 模型加载 */

async function loadModels(force) {
  if (loading) return;
  loading = true;
  $('#run').disabled = true;
  hideRetry();

  const RUNTIME = 14.2, DET = 10.1, CLS = 10.8;
  const TOTAL = RUNTIME + DET + CLS;
  let done = 0;

  try {
    if (!window.ort) throw new Error('ONNX Runtime 脚本未加载');

    // 1) 预取运行时：让 ORT 之后能命中浏览器 HTTP 缓存
    setStatus('正在准备运行时…');
    await downloadWithRetry(
      new URL('../ort/ort-wasm-simd-threaded.wasm', location.href).href,
      (got, total) => {
        const mb = (got / 1048576).toFixed(1);
        setProgress((done + (total ? got / total : 1) * RUNTIME) / TOTAL * 100,
                    `运行时 ${mb} / ${RUNTIME} MB`);
      },
      '运行时', 2);

    ort.env.wasm.wasmPaths = new URL('../ort/', location.href).href;
    ort.env.wasm.numThreads = 1;   // GitHub Pages 无 COOP/COEP，无法用 SharedArrayBuffer
    ort.env.wasm.initTimeout = 180000;
    ort.env.logLevel = 'error';

    // 2) 两个模型并行下载（串行会白等一倍时间）
    setStatus('正在下载模型…');
    const detUrl = new URL('../models/detector_merged.onnx', location.href).href;
    const clsUrl = new URL('../models/classifier_merged_int8.onnx', location.href).href;
    let detGot = 0, clsGot = 0;

    const detP = (got, total) => {
      detGot = total ? got / total : 1;
      setProgress((RUNTIME + detGot * DET + clsGot * CLS) / TOTAL * 100,
                  `检测器 ${(got / 1048576).toFixed(1)} / ${DET} MB` +
                  `　分类器 ${(clsGot * CLS).toFixed(1)} / ${CLS} MB`);
    };
    const clsP = (got, total) => {
      clsGot = total ? got / total : 1;
      setProgress((RUNTIME + detGot * DET + clsGot * CLS) / TOTAL * 100,
                  `检测器 ${(detGot * DET).toFixed(1)} / ${DET} MB` +
                  `　分类器 ${(got / 1048576).toFixed(1)} / ${CLS} MB`);
    };

    const [detBuf, clsBuf] = await Promise.all([
      loadModelBytes(detUrl, detP, '检测器'),
      loadModelBytes(clsUrl, clsP, '分类器'),
    ]);
    done = RUNTIME + DET + CLS;
    setProgress((done) / TOTAL * 100, '模型下载完成');

    // 3) 建会话（graph 优化比较耗时，给个明确提示）
    setStatus('正在初始化模型（首次较慢）…');
    sessions.det = await ort.InferenceSession.create(detBuf, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    setStatus('正在初始化分类器…');
    sessions.cls = await ort.InferenceSession.create(clsBuf, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });

    setProgress(100, '模型就绪');
    setStatus('模型就绪，选一张图片开始', 'ok');
    $('#run').disabled = false;
  } catch (err) {
    console.error(err);
    setStatus('模型加载失败：' + friendlyError(err) +
              '（检查网络后点「重试加载模型」；已下载成功的模型会被缓存，不用重下）', 'err');
    showRetry('重试加载模型');
  } finally {
    loading = false;
  }
}

/* ---------------------------------------------------------------- 输入 */

const drop = $('#drop');
const fileInput = $('#file');

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('drag');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

// 独立的「选择图片」按钮（避免 label 转发导致文件框弹两次）
$('#pick-file').addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

// 手机上直接调用后置摄像头拍草莓
$('#take-photo').addEventListener('click', (e) => {
  e.stopPropagation();
  const cam = document.createElement('input');
  cam.type = 'file';
  cam.accept = 'image/*';
  cam.capture = 'environment';   // 手机优先调用后置摄像头
  cam.addEventListener('change', () => {
    if (cam.files && cam.files.length) handleFile(cam.files[0]);
  });
  cam.click();
});

async function handleFile(file) {
  if (!/^image\//.test(file.type) && !/\.(jpe?g|png|bmp|webp)$/i.test(file.name)) {
    setStatus('不是图片文件：' + file.name, 'err');
    return;
  }
  try {
    currentBitmap = await createImageBitmap(file);
    $('#run').disabled = false;
    setStatus(`已选择 ${file.name}（${currentBitmap.width}×${currentBitmap.height}），分析中…`, 'ok');
    await run();
  } catch (err) {
    setStatus('图片读取失败：' + err.message, 'err');
  }
}

async function loadSample(url, name) {
  setStatus(`正在下载样例：${name}…`);
  try {
    const resp = await fetch(url, { cache: 'force-cache' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const blob = await resp.blob();
    currentBitmap = await createImageBitmap(blob);
    setStatus(`已载入样例：${name}`, 'ok');
    await run();
  } catch (err) {
    setStatus('样例加载失败：' + err.message + '（请检查网络后重试）', 'err');
  }
}

function renderSamples() {
  $('#samples').innerHTML = SAMPLES.map((s, i) => `
    <button class="sample" data-i="${i}">${s.name}</button>`).join('');
  Array.from(document.querySelectorAll('.sample')).forEach((btn) => {
    btn.addEventListener('click', () => {
      const s = SAMPLES[Number(btn.dataset.i)];
      loadSample(s.url, s.name);
    });
  });
}

/* ---------------------------------------------------------------- 分析 */

$('#conf').addEventListener('input', (e) => {
  $('#confVal').textContent = Number(e.target.value).toFixed(2);
});
$('#iou').addEventListener('input', (e) => {
  $('#iouVal').textContent = Number(e.target.value).toFixed(2);
});
$('#run').addEventListener('click', run);

async function run() {
  if (!currentBitmap || !sessions.det || !sessions.cls) {
    setStatus('模型还没加载好，请稍候或点「重试加载模型」', 'err');
    return;
  }
  const btn = $('#run');
  btn.disabled = true;
  setStatus('推理中…');
  try {
    const report = await analyze(currentBitmap, sessions, {
      conf: Number($('#conf').value),
      iou: Number($('#iou').value),
    });
    showResult(report);
    setStatus(`完成：${report.total} 颗，用时 ${report.latencyMs.toFixed(0)} ms`, 'ok');
  } catch (err) {
    console.error(err);
    setStatus('推理失败：' + err.message, 'err');
  } finally {
    btn.disabled = false;
  }
}

function showResult(report) {
  $('#empty').classList.add('hidden');
  $('#result').classList.remove('hidden');

  const canvas = $('#canvas');
  renderResult(canvas, currentBitmap, report, {
    conf: Number($('#conf').value), iou: Number($('#iou').value),
  });
  currentCanvas = canvas;

  const total = Math.max(1, report.total);
  const cards = [
    { cls: 'total', label: '检测到', value: report.total, sub: '颗草莓' },
    { cls: 'mat', label: '成熟度指数', value: (report.maturity * 100).toFixed(0) + '%', sub: '成熟1/半熟.5/未熟0' },
  ];
  ['ripe', 'half', 'unripe'].forEach((name) => {
    const n = report.counts[name] || 0;
    cards.push({
      cls: name, label: { ripe: '成熟', half: '半熟', unripe: '未熟' }[name], value: n,
      sub: ((n / total) * 100).toFixed(0) + '% 占比',
    });
  });
  cards.push({ cls: 'total', label: '耗时', value: report.latencyMs.toFixed(0), sub: 'ms/图' });
  $('#stats').innerHTML = cards.map((c) => `
    <div class="stat ${c.cls}">
      <div class="label">${c.label}</div><div class="value">${c.value}</div><div class="sub">${c.sub}</div>
    </div>`).join('');

  $('#rows').innerHTML = report.fruits.length ? report.fruits.map((f, i) => `
    <tr>
      <td class="num">${i + 1}</td>
      <td><span class="chip ${f.state}">${f.stateCn}</span></td>
      <td class="num">${(f.stateConf * 100).toFixed(0)}%</td>
      <td class="num">${((f.probs.ripe || 0) * 100).toFixed(0)}%</td>
      <td class="num">${((f.probs.half || 0) * 100).toFixed(0)}%</td>
      <td class="num">${((f.probs.unripe || 0) * 100).toFixed(0)}%</td>
    </tr>`).join('')
    : '<tr><td colspan="6" style="color:var(--muted)">没有检测到草莓</td></tr>';
}

$('#download').addEventListener('click', (e) => {
  if (!currentCanvas) { e.preventDefault(); return; }
  e.target.href = currentCanvas.toDataURL('image/jpeg', 0.92);
});

/* ---------------------------------------------------------------- 自检 */

/** 逐项检查资源可达性，方便用户/我快速定位是网络问题还是代码问题 */
async function selfCheck() {
  const out = $('#selfcheck-out');
  out.innerHTML = '<div class="sc-item">检查中…</div>';
  const urls = [
    ['ort.min.js', '../ort/ort.wasm.min.js'],
    ['ort wasm', '../ort/ort-wasm-simd-threaded.wasm'],
    ['ort loader', '../ort/ort-wasm-simd-threaded.mjs'],
    ['检测器', '../models/detector_merged.onnx'],
    ['分类器', '../models/classifier_merged_int8.onnx'],
    ['样例图', '../samples/v1/detection/images/image_001.jpg'],
  ];
  const lines = [];
  for (const [name, url] of urls) {
    const t0 = performance.now();
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 20000);
      const resp = await fetch(new URL(url, location.href).href, { signal: ctrl.signal });
      clearTimeout(timer);
      const len = parseInt(resp.headers.get('content-length') || '0', 10);
      // 只读前几 KB，确认能拿到数据即可
      const reader = resp.body && resp.body.getReader();
      if (reader) { await reader.read(); await reader.cancel(); }
      lines.push(`<div class="sc-item ok">✓ ${name} — HTTP ${resp.status} · ${(len / 1048576).toFixed(1)} MB · ${(performance.now() - t0).toFixed(0)} ms</div>`);
    } catch (err) {
      lines.push(`<div class="sc-item bad">✗ ${name} — ${err.message}</div>`);
    }
  }
  lines.push(`<div class="sc-item">Cache Storage：${typeof caches !== 'undefined' ? '可用' : '不可用'}</div>`);
  lines.push(`<div class="sc-item">模型会话：det=${sessions.det ? '已加载' : '未加载'} / cls=${sessions.cls ? '已加载' : '未加载'}</div>`);
  out.innerHTML = lines.join('');
}

$('#selfcheck').addEventListener('click', selfCheck);
$('#retry').addEventListener('click', () => loadModels(true));

/* ---------------------------------------------------------------- 启动 */

renderSamples();
loadModels();
