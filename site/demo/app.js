/* 浏览器端 Demo 的界面逻辑 */

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

function setStatus(text, kind) {
  const el = $('#status');
  el.textContent = text;
  el.className = 'status' + (kind ? ' ' + kind : '');
}

function setProgress(pct) {
  const el = $('#progress');
  if (pct === null) { el.classList.add('hidden'); return; }
  el.classList.remove('hidden');
  $('#bar').style.width = pct + '%';
}

/* ---------------- 模型加载 ---------------- */

async function loadModels() {
  if (!window.ort) {
    setStatus('ONNX Runtime 加载失败（检查 ort/ 目录）', 'err');
    return;
  }
  ort.env.wasm.wasmPaths = new URL('../ort/', location.href).href;
  ort.env.wasm.numThreads = 1;   // GitHub Pages 无 COOP/COEP，无法用 SharedArrayBuffer
  ort.env.logLevel = 'error';

  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  try {
    setStatus('加载检测器…');
    sessions.det = await ort.InferenceSession.create(
      new URL('../models/detector_merged.onnx', location.href).href, opts);
    setProgress(55);
    setStatus('加载分类器…');
    sessions.cls = await ort.InferenceSession.create(
      new URL('../models/classifier_merged_int8.onnx', location.href).href, opts);
    setProgress(100);
    setStatus('模型就绪，选一张图片开始', 'ok');
    $('#run').disabled = false;
  } catch (err) {
    setStatus('模型加载失败：' + err.message, 'err');
  }
}

/* ---------------- 输入 ---------------- */

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

async function handleFile(file) {
  try {
    currentBitmap = await createImageBitmap(file);
    $('#run').disabled = false;
    setStatus(`已选择 ${file.name}（${currentBitmap.width}×${currentBitmap.height}），点「开始分析」`, 'ok');
    await run();
  } catch (err) {
    setStatus('图片读取失败：' + err.message, 'err');
  }
}

async function loadSample(url, name) {
  setStatus('下载样例…');
  try {
    const resp = await fetch(url);
    const blob = await resp.blob();
    currentBitmap = await createImageBitmap(blob);
    setStatus(`已载入样例：${name}`, 'ok');
    await run();
  } catch (err) {
    setStatus('样例加载失败：' + err.message, 'err');
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

/* ---------------- 分析 ---------------- */

$('#conf').addEventListener('input', (e) => {
  $('#confVal').textContent = Number(e.target.value).toFixed(2);
});
$('#iou').addEventListener('input', (e) => {
  $('#iouVal').textContent = Number(e.target.value).toFixed(2);
});
$('#run').addEventListener('click', run);

async function run() {
  if (!currentBitmap || !sessions.det) return;
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

/* ---------------- 启动 ---------------- */
renderSamples();
loadModels();
