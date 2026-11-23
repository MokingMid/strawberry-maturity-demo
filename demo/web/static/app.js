/* 草莓成熟度检测 Demo —— 前端逻辑 */
'use strict';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const CN = { ripe: '成熟', half: '半熟', unripe: '未熟', unknown: '未知' };
const CSS_VAR = { ripe: '--ripe', half: '--half', unripe: '--unripe', unknown: '--muted' };

const state = {
  src: 'upload',
  dataset: 'v1',
  samples: [],
  sampleIndex: 0,
  page: 1,
  pageSize: 24,
  selectedFile: null,
};

/* ---------------- 通用 ---------------- */

function showLoading(text) {
  $('#loading-text').textContent = text || '处理中…';
  $('#loading').classList.remove('hidden');
}
function hideLoading() { $('#loading').classList.add('hidden'); }

function setStatus(sel, text, kind) {
  const el = $(sel);
  el.textContent = text || '';
  el.className = 'status' + (kind ? ' ' + kind : '');
}

function params() {
  return {
    detector: $('#sel-detector').value,
    classifier: $('#sel-classifier').value,
    conf: $('#conf').value,
    iou: $('#iou').value,
  };
}
function query(extra) {
  const p = new URLSearchParams(params());
  Object.entries(extra || {}).forEach(([k, v]) => p.set(k, v));
  return p.toString();
}

async function api(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({ error: '响应不是合法 JSON' }));
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}

/* ---------------- 统计卡片 ---------------- */

function renderStats(container, report) {
  const total = report.total || 0;
  const mat = Math.round((report.maturity || 0) * 100);
  const cards = [
    { cls: 'total', label: '检测到', value: total, sub: '颗草莓' },
    { cls: 'mat', label: '成熟度指数', value: mat + '%', sub: '成熟1/半熟.5/未熟0' },
  ];
  ['ripe', 'half', 'unripe'].forEach((name) => {
    const n = report.counts[name] || 0;
    cards.push({
      cls: name, label: CN[name], value: n,
      sub: total ? Math.round((n / total) * 100) + '% 占比' : '—',
    });
  });
  cards.push({ cls: 'total', label: '耗时', value: Math.round(report.latency_ms || 0), sub: 'ms/图' });

  container.innerHTML = cards.map((c) => `
    <div class="stat ${c.cls}">
      <div class="label">${c.label}</div>
      <div class="value">${c.value}</div>
      <div class="sub">${c.sub}</div>
    </div>`).join('');
}

function renderTable(tableId, fruits) {
  const table = $(tableId);
  const tbody = table.querySelector('tbody');
  if (!fruits || !fruits.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted)">没有检测到草莓</td></tr>';
    return;
  }
  tbody.innerHTML = fruits.map((f, i) => `
    <tr>
      <td class="num">${i + 1}</td>
      <td><span class="chip ${f.state}">${f.state_cn || CN[f.state] || f.state}</span></td>
      <td class="num">${(f.state_conf * 100).toFixed(0)}%</td>
      <td class="num">${((f.probs.ripe || 0) * 100).toFixed(0)}%</td>
      <td class="num">${((f.probs.half || 0) * 100).toFixed(0)}%</td>
      <td class="num">${((f.probs.unripe || 0) * 100).toFixed(0)}%</td>
    </tr>`).join('');
}

/* ---------------- Tab 切换 ---------------- */

$$('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.tab').forEach((b) => b.classList.remove('active'));
    $$('.tab-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    $('#tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'metrics') loadMetrics();
    if (btn.dataset.tab === 'samples') loadSamples();
  });
});

/* ---------------- 参数滑块 ---------------- */

$('#conf').addEventListener('input', (e) => { $('#conf-val').textContent = Number(e.target.value).toFixed(2); });
$('#iou').addEventListener('input', (e) => { $('#iou-val').textContent = Number(e.target.value).toFixed(2); });

/* ---------------- 输入源切换 ---------------- */

$$('.seg-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.seg-btn').forEach((b) => b.classList.remove('active'));
    $$('.src-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    state.src = btn.dataset.src;
    $('#src-' + state.src).classList.add('active');
  });
});

/* ---------------- 文件上传 ---------------- */

const dropzone = $('#dropzone');
const fileInput = $('#file-input');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});
$('#clear-upload').addEventListener('click', () => {
  state.selectedFile = null;
  $('#upload-preview').classList.add('hidden');
  $('#dropzone').classList.remove('hidden');
  fileInput.value = '';
});

function handleFile(file) {
  state.selectedFile = file;
  $('#dropzone').classList.add('hidden');
  $('#upload-preview').classList.remove('hidden');
  const url = URL.createObjectURL(file);
  $('#upload-thumb').src = url;
  setStatus('#play-status', '已选择 ' + file.name + '，点击「开始分析」', 'ok');
}

/* ---------------- 数据集下拉 ---------------- */

async function initDatasets() {
  const opts = [];
  $$('#pick-dataset, #browse-dataset, #batch-dataset').forEach((sel) => { sel.innerHTML = ''; });
  const datasets = await api('/api/samples?dataset=v1&size=1').then(() => null).catch(() => null);
  // 从页面注入的 DATASETS 读取（index.html 里没直接给，改用已知键）
  const known = { v1: '网络单果特写（64 张）', archive: '温室大棚场景（566 张）' };
  Object.entries(known).forEach(([k, label]) => {
    opts.push(`<option value="${k}">${label}</option>`);
  });
  $$('#pick-dataset, #browse-dataset, #batch-dataset').forEach((sel) => {
    sel.innerHTML = opts.join('');
  });
  $('#pick-dataset').addEventListener('change', loadSampleOptions);
  $('#browse-dataset').addEventListener('change', () => { state.page = 1; loadSamples(); });
  $('#batch-dataset').addEventListener('change', () => {});
  await loadSampleOptions();
}

async function loadSampleOptions() {
  state.dataset = $('#pick-dataset').value;
  setStatus('#play-status', '加载样例列表…');
  const data = await api('/api/samples?dataset=' + state.dataset + '&size=120');
  state.samples = data.items;
  $('#pick-sample').innerHTML = data.items.map((it, i) =>
    `<option value="${i}">${it.name}</option>`).join('');
  state.sampleIndex = 0;
  setStatus('#play-status', `共 ${data.total} 张样例`, 'ok');
}

$('#pick-sample').addEventListener('change', (e) => { state.sampleIndex = Number(e.target.value); });
$('#pick-random').addEventListener('click', () => {
  if (!state.samples.length) return;
  state.sampleIndex = Math.floor(Math.random() * state.samples.length);
  $('#pick-sample').value = state.sampleIndex;
  setStatus('#play-status', '已随机选择：' + state.samples[state.sampleIndex].name, 'ok');
});

/* ---------------- 单张分析 ---------------- */

$('#btn-analyze').addEventListener('click', async () => {
  setStatus('#play-status', '');
  showLoading('正在推理（首次会加载模型，请稍候）…');
  try {
    let data;
    if (state.src === 'upload') {
      if (!state.selectedFile) {
        hideLoading();
        setStatus('#play-status', '请先选择一张图片', 'err');
        return;
      }
      const fd = new FormData();
      fd.append('file', state.selectedFile);
      data = await api('/api/upload?' + query(), { method: 'POST', body: fd });
    } else {
      const item = state.samples[state.sampleIndex];
      if (!item) { hideLoading(); setStatus('#play-status', '没有可用样例', 'err'); return; }
      data = await api(item.url + '?' + query());
    }
    showResult(data);
    setStatus('#play-status', '完成', 'ok');
  } catch (err) {
    setStatus('#play-status', '出错了：' + err.message, 'err');
  } finally {
    hideLoading();
  }
});

function showResult(data) {
  $('#result-empty').classList.add('hidden');
  $('#result-body').classList.remove('hidden');
  $('#result-image').src = data.image_url;
  $('#btn-download').href = data.image_url;
  renderStats($('#result-stats'), data);
  renderTable('#result-table', data.fruits);
}

/* ---------------- 数据集浏览 ---------------- */

async function loadSamples() {
  const ds = $('#browse-dataset').value;
  showLoading('加载样例…');
  try {
    const data = await api(`/api/samples?dataset=${ds}&page=${state.page}&size=${state.pageSize}`);
    $('#thumbs').innerHTML = data.items.map((it) => `
      <div class="thumb" data-url="${it.url}" data-thumb="${it.thumb}">
        <img loading="lazy" src="${it.thumb}" alt="${it.name}">
        <div class="cap">${it.name.replace(/^(archive_)?image_/, '#')}</div>
      </div>`).join('');
    $('#browse-info').textContent =
      `第 ${data.page} / ${Math.ceil(data.total / data.size)} 页 · 共 ${data.total} 张`;
    $$('#thumbs .thumb').forEach((el) => {
      el.addEventListener('click', () => openBrowse(el));
    });
  } catch (err) {
    setStatus('#play-status', '加载失败：' + err.message, 'err');
  } finally {
    hideLoading();
  }
}

async function openBrowse(el) {
  $$('#thumbs .thumb').forEach((t) => t.classList.remove('sel'));
  el.classList.add('sel');
  showLoading('分析中…');
  try {
    const data = await api(el.dataset.url + '?' + query());
    $('#browse-detail').hidden = false;
    $('#browse-image').src = data.image_url;
    renderStats($('#browse-stats'), data);
    $('#browse-detail').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    setStatus('#play-status', '分析失败：' + err.message, 'err');
  } finally {
    hideLoading();
  }
}

$('#btn-browse').addEventListener('click', loadSamples);
$('#page-prev').addEventListener('click', () => { if (state.page > 1) { state.page--; loadSamples(); } });
$('#page-next').addEventListener('click', () => { state.page++; loadSamples(); });

/* ---------------- 批量评测 ---------------- */

$('#btn-batch').addEventListener('click', async () => {
  const ds = $('#batch-dataset').value;
  const count = $('#batch-count').value;
  const offset = $('#batch-offset').value;
  setStatus('#batch-status', '');
  showLoading(`批量分析 ${count} 张图片…`);
  try {
    const data = await api(`/api/batch?dataset=${ds}&count=${count}&offset=${offset}&` + params());
    renderStats($('#batch-stats'), {
      total: data.total_fruits,
      counts: data.counts,
      maturity: 0,
      latency_ms: 0,
    });
    $('#batch-stats').innerHTML = `
      <div class="stat total"><div class="label">处理图片</div><div class="value">${data.items.length}</div><div class="sub">张</div></div>
      <div class="stat total"><div class="label">检出草莓</div><div class="value">${data.total_fruits}</div><div class="sub">颗</div></div>
      <div class="stat ripe"><div class="label">成熟</div><div class="value">${data.counts.ripe}</div><div class="sub">颗</div></div>
      <div class="stat half"><div class="label">半熟</div><div class="value">${data.counts.half}</div><div class="sub">颗</div></div>
      <div class="stat unripe"><div class="label">未熟</div><div class="value">${data.counts.unripe}</div><div class="sub">颗</div></div>
      <div class="stat mat"><div class="label">与 GT 状态一致率</div><div class="value">${data.state_acc === null ? '—' : Math.round(data.state_acc * 100) + '%'}</div><div class="sub">${data.state_total} 颗可比</div></div>
      <div class="stat mat"><div class="label">检测 P / R</div><div class="value">${data.det_precision.toFixed(2)} / ${data.det_recall.toFixed(2)}</div><div class="sub">TP${data.det_tp} FP${data.det_fp} FN${data.det_fn}</div></div>`;
    $('#batch-thumbs').innerHTML = data.items.map((it) => `
      <div class="thumb"><img loading="lazy" src="${it.image_url}" alt="${it.name}">
        <div class="cap">${it.total} 颗 · ${Math.round(it.maturity * 100)}% · ${Math.round(it.latency_ms)}ms</div></div>`).join('');
    setStatus('#batch-status', '完成', 'ok');
  } catch (err) {
    setStatus('#batch-status', '出错了：' + err.message, 'err');
  } finally {
    hideLoading();
  }
});

/* ---------------- 模型指标 ---------------- */

async function loadMetrics() {
  try {
    const data = await api('/api/stats');
    const cn = data.class_names;

    if (data.classifier && data.classifier.length) {
      const rows = data.classifier.map((r) => `
        <tr><td>${r.model}</td><td>${r.eval_set}</td>
          <td class="num">${r.n}</td><td class="num">${(r.acc * 100).toFixed(1)}%</td>
          <td class="num">${(r.macro_f1 * 100).toFixed(1)}%</td>
          <td class="num">${(r.half_f1 * 100).toFixed(1)}%</td>
          <td class="num">${r.half_support}</td></tr>`).join('');
      $('#cls-table').innerHTML =
        `<thead><tr><th>模型</th><th>测试集</th><th>样本</th><th>准确率</th>
          <th>macro F1</th><th>half F1</th><th>half 支持数</th></tr></thead><tbody>${rows}</tbody>`;
    }

    if (data.end2end && data.end2end.length) {
      const rows = data.end2end.map((r) => `
        <tr><td>${r.model}</td><td>${data.datasets[r.dataset] || r.dataset}</td>
          <td class="num">${r.images}</td>
          <td class="num">${(r.state_acc * 100).toFixed(1)}%</td>
          <td class="num">${r.state_correct}/${r.state_total}</td>
          <td class="num">${r.det_precision.toFixed(3)}</td>
          <td class="num">${r.det_recall.toFixed(3)}</td>
          <td class="num">${Math.round(r.latency_ms)}ms</td></tr>`).join('');
      $('#e2e-table').innerHTML =
        `<thead><tr><th>模型</th><th>数据集</th><th>图片</th><th>状态正确率</th>
          <th>正确/总数</th><th>检测 P</th><th>检测 R</th><th>耗时</th></tr></thead><tbody>${rows}</tbody>`;
    }

    $('#gallery-img').src = '/api/gallery?t=' + Date.now();
  } catch (err) {
    setStatus('#play-status', '指标加载失败：' + err.message, 'err');
  }
}

/* ---------------- 启动 ---------------- */

initDatasets().catch((e) => setStatus('#play-status', '初始化失败：' + e.message, 'err'));
