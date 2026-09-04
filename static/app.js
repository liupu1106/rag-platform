// ============ RAG 实操平台 前端（v2：集合/2D/流式/上传/评测） ============
const $ = (s, r = document) => r.querySelector(s);
const api = (path, body) => fetch(path, {
  method: body ? 'POST' : 'GET',
  headers: { 'Content-Type': 'application/json' },
  body: body ? JSON.stringify(body) : undefined,
}).then(r => r.json());

let CID = null;
const form = {
  chunk_size: 80, overlap: 20,
  query: '拍照答疑功能怎么用？识别不准怎么办？', top_k: 3,
  hybrid: false, rerank: false, rewrite: false,
  emb_kind: 'tfidf', emb_base: 'https://api.openai.com/v1', emb_key: '', emb_model: 'text-embedding-3-small',
  llm_base: '', llm_key: '', llm_model: 'gpt-3.5-turbo', llm_provider: 'openai', temperature: 0.2,
  eval_k: 3, eval_hybrid: false, eval_rerank: false,
};
const C = { chunks: [], query: '', results: [], prompt: '', proj: [], query_pt: null, method: '', latency: 0, rewrote: false, rewritten: '' };
const done = new Set();
const outputs = {};
let cur = 'intro';

const STEPS = [
  { key: 'intro', ico: '📚', t: '认识 RAG', s: '概念',
    info: 'RAG = 检索增强生成：先检索外部资料，再把资料和问题一起交给大模型生成答案，让回答有依据、减少幻觉。本平台让你把这条链路（建库→检索→拼接→生成）亲手跑通，还能上传自己的文档、看 2D 向量图、做评测。',
    explain: '全流程：建库（切分→向量化→入库）→ 提问 → 检索 → 拼接 → 生成 → 评测。点左侧节点逐步操作。',
    inputs: [], run: null },

  { key: 'load', ico: '📄', t: '加载与切分', s: '① 建库',
    info: '长文档直接检索不精准也超长。切成小段（chunk）后每段语义更聚焦，检索能命中“最相关的那一块”。窗口大小与重叠控制切法。也可直接“上传文档”建独立知识库。',
    explain: '第一步：把内置示例文档切成小段；或点顶部「上传文档」导入你自己的 .txt/.md/.json/.pdf。',
    inputs: [
      { name: 'chunk_size', label: '片段长度(字)', type: 'number', min: 20 },
      { name: 'overlap', label: '重叠(字)', type: 'number', min: 0 },
    ], run: runLoad },

  { key: 'embed', ico: '🔢', t: '向量化建库', s: '② 建库',
    info: 'Embedding 把文字变成一串数字（向量），语义相近则向量相近。默认 TF-IDF（离线教学近似）；可切换为真实语义 Embedding（OpenAI 兼容 /embeddings）。下方 2D 图把所有片段投影到平面，让你“看见”相似度。',
    explain: '第二步：把每段向量化并存向量库。运行后看词汇表/维度，以及把所有片段投影到 2D 平面的散点图（同色=同一文档）。',
    inputs: [
      { name: 'emb_kind', label: 'Embedding 方式', type: 'select', opts: [['tfidf', 'TF-IDF（离线）'], ['openai', '真实 Embedding API']] },
      { name: 'emb_base', label: 'Embedding Base URL', type: 'text', ph: 'https://api.openai.com/v1', showIf: 'emb_kind=openai' },
      { name: 'emb_key', label: 'Embedding API Key', type: 'text', ph: 'sk-...', showIf: 'emb_kind=openai' },
      { name: 'emb_model', label: 'Embedding 模型', type: 'text', showIf: 'emb_kind=openai' },
    ], run: runEmbed },

  { key: 'query', ico: '🔍', t: '相似度检索', s: '③ 检索',
    info: '把问题也向量化，用余弦相似度（越近 1 越相关）找回 Top-K。进阶：开“混合检索”= BM25+向量 RRF 融合召回；“Rerank”= 用 Cross-Encoder（或 BM25 启发式）对召回结果精排。开“查询改写”会用 LLM 先把问题改写成更易检索的查询。',
    explain: '第三步：输入问题检索。可开启 混合检索 / Rerank / 查询改写。2D 图里黄色点是你的问题，绿圈是召回的 Top-K。',
    inputs: [
      { name: 'query', label: '你的问题', type: 'text', cls: 'query-in' },
      { name: 'top_k', label: '召回 K', type: 'number', min: 1 },
      { name: 'hybrid', label: '混合检索(BM25+向量)', type: 'switch' },
      { name: 'rerank', label: 'Rerank 精排', type: 'switch' },
      { name: 'rewrite', label: '查询改写', type: 'switch' },
    ], run: runQuery },

  { key: 'assemble', ico: '📝', t: '拼接提示词', s: '④ 增强',
    info: '拼接（Augmented）把召回资料与问题组合成 Prompt：“这是题目，这是参考资料，请基于资料作答”。这是 RAG 里“Augmented”的来源。',
    explain: '第四步：把召回资料和问题拼成完整 Prompt，交给大模型。',
    inputs: [], run: runAssemble },

  { key: 'generate', ico: '🤖', t: '大模型生成', s: '⑤ 生成',
    info: '大模型基于带资料的 Prompt 生成答案。默认离线检索增强摘要；填 API 后接真实大模型并支持流式输出（OpenAI/Ollama/Anthropic 可选）。',
    explain: '第五步：生成答案。填 LLM 配置后运行，答案逐字流式出现。',
    inputs: [
      { name: 'llm_provider', label: 'Provider', type: 'select', opts: [['openai', 'OpenAI / Ollama'], ['anthropic', 'Anthropic']] },
      { name: 'llm_base', label: 'API Base URL', type: 'text', ph: 'https://api.openai.com/v1 或 http://localhost:11434/v1' },
      { name: 'llm_key', label: 'API Key', type: 'text', ph: 'sk-...' },
      { name: 'llm_model', label: '模型名', type: 'text' },
      { name: 'temperature', label: '温度', type: 'number', min: 0, max: 2, step: 0.1 },
    ], run: runGenerate },

  { key: 'eval', ico: '📊', t: '评测', s: '⑥ 验证',
    info: '用内置金标准问答（问题→应命中文档）计算 recall@k：检索是否召回到正确文档。接入大模型后可进一步评答案忠实度(faithfulness)。这是判断“优化有没有用”的量化依据。',
    explain: '第六步：跑离线评测，看 recall@k 与逐题命中情况。改了切分/混合/Rerank 后重新评测对比。',
    inputs: [
      { name: 'eval_k', label: 'Top-K', type: 'number', min: 1 },
      { name: 'eval_hybrid', label: '混合检索', type: 'switch' },
      { name: 'eval_rerank', label: 'Rerank', type: 'switch' },
    ], run: runEval },
];

function stepBy(k) { return STEPS.find(s => s.key === k); }
function canRun(k) {
  if (k === 'load') return true;
  if (k === 'embed') return done.has('load');
  if (k === 'query') return done.has('embed');
  if (k === 'assemble') return done.has('query');
  if (k === 'generate') return done.has('assemble');
  if (k === 'eval') return done.has('embed');
  return true;
}

function renderPipeline() {
  $('#pipeline').innerHTML = STEPS.map(s => `
    <div class="node ${s.key === cur ? 'active' : ''} ${done.has(s.key) ? 'done' : ''}" data-key="${s.key}">
      <span class="done">✓</span><div class="ico">${s.ico}</div><div class="t">${s.t}</div><div class="s">${s.s}</div>
    </div>`).join('');
  $('#pipeline').querySelectorAll('.node').forEach(n => n.onclick = () => setCur(n.dataset.key));
}

function setCur(k) { cur = k; renderPipeline(); renderStep(); }

function showIfShow(inp, form) {
  if (!inp.showIf) return true;
  const [f, v] = inp.showIf.split('=');
  return String(form[f]) === v;
}

function renderStep() {
  const s = stepBy(cur);
  const inputsHtml = s.inputs.map(inp => {
    if (!showIfShow(inp, form)) return '';
    if (inp.type === 'switch') {
      return `<label class="switch">${inp.label}<input type="checkbox" data-f="${inp.name}" ${form[inp.name] ? 'checked' : ''}></label>`;
    }
    if (inp.type === 'select') {
      const o = inp.opts.map(([v, t]) => `<option value="${v}" ${form[inp.name] == v ? 'selected' : ''}>${t}</option>`).join('');
      return `<label>${inp.label}<select data-f="${inp.name}">${o}</select></label>`;
    }
    return `<label>${inp.label}<input class="${inp.cls || ''}" type="${inp.type}" data-f="${inp.name}"
      value="${form[inp.name]}" ${inp.min != null ? `min="${inp.min}"` : ''} ${inp.max != null ? `max="${inp.max}"` : ''}
      ${inp.step ? `step="${inp.step}"` : ''} placeholder="${inp.ph || ''}"></label>`;
  }).join('');
  const runBtn = s.run ? `<button class="btn" id="runBtn" ${canRun(s.key) ? '' : 'disabled'}>▶ 运行这一步</button>` : '';
  const hint = (!s.run || canRun(s.key)) ? '' :
    `<div class="tip" style="border-left-color:var(--amber)">⚠️ 请先完成上一步（点左侧前一个节点并运行）。</div>`;
  $('#stepHost').innerHTML = `
    <div class="card">
      <h2>${s.ico} ${s.t} <span class="tag">${s.s}</span><button class="info" data-info="${encodeURIComponent(s.info)}">i</button></h2>
      <div class="lead">${s.explain}</div>
      ${inputsHtml ? `<div class="row">${inputsHtml}</div>` : ''}
      <div class="row">${runBtn}</div>
      ${hint}
      <div class="out" id="out-${s.key}">${outputs[s.key] || '<span style="color:var(--muted)">尚未运行，点击上方按钮 👆</span>'}</div>
    </div>`;
  const ib = $('#stepHost .info'); if (ib) ib.onclick = e => showPop(e.target, decodeURIComponent(ib.dataset.info));
  $('#stepHost').querySelectorAll('input[data-f],select[data-f]').forEach(el => {
    el.oninput = el.onchange = () => {
      const f = el.dataset.f;
      form[f] = el.type === 'checkbox' ? el.checked : (el.tagName === 'SELECT' ? el.value : el.value);
      // emb_kind 切换时重渲染以显隐 API 字段
      if (f === 'emb_kind') renderStep();
    };
  });
  const rb = $('#runBtn'); if (rb) rb.onclick = () => s.run();
}

const esc = s => (s || '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
function setOut(key, html) { outputs[key] = html; const o = $('#out-' + key); if (o) o.innerHTML = html; }

// ---------- 各步骤运行 ----------
async function runLoad() {
  const r = await api('/api/load', { chunk_size: +form.chunk_size, overlap: +form.overlap });
  if (!r.ok) return setOut('load', `<h3>出错</h3>${esc(r.error)}`);
  C.chunks = r.chunks; done.add('load');
  const docs = r.docs.map(d => `<span class="chip">📄 ${esc(d.title)}（${d.len}字）</span>`).join('');
  const ch = r.chunks.slice(0, 10).map(c => `<div class="res"><div class="tx">${esc(c.text)}</div><div class="src">▸ ${esc(c.doc_title)} · ${esc(c.id)}</div></div>`).join('');
  setOut('load', `<h3>✅ 已加载 ${r.doc_count} 篇文档，切成 ${r.chunk_count} 个片段</h3><div class="kv">${docs}</div>
    <div style="margin-top:8px;color:var(--muted);font-size:12.5px">片段预览（前 10 个）：</div>${ch}${r.chunk_count > 10 ? `<div style="color:var(--muted);font-size:12px">…还有 ${r.chunk_count - 10} 个</div>` : ''}`);
  renderPipeline(); renderStep();
}

async function runUpload(files) {
  const fd = new FormData();
  for (const f of files) fd.append('file', f);
  const r = await fetch('/api/upload', { method: 'POST', body: fd }).then(x => x.json());
  if (!r.ok) return ($('#upMsg').textContent = '上传失败：' + (r.error || ''), $('#upMsg').style.color = 'var(--red)');
  CID = r.cid; form.emb_kind = 'tfidf';
  done.add('load'); done.add('embed');
  $('#curCol').textContent = r.name || r.cid;
  $('#upMsg').textContent = `已建库「${r.name || r.cid}」(${r.chunk_count} 片段)`;
  const ch = (r.chunks || []).slice(0, 8).map(c => `<div class="res"><div class="tx">${esc(c.text)}</div><div class="src">▸ ${esc(c.doc_title)}</div></div>`).join('');
  setOut('load', `<h3>✅ 已上传并向量化：${esc(r.name || r.cid)}（${r.chunk_count} 片段）</h3><div style="margin-top:8px">${ch}</div>`);
  renderPipeline(); renderStep();
}

async function runEmbed() {
  const r = await api('/api/embed', {
    cid: CID, emb_kind: form.emb_kind, emb_base: form.emb_base, emb_key: form.emb_key, emb_model: form.emb_model,
  });
  if (!r.ok) return setOut('embed', `<h3>出错</h3>${esc(r.error)}`);
  C.proj = r.proj || []; done.add('embed');
  const vec = (r.sample_vec || []).map(v => `<span class="v"><b>${esc(v.term)}</b>: ${v.val}</span>`).join('');
  const scatter = `<canvas class="scatter" id="scatterE"></canvas>
    <div class="legend"><span><i style="background:#34d399"></i>片段（同色=同文档）</span><span style="color:var(--amber)">（无黄点：尚未提问）</span></div>`;
  setOut('embed', `<h3>✅ 向量库已建好（${esc(r.emb_kind)}）</h3><div class="kv">
      <span class="chip">词汇表：${r.vocab_size}</span><span class="chip">向量维度：${r.vector_dim}</span><span class="chip">片段数：${r.chunk_count}</span></div>
    ${vec ? `<div style="margin-top:8px;color:var(--muted);font-size:12.5px">示例片段向量（按绝对值取前 ${r.sample_vec.length} 维）：</div><div class="vec">${vec}</div>` : ''}
    <div style="margin-top:10px;color:var(--muted);font-size:12.5px">2D 投影（PCA）：每个点是一个片段，越近越相似</div>${scatter}`);
  renderPipeline(); renderStep();
  drawScatter('#scatterE', C.proj, null, null);
}

async function runQuery() {
  const r = await api('/api/query', {
    cid: CID, query: form.query, top_k: +form.top_k, hybrid: form.hybrid, rerank: form.rerank, rewrite: form.rewrite,
  });
  if (!r.ok) return setOut('query', `<h3>出错</h3>${esc(r.error)}`);
  C.query = r.query; C.results = r.results; C.proj = r.proj; C.query_pt = r.query_pt;
  C.method = r.method; C.latency = r.latency_ms; C.rewrote = r.rewrote; C.rewritten = r.rewritten;
  done.add('query');
  const topSet = new Set(r.results.map(x => x.chunk_id));
  const res = r.results.map(x => `<div class="res"><div class="top"><span class="rk">Top ${x.rank}</span><span class="sc">相关度 ${x.score}</span></div>
    <div class="src">▸ 《${esc(x.doc_title)}》· ${esc(x.chunk_id)}</div><div class="tx">${esc(x.text)}</div></div>`).join('');
  const scatter = `<canvas class="scatter" id="scatterQ"></canvas>
    <div class="legend"><span><i style="background:#34d399"></i>召回 Top-K</span><span style="background:#fbbf24;border-radius:50%;width:10px;height:10px;display:inline-block"></span><span style="color:var(--amber)">你的问题</span></div>`;
  setOut('query', `<h3>✅ 召回 Top-${r.top_k}（${esc(r.method)}，耗时 ${r.latency_ms}ms）</h3>
    ${r.rewrote ? `<div class="bubble" style="border-left-color:var(--purple)">🔁 查询已改写为：<b>${esc(r.rewritten)}</b></div>` : ''}
    ${res}${scatter}`);
  renderPipeline(); renderStep();
  drawScatter('#scatterQ', C.proj, C.query_pt, topSet);
}

async function runAssemble() {
  if (!C.results.length) return setOut('assemble', `<h3>⚠️ 请先在「相似度检索」运行</h3>`);
  const r = await api('/api/assemble', { query: C.query, results: C.results });
  if (!r.ok) return setOut('assemble', `<h3>出错</h3>${esc(r.error)}`);
  C.prompt = r.prompt; done.add('assemble');
  setOut('assemble', `<h3>✅ 已拼接提示词</h3><pre>${esc(r.prompt)}</pre>`);
  renderPipeline(); renderStep();
}

async function runGenerate() {
  if (!C.prompt) return setOut('generate', `<h3>⚠️ 请先完成「拼接提示词」</h3>`);
  await api('/api/config', {
    llm_base: form.llm_base, llm_key: form.llm_key, llm_model: form.llm_model,
    llm_provider: form.llm_provider, temperature: +form.temperature,
    emb_base: form.emb_base, emb_key: form.emb_key, emb_model: form.emb_model, emb_kind: form.emb_kind,
  });
  setOut('generate', `<h3>⏳ 生成中…（流式）</h3><pre id="genOut"></pre>`);
  const genOut = $('#genOut'); let full = '';
  try {
    const resp = await fetch('/api/generate_stream', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: C.prompt, results: C.results, query: C.query }),
    });
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = '';
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n'); buf = parts.pop();
      for (const p of parts) {
        if (!p.startsWith('data:')) continue;
        const data = p.slice(5).trim();
        if (data === '[DONE]') continue;
        try { const o = JSON.parse(data); if (o.token) { full += o.token; genOut.textContent = full; } if (o.error) full += '\n[错误]' + o.error; } catch (e) {}
      }
    }
    const mode = form.llm_key && form.llm_base ? '<span class="badge llm">真实大模型·流式</span>' : '<span class="badge offline">离线摘要</span>';
    setOut('generate', `<h3>✅ 生成完成 ${mode}</h3><pre>${esc(full)}</pre>
      <div style="color:var(--muted);font-size:12px;margin-top:6px">想看真实大模型？填写上方 LLM 配置后重运行。</div>`);
    done.add('generate');
  } catch (e) {
    setOut('generate', `<h3>❌ 生成出错</h3><pre>${esc(e.message)}</pre>`);
  }
  renderPipeline(); renderStep();
}

async function runEval() {
  const r = await api('/api/eval', { cid: CID, top_k: +form.eval_k, hybrid: form.eval_hybrid, rerank: form.eval_rerank });
  if (!r.ok) return setOut('eval', `<h3>出错</h3>${esc(r.error)}`);
  const rows = r.detail.map(d => `<tr><td>${esc(d.question)}</td><td>${esc(d.gold)}</td>
    <td class="${d.hit ? 'ok' : 'no'}">${d.hit ? '✓' : '✗'}</td><td>${esc(d.top1)}</td></tr>`).join('');
  setOut('eval', `<h3>✅ 离线评测（${esc(r.method)}）</h3>
    <div class="metric"><div><b>${r['recall@k']}</b><br>recall@${r.k}</div><div><b>${r.hits}/${r.total}</b><br>命中题数</div></div>
    <div style="color:var(--muted);font-size:12px">${esc(r.note)}</div>
    <table class="eval"><tr><th>问题</th><th>金标准文档</th><th>命中</th><th>实际 Top1</th></tr>${rows}</table>`);
  renderPipeline(); renderStep();
}

// ---------- 2D 散点 ----------
function drawScatter(sel, proj, qpt, topSet) {
  const cv = $(sel); if (!cv || !proj) return;
  const ctx = cv.getContext('2d');
  const W = cv.width = cv.clientWidth || 600, H = cv.height = 300;
  ctx.clearRect(0, 0, W, H);
  const sx = x => (x * 0.5 + 0.5) * W, sy = y => (1 - (y * 0.5 + 0.5)) * H;
  proj.forEach(p => {
    const X = sx(p.x), Y = sy(p.y), top = topSet && topSet.has(p.cid);
    ctx.beginPath(); ctx.arc(X, Y, top ? 6 : 3.5, 0, 7);
    ctx.fillStyle = top ? '#34d399' : 'rgba(120,140,200,.7)'; ctx.fill();
    if (top) { ctx.strokeStyle = '#34d399'; ctx.lineWidth = 2; ctx.stroke(); }
  });
  if (qpt) {
    const X = sx(qpt.x), Y = sy(qpt.y);
    ctx.beginPath(); ctx.arc(X, Y, 7, 0, 7); ctx.fillStyle = '#fbbf24'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
  }
}

// ---------- 信息气泡 ----------
const pop = $('#pop');
function showPop(target, text) {
  pop.textContent = text;
  const r = target.getBoundingClientRect();
  pop.style.left = Math.min(r.left, window.innerWidth - 360) + 'px';
  pop.style.top = (r.bottom + 8) + 'px'; pop.classList.add('show');
  clearTimeout(pop._t); pop._t = setTimeout(() => pop.classList.remove('show'), 7000);
}
document.addEventListener('click', e => { if (!pop.contains(e.target) && !e.target.classList.contains('info')) pop.classList.remove('show'); });

// ============ 实时问答助手 ============
const chatBody = $('#chatBody');
const stepName = { intro: '认识 RAG', load: '加载与切分', embed: '向量化建库', query: '相似度检索', assemble: '拼接提示词', generate: '大模型生成', eval: '评测' };
function addMsg(role, html) {
  const d = document.createElement('div'); d.className = 'msg ' + role; d.innerHTML = html; chatBody.appendChild(d);
  if (role === 'a') { const g = d.querySelector('.goto'); if (g) g.onclick = () => setCur(g.dataset.k); }
  chatBody.scrollTop = chatBody.scrollHeight;
}
async function ask(q) {
  q = (q || '').trim(); if (!q) return;
  addMsg('u', esc(q)); $('#chatIn').value = '';
  try {
    const r = await api('/api/ask', { question: q });
    if (!r.ok) return addMsg('a', esc(r.error || '出错了'));
    let extra = r.goto_step ? `<br><span class="goto" data-k="${r.goto_step}">前往「${stepName[r.goto_step] || r.goto_step}」→</span>` : '';
    const tag = r.mode === 'llm' ? '（大模型回答）' : r.mode === 'faq' ? `（命中：${esc(r.matched)}）` : '（建议）';
    addMsg('a', esc(r.answer) + extra + `<div style="color:var(--muted);font-size:11px;margin-top:4px">${tag}</div>`);
  } catch (e) { addMsg('a', '网络异常：' + esc(e.message)); }
}
$('#chatSend').onclick = () => ask($('#chatIn').value);
$('#chatIn').addEventListener('keydown', e => { if (e.key === 'Enter') ask($('#chatIn').value); });
$('#chatQuick').querySelectorAll('span').forEach(s => s.onclick = () => ask(s.textContent));
$('#chatHd').onclick = () => $('#chat').classList.toggle('min');
addMsg('a', '你好！我是 RAG 教学助手 👋 可问任何概念，或说出实操中遇到的报错/困惑，我会帮你定位到对应步骤。');

// ============ 启动 ============
async function init() {
  const st = await api('/api/state');
  CID = st.default_cid;
  $('#curCol').textContent = '内置示例文档';
  renderPipeline(); renderStep();
  $('#fileIn').addEventListener('change', e => { if (e.target.files.length) runUpload(e.target.files); });
}
init();
