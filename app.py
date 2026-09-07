#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""RAG 全流程实操平台 —— 后端编排（Flask + SSE 流式 + 持久化向量库）。"""
import os, io, time, json
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import numpy as np

from embed import make_embedder, TfidfEmbedder, pca2_fit, pca2_project
from store import build_default, load_collection, save_collection, list_collections, DEFAULT_CID, STORE_DIR
from datautil import load_default_docs, build_chunks, chunk_text, GOLD
from retrieve import search
from rewrite import rewrite_query
from generate import stream_generate, offline_answer
from evalutil import evaluate
from auth import guard
import llm_client
from llm_client import AllModelsFailed, AllModelsUnavailable

app = Flask(__name__, static_folder='static', static_url_path='/static')

# 配置：优先读环境变量（云端用 Secret 注入），UI 填写仅覆盖本次进程内存
def _env(name, dflt):
    return os.environ.get(name, dflt)
CFG = {
    'llm_base': _env('RAG_LLM_BASE', ''),
    'llm_key': _env('RAG_LLM_KEY', ''),
    'llm_model': _env('RAG_LLM_MODEL', 'gpt-3.5-turbo'),
    'llm_provider': _env('RAG_LLM_PROVIDER', 'openai'),
    'temperature': float(_env('RAG_TEMPERATURE', '0.2')),
    # —— 多模型降级（可选）——
    # RAG_LLM_MODELS: JSON 数组，每个元素 {name,provider,base_url,api_key,model,priority,timeout,max_retries}
    #   留空则使用下方单模型配置（向后兼容）。priority 越小越优先（主模型）。
    'llm_models': _env('RAG_LLM_MODELS', ''),
    'llm_alert_webhook': _env('RAG_LLM_ALERT_WEBHOOK', ''),  # 切换/恢复告警 webhook
    'llm_timeout': int(_env('RAG_LLM_TIMEOUT', '120')),
    'llm_max_retries': int(_env('RAG_LLM_MAX_RETRIES', '2')),
    'llm_failure_threshold': int(_env('RAG_LLM_FAILURE_THRESHOLD', '3')),
    'llm_cooldown': float(_env('RAG_LLM_COOLDOWN', '60.0')),
    'llm_degrade_threshold': int(_env('RAG_LLM_DEGRADE_THRESHOLD', '1')),
    'llm_total_timeout': float(_env('RAG_LLM_TOTAL_TIMEOUT', '120.0')),
    'emb_base': _env('RAG_EMB_BASE', 'https://api.openai.com/v1'),
    'emb_key': _env('RAG_EMB_KEY', ''),
    'emb_model': _env('RAG_EMB_MODEL', 'text-embedding-3-small'),
    'emb_kind': _env('RAG_EMB_KIND', 'tfidf'),
}

# 启动时根据配置构建降级管理器（无大模型配置时 get_client() 返回 None -> 离线模式）
llm_client.rebuild(CFG)

# 启动：确保默认集合存在
build_default(emb_kind='tfidf')

FAQ = [
    {"q": "RAG 是什么", "kw": ["rag", "是什么", "检索增强", "原理", "定义"],
     "a": "RAG = Retrieval-Augmented Generation（检索增强生成）。先检索外部资料，再把资料和问题一起交给大模型生成答案，让回答有依据、减少幻觉。"},
    {"q": "为什么要切分文档", "kw": ["切分", "chunk", "分段", "分块", "为什么"],
     "a": "长文档直接检索会不精准也超长。切成小段（chunk）后，每段语义更聚焦，检索能命中“最相关的那一块”，也更容易塞进模型上下文。"},
    {"q": "什么是向量化/Embedding", "kw": ["向量", "embedding", "嵌入", "坐标", "怎么变数字"],
     "a": "Embedding 把文字变成一串数字（向量），语义相近的文字向量也更接近。TF-IDF 是离线教学近似；真实语义 Embedding 用神经网络（如 bge / text-embedding-3）。平台支持在“向量化”步骤切换。"},
    {"q": "向量库是什么", "kw": ["向量库", "数据库", "存储", "faiss", "milvus", "pgvector", "chroma"],
     "a": "向量库把每段文字的向量存起来，并提供“给向量返回最近 K 个”的检索。工业界常用 Chroma、Milvus、pgvector、FAISS。"},
    {"q": "检索怎么算相似", "kw": ["相似", "余弦", "距离", "打分", "score", "topk", "top-k", "排序", "rerank", "混合"],
     "a": "把问题和资料都转成向量后，用余弦相似度（越接近 1 越相关）比较；可取 Top-K。进阶做法：混合检索（BM25+向量 RRF 融合）召回后再用 Cross-Encoder Rerank 精排——平台“检索”步骤可开启这两个开关。"},
    {"q": "为什么要拼接提示词", "kw": ["拼接", "prompt", "提示词", "augment", "增强", "组装"],
     "a": "拼接（Augmented）把召回的资料和问题组合成一段 Prompt：“这是题目，这是参考资料，请基于资料作答”。这是 RAG 里“Augmented”的来源。"},
    {"q": "生成那一步做什么", "kw": ["生成", "generate", "llm", "大模型", "回答", "离线", "流式"],
     "a": "大模型拿到带资料的 Prompt 后生成最终答案。平台默认离线检索增强摘要，可填 API 接真实大模型并支持流式输出。"},
    {"q": "什么是幻觉", "kw": ["幻觉", "胡说", "编造", "hallucin"],
     "a": "幻觉指大模型一本正经地编造错误内容。RAG 通过给模型提供可查证的外部资料显著降低幻觉，因为答案需要“有出处”。"},
    {"q": "topk 怎么选", "kw": ["topk", "top-k", "k值", "几段", "数量"],
     "a": "Top-K 是召回的资料段数，常见 3~5。K 太小可能漏信息，太大引入噪声、占用上下文。"},
    {"q": "怎么接真实大模型", "kw": ["api", "接入", "openai", "key", "真实", "配置", "密钥", "ollama", "anthropic"],
     "a": "在“大模型生成”步骤填入 API Base / Key / 模型名，可选 provider（openai / ollama / anthropic）。Ollama 用其 OpenAI 兼容地址（如 http://localhost:11434/v1）。密钥仅存本次会话内存。"},
    {"q": "怎么上传自己的文档", "kw": ["上传", "自己的", "文档", "pdf", "txt", "语料"],
     "a": "在顶部“上传文档”选择 .txt/.md/.json/.pdf，平台会建一个独立集合（collection）并向量化，之后所有步骤都基于你的资料。可在“评测”里看检索质量。"},
    {"q": "如何评测 RAG", "kw": ["评测", "评估", "recall", "指标", "质量", "ragas"],
     "a": "离线可算 recall@k（检索是否召回到金标准文档）与答案忠实度(faithfulness)。平台“评测”步骤给出 recall@k；接入大模型后可进一步用 Ragas 等评答案质量。"},
]


def get_embedder(cid):
    """取得可用于“查询向量化”的 embedder（openai 集合重载时按需重建）。"""
    col = load_collection(cid)
    if col and col['embedder'] is not None:
        return col['embedder']
    if CFG['emb_key'] and CFG['emb_base']:
        return make_embedder('openai', CFG)
    return None


def answer_faq(question):
    q_toks = set(__import__('embed').tokenize(question))
    best, best_score = None, 0
    for item in FAQ:
        s = sum(1 for k in item['kw'] if k in question.lower())
        s += sum(1 for t in q_toks if t in set(__import__('embed').tokenize(' '.join(item['kw']))))
        if s > best_score:
            best, best_score = item, s
    return best if best and best_score > 0 else None


def extract_text(file):
    name = file.filename.lower()
    raw = file.read()
    if name.endswith('.pdf'):
        try:
            from pypdf import PdfReader
            r = PdfReader(io.BytesIO(raw))
            return '\n'.join(p.extract_text() or '' for p in r.pages)
        except Exception as e:
            return f'[PDF 解析失败：{e}]'
    if name.endswith('.json'):
        try:
            obj = json.loads(raw.decode('utf-8', 'ignore'))
            if isinstance(obj, list):
                return '\n'.join(d.get('text') or d.get('content') or '' for d in obj)
            return obj.get('text') or json.dumps(obj, ensure_ascii=False)
        except Exception:
            return raw.decode('utf-8', 'ignore')
    return raw.decode('utf-8', 'ignore')


# ---------------- 页面 ----------------
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/state')
def api_state():
    col = load_collection(DEFAULT_CID)
    return jsonify({
        'ok': True, 'default_cid': DEFAULT_CID,
        'collections': list_collections(),
        'has_default': bool(col),
        'config': {'llm': llm_client.get_client() is not None,
                   'emb_openai': bool(CFG['emb_key'] and CFG['emb_base']),
                   'emb_kind': CFG['emb_kind']},
    })


@app.route('/api/config', methods=['POST'])
def api_config():
    d = request.get_json(silent=True) or {}
    for k in ('llm_base', 'llm_key', 'llm_model', 'llm_provider', 'temperature',
             'llm_models', 'llm_alert_webhook', 'llm_timeout', 'llm_max_retries',
             'llm_failure_threshold', 'llm_cooldown', 'llm_degrade_threshold', 'llm_total_timeout',
             'emb_base', 'emb_key', 'emb_model', 'emb_kind'):
        if k in d:
            CFG[k] = d[k]
    # 配置变化后重建降级管理器（清掉旧熔断状态，重读多模型配置）
    llm_client.rebuild(CFG)
    return jsonify({'ok': True, 'cfg': {k: ('***' if 'key' in k and v else v) for k, v in CFG.items()}})


# ---------------- 加载 / 切分 ----------------
@app.route('/api/load', methods=['POST'])
def api_load():
    g = guard()
    if g:
        return g
    d = request.get_json(silent=True) or {}
    size, overlap = int(d.get('chunk_size', 80)), int(d.get('overlap', 20))
    docs = load_default_docs()
    chunks = build_chunks(docs, size, overlap)
    col = load_collection(DEFAULT_CID)
    col['chunks'] = chunks
    col['embedder'] = TfidfEmbedder().fit([c['text'] for c in chunks])
    col['matrix'] = col['embedder'].matrix
    save_collection(DEFAULT_CID, '内置示例文档', chunks, col['embedder'], col['matrix'])
    return jsonify({'ok': True, 'cid': DEFAULT_CID, 'doc_count': len(docs), 'chunk_count': len(chunks),
                    'chunk_size': size, 'overlap': overlap,
                    'docs': [{'id': x['id'], 'title': x['title'], 'len': len(x['text'])} for x in docs],
                    'chunks': chunks})


# ---------------- 上传文档 ----------------
@app.route('/api/upload', methods=['POST'])
def api_upload():
    g = guard()
    if g:
        return g
    files = request.files.getlist('file')
    if not files:
        return jsonify({'ok': False, 'error': '请选择文件'}), 400
    texts = []
    for f in files:
        t = extract_text(f)
        if t.strip():
            texts.append(t)
    if not texts:
        return jsonify({'ok': False, 'error': '未能从文件中提取到文本'}), 400
    cid = 'u' + str(int(time.time()))[4:]
    chunks = []
    for ti, t in enumerate(texts):
        for ci, piece in enumerate(chunk_text(t, 80, 20)):
            chunks.append({'id': f"{cid}-d{ti}-c{ci}", 'doc_id': f"doc{ti}",
                           'doc_title': (files[min(ti, len(files)-1)].filename or f'文档{ti}')[:40],
                           'text': piece})
    emb = TfidfEmbedder().fit([c['text'] for c in chunks])
    save_collection(cid, f'上传文档({len(files)}个)', chunks, emb, emb.matrix)
    return jsonify({'ok': True, 'cid': cid, 'chunk_count': len(chunks), 'chunks': chunks[:20],
                    'name': f'上传文档({len(files)}个)'})


# ---------------- 向量化建库 ----------------
@app.route('/api/embed', methods=['POST'])
def api_embed():
    g = guard()
    if g:
        return g
    d = request.get_json(silent=True) or {}
    cid = d.get('cid') or DEFAULT_CID
    kind = d.get('emb_kind') or CFG.get('emb_kind') or 'tfidf'
    col = load_collection(cid)
    if not col:
        return jsonify({'ok': False, 'error': '集合不存在，请先加载/上传'}), 400
    if kind == 'openai':
        if not (CFG['emb_key'] and CFG['emb_base']):
            return jsonify({'ok': False, 'error': '请先在配置中填写 Embedding API'}), 400
        emb = make_embedder('openai', CFG)
        emb.fit([c['text'] for c in col['chunks']])
        col['embedder'] = emb
        col['matrix'] = emb.matrix
        save_collection(cid, col['name'], col['chunks'], emb, emb.matrix)
    else:
        if col['embedder'] is None or col['embedder'].type != 'tfidf':
            emb = TfidfEmbedder().fit([c['text'] for c in col['chunks']])
            col['embedder'] = emb
            col['matrix'] = emb.matrix
            save_collection(cid, col['name'], col['chunks'], emb, emb.matrix)
    CFG['emb_kind'] = kind
    # 2D 投影
    coords, transformer = pca2_fit(col['matrix'])
    points = [{'i': int(i), 'x': round(float(coords[i][0]), 4), 'y': round(float(coords[i][1]), 4),
               'doc_title': col['chunks'][i]['doc_title'], 'cid': col['chunks'][i]['id']} for i in range(len(coords))]
    sample_vec = []
    if col['embedder'].type == 'tfidf':
        v0 = col['embedder'].matrix[0]
        terms = list(col['embedder'].vocab.keys())
        idx = np.argsort(-np.abs(v0))[:12]
        sample_vec = [{'term': terms[i], 'val': round(float(v0[i]), 4)} for i in idx if abs(v0[i]) > 1e-9]
    return jsonify({'ok': True, 'cid': cid, 'emb_kind': kind, 'vocab_size': len(col['embedder'].vocab) if col['embedder'].type=='tfidf' else 0,
                    'vector_dim': int(col['matrix'].shape[1]), 'chunk_count': len(col['chunks']),
                    'sample_vec': sample_vec, 'proj': points})


# ---------------- 2D 投影（含查询点） ----------------
@app.route('/api/project', methods=['POST'])
def api_project():
    g = guard()
    if g:
        return g
    d = request.get_json(silent=True) or {}
    cid = d.get('cid') or DEFAULT_CID
    col = load_collection(cid)
    coords, transformer = pca2_fit(col['matrix'])
    points = [{'i': int(i), 'x': round(float(coords[i][0]), 4), 'y': round(float(coords[i][1]), 4),
               'doc_title': col['chunks'][i]['doc_title'], 'cid': col['chunks'][i]['id']} for i in range(len(coords))]
    q = d.get('query')
    qpt = None
    if q:
        emb = get_embedder(cid)
        if emb:
            qpt = pca2_project(emb.embed(q), transformer)
            qpt = {'x': round(float(qpt[0]), 4), 'y': round(float(qpt[1]), 4)}
    return jsonify({'ok': True, 'proj': points, 'query': qpt})


# ---------------- 检索 ----------------
@app.route('/api/query', methods=['POST'])
def api_query():
    g = guard()
    if g:
        return g
    d = request.get_json(silent=True) or {}
    cid = d.get('cid') or DEFAULT_CID
    q = (d.get('query') or '').strip()
    if not q:
        return jsonify({'ok': False, 'error': '请输入查询问题'}), 400
    top_k = max(1, min(int(d.get('top_k', 3)), 30))
    hybrid = bool(d.get('hybrid'))
    rerank = bool(d.get('rerank'))
    col = load_collection(cid)
    if col is None or col['matrix'] is None:
        return jsonify({'ok': False, 'error': '请先完成向量化建库'}), 400
    t0 = time.time()
    rq, used = rewrite_query(q, CFG) if d.get('rewrite') else (q, False)
    results, method = search(col, rq, top_k=top_k, hybrid=hybrid, rerank=rerank)
    dt = round((time.time() - t0) * 1000, 1)
    # 2D 投影（含查询点）
    coords, transformer = pca2_fit(col['matrix'])
    points = [{'i': int(i), 'x': round(float(coords[i][0]), 4), 'y': round(float(coords[i][1]), 4),
               'doc_title': col['chunks'][i]['doc_title'], 'cid': col['chunks'][i]['id']} for i in range(len(coords))]
    emb = get_embedder(cid)
    qpt = pca2_project(emb.embed(rq), transformer) if emb else None
    return jsonify({'ok': True, 'cid': cid, 'query': q, 'rewritten': rq, 'rewrote': used, 'top_k': top_k,
                    'method': method, 'latency_ms': dt, 'results': results,
                    'proj': points, 'query_pt': ({'x': round(float(qpt[0]),4),'y': round(float(qpt[1]),4)} if qpt is not None else None)})


# ---------------- 拼接 ----------------
@app.route('/api/assemble', methods=['POST'])
def api_assemble():
    g = guard()
    if g:
        return g
    d = request.get_json(silent=True) or {}
    results, query = d.get('results'), d.get('query')
    if not results or not query:
        return jsonify({'ok': False, 'error': '缺少检索结果或问题'}), 400
    ctx = '\n\n'.join(f"【资料{r['rank']}】（来自《{r['doc_title']}》）：\n{r['text']}" for r in results)
    prompt = ("你是严谨的问答助手。请只依据下面提供的资料回答用户问题，若资料中没有相关信息，请明确说明“资料中未提及”。\n\n"
              "===== 参考资料 =====\n" + ctx + "\n\n===== 用户问题 =====\n" + query + "\n\n===== 回答 =====")
    return jsonify({'ok': True, 'prompt': prompt})


# ---------------- 流式生成 ----------------
@app.route('/api/generate_stream', methods=['POST'])
def api_generate_stream():
    g = guard()
    if g:
        return g
    d = request.get_json(silent=True) or {}
    prompt = d.get('prompt', '')
    results = d.get('results', [])
    query = d.get('query', '')
    if not prompt:
        return jsonify({'ok': False, 'error': '缺少提示词'}), 400

    def gen():
        try:
            for piece in stream_generate(prompt, CFG, results, query):
                yield f"data: {json.dumps({'token': piece}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return Response(stream_with_context(gen()), mimetype='text/event-stream')


# ---------------- 评测 ----------------
@app.route('/api/eval', methods=['POST'])
def api_eval():
    g = guard()
    if g:
        return g
    d = request.get_json(silent=True) or {}
    cid = d.get('cid') or DEFAULT_CID
    col = load_collection(cid)
    if not col:
        return jsonify({'ok': False, 'error': '集合不存在'}), 400
    top_k = int(d.get('top_k', 3))
    hybrid = bool(d.get('hybrid'))
    rerank = bool(d.get('rerank'))
    return jsonify({'ok': True, **evaluate(col, top_k=top_k, hybrid=hybrid, rerank=rerank)})


# ---------------- 实时问答助手 ----------------
@app.route('/api/ask', methods=['POST'])
def api_ask():
    g = guard()
    if g:
        return g
    d = request.get_json(silent=True) or {}
    q = (d.get('question') or '').strip()
    if not q:
        return jsonify({'ok': False, 'error': '请输入问题'}), 400
    hit = answer_faq(q)
    client = llm_client.get_client()
    if client is not None:
        try:
            kb = '\n'.join(f"- {f['q']}: {f['a']}" for f in FAQ)
            ans = llm_client.chat([
                {'role': 'system', 'content': '你是 RAG 教学助手，用简洁中文回答。\n知识库：\n' + kb},
                {'role': 'user', 'content': q},
            ], stream=False, temperature=0.3)
            if isinstance(ans, str) and ans.strip():
                return jsonify({'ok': True, 'mode': 'llm', 'answer': ans.strip(),
                               'model': client.last_used_model})
        except (AllModelsFailed, AllModelsUnavailable):
            pass  # 全部模型不可用：降级到 FAQ / 兜底回答
    if hit:
        return jsonify({'ok': True, 'mode': 'faq', 'answer': hit['a'], 'goto_step': _step_of(hit['q']), 'matched': hit['q']})
    return jsonify({'ok': True, 'mode': 'fallback', 'answer': '我暂时没有匹配到确切答案。你可以问：RAG 是什么 / 为什么要切分 / 向量化怎么做 / 检索怎么算相似 / 怎么接真实大模型 / 怎么上传自己的文档。'})


def _step_of(qtext):
    m = {'rag': 'intro', '切分': 'load', '向量': 'embed', '向量库': 'embed', '相似': 'query',
         '拼接': 'assemble', '生成': 'generate', '幻觉': 'intro', 'topk': 'query',
         '上传': 'load', '评测': 'eval'}
    for k, v in m.items():
        if k in qtext:
            return v
    return 'intro'


@app.route('/healthz')
def healthz():
    return jsonify({'ok': True})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
