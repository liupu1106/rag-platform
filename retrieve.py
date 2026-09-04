# -*- coding: utf-8 -*-
"""检索：余弦相似度 + 混合检索(BM25+向量 RRF) + Rerank。"""
import math
import numpy as np
from embed import tokenize


def bm25(chunks, query, k1=1.5, b=0.75):
    """Okapi BM25（词面相关性），离线可算。"""
    q = tokenize(query)
    docs = [tokenize(c['text']) for c in chunks]
    N = len(docs) or 1
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(len(d) for d in docs) / N
    scores = np.zeros(len(docs))
    for i, d in enumerate(docs):
        dl = len(d) or 1
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in set(q):
            if t not in df:
                continue
            f = tf.get(t, 0)
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


def _rerank_scores(query, texts):
    """优先用 Cross-Encoder 精排；不可用时退回 BM25 启发式（离线可用）。"""
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [(query, t) for t in texts]
        return np.array(model.predict(pairs, show_progress_bar=False)), 'cross-encoder'
    except Exception:
        # 启发式：以 BM25 作为精排代理
        fake = [{'text': t} for t in texts]
        return bm25(fake, query), '启发式(BM25)'


def search(collection, query, top_k=3, hybrid=False, rerank=False):
    """返回 (results, method)。results 含 rank/score/doc_title/text/chunk_id。"""
    embedder = collection['embedder']
    matrix = collection['matrix']
    chunks = collection['chunks']
    qv = embedder.embed(query)
    cos = matrix @ qv  # 已归一化 -> 即余弦
    cand = np.argsort(-cos)
    method = 'cosine(TF-IDF)' if embedder.type == 'tfidf' else 'cosine(向量)'

    if hybrid:
        bm = bm25(chunks, query)
        k2 = max(top_k * 3, 10)
        rc = {i: 1 / (r + 60) for r, i in enumerate(cand[:k2])}
        rb = {i: 1 / (r + 60) for r, i in enumerate(np.argsort(-bm)[:k2])}
        rrf = {i: rc.get(i, 0) + rb.get(i, 0) for i in set(rc) | set(rb)}
        cand = np.array(sorted(rrf, key=rrf.get, reverse=True))
        method = 'hybrid(RRF: BM25+向量)'

    if rerank:
        pool = cand[:max(top_k * 4, 12)]
        scores, rmethod = _rerank_scores(query, [chunks[i]['text'] for i in pool])
        order = np.argsort(-np.array(scores))
        final = pool[order][:top_k]
        method = f"{method}+rerank({rmethod})"
    else:
        final = cand[:top_k]

    results = []
    for rank, i in enumerate(final):
        i = int(i)
        results.append({
            'rank': rank + 1,
            'chunk_id': chunks[i]['id'],
            'doc_id': chunks[i]['doc_id'],
            'doc_title': chunks[i]['doc_title'],
            'text': chunks[i]['text'],
            'score': round(float(cos[i]), 4),
        })
    return results, method
