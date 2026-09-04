# -*- coding: utf-8 -*-
"""向量库（按集合 collection 持久化到磁盘，重启不丢）。"""
import os, json, time
import numpy as np
from embed import TfidfEmbedder
from datautil import load_default_docs, build_chunks

BASE = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(BASE, 'store')
DEFAULT_CID = 'default'

COLLECTIONS = {}  # cid -> dict(chunks, embedder, matrix, name, created)


def _ensure():
    os.makedirs(STORE_DIR, exist_ok=True)


def save_collection(cid, name, chunks, embedder, matrix):
    _ensure()
    d = os.path.join(STORE_DIR, cid)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'matrix.npy'), matrix)
    meta = {'cid': cid, 'name': name, 'created': time.time(),
            'chunks': chunks, 'embedder': embedder.to_dict() if hasattr(embedder, 'to_dict') else {'type': embedder.type}}
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), ensure_ascii=False)
    COLLECTIONS[cid] = {'chunks': chunks, 'embedder': embedder, 'matrix': matrix, 'name': name}


def load_collection(cid):
    if cid in COLLECTIONS:
        return COLLECTIONS[cid]
    d = os.path.join(STORE_DIR, cid)
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, 'meta.json')):
        return None
    meta = json.load(open(os.path.join(d, 'meta.json')))
    matrix = np.load(os.path.join(d, 'matrix.npy'))
    emb = meta['embedder']
    if emb['type'] == 'tfidf':
        e = TfidfEmbedder.from_dict(emb)
        e.matrix = matrix
    else:
        e = None  # OpenAI 集合需密钥才能重新 embed 查询，矩阵仍可用于展示
    COLLECTIONS[cid] = {'chunks': meta['chunks'], 'embedder': e, 'matrix': matrix, 'name': meta['name']}
    return COLLECTIONS[cid]


def list_collections():
    _ensure()
    out = []
    for cid in os.listdir(STORE_DIR):
        c = load_collection(cid)
        if c:
            out.append({'cid': cid, 'name': c['name'], 'chunk_count': len(c['chunks'])})
    return out


def build_default(emb_kind='tfidf', cfg=None, size=80, overlap=20):
    """启动时可调用：若磁盘已有默认集合则加载，否则用内置文档建库。"""
    existing = load_collection(DEFAULT_CID)
    if existing and existing['embedder'] is not None:
        return DEFAULT_CID
    docs = load_default_docs()
    chunks = build_chunks(docs, size, overlap)
    embedder = TfidfEmbedder().fit([c['text'] for c in chunks]) if emb_kind == 'tfidf' else None
    # OpenAI 集合：在线 embed 后保存（此处仅 tfidf 在启动时自动建）
    matrix = embedder.matrix
    save_collection(DEFAULT_CID, '内置示例文档', chunks, embedder, matrix)
    return DEFAULT_CID


def embed_and_save(cid, name, chunks, embedder):
    matrix = embedder.fit([c['text'] for c in chunks]).matrix if embedder.type == 'tfidf' else embedder.matrix
    save_collection(cid, name, chunks, embedder, matrix)
    return cid
