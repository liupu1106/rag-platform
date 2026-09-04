# -*- coding: utf-8 -*-
"""Embedding 抽象：TF-IDF（默认/离线）与 OpenAI Embedding（可插拔）双后端。"""
import math, re
import numpy as np
import jieba
jieba.setLogLevel(20)


def tokenize(text):
    """中文按词、英文/数字按词，去标点小写。"""
    text = re.sub(r'\s+', ' ', text or '')
    out = []
    for w in jieba.cut(text):
        w = w.strip().lower()
        if not w:
            continue
        if re.match(r'^[一-鿿]+$', w):
            out.append(w)
        elif re.match(r'^[a-z0-9]+$', w):
            out.append(w)
    return out


class TfidfEmbedder:
    """离线、零依赖的近似 Embedding（教学用）。"""
    type = 'tfidf'

    def __init__(self):
        self.vocab = {}
        self.idf = None
        self.matrix = None  # (n, dim) 已 L2 归一化

    def fit(self, texts):
        tok_lists = [tokenize(t) for t in texts]
        df = {}
        for toks in tok_lists:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, t in enumerate(df)}
        N = len(tok_lists)
        self.idf = np.array([math.log((N + 1) / (df[t] + 1)) + 1.0 for t in self.vocab], dtype=float)
        rows = []
        for toks in tok_lists:
            L = len(toks) or 1
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            v = np.zeros(len(self.vocab), dtype=float)
            for t, c in tf.items():
                v[self.vocab[t]] = (c / L) * self.idf[self.vocab[t]]
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            rows.append(v)
        self.matrix = np.vstack(rows) if rows else np.zeros((0, len(self.vocab)))
        return self

    def embed(self, text):
        q = np.zeros(len(self.vocab), dtype=float)
        if not self.vocab:
            return q
        toks = tokenize(text)
        L = len(toks) or 1
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            if t in self.vocab:
                q[self.vocab[t]] = (c / L) * self.idf[self.vocab[t]]
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        return q

    def dim(self):
        return int(self.matrix.shape[1]) if self.matrix is not None else 0

    def to_dict(self):
        return {'type': 'tfidf', 'vocab': self.vocab, 'idf': self.idf.tolist()}

    @classmethod
    def from_dict(cls, d):
        e = cls()
        e.vocab = d['vocab']
        e.idf = np.array(d['idf'], dtype=float)
        return e  # matrix 由外部赋值


class OpenAIEmbedder:
    """真实语义 Embedding（OpenAI 兼容 /embeddings 接口）。"""
    type = 'openai'

    def __init__(self, base, key, model):
        self.base = base.rstrip('/')
        self.key = key
        self.model = model
        self.matrix = None
        self._dim = None

    def _one(self, text):
        import requests
        r = requests.post(self.base + '/embeddings', json={'input': text, 'model': self.model},
                          headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'},
                          timeout=60)
        r.raise_for_status()
        vec = r.json()['data'][0]['embedding']
        self._dim = len(vec)
        return np.array(vec, dtype=float)

    def fit(self, texts):
        self.matrix = np.vstack([self._one(t) for t in texts])
        return self

    def embed(self, text):
        return self._one(text)

    def dim(self):
        return int(self._dim or 0)


def make_embedder(kind, cfg):
    """工厂：kind='tfidf' 或 'openai'。"""
    if kind == 'openai':
        return OpenAIEmbedder(cfg.get('emb_base', ''), cfg.get('emb_key', ''), cfg.get('emb_model', 'text-embedding-3-small'))
    return TfidfEmbedder()


def pca2_fit(X):
    """把 (n,dim) 投影到 2D。返回 (coords(n,2), transformer)。transformer 可继续投影新点。"""
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.zeros((max(X.shape[0], 1), 2)), (None, None, 1.0)
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    scale = np.linalg.norm(coords, axis=1).max() or 1.0
    return coords / scale, (mean, Vt[:2], scale)


def pca2_project(qv, transformer):
    mean, Vt, scale = transformer
    if Vt is None:
        return np.array([0.0, 0.0])
    return ((np.asarray(qv, float) - mean) @ Vt.T) / scale
