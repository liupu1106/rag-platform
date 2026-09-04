# -*- coding: utf-8 -*-
"""文档加载、切分、以及评测用金标准问答。"""
import os, re, json

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE, 'data', 'docs.json')


def load_default_docs():
    with open(DOCS_PATH, encoding='utf-8') as f:
        return json.load(f)


def chunk_text(text, size=80, overlap=20):
    """按字符滑动窗口切分（中文友好）。"""
    size, overlap = max(20, int(size)), max(0, int(overlap))
    text = re.sub(r'\s+', '', text)
    if not text:
        return []
    step = max(1, size - overlap)
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += step
    return out


def build_chunks(docs, size=80, overlap=20):
    chunks = []
    for d in docs:
        for ci, piece in enumerate(chunk_text(d['text'], size, overlap)):
            chunks.append({'id': f"{d['id']}-c{ci}", 'doc_id': d['id'],
                           'doc_title': d['title'], 'text': piece})
    return chunks


# 评测金标准：问题与"应命中文档"的对应（用于 recall@k）
GOLD = [
    ('拍照答疑功能怎么用？识别不准怎么办？', 'doc3'),
    ('会员一个月多少钱？年付多少？', 'doc5'),
    ('错题本如何生成复习计划？', 'doc4'),
    ('智学助手怎么安装和登录？', 'doc2'),
    ('智学助手是做什么的产品？', 'doc1'),
    ('RAG 是什么技术？', 'doc6'),
]
