# -*- coding: utf-8 -*-
"""文档加载、切分（RAGFlow 风格：标题层级识别 + Token 预算聚合）、以及评测用金标准问答。"""
import os
import re
import json
import logging

import jieba
jieba.setLogLevel(logging.ERROR)

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE, 'data', 'docs.json')


def load_default_docs():
    with open(DOCS_PATH, encoding='utf-8') as f:
        return json.load(f)


# ===================== RAGFlow 风格切分 =====================
# 核心思路（对齐 RAGFlow 的 naive/内容感知切分）：
#   1) 用正则识别文档大纲（markdown #、第X章、1.2.3 编号、一、二、等），构建标题层级树；
#   2) 以 Token 预算为上限，把相邻内容聚合成块，尽量不打断章节；
#   3) 每个块拼接祖先标题链作为上下文（RAGFlow 的“父标题带上下文”）。
# Token 用 jieba 分词数近似（中文友好）。

# 文档类型预设：(展示名, 默认 Token 预算)
# 注意：内置示例每篇约 90~123 字(≈token)，预算需明显低于/高于单篇长度，
# 才能让“切换文档类型”在样本上可见地改变分块数（简历保整篇，通用/论文/手册逐级更碎）。
DOC_TYPES = {
    'general': ('通用', 60),
    'paper':   ('论文', 40),
    'manual':  ('手册', 28),
    'resume':  ('简历', 256),
}


def count_tokens(text):
    return len(jieba.lcut(text or ''))


def detect_title(line):
    """识别一行是否为标题，返回 (层级, 标题文本)；否则返回 None。"""
    s = line.strip()
    if not s or len(s) > 80:
        return None
    # markdown 标题 # / ## / ...
    m = re.match(r'^(#{1,6})\s+(.*)$', s)
    if m:
        return (len(m.group(1)), m.group(2).strip())
    # 第一章 / 第一节
    if re.match(r'^第[零一二三四五六七八九十百千0-9]+章', s):
        return (1, s)
    if re.match(r'^第[零一二三四五六七八九十百千0-9]+节', s):
        return (2, s)
    # 1.2.3 / 1.2（至少两级编号）
    m = re.match(r'^(\d+(?:\.\d+){1,})\s', s)
    if m:
        return (m.group(1).count('.') + 1, s)
    # 1. / 1、 / 1) 列表式
    if re.match(r'^(\d+)[、.)）]', s):
        return (2, s)
    # 一、 二、 / （一）（二）
    if re.match(r'^[一二三四五六七八九十]+[、.．]', s):
        return (2, s)
    if re.match(r'^（[一二三四五六七八九十]+）', s):
        return (2, s)
    # Chapter / Section
    if re.match(r'^Chapter\s+[\dA-Z]', s, re.I):
        return (1, s)
    if re.match(r'^Section\s+[\dA-Z]', s, re.I):
        return (2, s)
    return None


def parse_outline(text):
    """把文本解析为标题层级树；无标题正文归到根节点下。"""
    root = {'level': 0, 'title': '', 'body': [], 'children': []}
    stack = [root]
    for line in text.split('\n'):
        t = detect_title(line)
        if t:
            level, title = t
            node = {'level': level, 'title': title, 'body': [], 'children': []}
            while stack and stack[-1]['level'] >= level:
                stack.pop()
            stack[-1]['children'].append(node)
            stack.append(node)
        else:
            stack[-1]['body'].append(line)
    for n in _iter_nodes(root):
        n['body'] = '\n'.join(n['body']).strip()
    return root


def _iter_nodes(node):
    yield node
    for c in node['children']:
        yield from _iter_nodes(c)


def _iter_preorder(node, ancestors):
    yield node, ancestors
    for c in node['children']:
        yield from _iter_preorder(c, ancestors + [node['title']])


def _split_long(text, token_num):
    """超长单块按标点/字符切小，保证每块不超过 Token 预算。"""
    if count_tokens(text) <= token_num:
        return [text]
    pieces, buf = [], ''
    for ch in text:
        buf += ch
        if ch in '。！？!?\n；;' and count_tokens(buf) >= token_num:
            pieces.append(buf)
            buf = ''
    if buf:
        pieces.append(buf)
    return pieces or [text]


def ragflow_chunk(text, token_num=128, method='general'):
    """RAGFlow 风格分块：标题层级树 + Token 预算聚合 + 祖先标题上下文。"""
    root = parse_outline(text)
    chunks = []
    buf, buf_tok = '', 0
    for node, ancestors in _iter_preorder(root, []):
        node_text = (node['title'] + ('\n' + node['body'] if node['body'] else '')).strip()
        if not node_text:
            continue
        tok = count_tokens(node_text)
        # 单块超预算：直接切小，并附带祖先标题作上下文
        if tok > token_num and buf_tok == 0:
            ctx = ' '.join(ancestors)
            for piece in _split_long(node_text, token_num):
                chunks.append((ctx + '\n' + piece).strip() if ctx else piece)
            continue
        # 超过预算且已有内容：先落盘当前块，新起一块
        # 注：普通合并不再重复拼接祖先标题，因为节点文本已自带其标题路径
        if buf_tok + tok > token_num and buf:
            chunks.append(buf.strip())
            buf, buf_tok = node_text, tok
        else:
            buf = (buf + '\n' + node_text) if buf else node_text
            buf_tok += tok
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def split_document(text, method='general', token_num=None):
    tn = token_num or DOC_TYPES.get(method, DOC_TYPES['general'])[1]
    return ragflow_chunk(text, tn, method)


def build_chunks(docs, token_num=128, method='general'):
    chunks = []
    for d in docs:
        for ci, piece in enumerate(split_document(d['text'], method, token_num)):
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
