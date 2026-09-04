# -*- coding: utf-8 -*-
"""鉴权（可选 Bearer Token，由环境变量 RAG_TOKEN 开启）与简单限流。"""
import os, time
from flask import request, jsonify

RATE = {}


def rate_ok(key, limit=60, window=60):
    now = time.time()
    RATE.setdefault(key, [])
    RATE[key] = [t for t in RATE[key] if now - t < window]
    if len(RATE[key]) >= limit:
        return False
    RATE[key].append(now)
    return True


def guard():
    """返回 None 表示放行；否则返回 (json, status)。"""
    token = os.environ.get('RAG_TOKEN')
    if token:
        h = request.headers.get('Authorization', '')
        if h != f'Bearer {token}':
            return jsonify({'ok': False, 'error': '未授权：缺少正确的 Bearer Token'}), 401
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if not rate_ok(ip, limit=int(os.environ.get('RAG_RATE', 120)), window=60):
        return jsonify({'ok': False, 'error': '请求过于频繁，请稍后再试'}), 429
    return None
