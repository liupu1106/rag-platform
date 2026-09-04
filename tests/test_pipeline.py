# -*- coding: utf-8 -*-
"""RAG 平台端到端校验（离线，Flask 测试客户端）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from app import app

client = app.test_client()


def test_health():
    assert client.get('/healthz').get_json()['ok']


def test_state_and_default():
    r = client.get('/api/state').get_json()
    assert r['ok'] and r['default_cid']
    assert r['has_default']


def test_load():
    r = client.post('/api/load', json={'chunk_size': 80, 'overlap': 20}).get_json()
    assert r['ok'] and r['doc_count'] == 6 and r['chunk_count'] > 0


def test_embed_proj():
    r = client.post('/api/embed', json={'emb_kind': 'tfidf'}).get_json()
    assert r['ok']
    assert r['vocab_size'] > 0 and r['vector_dim'] > 0
    assert len(r['proj']) == r['chunk_count']          # 每个片段一个 2D 点
    assert all('cid' in p for p in r['proj'])           # 点带 chunk_id（高亮用）
    assert all(-1.2 <= p['x'] <= 1.2 and -1.2 <= p['y'] <= 1.2 for p in r['proj'])
    assert r['sample_vec']


def test_query_top1_and_proj():
    r = client.post('/api/query', json={'query': '拍照答疑功能怎么用？识别不准怎么办？', 'top_k': 3}).get_json()
    assert r['ok']
    scores = [x['score'] for x in r['results']]
    assert all(0 <= s <= 1 for s in scores) and scores == sorted(scores, reverse=True)
    assert '拍照答疑怎么用' in r['results'][0]['doc_title']      # 黄金文档居 Top1
    assert r['query_pt'] and 'x' in r['query_pt']               # 2D 含查询点
    assert 'cosine' in r['method']


def test_query_hybrid_rerank_flags():
    r = client.post('/api/query', json={'query': '会员多少钱', 'top_k': 3, 'hybrid': True, 'rerank': True}).get_json()
    assert r['ok']
    assert 'hybrid' in r['method'] and 'rerank' in r['method']
    assert '会员与隐私说明' in r['results'][0]['doc_title']


def test_assemble_contains_ctx():
    q = client.post('/api/query', json={'query': '错题本怎么复习', 'top_k': 3}).get_json()
    r = client.post('/api/assemble', json={'query': q['query'], 'results': q['results']}).get_json()
    assert r['ok']
    assert q['query'] in r['prompt'] and q['results'][0]['text'] in r['prompt']


def test_generate_stream_offline():
    q = client.post('/api/query', json={'query': '错题本怎么复习', 'top_k': 3}).get_json()
    a = client.post('/api/assemble', json={'query': q['query'], 'results': q['results']}).get_json()
    r = client.post('/api/generate_stream', json={'prompt': a['prompt'], 'results': q['results'], 'query': q['query']})
    data = r.data.decode('utf-8')
    assert 'data: [DONE]' in data
    assert '检索增强摘要' in data                          # 离线摘要已流式返回


def test_eval_recall():
    r = client.post('/api/eval', json={'top_k': 3}).get_json()
    assert r['ok']
    assert isinstance(r['recall@k'], float) and 0 <= r['recall@k'] <= 1
    assert r['total'] == 6 and r['hits'] > 0


def test_upload_txt():
    import io
    content = 'RAG 是检索增强生成。\n向量数据库用于存储嵌入。'.encode('utf-8')
    data = {'file': (io.BytesIO(content), 'a.txt', 'text/plain')}
    r = client.post('/api/upload', data=data, content_type='multipart/form-data').get_json()
    assert r['ok'] and r['cid'] and r['chunk_count'] > 0


def test_auth_required_when_token_set(monkeypatch):
    monkeypatch.setenv('RAG_TOKEN', 'secret123')
    r = client.post('/api/load', json={})
    assert r.status_code == 401
    r2 = client.post('/api/load', json={}, headers={'Authorization': 'Bearer secret123'})
    assert r2.status_code == 200
