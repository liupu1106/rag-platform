#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""RAG 平台全流程校验：逐端点断言每步/每条线的正确性。"""
import requests, sys
BASE = 'http://localhost:8080'

def check(name, cond, extra=''):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {extra}" if extra else ''))
    return cond

ok = True
def api(path, body=None):
    return requests.post(BASE+path, json=body).json()

# 1) 加载与切分
r = api('/api/load', {'chunk_size':80, 'overlap':20})
ok &= check('加载切分: ok', r.get('ok'))
ok &= check('加载切分: 6 篇文档', r.get('doc_count')==6, f"doc_count={r.get('doc_count')}")
ok &= check('加载切分: 切出片段>0', r.get('chunk_count',0)>0, f"chunk_count={r.get('chunk_count')}")
ok &= check('加载切分: 片段含文本', bool(r.get('chunks')) and len(r['chunks'][0]['text'])>0)

# 2) 向量化建库
r = api('/api/embed', {})
ok &= check('向量化: ok', r.get('ok'))
ok &= check('向量化: 词汇表>0', r.get('vocab_size',0)>0, f"vocab={r.get('vocab_size')}")
ok &= check('向量化: 维度>0', r.get('vector_dim',0)>0, f"dim={r.get('vector_dim')}")
ok &= check('向量化: 示例向量非空', bool(r.get('sample_vec')))

# 3) 检索：相关度 + 排序 + 范围 + 语义命中
def query(q, k=3):
    return api('/api/query', {'query':q, 'top_k':k})

# 3a 拍照答疑：黄金文档《拍照答疑怎么用》应居 Top-1
r = query('拍照答疑功能怎么用？识别不准怎么办？', 3)
ok &= check('检索: ok', r.get('ok'))
sc = [x['score'] for x in r['results']]
ok &= check('检索: 分数∈[0,1]', all(0<=s<=1 for s in sc), f"scores={sc}")
ok &= check('检索: 分数降序', sc==sorted(sc, reverse=True), f"scores={sc}")
ok &= check('检索: “拍照答疑”黄金文档居 Top-1', '拍照答疑怎么用' in r['results'][0]['doc_title'],
            f"top1={r['results'][0]['doc_title']}, scores={sc}")

# 3b 会员价格 应命中 doc5
r = query('会员一个月多少钱？年付多少？', 3)
ok &= check('检索: “会员价格”命中《会员与隐私说明》', '会员' in r['results'][0]['doc_title'], f"top1={r['results'][0]['doc_title']}")

# 3c 错题本 应命中 doc4
r = query('错题本如何生成复习计划？', 3)
ok &= check('检索: “错题本”命中《错题本与复习计划》', '错题' in r['results'][0]['doc_title'], f"top1={r['results'][0]['doc_title']}")

# 3d top_k 边界
r = query('智学助手是什么', 1)
ok &= check('检索: top_k=1 只返回 1 条', len(r['results'])==1)

# 4) 拼接提示词
r = query('智学助手怎么拍照答疑？', 3)
r2 = api('/api/assemble', {'query': r['query'], 'results': r['results']})
ok &= check('拼接: ok', r2.get('ok'))
ok &= check('拼接: 提示词含用户问题', r['query'] in r2['prompt'])
ok &= check('拼接: 提示词含召回资料', r['results'][0]['text'] in r2['prompt'])
ok &= check('拼接: 提示词含“参考资料”标记', '参考资料' in r2['prompt'])

# 5) 生成（离线）
r3 = api('/api/generate', {'prompt': r2['prompt'], 'results': r['results'], 'query': r['query']})
ok &= check('生成: ok', r3.get('ok'))
ok &= check('生成: 离线模式', r3.get('mode')=='offline', f"mode={r3.get('mode')}")
ok &= check('生成: 答案非空且含问题', bool(r3.get('answer')) and r['query'] in r3['answer'])

# 6) 实时问答助手
a = api('/api/ask', {'question':'RAG 是什么'})
ok &= check('问答: RAG是什么 -> faq 且跳 intro', a.get('mode')=='faq' and a.get('goto_step')=='intro', f"mode={a.get('mode')},goto={a.get('goto_step')}")
a = api('/api/ask', {'question':'检索相似度怎么算的'})
ok &= check('问答: 相似度 -> 命中 query 步骤', a.get('goto_step')=='query', f"goto={a.get('goto_step')}")
a = api('/api/ask', {'question':'asdfqwerty 乱码测试'})
ok &= check('问答: 无意义问题 -> fallback', a.get('mode')=='fallback', f"mode={a.get('mode')}")

# 7) 状态接口
s = requests.get(BASE+'/api/state').json()
ok &= check('状态: 已建索引', s.get('has_index') is True)

print('\n==== 结果:', '全部通过 ✅' if ok else '存在失败 ❌', '====')
sys.exit(0 if ok else 1)
