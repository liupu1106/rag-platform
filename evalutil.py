# -*- coding: utf-8 -*-
"""离线评测：recall@k（检索质量）与金标准覆盖度。faithfulness 需 LLM，留接口说明。"""
from datautil import GOLD
from retrieve import search


def evaluate(collection, top_k=3, hybrid=False, rerank=False):
    hits, total = 0, 0
    detail = []
    for q, gold_doc in GOLD:
        results, method = search(collection, q, top_k=top_k, hybrid=hybrid, rerank=rerank)
        total += 1
        hit = any(r['doc_id'] == gold_doc for r in results)
        hits += 1 if hit else 0
        detail.append({'question': q, 'gold': gold_doc,
                       'hit': hit, 'top1': results[0]['doc_title'] if results else ''})
    covered = hits / total
    return {
        'recall@k': round(covered, 3), 'k': top_k, 'total': total, 'hits': hits,
        'method': method, 'detail': detail,
        'note': 'recall@k 衡量检索是否召回到金标准文档；答案忠实度(faithfulness)需接入大模型后评测。',
    }
