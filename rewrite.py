# -*- coding: utf-8 -*-
"""查询改写：用 LLM 把口语化/有指代的问题改写成更易检索的查询。无 LLM 配置则返回原句。"""
import requests


def rewrite_query(query, cfg):
    """返回 (new_query, used)。"""
    if not (cfg.get('llm_key') and cfg.get('llm_base')):
        return query, False
    try:
        url = cfg['llm_base'].rstrip('/') + '/chat/completions'
        resp = requests.post(url, json={
            'model': cfg.get('llm_model', 'gpt-3.5-turbo'),
            'messages': [{'role': 'system',
                          'content': '你是检索系统助手。请把用户的问题改写成一段更适合向量检索的查询，'
                                     '保留关键实体，去掉口语和问候，只输出改写后的查询本身，不要解释。'},
                         {'role': 'user', 'content': query}],
            'temperature': 0.1,
        }, headers={'Authorization': f"Bearer {cfg['llm_key']}", 'Content-Type': 'application/json'}, timeout=60)
        if resp.status_code == 200:
            q = resp.json()['choices'][0]['message']['content'].strip()
            return (q or query), True
    except Exception:
        pass
    return query, False
