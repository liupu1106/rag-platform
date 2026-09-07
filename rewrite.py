# -*- coding: utf-8 -*-
"""查询改写：用 LLM 把口语化/有指代的问题改写成更易检索的查询。无 LLM 配置则返回原句。

已接入 fallback_manager：多模型主备降级、失败重试、熔断、自动回切对调用方透明。
"""
from llm_client import chat, AllModelsFailed, AllModelsUnavailable


def rewrite_query(query, cfg):
    """返回 (new_query, used)。"""
    try:
        text = chat([
            {'role': 'system',
             'content': '你是检索系统助手。请把用户的问题改写成一段更适合向量检索的查询，'
                        '保留关键实体，去掉口语和问候，只输出改写后的查询本身，不要解释。'},
            {'role': 'user', 'content': query},
        ], stream=False, temperature=0.1)
    except (AllModelsFailed, AllModelsUnavailable):
        # 全部模型不可用：优雅降级为「不改写」，不影响检索流程
        return query, False
    if isinstance(text, str) and text.strip():
        return text.strip(), True
    return query, False
