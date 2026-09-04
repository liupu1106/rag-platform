# -*- coding: utf-8 -*-
"""生成：多 Provider（OpenAI / Ollama / Anthropic）流式输出；离线检索增强摘要。"""
import requests


def offline_answer(query, results):
    if not results:
        return '（离线模式未提供召回资料，无法生成摘要。请依次完成检索与拼接步骤。）'
    bullet = '\n'.join(f"• {r['text']}（出处：《{r['doc_title']}》，相关度 {r['score']}）" for r in results)
    return (f"【离线模式·检索增强摘要】\n针对你的问题「{query}」，系统从知识库召回了以下 {len(results)} 段最相关资料：\n\n"
            f"{bullet}\n\n（说明：当前为离线演示，未调用大模型。接入大模型 API 后，模型会基于以上资料生成通顺的自然语言回答。）")


def _chunks(s, n=24):
    for i in range(0, len(s), n):
        yield s[i:i + n]


def _stream_openai(prompt, cfg):
    """OpenAI / Ollama（均兼容 /chat/completions SSE）。"""
    url = cfg['llm_base'].rstrip('/') + '/chat/completions'
    resp = requests.post(url, json={
        'model': cfg.get('llm_model', 'gpt-3.5-turbo'),
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': float(cfg.get('temperature', 0.2)),
        'stream': True,
    }, headers={'Authorization': f"Bearer {cfg['llm_key']}", 'Content-Type': 'application/json'},
        timeout=120, stream=True)
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode('utf-8') if isinstance(line, bytes) else line
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            break
        try:
            obj = __import__('json').loads(data)
            delta = obj['choices'][0]['delta'].get('content')
            if delta:
                yield delta
        except Exception:
            continue


def _stream_anthropic(prompt, cfg):
    """Anthropic Messages Streaming（轻量实现）。"""
    url = (cfg['llm_base'].rstrip('/') + '/v1/messages') if 'messages' not in cfg['llm_base'] else cfg['llm_base']
    resp = requests.post(url, json={
        'model': cfg.get('llm_model', 'claude-3-5-haiku-latest'),
        'max_tokens': 1024,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': True,
    }, headers={'x-api-key': cfg['llm_key'], 'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'}, timeout=120, stream=True)
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode('utf-8') if isinstance(line, bytes) else line
        if not line.startswith('data:'):
            continue
        try:
            obj = __import__('json').loads(line[5:].strip())
            if obj.get('type') == 'content_block_delta':
                yield obj['delta'].get('text', '')
        except Exception:
            continue


def stream_generate(prompt, cfg, results=None, query=None):
    """生成器：逐段 yield 文本（供 SSE 封装）。"""
    provider = (cfg.get('llm_provider') or 'openai').lower()
    if not (cfg.get('llm_key') and cfg.get('llm_base')):
        for piece in _chunks(offline_answer(query, results)):
            yield piece
        return
    if provider in ('openai', 'ollama'):
        yield from _stream_openai(prompt, cfg)
    elif provider == 'anthropic':
        yield from _stream_anthropic(prompt, cfg)
    else:
        yield from _stream_openai(prompt, cfg)
