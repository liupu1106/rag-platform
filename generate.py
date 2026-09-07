# -*- coding: utf-8 -*-
"""生成：多 Provider（OpenAI / Ollama / Anthropic）流式输出；离线检索增强摘要。

已接入 fallback_manager：主备多模型自动降级、失败重试、熔断、自动回切对上层透明。
流式场景的降级边界：仅覆盖「建连 / 首字节前失败」即切换到备用模型；一旦开始吐字，
中途断流无法对客户端透明重试（已吐出的 token 收不回），由上层 api_generate_stream
的 gen() 统一转成 error 事件。
"""
from llm_client import get_client, chat, AllModelsFailed, AllModelsUnavailable


def offline_answer(query, results):
    if not results:
        return '（离线模式未提供召回资料，无法生成摘要。请依次完成检索与拼接步骤。）'
    bullet = '\n'.join(f"• {r['text']}（出处：《{r['doc_title']}》，相关度 {r['score']}）" for r in results)
    return (f"【离线模式·检索增强摘要】\n针对你的问题「{query}」，系统从知识库召回了以下 {len(results)} 段最相关资料：\n\n"
            f"{bullet}\n\n（说明：当前为离线演示，未调用大模型。接入大模型 API 后，模型会基于以上资料生成通顺的自然语言回答。）")


def _chunks(s, n=24):
    for i in range(0, len(s), n):
        yield s[i:i + n]


def stream_generate(prompt, cfg, results=None, query=None):
    """生成器：逐段 yield 文本（供 SSE 封装）。"""
    client = get_client()
    # 未配置大模型 -> 离线检索增强摘要
    if client is None:
        for piece in _chunks(offline_answer(query, results)):
            yield piece
        return

    try:
        # 仅这一行可能同步抛 AllModelsFailed/AllModelsUnavailable（建连/选择阶段），
        # 此时还没吐任何 token，可安全回退到离线摘要。
        stream_iter = chat([{'role': 'user', 'content': prompt}],
                           stream=True, temperature=float(cfg.get('temperature', 0.2)))
    except (AllModelsFailed, AllModelsUnavailable):
        for piece in _chunks(offline_answer(query, results)):
            yield piece
        yield "\n\n[⚠️ 所有大模型均不可用，已切换为离线摘要模式]"
        return

    # 进入真正流式：此处之后的异常（含中途断流）由上层 gen() 处理为 error 事件。
    yield from stream_iter
