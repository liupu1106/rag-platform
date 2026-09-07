# -*- coding: utf-8 -*-
"""LLM 调用桥接层：把 rag-platform 裸 ``requests`` 调用接到 fallback_manager。

对上层（generate / rewrite / app）完全透明，只暴露：

    rebuild(cfg)   # 配置变化时重建（清熔断、重读多模型配置）
    get_client()   # 当前 client；无大模型配置时返回 None（走离线）
    chat(messages, *, stream=False, temperature=0.2) -> str | 生成器[str] | None

降级能力全部来自 fallback_manager（主备优先级、重试退避、熔断、自动回切、
切换/恢复告警）。本层只负责两件事：
1. 把 CFG / 环境变量翻译成 ModelRegistry（主+备多模型，priority 越小越优先）；
2. 用基于 requests 的 RagChatCaller 做真实 HTTP，并把 requests 的
   超时/连接错误/HTTP 状态码翻译成 fallback_manager 的 typed 异常。
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Iterator, Optional

import requests

from fallback_manager import (
    AlertKind,
    CompositeAlertSink,
    LoggingAlertSink,
    ModelConfig,
    ModelFallbackManager,
    ModelConnectionError,
    ModelFatalError,
    ModelRateLimitError,
    ModelRegistry,
    ModelServerError,
    ModelTimeoutError,
    WebhookAlertSink,
)
from fallback_manager.exceptions import AllModelsFailed, AllModelsUnavailable

log = logging.getLogger("rag.llm")


# ---------------------------------------------------------------------------
# 配置翻译：CFG / 环境变量 -> ModelRegistry
# ---------------------------------------------------------------------------
def _models_from_cfg(cfg: dict) -> Optional[list[dict]]:
    """返回模型列表（dict），或 None 表示未配置大模型（走离线）。"""
    raw = cfg.get("llm_models")  # 由 RAG_LLM_MODELS(JSON) 注入
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list) and parsed:
                return parsed
        except Exception:
            log.warning("RAG_LLM_MODELS 解析失败，忽略，回退到单模型配置")

    # 单模型（向后兼容旧环境变量 RAG_LLM_BASE/KEY/MODEL/PROVIDER）
    if not (cfg.get("llm_key") and cfg.get("llm_base")):
        return None
    return [{
        "name": cfg.get("llm_model") or "primary",
        "provider": (cfg.get("llm_provider") or "openai").lower(),
        "base_url": cfg["llm_base"],
        "api_key": cfg["llm_key"],
        "model": cfg.get("llm_model") or "gpt-3.5-turbo",
        "priority": 0,
        "timeout": int(cfg.get("llm_timeout", 120)),
        "max_retries": int(cfg.get("llm_max_retries", 2)),
    }]


def build_client(cfg: dict) -> Optional[ModelFallbackManager]:
    """根据配置构建降级管理器；无大模型配置时返回 None。"""
    models = _models_from_cfg(cfg)
    if not models:
        return None
    try:
        registry = ModelRegistry([ModelConfig.from_dict(m) for m in models])
    except Exception as e:  # 配置错误不应让整个服务起不来
        log.error("构建模型注册表失败：%s", e)
        return None

    sinks: list[Any] = [LoggingAlertSink(logging.getLogger("rag.llm"))]
    webhook = cfg.get("llm_alert_webhook")
    if webhook:
        sinks.append(WebhookAlertSink(webhook))

    try:
        mgr = ModelFallbackManager(
            registry,
            caller_factory=rag_caller_factory,
            failure_threshold=int(cfg.get("llm_failure_threshold", 3)),
            cooldown=float(cfg.get("llm_cooldown", 60.0)),
            degrade_threshold=int(cfg.get("llm_degrade_threshold", 1)),
            total_timeout=float(cfg.get("llm_total_timeout", 120.0)),
            alert_sink=CompositeAlertSink(sinks),
            logger=logging.getLogger("rag.llm"),
        )
    except Exception as e:
        log.error("构建降级管理器失败：%s", e)
        return None
    return mgr


# ---------------------------------------------------------------------------
# 调用方：基于 requests，OpenAI 兼容 + Anthropic
# ---------------------------------------------------------------------------
class RagChatCaller:
    """一个模型端点的真实 HTTP 调用方，供 fallback_manager 调度。"""

    def __init__(self, provider: str, base_url: str, api_key: str, model: str, timeout: float):
        self.provider = (provider or "openai").lower()
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def __call__(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any):
        stream = bool(kwargs.get("stream"))
        temperature = float(kwargs.get("temperature", 0.2))
        if self.provider == "anthropic":
            return self._anthropic(messages, stream, temperature)
        return self._openai(messages, stream, temperature)

    # ---- OpenAI 兼容（/chat/completions；ollama/deepseek/qwen 同构）----
    def _openai(self, messages, stream, temperature):
        url = self.base_url + "/chat/completions"
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "stream": stream}
        try:
            resp = requests.post(
                url, json=body,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                timeout=self.timeout, stream=stream,
            )
        except requests.exceptions.Timeout:
            raise ModelTimeoutError("upstream timeout", model=self.model)
        except requests.exceptions.ConnectionError:
            raise ModelConnectionError("connection error", model=self.model)
        except requests.exceptions.RequestException as e:
            raise ModelConnectionError(f"request error: {e}", model=self.model)

        code = resp.status_code
        if code == 429:
            raise ModelRateLimitError("rate limited (429)", model=self.model)
        if code in (500, 502, 503, 504):
            raise ModelServerError(f"upstream 5xx ({code})", status_code=code, model=self.model)
        if 400 <= code < 500:
            # 400/401/403 是请求本身或鉴权问题，切到备用模型必同样失败 -> 致命，立即中止
            raise ModelFatalError(f"request rejected ({code})", status_code=code, model=self.model)
        if code != 200:
            raise ModelServerError(f"unexpected status {code}", status_code=code, model=self.model)

        if stream:
            return self._iter_openai_stream(resp)
        try:
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise ModelServerError(f"bad response: {e}", model=self.model)

    @staticmethod
    def _iter_openai_stream(resp) -> Iterator[str]:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
            except Exception:
                continue

    # ---- Anthropic（/v1/messages，消息结构不同）----
    def _anthropic(self, messages, stream, temperature):
        url = self.base_url + ("/v1/messages" if "/messages" not in self.base_url else "")
        sys = next((m["content"] for m in messages if m.get("role") == "system"), None)
        msgs = [m for m in messages if m.get("role") != "system"]
        body = {"model": self.model, "max_tokens": 1024,
                "messages": msgs, "temperature": temperature, "stream": stream}
        if sys:
            body["system"] = sys
        try:
            resp = requests.post(
                url, json=body,
                headers={"x-api-key": self.api_key,
                         "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                timeout=self.timeout, stream=stream,
            )
        except requests.exceptions.Timeout:
            raise ModelTimeoutError("upstream timeout", model=self.model)
        except requests.exceptions.ConnectionError:
            raise ModelConnectionError("connection error", model=self.model)
        except requests.exceptions.RequestException as e:
            raise ModelConnectionError(f"request error: {e}", model=self.model)

        code = resp.status_code
        if code == 429:
            raise ModelRateLimitError("rate limited (429)", model=self.model)
        if code in (500, 502, 503, 504):
            raise ModelServerError(f"upstream 5xx ({code})", status_code=code, model=self.model)
        if 400 <= code < 500:
            raise ModelFatalError(f"request rejected ({code})", status_code=code, model=self.model)
        if stream:
            return self._iter_anthropic_stream(resp)
        try:
            data = resp.json()
            return "".join(b.get("text", "") for b in data.get("content", []))
        except Exception as e:
            raise ModelServerError(f"bad response: {e}", model=self.model)

    @staticmethod
    def _iter_anthropic_stream(resp) -> Iterator[str]:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line.startswith("data:"):
                continue
            try:
                obj = json.loads(line[5:].strip())
                if obj.get("type") == "content_block_delta":
                    yield obj["delta"].get("text", "")
            except Exception:
                continue


def rag_caller_factory(config) -> RagChatCaller:
    return RagChatCaller(
        provider=getattr(config, "provider", "openai"),
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model or config.name,
        timeout=config.timeout,
    )


# ---------------------------------------------------------------------------
# 进程级单例：CFG 在运行时可能经 /api/config 改变，需重建
# ---------------------------------------------------------------------------
_CLIENT: Optional[ModelFallbackManager] = None
_CLIENT_LOCK = threading.Lock()


def rebuild(cfg: dict) -> None:
    """根据最新 CFG 重建降级管理器（会清掉旧的熔断状态）。"""
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = build_client(cfg)


def get_client() -> Optional[ModelFallbackManager]:
    return _CLIENT


def chat(messages: list[dict[str, str]], *, stream: bool = False,
         temperature: float = 0.2) -> Any:
    """对上层透明的入口。无大模型配置返回 None（调用方走离线）。

    - 非流式：返回 str。
    - 流式：返回 生成器[str]（一旦开始吐字，中途失败由上层 gen() 处理）。
    - 全部模型不可用：抛 AllModelsFailed / AllModelsUnavailable（调用方捕获后降级）。
    """
    client = _CLIENT
    if client is None:
        return None
    return client.chat(messages, stream=stream, temperature=temperature)


__all__ = [
    "build_client", "rebuild", "get_client", "chat",
    "RagChatCaller", "rag_caller_factory",
    "AllModelsFailed", "AllModelsUnavailable", "AlertKind",
]
