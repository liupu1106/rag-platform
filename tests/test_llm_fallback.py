# -*- coding: utf-8 -*-
"""llm_client ↔ fallback_manager 的真实集成测试。

通过 monkeypatch requests.post 模拟主/备模型的不同 HTTP 响应，覆盖：
1. 主模型失败自动切到备用（非流式）
2. 主模型熔断后，下一请求跳过它直接用备用
3. 熔断冷却后探活成功 -> 自动回切到主模型 + model_recovered 告警
4. 全部模型失败 -> AllModelsFailed（带降级路径）
5. 全部熔断冷却中 -> AllModelsUnavailable（带 retry_after）
6. 流式降级：主模型 500、备用 SSE 流式吐字
7. 致命错误（401）不切备用、立即中止
8. 离线模式：无大模型配置时 rewrite / stream_generate 优雅降级
9. 端到端：RAG_LLM_MODELS(JSON) -> 配置解析 -> 降级切换 -> 返回备用结果
"""
import json
import os
import sys

# 让 tests/ 能 import 到仓库根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

import requests

from fallback_manager import ModelConfig, ModelFallbackManager, ModelRegistry
from fallback_manager.exceptions import AllModelsFailed, AllModelsUnavailable, ModelFatalError

from llm_client import rag_caller_factory, rebuild, chat, get_client
import llm_client
import rewrite
import generate

MSG = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# 假 HTTP 层
# ---------------------------------------------------------------------------
class FakeResp:
    def __init__(self, status_code=200, json_data=None, lines=None):
        self.status_code = status_code
        self._json = json_data or {}
        self._lines = lines or []

    def json(self):
        return self._json

    def iter_lines(self):
        for ln in self._lines:
            yield ln


class ScriptedPost:
    """requests.post 替身：plan 把 url 子串映射到「依次返回的响应/异常」。"""

    def __init__(self, plan):
        self.plan = {k: list(v) for k, v in plan.items()}
        self.calls = []

    def __call__(self, url, **kw):
        self.calls.append(url)
        for sub, q in self.plan.items():
            if sub in url and q:
                item = q.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
        return FakeResp(200, {"choices": [{"message": {"content": "default"}}]})


def sse_lines(tokens):
    out = [("data: " + json.dumps({"choices": [{"delta": {"content": t}}]})).encode("utf-8")
           for t in tokens]
    out.append(b"data: [DONE]")
    return out


class CollectSink:
    def __init__(self):
        self.events = []

    def send(self, event):
        self.events.append(event)

    def kinds(self):
        return [e.kind.value for e in self.events]


class MutableClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_mgr(plan, *, clock=None, failure_threshold=1, cooldown=60.0,
             max_retries=0, total_timeout=30.0):
    registry = ModelRegistry([
        ModelConfig(name="primary", priority=0, base_url="http://primary/v1",
                    api_key="k", model="gpt-primary", timeout=5, max_retries=max_retries),
        ModelConfig(name="backup", priority=1, base_url="http://backup/v1",
                    api_key="k", model="gpt-backup", timeout=5, max_retries=max_retries),
    ])
    sink = CollectSink()
    mgr = ModelFallbackManager(
        registry, caller_factory=rag_caller_factory,
        failure_threshold=failure_threshold, cooldown=cooldown,
        total_timeout=total_timeout, alert_sink=sink,
        clock=clock or (lambda: 1000.0),
    )
    return mgr, sink


# ---------------------------------------------------------------------------
# 1. 主模型失败 -> 自动切到备用（非流式）
# ---------------------------------------------------------------------------
@patch("requests.post")
def test_switch_to_backup_on_failure(patch_post):
    post = ScriptedPost({
        "primary": [FakeResp(500)],
        "backup": [FakeResp(200, {"choices": [{"message": {"content": "backup ok"}}]})],
    })
    patch_post.side_effect = post
    mgr, sink = make_mgr(post.plan)

    out = mgr.chat(MSG)

    assert out == "backup ok"
    assert mgr.last_used_model == "backup"
    assert "switch" in sink.kinds()
    assert "model_down" in sink.kinds()
    assert any("primary" in c for c in post.calls)


# ---------------------------------------------------------------------------
# 2. 主模型熔断后，下一请求跳过它直接用备用（无 switch 告警）
# ---------------------------------------------------------------------------
@patch("requests.post")
def test_skip_opened_primary(patch_post):
    post = ScriptedPost({
        "primary": [FakeResp(500)],
        "backup": [FakeResp(200, {"choices": [{"message": {"content": "backup ok"}}]}),
                   FakeResp(200, {"choices": [{"message": {"content": "backup ok"}}]})],
    })
    patch_post.side_effect = post
    mgr, sink = make_mgr(post.plan)

    assert mgr.chat(MSG) == "backup ok"      # 主挂 -> 切备用
    assert mgr.chat(MSG) == "backup ok"      # 主仍在熔断 -> 直接走备用
    assert mgr.last_used_model == "backup"
    # 第二次没有 switch 告警（主是被跳过，不是切走）
    assert sink.kinds().count("switch") == 1


# ---------------------------------------------------------------------------
# 3. 熔断冷却后探活成功 -> 自动回切主模型 + model_recovered 告警
# ---------------------------------------------------------------------------
@patch("requests.post")
def test_failback_after_probe(patch_post):
    clock = MutableClock(1000.0)
    post = ScriptedPost({
        "primary": [FakeResp(500), FakeResp(200, {"choices": [{"message": {"content": "primary ok"}}]})],
        "backup": [FakeResp(200, {"choices": [{"message": {"content": "backup ok"}}]})],
    })
    patch_post.side_effect = post
    mgr, sink = make_mgr(post.plan, clock=clock, cooldown=10.0)

    assert mgr.chat(MSG) == "backup ok"      # 主挂 -> 备用
    clock.t = 1030                           # 超过 cooldown，主进入探活窗口
    assert mgr.chat(MSG) == "primary ok"     # 探活成功 -> 回切主模型
    assert mgr.last_used_model == "primary"
    assert "model_recovered" in sink.kinds()


# ---------------------------------------------------------------------------
# 4. 全部模型失败 -> AllModelsFailed（带降级路径）
# ---------------------------------------------------------------------------
@patch("requests.post")
def test_all_fail_raises(patch_post):
    post = ScriptedPost({
        "primary": [FakeResp(500)],
        "backup": [FakeResp(500)],
    })
    patch_post.side_effect = post
    mgr, sink = make_mgr(post.plan)

    try:
        mgr.chat(MSG)
        assert False, "应当抛出 AllModelsFailed"
    except AllModelsFailed as e:
        assert e.attempted == ["primary", "backup"]


# ---------------------------------------------------------------------------
# 5. 全部熔断冷却中 -> AllModelsUnavailable（带 retry_after）
# ---------------------------------------------------------------------------
@patch("requests.post")
def test_all_unavailable(patch_post):
    post = ScriptedPost({
        "primary": [FakeResp(500)],
        "backup": [FakeResp(500)],
    })
    patch_post.side_effect = post
    mgr, sink = make_mgr(post.plan, cooldown=60.0)

    try:
        mgr.chat(MSG)
    except AllModelsFailed:
        pass
    # 第二次：两者都在冷却 -> 无可用模型
    try:
        mgr.chat(MSG)
        assert False, "应当抛出 AllModelsUnavailable"
    except AllModelsUnavailable as e:
        assert e.retry_after is not None
        assert e.retry_after > 0


# ---------------------------------------------------------------------------
# 6. 流式降级：主模型 500、备用 SSE 流式吐字
# ---------------------------------------------------------------------------
@patch("requests.post")
def test_streaming_fallback(patch_post):
    post = ScriptedPost({
        "primary": [FakeResp(500)],
        "backup": [FakeResp(200, lines=sse_lines(["你", "好", "世界"]))],
    })
    patch_post.side_effect = post
    mgr, sink = make_mgr(post.plan)

    gen = mgr.chat(MSG, stream=True)
    tokens = list(gen)
    assert tokens == ["你", "好", "世界"]
    assert mgr.last_used_model == "backup"


# ---------------------------------------------------------------------------
# 7. 致命错误（401）不切备用、立即中止
# ---------------------------------------------------------------------------
@patch("requests.post")
def test_fatal_no_switch(patch_post):
    post = ScriptedPost({
        "primary": [FakeResp(401)],
        "backup": [FakeResp(200, {"choices": [{"message": {"content": "backup ok"}}]})],
    })
    patch_post.side_effect = post
    mgr, sink = make_mgr(post.plan)

    try:
        mgr.chat(MSG)
        assert False, "应当抛出 ModelFatalError"
    except ModelFatalError:
        pass
    # 主模型 401 后立即中止，绝不应调用备用
    assert not any("backup" in c for c in post.calls)


# ---------------------------------------------------------------------------
# 8. 离线模式：无大模型配置时优雅降级
# ---------------------------------------------------------------------------
def test_offline_mode():
    rebuild({})  # 空配置 -> get_client() 为 None
    assert get_client() is None

    # 改写：仍返回原句 + used=False
    q, used = rewrite.rewrite_query("什么是 RAG？", {})
    assert q == "什么是 RAG？" and used is False

    # 流式生成：走离线检索增强摘要
    chunks = list(generate.stream_generate("prompt", {}, results=[], query="RAG"))
    text = "".join(chunks)
    assert "离线" in text


# ---------------------------------------------------------------------------
# 9. 端到端：RAG_LLM_MODELS(JSON) -> 配置解析 -> 降级切换
# ---------------------------------------------------------------------------
@patch("requests.post")
def test_end_to_end_json_config(patch_post):
    post = ScriptedPost({
        "primary": [FakeResp(500)],
        "backup": [FakeResp(200, {"choices": [{"message": {"content": "bridge backup"}}]})],
    })
    patch_post.side_effect = post

    models = [
        {"name": "primary", "provider": "openai", "base_url": "http://primary/v1",
         "api_key": "k", "model": "gpt-primary", "priority": 0, "timeout": 5, "max_retries": 0},
        {"name": "backup", "provider": "openai", "base_url": "http://backup/v1",
         "api_key": "k", "model": "gpt-backup", "priority": 1, "timeout": 5, "max_retries": 0},
    ]
    cfg = {
        "llm_models": json.dumps(models),
        "llm_failure_threshold": "1",
        "llm_max_retries": "0",
        "llm_cooldown": "60",
        "llm_total_timeout": "30",
    }
    try:
        rebuild(cfg)
        assert get_client() is not None
        out = chat(MSG, stream=False, temperature=0.2)
        assert out == "bridge backup"
        assert get_client().last_used_model == "backup"
    finally:
        rebuild({})  # 还原为离线，避免影响其它测试
