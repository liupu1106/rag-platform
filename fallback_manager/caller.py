"""Caller abstraction so the manager stays SDK-agnostic.

The manager never talks to an SDK directly. It asks a *caller* to perform one
invocation for a given model. This makes the whole strategy testable with fake
callers and lets you wrap OpenAI, DashScope, or any home-grown endpoint.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

__all__ = [
    "ChatRequest",
    "ModelCaller",
    "OpenAICaller",
    "CallableCaller",
    "default_caller_factory",
]


class ChatRequest(Protocol):
    """Structural type for the per-call kwargs (messages=..., **openai kwargs)."""

    messages: list[dict[str, str]]


class ModelCaller(Protocol):
    """Callable that performs one model invocation.

    Must accept ``model`` (the configured model name) and the openai-style
    ``messages`` plus any extra kwargs, and return whatever the SDK returns.
    ANY exception it raises is classified by the manager's classifier.
    """

    def __call__(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        ...


class CallableCaller:
    """Wrap an arbitrary ``f(model=..., messages=..., **kw)`` as a ModelCaller."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func

    def __call__(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        return self._func(model=model, messages=messages, **kwargs)


class OpenAICaller:
    """OpenAI-compatible caller built from a :class:`ModelConfig`.

    Requires the ``openai`` package (``pip install openai``). One client is
    created per model config so different base_url/api_key work side by side.
    Any SDK exception (timeout / connection / rate-limit / status) is allowed
    to propagate — the manager classifies it via ``default_classifier``.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        **client_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "OpenAICaller needs the 'openai' package: pip install openai"
            ) from exc
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, **client_kwargs
        )

    def __call__(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        return self._client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )


def default_caller_factory(config) -> ModelCaller:
    """Build a caller for a config.

    If the config carries a pre-built ``caller`` attribute it is returned
    as-is (lets tests and advanced users inject fakes). Otherwise an
    :class:`OpenAICaller` is constructed from the config's connection fields.
    """
    explicit = getattr(config, "caller", None)
    if explicit is not None:
        return explicit
    return OpenAICaller(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
        **config.client_kwargs,
    )
