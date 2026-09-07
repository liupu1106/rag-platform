"""Default failure classifier: maps any exception into our typed hierarchy.

Importing optional SDKs (``openai``, ``httpx``) is done lazily inside the
function so the core package stays importable with zero dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .exceptions import (
    ModelCallError,
    ModelConnectionError,
    ModelFatalError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
)

if TYPE_CHECKING:  # pragma: no cover
    pass


def _openai_classify(exc: BaseException) -> "ModelCallError | None":
    try:
        import openai
    except ImportError:
        return None
    if isinstance(exc, openai.APITimeoutError):
        return ModelTimeoutError("upstream timeout", cause=exc)
    if isinstance(exc, openai.APIConnectionError):
        return ModelConnectionError("upstream connection error", cause=exc)
    if isinstance(exc, openai.RateLimitError):
        retry_after = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            rh = getattr(resp, "headers", None)
            if rh and "retry-after" in rh:
                try:
                    retry_after = float(rh["retry-after"])
                except (TypeError, ValueError):
                    retry_after = None
        return ModelRateLimitError("rate limited (429)", retry_after=retry_after, cause=exc)
    if isinstance(exc, openai.APIStatusError):
        code = getattr(exc, "status_code", None)
        if code == 429:
            return ModelRateLimitError("rate limited (429)", cause=exc)
        if isinstance(code, int) and 500 <= code < 600:
            return ModelServerError(f"upstream 5xx ({code})", status_code=code, cause=exc)
        if isinstance(code, int) and 400 <= code < 500:
            return ModelFatalError(f"request rejected ({code})", status_code=code, cause=exc)
        return ModelServerError(f"upstream error ({code})", status_code=code, cause=exc)
    if isinstance(exc, openai.APIError):
        return ModelServerError("upstream API error", cause=exc)
    return None


def _httpx_classify(exc: BaseException) -> "ModelCallError | None":
    try:
        import httpx
    except ImportError:
        return None
    if isinstance(exc, httpx.TimeoutException):
        return ModelTimeoutError("http timeout", cause=exc)
    if isinstance(exc, (httpx.ConnectError, httpx.TransportError, httpx.NetworkError)):
        return ModelConnectionError("http connection error", cause=exc)
    if isinstance(exc, httpx.HTTPStatusError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code == 429:
            return ModelRateLimitError("rate limited (429)", cause=exc)
        if isinstance(code, int) and 500 <= code < 600:
            return ModelServerError(f"upstream 5xx ({code})", status_code=code, cause=exc)
        if isinstance(code, int) and 400 <= code < 500:
            return ModelFatalError(f"request rejected ({code})", status_code=code, cause=exc)
    return None


def default_classify(exc: BaseException) -> ModelCallError:
    """Normalize *any* exception into a :class:`ModelCallError`.

    Already-typed errors pass through untouched. Known SDKs (openai, httpx) are
    mapped by status/type. Generic ``TimeoutError``/``ConnectionError`` become
    timeout/connection. Everything else is treated as a transient server error
    (safest default: we'd rather retry/switch than abort a good request).
    """
    if isinstance(exc, ModelCallError):
        return exc

    for fn in (_openai_classify, _httpx_classify):
        typed = fn(exc)
        if typed is not None:
            return typed

    if isinstance(exc, TimeoutError):
        return ModelTimeoutError("timeout", cause=exc)
    if isinstance(exc, ConnectionError):
        return ModelConnectionError("connection error", cause=exc)
    if isinstance(exc, (OSError,)) and getattr(exc, "errno", None) in (101, 111, 113, 104):
        return ModelConnectionError("connection error", cause=exc)

    return ModelServerError(f"unclassified error: {exc}", cause=exc)
