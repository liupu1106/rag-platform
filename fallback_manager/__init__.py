"""fallback_manager — resilient model-calling with automatic fallback.

Drop-in wrapper around any OpenAI-compatible chat call. On failure (timeout,
connection error, rate limit, 5xx) it retries with backoff, switches to the next
model in priority order, opens a circuit breaker to shed a dead model, and
automatically fails back to the primary once it recovers. To the caller it is
fully transparent: ``manager.chat(messages=...)`` just returns the response.
"""

from __future__ import annotations

from .alerting import (
    AlertEvent,
    AlertKind,
    AlertSink,
    CompositeAlertSink,
    LoggingAlertSink,
    WebhookAlertSink,
)
from .caller import (
    CallableCaller,
    ModelCaller,
    OpenAICaller,
    default_caller_factory,
)
from .circuit import CircuitBreaker
from .classify import default_classify
from .config import HealthState, ModelConfig, ModelHealth, ModelRegistry
from .exceptions import (
    AllModelsFailed,
    AllModelsUnavailable,
    FailureKind,
    ModelCallError,
    ModelConnectionError,
    ModelFatalError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
)
from .manager import ModelFallbackManager
from .retry import compute_backoff, next_retry_delay, with_jitter

__all__ = [
    "ModelFallbackManager",
    "ModelConfig",
    "ModelRegistry",
    "ModelHealth",
    "HealthState",
    "CircuitBreaker",
    "ModelCaller",
    "OpenAICaller",
    "CallableCaller",
    "default_caller_factory",
    "default_classify",
    "compute_backoff",
    "next_retry_delay",
    "with_jitter",
    "AlertSink",
    "AlertEvent",
    "AlertKind",
    "LoggingAlertSink",
    "WebhookAlertSink",
    "CompositeAlertSink",
    "ModelCallError",
    "ModelTimeoutError",
    "ModelConnectionError",
    "ModelRateLimitError",
    "ModelServerError",
    "ModelFatalError",
    "AllModelsFailed",
    "AllModelsUnavailable",
    "FailureKind",
]
