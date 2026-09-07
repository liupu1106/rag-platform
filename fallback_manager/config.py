"""Model configuration, priority management, and per-model health state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Optional

__all__ = [
    "HealthState",
    "ModelConfig",
    "ModelHealth",
    "ModelRegistry",
]


class HealthState(str, Enum):
    """Per-model health, drives availability and failback probing."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # some recent failures, still in use
    UNHEALTHY = "unhealthy"  # circuit open, cooling down / probing


@dataclass
class ModelConfig:
    """One model endpoint and its per-model policies.

    Lower ``priority`` wins. Ties are broken by insertion order for stability.
    All time/retry fields are *per single request* budgets; circuit-level
    behavior (across requests) lives in :class:`ModelHealth`.
    """

    name: str
    priority: int = 0
    #: Logical id used for health/circuit tracking. Distinct from ``model``
    #: so you can register e.g. name="primary" but call model="gpt-4o-mini".
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    #: The actual model id sent to the upstream API (falls back to ``name``).
    model: Optional[str] = None
    #: Protocol family: "openai" (covers ollama/deepseek/qwen/.../chat/completions)
    #: or "anthropic" (/v1/messages). The caller factory dispatches on this.
    provider: str = "openai"
    timeout: float = 30.0
    max_retries: int = 2
    enabled: bool = True
    #: Free-form tags (e.g. "primary", "backup", "cheap") — handy for alerting.
    tags: dict[str, str] = field(default_factory=dict)
    #: Extra kwargs forwarded to the SDK client (e.g. organization, default_headers).
    client_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ModelHealth:
    """Mutable runtime state for one model (guarded by the manager's lock)."""

    state: HealthState = HealthState.HEALTHY
    consecutive_failures: int = 0
    total_success: int = 0
    total_failure: int = 0
    last_failure_at: Optional[float] = None
    last_success_at: Optional[float] = None
    #: When the circuit re-opens for probing; None means not open.
    circuit_open_until: Optional[float] = None
    #: True while a single probe attempt is in flight (prevents probe stampede).
    probing: bool = False


class ModelRegistry:
    """Holds the ordered set of models and their live health.

    Thread-safety note: mutating health counters is the manager's job and is
    done under its lock. The registry only provides ordered iteration and
    lookups; it does not mutate health on its own.
    """

    def __init__(self, models: Iterable[ModelConfig]) -> None:
        self._models: list[ModelConfig] = []
        self._health: dict[str, ModelHealth] = {}
        for m in models:
            self.add(m)

    def add(self, model: ModelConfig) -> None:
        if any(m.name == model.name for m in self._models):
            raise ValueError(f"Duplicate model name: {model.name!r}")
        self._models.append(model)
        self._health[model.name] = ModelHealth()

    # -- ordered access -------------------------------------------------
    def iter_priority(self) -> Iterator[ModelConfig]:
        """Yield enabled models in (priority, insertion) order."""
        yield from sorted(
            (m for m in self._models if m.enabled),
            key=lambda m: (m.priority, self._models.index(m)),
        )

    def health(self, name: str) -> ModelHealth:
        return self._health[name]

    def all_health(self) -> dict[str, ModelHealth]:
        return dict(self._health)

    @property
    def size(self) -> int:
        return len(self._models)

    def enabled_size(self) -> int:
        return sum(1 for m in self._models if m.enabled)

    @classmethod
    def from_list(cls, data: Iterable[dict[str, Any]]) -> "ModelRegistry":
        return cls([ModelConfig.from_dict(d) for d in data])

    def is_eligible(self, model: ModelConfig, now: Optional[float] = None) -> bool:
        """A model is eligible if enabled and its circuit is not cooling.

        When the circuit's cooldown has elapsed it becomes *eligible as a probe*
        (the manager will attempt exactly one trial call to test recovery).
        """
        if not model.enabled:
            return False
        h = self._health[model.name]
        if h.circuit_open_until is None:
            return True
        now = time.time() if now is None else now
        return now >= h.circuit_open_until
