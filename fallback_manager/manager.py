"""The orchestrator: picks a model, retries, switches, and fails back.

``ModelFallbackManager`` is the single object callers interact with. The public
surface is deliberately tiny:

    resp = manager.chat(messages=[...])          # transparent to the caller
    manager.last_used_model                      # which model actually served it
    manager.stats()                              # observability

Everything else (priority ordering, retries, circuit breaking, failback,
alerting) is internal. A caller that worked against a single OpenAI model keeps
working after you wrap it in this manager — that is the "transparent" guarantee.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Any, Callable, Optional

from .alerting import (
    AlertEvent,
    AlertKind,
    AlertSink,
    CompositeAlertSink,
    LoggingAlertSink,
)
from .caller import ModelCaller, default_caller_factory
from .circuit import CircuitBreaker
from .classify import default_classify
from .config import HealthState, ModelConfig, ModelRegistry
from .exceptions import (
    AllModelsFailed,
    AllModelsUnavailable,
    FailureKind,
    ModelCallError,
    ModelFatalError,
)
from .retry import compute_backoff

__all__ = ["ModelFallbackManager"]


class ModelFallbackManager:
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        caller_factory: Callable[[ModelConfig], ModelCaller] = default_caller_factory,
        classifier: Callable[[BaseException], ModelCallError] = default_classify,
        alert_sink: Optional[AlertSink] = None,
        # circuit / policy
        failure_threshold: int = 3,
        cooldown: float = 60.0,
        degrade_threshold: int = 1,
        # retry / timing
        total_timeout: Optional[float] = 120.0,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
        backoff_max: float = 8.0,
        backoff_jitter: float = 0.1,
        # misc
        logger: Optional[logging.Logger] = None,
        request_id_factory: Callable[[], str] = lambda: f"rid-{random.randint(10**8, 10**9 - 1)}",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if registry.enabled_size() == 0:
            raise ValueError("registry must contain at least one enabled model")

        self._registry = registry
        self._caller_factory = caller_factory
        self._classify = classifier
        self._circuit = CircuitBreaker(
            registry,
            failure_threshold=failure_threshold,
            cooldown=cooldown,
            degrade_threshold=degrade_threshold,
            clock=clock,
        )
        self._alert = CompositeAlertSink(
            [alert_sink] if alert_sink is not None else [LoggingAlertSink(logger)]
        )
        self.total_timeout = total_timeout
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self.backoff_max = backoff_max
        self.backoff_jitter = backoff_jitter
        self.logger = logger or logging.getLogger("fallback_manager")
        self._request_id = request_id_factory
        self._sleep = sleep
        self._clock = clock

        self._callers: dict[str, ModelCaller] = {}
        self._callers_lock = threading.Lock()
        self._last_used_model: Optional[str] = None
        self._last_meta_lock = threading.Lock()

    # -- public API -----------------------------------------------------
    @property
    def last_used_model(self) -> Optional[str]:
        return self._last_used_model

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """OpenAI-compatible entry point. Returns whatever the caller returns."""
        return self._invoke(messages=messages, **kwargs)

    def invoke(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """Generic entry point with explicit keyword args."""
        return self._invoke(messages=messages, **kwargs)

    def stats(self) -> dict[str, Any]:
        """Observability snapshot: per-model health + last served model."""
        return {
            "last_used_model": self._last_used_model,
            "models": self._circuit.snapshot(),
        }

    def mark_healthy(self, name: str) -> None:
        """Ops escape hatch: manually clear a model's circuit (e.g. after fix)."""
        with self._circuit._lock:
            h = self._registry.health(name)
            h.state = HealthState.HEALTHY
            h.consecutive_failures = 0
            h.circuit_open_until = None
            h.probing = False

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Take a model in/out of rotation without rebuilding the manager."""
        for m in self._registry._models:
            if m.name == name:
                m.enabled = enabled
                return
        raise KeyError(name)

    # -- internals ------------------------------------------------------
    def _caller_for(self, cfg: ModelConfig) -> ModelCaller:
        with self._callers_lock:
            c = self._callers.get(cfg.name)
            if c is None:
                c = self._caller_factory(cfg)
                self._callers[cfg.name] = c
            return c

    def _backoff(self, attempt: int) -> float:
        base = compute_backoff(
            attempt, base=self.backoff_base, factor=self.backoff_factor,
            max_backoff=self.backoff_max,
        )
        spread = base * self.backoff_jitter
        # deterministic-ish jitter in [base-spread, base+spread]
        return max(0.0, base - spread + 2 * spread * random.random())

    def _emit(self, kind: AlertKind, **kw: Any) -> None:
        self._alert.send(AlertEvent(kind=kind, **kw))

    def _set_last(self, name: str) -> None:
        with self._last_meta_lock:
            self._last_used_model = name

    def _invoke(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        rid = self._request_id()
        now = self._clock()
        deadline = now + self.total_timeout if self.total_timeout else math.inf
        attempted: list[str] = []
        last_exc: Optional[ModelCallError] = None

        for cfg in self._registry.iter_priority():
            # Circuit open & still cooling -> skip without calling.
            if not self._circuit.is_available(cfg.name):
                continue

            probe = self._circuit.reserve_probe(cfg.name)
            attempted.append(cfg.name)
            caller = self._caller_for(cfg)

            for attempt in range(cfg.max_retries + 1):
                if self._clock() > deadline:
                    break
                try:
                    resp = caller(model=cfg.name, messages=messages, **kwargs)
                except Exception as raw:  # classifier decides what this means
                    err = self._classify(raw)

                    if err.kind == FailureKind.FATAL:
                        # Request-level problem: will fail on every model.
                        # Abort immediately; do not burn backups.
                        self._circuit.record_failure(cfg.name, transient=False)
                        if probe:
                            self._circuit.release_probe(cfg.name)
                        raise ModelFatalError(
                            str(err), model=cfg.name,
                            status_code=err.status_code, cause=raw,
                        ) from raw

                    # Transient: record, maybe retry, else switch to next model.
                    prev = self._registry.health(cfg.name).state
                    new_state = self._circuit.record_failure(cfg.name, transient=True, probe=probe)
                    last_exc = err
                    if prev != HealthState.UNHEALTHY and new_state == HealthState.UNHEALTHY and not probe:
                        self._emit(
                            AlertKind.MODEL_DOWN, model=cfg.name, request_id=rid,
                            detail="primary/healthy model became unavailable",
                            extra={"attempt": attempt},
                        )
                    if attempt < cfg.max_retries:
                        delay = self._backoff(attempt)
                        if self._clock() + delay <= deadline:
                            self._sleep(delay)
                            continue
                    # retries exhausted on this model -> break to switch
                    break
                else:
                    # Success path: record health, failback alert, return.
                    recovered = self._circuit.record_success(cfg.name)
                    if probe:
                        self._circuit.release_probe(cfg.name)
                    if recovered:
                        self._emit(
                            AlertKind.MODEL_RECOVERED, model=cfg.name, request_id=rid,
                            detail="failback: model recovered, back in rotation",
                        )
                    self._set_last(cfg.name)
                    self.logger.info("rid=%s served by %s (probe=%s)", rid, cfg.name, probe)
                    return resp

            # We get here only if the model was exhausted (all retries failed).
            if probe:
                self._circuit.release_probe(cfg.name)

            remaining = [
                c.name for c in self._registry.iter_priority()
                if c.name != cfg.name and self._circuit.is_available(c.name)
            ]
            if remaining:
                self._emit(
                    AlertKind.SWITCH, model=cfg.name, request_id=rid,
                    detail=f"switched away from {cfg.name} to backups", tried=attempted,
                )
            # continue to next model in priority order

        # No model succeeded.
        if not attempted:
            retry_after = min(
                (self._circuit.retry_after(c.name) for c in self._registry.iter_priority()
                 if self._circuit.retry_after(c.name) is not None),
                default=None,
            )
            self._emit(
                AlertKind.ALL_UNAVAILABLE, request_id=rid,
                detail="all models cooling (circuits open)", tried=[],
                extra={"retry_after": retry_after},
            )
            raise AllModelsUnavailable([], retry_after=retry_after)

        self._emit(
            AlertKind.ALL_FAILED, request_id=rid,
            detail="all models failed", tried=attempted,
            extra={"last_error": str(last_exc)},
        )
        raise AllModelsFailed(attempted, cause=last_exc)
