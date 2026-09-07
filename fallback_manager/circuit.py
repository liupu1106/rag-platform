"""Per-model circuit breaker + recovery probing.

This module is intentionally *stateful but dumb*: it only answers two questions
under a lock — "is this model available to try right now?" and "please record a
result". The orchestration (which model to pick next, sleeping, alerting) lives
in the manager. Keeping the state here makes it easy to reason about and test.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .config import HealthState, ModelHealth, ModelRegistry

__all__ = ["CircuitBreaker"]


class CircuitBreaker:
    """Tracks health per model and decides availability + failback probing.

    Policy:
      * After ``failure_threshold`` *consecutive transient* failures the model
        is marked UNHEALTHY and its circuit opens for ``cooldown`` seconds.
      * While open and cooling, ``is_available`` returns False (skip it).
      * When cooldown elapses, the model becomes a *probe candidate*: the next
        single attempt is allowed (and flagged ``probing``) to test recovery.
      * A success resets everything to HEALTHY -> this is the automatic failback
        (the highest-priority model is now chosen first again).
      * A probe failure re-opens the circuit for another cooldown (quietly).
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        failure_threshold: int = 3,
        cooldown: float = 60.0,
        degrade_threshold: int = 1,
        clock: Optional[callable] = None,
    ) -> None:
        self._registry = registry
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.degrade_threshold = degrade_threshold
        self._clock = clock or time.time
        self._lock = threading.Lock()

    # -- queries --------------------------------------------------------
    def is_available(self, name: str) -> bool:
        """Whether the model may be attempted now (healthy, or probe-ready)."""
        h = self._registry.health(name)
        if h.circuit_open_until is None:
            return True
        return self._clock() >= h.circuit_open_until

    def reserve_probe(self, name: str) -> bool:
        """Atomically claim the single probe slot if the circuit is ready.

        Returns True exactly once per cooldown window. Callers that get True
        should attempt one trial call and then call ``record_success`` or
        ``record_failure`` to release the slot.
        """
        with self._lock:
            h = self._registry.health(name)
            if h.circuit_open_until is None or self._clock() < h.circuit_open_until:
                return False
            if h.probing:
                return False
            h.probing = True
            return True

    def release_probe(self, name: str) -> None:
        with self._lock:
            self._registry.health(name).probing = False

    # -- mutations ------------------------------------------------------
    def record_success(self, name: str) -> bool:
        """Record a success; returns True if this was a *recovery* (failback)."""
        with self._lock:
            h = self._registry.health(name)
            was_down = h.state != HealthState.HEALTHY
            h.consecutive_failures = 0
            h.circuit_open_until = None
            h.probing = False
            h.state = HealthState.HEALTHY
            h.total_success += 1
            h.last_success_at = self._clock()
            return was_down

    def record_failure(self, name: str, *, transient: bool, probe: bool = False) -> str:
        """Record a failure; returns the resulting state name.

        For transient failures we increment the consecutive counter; once it
        reaches ``failure_threshold`` the circuit opens. For non-transient
        (fatal) failures we do not open the circuit (the request is aborted
        anyway) but we still count it.

        A *failed probe* (a trial call made while the circuit was cooling)
        immediately re-opens the circuit for another cooldown, so a still-down
        model does not get hammered every request.
        """
        with self._lock:
            h = self._registry.health(name)
            h.total_failure += 1
            h.last_failure_at = self._clock()
            if not transient:
                h.probing = False
                return h.state.value
            if probe:
                # Trial call while cooling failed -> re-open for another cooldown.
                h.state = HealthState.UNHEALTHY
                h.circuit_open_until = self._clock() + self.cooldown
                h.consecutive_failures = 0
                h.probing = False
                return h.state.value
            h.consecutive_failures += 1
            if h.state == HealthState.HEALTHY and h.consecutive_failures >= self.degrade_threshold:
                h.state = HealthState.DEGRADED
            if h.consecutive_failures >= self.failure_threshold:
                h.state = HealthState.UNHEALTHY
                h.circuit_open_until = self._clock() + self.cooldown
                h.consecutive_failures = 0
                h.probing = False
            return h.state.value

    def retry_after(self, name: str) -> Optional[float]:
        """Seconds until the circuit for ``name`` becomes probe-available."""
        h = self._registry.health(name)
        if h.circuit_open_until is None:
            return None
        return max(0.0, h.circuit_open_until - self._clock())

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {
                name: {
                    "state": h.state.value,
                    "consecutive_failures": h.consecutive_failures,
                    "total_success": h.total_success,
                    "total_failure": h.total_failure,
                    "circuit_open_until": h.circuit_open_until,
                    "last_failure_at": h.last_failure_at,
                    "last_success_at": h.last_success_at,
                }
                for name, h in self._registry.all_health().items()
            }
