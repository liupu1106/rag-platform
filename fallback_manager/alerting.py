"""Logging + alert notifications for switch / failback / total-failure events.

Alerts fire on *state transitions*, not on every single failure, so a flaky
model does not generate an alert storm. The manager decides *when* to alert;
this module decides *how* (log, webhook, or both).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, Sequence

__all__ = [
    "AlertKind",
    "AlertEvent",
    "AlertSink",
    "LoggingAlertSink",
    "WebhookAlertSink",
    "CompositeAlertSink",
]


class AlertKind(str, Enum):
    #: Primary/healthy model went down -> switched away (healthy->unhealthy).
    MODEL_DOWN = "model_down"
    #: An unhealthy model recovered -> failback path re-enabled (unhealthy->healthy).
    MODEL_RECOVERED = "model_recovered"
    #: A request moved from one model to the next within a single call.
    SWITCH = "switch"
    #: Every model failed for a request.
    ALL_FAILED = "all_failed"
    #: No model was even eligible (all circuits cooling).
    ALL_UNAVAILABLE = "all_unavailable"


@dataclass
class AlertEvent:
    kind: AlertKind
    model: Optional[str] = None
    request_id: Optional[str] = None
    detail: str = ""
    tried: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "model": self.model,
            "request_id": self.request_id,
            "detail": self.detail,
            "tried": self.tried,
            "timestamp": self.timestamp,
            **self.extra,
        }


class AlertSink(Protocol):
    def send(self, event: AlertEvent) -> None: ...  # pragma: no cover - interface


class LoggingAlertSink:
    """Default sink: emits structured records via the standard logging module."""

    def __init__(self, logger: Optional[logging.Logger] = None, level: int = logging.WARNING) -> None:
        self._logger = logger or logging.getLogger("fallback_manager.alerts")
        self._level = level

    def send(self, event: AlertEvent) -> None:
        msg = f"[{event.kind.value}] model={event.model or '-'} rid={event.request_id or '-'} {event.detail}"
        if event.tried:
            msg += f" | tried={','.join(event.tried)}"
        self._logger.log(self._level, msg)


class WebhookAlertSink:
    """POSTs each event as JSON to a URL (e.g. Slack/Feishu/Alertmanager)."""

    def __init__(self, url: str, *, timeout: float = 5.0, headers: Optional[dict] = None) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def send(self, event: AlertEvent) -> None:
        payload = json.dumps(event.to_dict(), default=str).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload, headers=self.headers, method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout)  # nosec - user-configured
        except Exception as exc:  # pragma: no cover - network
            logging.getLogger("fallback_manager.alerts").error(
                "webhook alert failed: %s", exc
            )


class CompositeAlertSink:
    """Fan an event out to multiple sinks."""

    def __init__(self, sinks: Sequence[AlertSink]) -> None:
        self._sinks = list(sinks)

    def add(self, sink: AlertSink) -> None:
        self._sinks.append(sink)

    def send(self, event: AlertEvent) -> None:
        for sink in self._sinks:
            try:
                sink.send(event)
            except Exception:  # pragma: no cover - sink isolation
                logging.getLogger("fallback_manager.alerts").exception(
                    "alert sink %r raised", sink
                )
