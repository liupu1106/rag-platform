"""Typed failure conditions for model calls.

Every error that can bubble up from a model invocation is normalized into one
of these classes so the manager can decide *what to do* (retry? switch? abort?)
without depending on any specific SDK's exception hierarchy.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
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


class FailureKind:
    """Classification of a failure, used by the manager to pick a strategy.

    TRANSIENT: worth retrying on the same model, and worth switching to a
               backup if retries are exhausted (timeout / connection / 429 / 5xx).
    FATAL:     request-level problem that will fail on *every* model with the
               same arguments (bad request, auth rejected, content policy).
               Abort immediately, do not switch.
    """

    TRANSIENT = "transient"
    FATAL = "fatal"


class ModelCallError(Exception):
    """Base class for all model-call failures."""

    #: Filled in by the classifier; lets the manager branch on kind.
    kind: str = FailureKind.TRANSIENT

    def __init__(
        self,
        message: str,
        *,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.status_code = status_code
        self.cause = cause

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        base = super().__str__()
        if self.model:
            return f"[{self.model}] {base}"
        return base


class ModelTimeoutError(ModelCallError):
    """The model did not respond within the allotted time."""


class ModelConnectionError(ModelCallError):
    """TCP/DNS/TLS level failure talking to the model endpoint."""


class ModelRateLimitError(ModelCallError):
    """The model returned 429 (or an SDK rate-limit signal)."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class ModelServerError(ModelCallError):
    """The model returned a 5xx (or an SDK server-error signal)."""


class ModelFatalError(ModelCallError):
    """Request-level error that will fail identically on every model.

    Examples: 400 bad request, 401/403 auth rejected, content-policy block.
    The manager aborts the whole call rather than burning backups.
    """

    kind = FailureKind.FATAL


class AllModelsFailed(ModelCallError):
    """Every configured model was tried and none succeeded.

    ``attempted`` lists the model names that were actually invoked (in order),
    so callers can see the full degradation path. ``cause`` carries the last
    underlying error if one exists.
    """

    def __init__(
        self,
        attempted: list[str],
        message: str = "All configured models failed",
        *,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message, cause=cause)
        self.attempted = attempted


class AllModelsUnavailable(AllModelsFailed):
    """No model was even eligible to be called (all circuits open & cooling).

    ``retry_after`` is a suggested backoff (seconds) before trying again.
    """

    def __init__(
        self,
        attempted: list[str],
        *,
        retry_after: Optional[float] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        msg = "No model available right now"
        if retry_after is not None:
            msg += f" (retry after ~{retry_after:.1f}s)"
        super().__init__(attempted, msg, cause=cause)
        self.retry_after = retry_after
