"""Retry / backoff helpers (pure, no I/O)."""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["compute_backoff"]


def compute_backoff(
    attempt: int,
    *,
    base: float = 0.5,
    factor: float = 2.0,
    max_backoff: float = 8.0,
    jitter: float = 0.1,
) -> float:
    """Exponential backoff with a touch of jitter.

    attempt 0 -> ~base, attempt 1 -> ~base*factor, capped at ``max_backoff``.
    ``jitter`` randomizes within +/- that fraction to avoid thundering herds
    when many callers share the same model. The jitter term is deterministic
    here (caller may pass a precomputed value) but the manager feeds in a
    random component.
    """
    exp = min(base * (factor ** max(attempt, 0)), max_backoff)
    # jitter is applied by the caller via the optional param below.
    return exp


def with_jitter(delay: float, jitter: float, rand: float) -> float:
    """Return ``delay`` perturbed by +/- ``jitter`` fraction.

    ``rand`` is expected in [0, 1) so this stays pure/testable.
    """
    spread = delay * jitter
    return max(0.0, delay - spread + 2 * spread * rand)


def next_retry_delay(
    attempt: int,
    *,
    base: float = 0.5,
    factor: float = 2.0,
    max_backoff: float = 8.0,
    jitter: float = 0.1,
    rand: float = 0.5,
) -> float:
    """Convenience: backoff + jitter for ``attempt`` (0-based)."""
    return with_jitter(compute_backoff(attempt, base=base, factor=factor,
                                       max_backoff=max_backoff), jitter, rand)
