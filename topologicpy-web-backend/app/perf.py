"""Lightweight timing helpers.

Enabled with ``PERF_LOG=1``.  Kept dependency-free on purpose: the perf
markers are the only way we can compare the Classic and Next pipelines on
the same machine.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator

from .config import settings


def log(route: str, **timings) -> None:
    if not settings.perf_log:
        return
    parts = []
    for key, value in timings.items():
        parts.append(f"{key}={value:.3f}s" if isinstance(value, float) else f"{key}={value}")
    print(f"PERF route={route} " + " ".join(parts), flush=True)


class Timer:
    """Accumulates named phase timings for a single request."""

    def __init__(self, route: str):
        self.route = route
        self._t0 = time.perf_counter()
        self.phases: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - start)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def emit(self, **extra) -> None:
        log(self.route, **self.phases, total=self.elapsed, **extra)
