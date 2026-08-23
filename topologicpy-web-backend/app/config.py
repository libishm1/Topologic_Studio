"""Runtime configuration for the Topologic Studio Next backend.

Everything here is driven by environment variables so the same image runs
locally, in Docker and on Render without code edits.  The classic backend
carried a hard-coded ``sys.path.append`` to a developer's machine; that is
deliberately gone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


@dataclass(frozen=True)
class Settings:
    title: str = "Topologic Studio Next API"
    version: str = "0.2.0"

    perf_log: bool = field(default_factory=lambda: _env_bool("PERF_LOG", False))

    #: Maximum number of graphs kept in memory at once (LRU eviction).
    graph_cache_size: int = field(default_factory=lambda: _env_int("GRAPH_CACHE_SIZE", 8))
    #: Seconds before an idle graph is evicted.
    graph_ttl_seconds: float = field(default_factory=lambda: _env_float("GRAPH_TTL_SECONDS", 3600.0))

    #: Hard ceiling on nodes in a navigation graph, protects the server from
    #: pathological uploads.
    max_graph_nodes: int = field(default_factory=lambda: _env_int("MAX_GRAPH_NODES", 60000))
    #: Hard ceiling on the raw upload size accepted by /upload-ifc, in MiB.
    max_upload_mib: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MIB", 256))

    @property
    def cors_origins(self) -> List[str]:
        origins = list(DEFAULT_ORIGINS)
        extra = os.getenv("CORS_ORIGINS", "")
        for item in extra.split(","):
            item = item.strip()
            if item and item not in origins:
                origins.append(item)
        return origins


settings = Settings()
