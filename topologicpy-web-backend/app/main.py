"""Topologic Studio Next - FastAPI application.

Run locally:

    uvicorn app.main:app --reload --port 8000

Classic's ``main.py`` was a single 2 800-line module that opened with a
``sys.path.append`` to one developer's home directory. This one is an app
factory that wires up routers; every piece of logic lives in a module that can
be imported and tested on its own.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .config import settings
from .models import HealthResponse
from .store import store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("topologic-studio")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.title,
        version=settings.version,
        description=(
            "Navigation-graph, egress-path and fire-simulation API for "
            "Topologic Studio Next."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Graph payloads are base64 arrays; gzip roughly halves them again. SSE is
    # excluded automatically because its chunks fall under the size threshold.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    from .routers import ifc, simulation

    app.include_router(ifc.router)
    app.include_router(simulation.router)

    # The legacy TopologicPy JSON contract needs the native topologic backend.
    # If that is missing the rest of the API must still start, so the import is
    # guarded and the failure is reported through /health.
    try:
        from .legacy import contract

        app.include_router(contract.router)
        app.state.legacy_available = True
        app.state.legacy_error = None
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Legacy topology router unavailable: %s", exc)
        app.state.legacy_available = False
        app.state.legacy_error = str(exc)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=settings.version,
            topologicpy=_topologicpy_info(),
            ifcopenshell=_ifcopenshell_version(),
            graphs=store.stats(),
        )

    @app.get("/api/capabilities", tags=["meta"])
    def capabilities() -> Dict[str, Any]:
        """What this backend can actually do right now.

        The frontend reads this on boot so it can disable features the server
        cannot serve, instead of surfacing a 500 when the user clicks.
        """
        topo = _topologicpy_info()
        return {
            "version": settings.version,
            "engines": ["fast"] + (["topologicpy"] if topo.get("usable") else []),
            "fire_models": ["radial", "flood", "temperature"],
            "legacy_topology": bool(getattr(app.state, "legacy_available", False)),
            "legacy_error": getattr(app.state, "legacy_error", None),
            "server_side_ifc": _ifcopenshell_version() is not None,
            "limits": {
                "max_graph_nodes": settings.max_graph_nodes,
                "max_upload_mib": settings.max_upload_mib,
                "graph_cache_size": settings.graph_cache_size,
            },
            "topologicpy": topo,
        }

    return app


def _topologicpy_info() -> Dict[str, Any]:
    try:
        from .graphs.topologic_engine import version_info

        return version_info()
    except Exception as exc:  # pragma: no cover
        return {"usable": False, "error": str(exc)}


def _ifcopenshell_version():
    try:
        import ifcopenshell

        return getattr(ifcopenshell, "version", "installed")
    except Exception:
        return None


app = create_app()
