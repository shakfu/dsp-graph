"""FastAPI application with SPA static serving."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dsp_graph.api import build as build_api
from dsp_graph.api import compile as compile_api
from dsp_graph.api import generate as generate_api
from dsp_graph.api import graph as graph_api
from dsp_graph.api import layout as layout_api
from dsp_graph.api import optimize as optimize_api
from dsp_graph.api import simulate as simulate_api
from dsp_graph.security import (
    SESSION_TOKEN,
    BodySizeLimitMiddleware,
    SessionTokenMiddleware,
)

# Static files (React build output)
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    if STATIC_DIR.is_dir() and (STATIC_DIR / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
    yield


app = FastAPI(title="dsp-graph", version="0.2.0", lifespan=lifespan)

# Reject oversized bodies first, then require a per-session token on
# state-changing requests (anti-CSRF for the localhost server). Middleware runs
# in reverse registration order, so the body-size check executes first.
app.add_middleware(SessionTokenMiddleware)
app.add_middleware(BodySizeLimitMiddleware)


@app.get("/api/session")
async def _session() -> dict[str, str]:
    """Return the per-process session token (readable only same-origin)."""
    return {"token": SESSION_TOKEN}


# Mount API routers
app.include_router(graph_api.router, prefix="/api/graph", tags=["graph"])
app.include_router(simulate_api.router, prefix="/api", tags=["simulate"])
app.include_router(optimize_api.router, prefix="/api", tags=["optimize"])
app.include_router(compile_api.router, prefix="/api", tags=["compile"])
app.include_router(generate_api.router, prefix="/api", tags=["generate"])
app.include_router(build_api.router, prefix="/api", tags=["build"])
app.include_router(layout_api.router, prefix="/api", tags=["layout"])


@app.get("/")
async def _index() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    from fastapi.responses import HTMLResponse

    return HTMLResponse(  # type: ignore[return-value]
        "<h3>dsp-graph</h3><p>Frontend not built. "
        "Run <code>make frontend-build</code> then reload.</p>"
    )


@app.get("/{path:path}")
async def _spa_fallback(path: str) -> FileResponse:
    """Serve static files or fall back to index.html for SPA routing."""
    static_root = STATIC_DIR.resolve()
    candidate = (static_root / path).resolve()
    # Reject path traversal: the resolved target must stay within STATIC_DIR.
    if candidate.is_relative_to(static_root) and candidate.is_file():
        return FileResponse(candidate)
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    from fastapi.responses import HTMLResponse

    return HTMLResponse(  # type: ignore[return-value]
        "<h3>dsp-graph</h3><p>Frontend not built.</p>"
    )
