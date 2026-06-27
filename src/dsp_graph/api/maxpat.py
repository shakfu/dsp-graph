"""Max/MSP ``.maxpat`` test-patch export endpoint.

Transpiles a graph to a gen~ codebox and wraps it in a ready-to-open Max test
patch (see :mod:`dsp_graph.maxpat`). Like the GenExpr transpiler it builds on,
this is experimental and only available when the server is started with
``--experimental``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from gen_dsp.graph.models import Graph
from gen_dsp.graph.transpile import GenExprUnsupportedError, transpile_to_genexpr
from pydantic import BaseModel

from dsp_graph.config import is_experimental
from dsp_graph.maxpat import graph_to_maxpat

router = APIRouter()


class MaxpatRequest(BaseModel):
    graph: dict[str, Any]


class MaxpatResponse(BaseModel):
    maxpat_json: str
    filename: str


def _safe_filename(name: str) -> str:
    """Turn a graph name into a safe ``<name>.maxpat`` filename."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "graph"
    return f"{stem}.maxpat"


@router.post("/export/maxpat", response_model=MaxpatResponse)
async def export_maxpat(req: MaxpatRequest) -> MaxpatResponse:
    """Transpile a Graph and return a Max ``.maxpat`` test patch as JSON text.

    The GenExpr transpiler is experimental, so this endpoint returns 404 unless
    the server was started with ``--experimental``.
    """
    if not is_experimental():
        raise HTTPException(
            status_code=404,
            detail="Max patch export is experimental; start the server with "
            "--experimental to enable it.",
        )

    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        genexpr_source = transpile_to_genexpr(g)
    except GenExprUnsupportedError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Graph cannot be transpiled to GenExpr: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        patch = graph_to_maxpat(g, genexpr_source)
    except Exception as exc:  # pragma: no cover - defensive: py2max build failure
        raise HTTPException(status_code=500, detail=f"Failed to build Max patch: {exc}") from exc

    return MaxpatResponse(
        maxpat_json=json.dumps(patch, indent=4),
        filename=_safe_filename(g.name or "graph"),
    )
