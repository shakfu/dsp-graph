"""GenExpr (gen~ codebox) transpile preview endpoint.

Re-emits a DSP graph as gen~ ``codebox`` source for pasting into Max/MSP, via
gen-dsp's experimental :func:`gen_dsp.graph.transpile.transpile_to_genexpr`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from gen_dsp.graph.models import Graph
from gen_dsp.graph.transpile import GenExprUnsupportedError, transpile_to_genexpr
from pydantic import BaseModel

from dsp_graph.config import is_experimental

router = APIRouter()


class GenExprRequest(BaseModel):
    graph: dict[str, Any]


class GenExprResponse(BaseModel):
    genexpr_source: str


@router.post("/genexpr", response_model=GenExprResponse)
async def genexpr_endpoint(req: GenExprRequest) -> GenExprResponse:
    """Transpile a Graph to gen~ codebox (GenExpr) source and return it.

    The GenExpr transpiler is experimental: it is only available when the server
    is started with ``--experimental``. Otherwise this endpoint returns 404 so
    the (also-hidden) UI tab has nothing to call.
    """
    if not is_experimental():
        raise HTTPException(
            status_code=404,
            detail="GenExpr transpile is experimental; start the server with "
            "--experimental to enable it.",
        )
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        source = transpile_to_genexpr(g)
    except GenExprUnsupportedError as exc:
        # Some node types (e.g. fast* approximations, sine-filled buffers) have no
        # faithful gen~ equivalent; report it as a clear, expected limitation.
        raise HTTPException(
            status_code=400,
            detail=f"Graph cannot be transpiled to GenExpr: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return GenExprResponse(genexpr_source=source)
