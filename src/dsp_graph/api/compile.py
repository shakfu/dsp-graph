"""C++ compilation preview endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from gen_dsp.graph.compile import compile_graph
from gen_dsp.graph.models import Graph
from pydantic import BaseModel

router = APIRouter()


class CompileRequest(BaseModel):
    graph: dict[str, Any]


class CompileResponse(BaseModel):
    cpp_source: str


@router.post("/compile", response_model=CompileResponse)
async def compile_endpoint(req: CompileRequest) -> CompileResponse:
    """Compile a Graph to C++ source and return it."""
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        cpp = compile_graph(g)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CompileResponse(cpp_source=cpp)
