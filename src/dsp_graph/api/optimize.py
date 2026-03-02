"""Optimization endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from gen_dsp.graph.models import Graph
from gen_dsp.graph.optimize import optimize_graph
from pydantic import BaseModel

from dsp_graph.convert import ReactFlowGraph, graph_to_reactflow

router = APIRouter()


class OptimizeRequest(BaseModel):
    graph: dict[str, Any]


class OptimizeResponse(BaseModel):
    original: ReactFlowGraph
    optimized: ReactFlowGraph
    stats: dict[str, Any]


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest) -> OptimizeResponse:
    """Optimize a graph and return before/after ReactFlow representations."""
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = optimize_graph(g)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return OptimizeResponse(
        original=graph_to_reactflow(g),
        optimized=graph_to_reactflow(result.graph),
        stats=result.stats._asdict(),
    )
