"""Optimization endpoint."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from gen_dsp.graph.models import Graph
from gen_dsp.graph.optimize import (
    constant_fold,
    eliminate_cse,
    eliminate_dead_nodes,
    optimize_graph,
    promote_control_rate,
)
from pydantic import BaseModel

from dsp_graph.convert import ReactFlowGraph, graph_to_reactflow

router = APIRouter()

PASSES: dict[str, Callable[[Graph], Graph]] = {
    "constant_fold": constant_fold,
    "eliminate_cse": eliminate_cse,
    "eliminate_dead_nodes": eliminate_dead_nodes,
    "promote_control_rate": promote_control_rate,
}


class OptimizeRequest(BaseModel):
    graph: dict[str, Any]


class OptimizeResponse(BaseModel):
    original: ReactFlowGraph
    optimized: ReactFlowGraph
    stats: dict[str, Any]


class PassRequest(BaseModel):
    graph: dict[str, Any]
    pass_name: str


class PassResponse(BaseModel):
    original: ReactFlowGraph
    optimized: ReactFlowGraph
    stats: dict[str, int]


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


@router.post("/optimize/pass", response_model=PassResponse)
async def optimize_pass(req: PassRequest) -> PassResponse:
    """Run a single optimization pass and return before/after."""
    if req.pass_name not in PASSES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown pass '{req.pass_name}'. Available: {', '.join(sorted(PASSES))}",
        )

    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        optimized = PASSES[req.pass_name](g)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    nodes_before = len(g.nodes)
    nodes_after = len(optimized.nodes)

    return PassResponse(
        original=graph_to_reactflow(g),
        optimized=graph_to_reactflow(optimized),
        stats={
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "nodes_removed": nodes_before - nodes_after,
        },
    )
