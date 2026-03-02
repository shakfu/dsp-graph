"""Graph load / validate / export / catalog endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from gen_dsp.graph.models import Graph
from gen_dsp.graph.validate import GraphValidationError, validate_graph
from gen_dsp.graph.visualize import graph_to_dot
from pydantic import BaseModel

from dsp_graph.convert import (
    OP_COLORS,
    ReactFlowGraph,
    graph_to_reactflow,
    reactflow_to_graph,
)

router = APIRouter()


class LoadJsonRequest(BaseModel):
    graph: dict[str, Any]


class LoadGdspRequest(BaseModel):
    source: str


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str]


class DotResponse(BaseModel):
    dot: str


@router.post("/load/json", response_model=ReactFlowGraph)
async def load_json(req: LoadJsonRequest) -> ReactFlowGraph:
    """Load a Graph from JSON and return ReactFlow representation."""
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return graph_to_reactflow(g)


@router.post("/load/gdsp", response_model=ReactFlowGraph)
async def load_gdsp(req: LoadGdspRequest) -> ReactFlowGraph:
    """Parse a .gdsp DSL source string and return ReactFlow representation."""
    try:
        from gen_dsp.graph.dsl import parse
    except ImportError as exc:
        raise HTTPException(
            status_code=501, detail="GDSP parser not available"
        ) from exc
    try:
        g = parse(req.source)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return graph_to_reactflow(g)


@router.post("/validate", response_model=ValidateResponse)
async def validate(req: LoadJsonRequest) -> ValidateResponse:
    """Validate a Graph JSON and return errors if any."""
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        return ValidateResponse(valid=False, errors=[str(exc)])
    try:
        validate_graph(g)
    except GraphValidationError as exc:
        return ValidateResponse(valid=False, errors=[str(exc)])
    return ValidateResponse(valid=True, errors=[])


@router.post("/export/json")
async def export_json(rf: ReactFlowGraph) -> dict[str, Any]:
    """Convert a ReactFlowGraph back to a Graph JSON dict."""
    g = reactflow_to_graph(rf)
    return g.model_dump()  # type: ignore[return-value]


@router.get("/node-types")
async def node_types() -> dict[str, Any]:
    """Return the catalog of node op types with their colors."""
    return {"colors": OP_COLORS}


@router.post("/dot", response_model=DotResponse)
async def dot(req: LoadJsonRequest) -> DotResponse:
    """Return Graphviz DOT source for a Graph."""
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DotResponse(dot=graph_to_dot(g))
