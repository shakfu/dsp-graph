"""Graph load / validate / export / catalog endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from gen_dsp.graph.models import Graph, Node
from gen_dsp.graph.validate import validate_graph
from gen_dsp.graph.visualize import graph_to_dot
from pydantic import BaseModel, TypeAdapter

from dsp_graph.convert import (
    OP_COLORS,
    ReactFlowGraph,
    graph_to_gdsp,
    graph_to_reactflow,
    reactflow_to_graph,
)

router = APIRouter()


class LoadJsonRequest(BaseModel):
    graph: dict[str, Any]


class LoadGdspRequest(BaseModel):
    source: str


class ValidationErrorDetail(BaseModel):
    message: str
    kind: str
    node_id: str | None = None
    field_name: str | None = None
    severity: str = "error"


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[ValidationErrorDetail]


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
        from gen_dsp.graph.dsl import GDSPSyntaxError, parse
    except ImportError as exc:
        raise HTTPException(status_code=501, detail="GDSP parser not available") from exc
    try:
        g = parse(req.source)
    except GDSPSyntaxError as exc:
        # Extract raw message by stripping the "<file>:<line>:<col>: " prefix
        full = str(exc)
        prefix = f"{exc.filename}:{exc.line}:{exc.col}: "
        msg = full[len(prefix) :] if full.startswith(prefix) else full
        raise HTTPException(
            status_code=422,
            detail={"message": msg, "line": exc.line, "col": exc.col},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return graph_to_reactflow(g)


@router.post("/validate", response_model=ValidateResponse)
async def validate(req: LoadJsonRequest) -> ValidateResponse:
    """Validate a Graph JSON and return errors if any."""
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        return ValidateResponse(
            valid=False,
            errors=[
                ValidationErrorDetail(
                    message=str(exc),
                    kind="parse_error",
                    severity="error",
                )
            ],
        )
    errs = validate_graph(g)
    if errs:
        details = [
            ValidationErrorDetail(
                message=str(e),
                kind=e.kind,
                node_id=e.node_id,
                field_name=e.field_name,
                severity=e.severity,
            )
            for e in errs
        ]
        return ValidateResponse(valid=False, errors=details)
    return ValidateResponse(valid=True, errors=[])


class GdspResponse(BaseModel):
    source: str


@router.post("/export/json")
async def export_json(rf: ReactFlowGraph) -> dict[str, Any]:
    """Convert a ReactFlowGraph back to a Graph JSON dict."""
    g = reactflow_to_graph(rf)
    return g.model_dump()


@router.post("/export/gdsp", response_model=GdspResponse)
async def export_gdsp(rf: ReactFlowGraph) -> GdspResponse:
    """Convert a ReactFlowGraph to .gdsp DSL source."""
    try:
        g = reactflow_to_graph(rf)
        source = graph_to_gdsp(g)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GdspResponse(source=source)


_SKIP_CATALOG = {"AudioInput", "AudioOutput", "Graph", "Param", "Buffer"}
_NODE_SCHEMA = TypeAdapter(Node).json_schema()
_NODE_DEFS = _NODE_SCHEMA.get("$defs", {})


def _build_catalog() -> dict[str, Any]:
    """Build a catalog of node types from the Node discriminated union schema."""
    catalog: dict[str, Any] = {}
    for class_name, defn in _NODE_DEFS.items():
        if class_name in _SKIP_CATALOG:
            continue
        props = defn.get("properties", {})
        op_prop = props.get("op", {})
        # Get op name(s) -- const for single-op, enum for multi-op (e.g. BinOp)
        op_const = op_prop.get("const")
        op_enum = op_prop.get("enum")
        required = set(defn.get("required", []))
        # Build fields dict, skipping id and op
        fields: dict[str, Any] = {}
        for fname, fschema in props.items():
            if fname in ("id", "op"):
                continue
            ftype = fschema.get("type", "")
            if not ftype:
                any_of = fschema.get("anyOf", [])
                ftype = "|".join(t.get("type", "?") for t in any_of)
            field_info: dict[str, Any] = {
                "type": ftype,
                "required": fname in required,
            }
            if "default" in fschema:
                field_info["default"] = fschema["default"]
            fields[fname] = field_info

        if op_const:
            catalog[op_const] = {
                "class": class_name,
                "fields": fields,
                "color": OP_COLORS.get(op_const, "#ffffff"),
            }
        elif op_enum:
            for op_name in op_enum:
                catalog[op_name] = {
                    "class": class_name,
                    "fields": fields,
                    "color": OP_COLORS.get(op_name, "#ffffff"),
                }
    return catalog


_CATALOG = _build_catalog()


@router.get("/node-types")
async def node_types() -> dict[str, Any]:
    """Return the catalog of node op types with their colors."""
    return {"colors": OP_COLORS, "catalog": _CATALOG}


@router.post("/dot", response_model=DotResponse)
async def dot(req: LoadJsonRequest) -> DotResponse:
    """Return Graphviz DOT source for a Graph."""
    try:
        g = Graph.model_validate(req.graph)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DotResponse(dot=graph_to_dot(g))
