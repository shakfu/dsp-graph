"""Auto-layout endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from dsp_graph.convert import (
    LAYER_X_SPACING,
    SLOT_Y_SPACING,
    ReactFlowGraph,
)

router = APIRouter()


@router.post("/layout", response_model=ReactFlowGraph)
async def layout(rf: ReactFlowGraph) -> ReactFlowGraph:
    """Re-compute positions for all nodes using a simple layered layout.

    Groups nodes by type: inputs at x=0, params at x=0, processing nodes
    in the middle, outputs at the rightmost layer.
    """
    inputs = [n for n in rf.nodes if n.type == "input"]
    params = [n for n in rf.nodes if n.type == "param"]
    dsp_nodes = [n for n in rf.nodes if n.type == "dsp_node"]
    outputs = [n for n in rf.nodes if n.type == "output"]

    max_layer = len(dsp_nodes) + 1
    y_slot = 0

    for node in inputs:
        node.position = {"x": 0, "y": y_slot * SLOT_Y_SPACING}
        y_slot += 1

    for node in params:
        node.position = {"x": 0, "y": y_slot * SLOT_Y_SPACING}
        y_slot += 1

    for i, node in enumerate(dsp_nodes):
        node.position = {
            "x": (i + 1) * LAYER_X_SPACING,
            "y": 0,
        }

    for i, node in enumerate(outputs):
        node.position = {
            "x": max_layer * LAYER_X_SPACING,
            "y": i * SLOT_Y_SPACING,
        }

    return rf
