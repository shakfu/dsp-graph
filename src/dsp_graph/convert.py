"""Convert between gen_dsp.graph.Graph and ReactFlow JSON representations."""

from __future__ import annotations

from typing import Any

from gen_dsp.graph._deps import is_feedback_edge
from gen_dsp.graph.models import (
    AudioInput,
    AudioOutput,
    Graph,
    Node,
    Param,
)
from gen_dsp.graph.toposort import toposort
from pydantic import BaseModel, TypeAdapter

# ---------------------------------------------------------------------------
# Color map: op string -> hex color (derived from gen_dsp.graph.visualize)
# ---------------------------------------------------------------------------

OP_COLORS: dict[str, str] = {
    # Arithmetic / logic
    "add": "#fff3cd",
    "sub": "#fff3cd",
    "mul": "#fff3cd",
    "div": "#fff3cd",
    "mod": "#fff3cd",
    "pow": "#fff3cd",
    "min": "#fff3cd",
    "max": "#fff3cd",
    "abs": "#fff3cd",
    "neg": "#fff3cd",
    "floor": "#fff3cd",
    "ceil": "#fff3cd",
    "sqrt": "#fff3cd",
    "exp": "#fff3cd",
    "log": "#fff3cd",
    "log2": "#fff3cd",
    "log10": "#fff3cd",
    "sin": "#fff3cd",
    "cos": "#fff3cd",
    "tan": "#fff3cd",
    "tanh": "#fff3cd",
    "sign": "#fff3cd",
    "atan2": "#fff3cd",
    "clamp": "#fff3cd",
    "wrap": "#fff3cd",
    "fold": "#fff3cd",
    "mix": "#fff3cd",
    "scale": "#fff3cd",
    "pass": "#fff3cd",
    "smoothstep": "#fff3cd",
    "gate_route": "#fff3cd",
    "gate_out": "#fff3cd",
    "selector": "#fff3cd",
    # Comparison / selection
    "gt": "#fff3cd",
    "lt": "#fff3cd",
    "gte": "#fff3cd",
    "lte": "#fff3cd",
    "eq": "#fff3cd",
    "neq": "#fff3cd",
    "select": "#fff3cd",
    # Constants
    "constant": "#e9ecef",
    "named_constant": "#e9ecef",
    "samplerate": "#e9ecef",
    # State / memory
    "history": "#fde0c8",
    "delay_line": "#fde0c8",
    "delay_read": "#fde0c8",
    "delay_write": "#fde0c8",
    "delta": "#fde0c8",
    "change": "#fde0c8",
    "sample_hold": "#fde0c8",
    "latch": "#fde0c8",
    "accum": "#fde0c8",
    "mulaccum": "#fde0c8",
    "counter": "#fde0c8",
    "elapsed": "#fde0c8",
    "rate_div": "#fde0c8",
    "smooth_param": "#fde0c8",
    "slide": "#fde0c8",
    "adsr": "#fde0c8",
    # Filters
    "biquad": "#fde0c8",
    "svf": "#fde0c8",
    "onepole": "#fde0c8",
    "dcblock": "#fde0c8",
    "allpass": "#fde0c8",
    # Oscillators
    "phasor": "#e2d5f1",
    "noise": "#e2d5f1",
    "sinosc": "#e2d5f1",
    "triosc": "#e2d5f1",
    "sawosc": "#e2d5f1",
    "pulseosc": "#e2d5f1",
    # Buffers / wavetables
    "buffer": "#fde0c8",
    "buf_read": "#fde0c8",
    "buf_write": "#fde0c8",
    "buf_size": "#fde0c8",
    "splat": "#fde0c8",
    "cycle": "#fde0c8",
    "wave": "#fde0c8",
    "lookup": "#fde0c8",
    # Utility
    "peek": "#d4edda",
    # Subgraph
    "subgraph": "#cce5ff",
}

INPUT_COLOR = "#d4edda"
OUTPUT_COLOR = "#f8d7da"
PARAM_COLOR = "#cce5ff"
DEFAULT_COLOR = "#ffffff"

# Layout constants
LAYER_X_SPACING = 250
SLOT_Y_SPACING = 100


# ---------------------------------------------------------------------------
# ReactFlow Pydantic models
# ---------------------------------------------------------------------------


class RFNodeData(BaseModel):
    """Data payload for a ReactFlow node."""

    label: str
    op: str | None = None
    color: str = DEFAULT_COLOR
    node_data: dict[str, Any] | None = None


class RFNode(BaseModel):
    """A ReactFlow node."""

    id: str
    type: str
    position: dict[str, float]
    data: RFNodeData


class RFEdge(BaseModel):
    """A ReactFlow edge."""

    id: str
    source: str
    target: str
    animated: bool = False
    label: str | None = None


class ReactFlowGraph(BaseModel):
    """Complete ReactFlow graph representation."""

    nodes: list[RFNode]
    edges: list[RFEdge]
    name: str = ""
    sample_rate: int = 44100
    control_interval: int = 0


# ---------------------------------------------------------------------------
# Graph -> ReactFlow
# ---------------------------------------------------------------------------


def _build_all_ids(graph: Graph) -> set[str]:
    """Collect all referenceable IDs in the graph."""
    ids: set[str] = set()
    for inp in graph.inputs:
        ids.add(inp.id)
    for out in graph.outputs:
        ids.add(out.id)
    for p in graph.params:
        ids.add(p.name)
    for node in graph.nodes:
        ids.add(node.id)
    return ids


def _topo_layers(graph: Graph) -> dict[str, int]:
    """Assign a layer index to each node ID based on topological order."""
    try:
        order = toposort(graph)
        order_ids = [n.id for n in order]
    except Exception:
        order_ids = [n.id for n in graph.nodes]
    layers: dict[str, int] = {}
    # Inputs at layer 0
    for inp in graph.inputs:
        layers[inp.id] = 0
    # Params at layer 0
    for p in graph.params:
        layers[p.name] = 0
    # Processing nodes at layers 1..N
    for i, nid in enumerate(order_ids):
        layers[nid] = i + 1
    # Outputs at last layer
    max_layer = max(layers.values(), default=0) + 1
    for out in graph.outputs:
        layers[out.id] = max_layer
    return layers


def graph_to_reactflow(graph: Graph) -> ReactFlowGraph:
    """Convert a gen-dsp Graph to a ReactFlowGraph for the frontend."""
    all_ids = _build_all_ids(graph)
    layers = _topo_layers(graph)
    rf_nodes: list[RFNode] = []
    rf_edges: list[RFEdge] = []

    # Track y-slot per layer for layout
    layer_slots: dict[int, int] = {}

    def _next_pos(layer: int) -> dict[str, float]:
        slot = layer_slots.get(layer, 0)
        layer_slots[layer] = slot + 1
        return {"x": layer * LAYER_X_SPACING, "y": slot * SLOT_Y_SPACING}

    # Audio inputs
    for inp in graph.inputs:
        rf_nodes.append(
            RFNode(
                id=inp.id,
                type="input",
                position=_next_pos(layers[inp.id]),
                data=RFNodeData(label=inp.id, color=INPUT_COLOR),
            )
        )

    # Params
    for p in graph.params:
        rf_nodes.append(
            RFNode(
                id=p.name,
                type="param",
                position=_next_pos(layers[p.name]),
                data=RFNodeData(
                    label=f"{p.name} [{p.min}, {p.max}] d={p.default}",
                    color=PARAM_COLOR,
                    node_data=p.model_dump(),
                ),
            )
        )

    # Processing nodes
    for node in graph.nodes:
        op = node.op
        color = OP_COLORS.get(op, DEFAULT_COLOR)
        rf_nodes.append(
            RFNode(
                id=node.id,
                type="dsp_node",
                position=_next_pos(layers.get(node.id, 1)),
                data=RFNodeData(
                    label=f"{node.id} ({op})",
                    op=op,
                    color=color,
                    node_data=node.model_dump(),
                ),
            )
        )

    # Audio outputs
    for out in graph.outputs:
        rf_nodes.append(
            RFNode(
                id=out.id,
                type="output",
                position=_next_pos(layers[out.id]),
                data=RFNodeData(label=out.id, color=OUTPUT_COLOR),
            )
        )

    # Edges from processing node fields
    edge_id = 0
    for node in graph.nodes:
        for field_name, value in node.__dict__.items():
            if field_name in ("id", "op"):
                continue
            if isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str) and v in all_ids:
                        feedback = is_feedback_edge(node, field_name)
                        rf_edges.append(
                            RFEdge(
                                id=f"e{edge_id}",
                                source=v,
                                target=node.id,
                                animated=feedback,
                                label="z^-1" if feedback else None,
                            )
                        )
                        edge_id += 1
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item in all_ids:
                        rf_edges.append(
                            RFEdge(
                                id=f"e{edge_id}",
                                source=item,
                                target=node.id,
                            )
                        )
                        edge_id += 1
                continue
            if not isinstance(value, str) or value not in all_ids:
                continue
            feedback = is_feedback_edge(node, field_name)
            rf_edges.append(
                RFEdge(
                    id=f"e{edge_id}",
                    source=value,
                    target=node.id,
                    animated=feedback,
                    label="z^-1" if feedback else None,
                )
            )
            edge_id += 1

    # Edges from outputs
    for out in graph.outputs:
        if isinstance(out.source, str) and out.source in all_ids:
            rf_edges.append(
                RFEdge(id=f"e{edge_id}", source=out.source, target=out.id)
            )
            edge_id += 1

    return ReactFlowGraph(
        nodes=rf_nodes,
        edges=rf_edges,
        name=graph.name,
        sample_rate=graph.sample_rate,
        control_interval=graph.control_interval,
    )


# ---------------------------------------------------------------------------
# ReactFlow -> Graph
# ---------------------------------------------------------------------------


_NODE_ADAPTER: TypeAdapter[Node] = TypeAdapter(Node)


def reactflow_to_graph(rf: ReactFlowGraph) -> Graph:
    """Convert a ReactFlowGraph back to a gen-dsp Graph.

    Reconstructs Graph from the node_data stored in each ReactFlow node.
    """
    inputs: list[AudioInput] = []
    outputs: list[AudioOutput] = []
    params: list[Param] = []
    nodes: list[Node] = []

    # Collect output edges: target -> source
    output_sources: dict[str, str] = {}
    for edge in rf.edges:
        output_sources[edge.target] = edge.source

    for rf_node in rf.nodes:
        if rf_node.type == "input":
            inputs.append(AudioInput(id=rf_node.id))
        elif rf_node.type == "output":
            source = output_sources.get(rf_node.id, "")
            outputs.append(AudioOutput(id=rf_node.id, source=source))
        elif rf_node.type == "param":
            if rf_node.data.node_data:
                params.append(Param.model_validate(rf_node.data.node_data))
            else:
                params.append(Param(name=rf_node.id))
        elif rf_node.type == "dsp_node":
            if rf_node.data.node_data:
                nodes.append(_NODE_ADAPTER.validate_python(rf_node.data.node_data))

    return Graph(
        name=rf.name,
        sample_rate=rf.sample_rate,
        control_interval=rf.control_interval,
        inputs=inputs,
        outputs=outputs,
        params=params,
        nodes=nodes,
    )
