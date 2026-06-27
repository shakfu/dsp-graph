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
from gen_dsp.graph.serialize import graph_to_gdsp
from gen_dsp.graph.toposort import toposort
from pydantic import BaseModel, TypeAdapter

# ---------------------------------------------------------------------------
# Color map: op string -> hex color
# ---------------------------------------------------------------------------
#
# Colors are keyed by node *class* (mirroring gen_dsp.graph.visualize._node_attrs,
# which colors by isinstance) and the per-op OP_COLORS map below is derived from
# this by walking the Node discriminated union. Deriving rather than hardcoding
# each op string keeps the editor in step with gen-dsp: a new op added to an
# existing class (e.g. another BinOp/UnaryOp) is colored automatically, and a
# brand-new node *class* fails loudly at import (see _build_op_colors) instead of
# silently rendering white.

CLASS_COLORS: dict[str, str] = {
    # Arithmetic / logic / routing
    "BinOp": "#fff3cd",
    "UnaryOp": "#fff3cd",
    "Clamp": "#fff3cd",
    "Compare": "#fff3cd",
    "Select": "#fff3cd",
    "Wrap": "#fff3cd",
    "Fold": "#fff3cd",
    "Mix": "#fff3cd",
    "Scale": "#fff3cd",
    "Pass": "#fff3cd",
    "Smoothstep": "#fff3cd",
    "GateRoute": "#fff3cd",
    "GateOut": "#fff3cd",
    "Selector": "#fff3cd",
    # Constants
    "Constant": "#e9ecef",
    "NamedConstant": "#e9ecef",
    "SampleRate": "#e9ecef",
    # State / memory / timing
    "History": "#fde0c8",
    "DelayLine": "#fde0c8",
    "DelayRead": "#fde0c8",
    "DelayWrite": "#fde0c8",
    "Delta": "#fde0c8",
    "Change": "#fde0c8",
    "SampleHold": "#fde0c8",
    "Latch": "#fde0c8",
    "Accum": "#fde0c8",
    "MulAccum": "#fde0c8",
    "Counter": "#fde0c8",
    "Elapsed": "#fde0c8",
    "RateDiv": "#fde0c8",
    "SmoothParam": "#fde0c8",
    "Slide": "#fde0c8",
    "ADSR": "#fde0c8",
    # Filters
    "Biquad": "#fde0c8",
    "SVF": "#fde0c8",
    "OnePole": "#fde0c8",
    "DCBlock": "#fde0c8",
    "Allpass": "#fde0c8",
    # Buffers / wavetables
    "Buffer": "#fde0c8",
    "BufRead": "#fde0c8",
    "BufWrite": "#fde0c8",
    "BufSize": "#fde0c8",
    "Splat": "#fde0c8",
    "Cycle": "#fde0c8",
    "Wave": "#fde0c8",
    "Lookup": "#fde0c8",
    # Oscillators
    "Phasor": "#e2d5f1",
    "Noise": "#e2d5f1",
    "SinOsc": "#e2d5f1",
    "TriOsc": "#e2d5f1",
    "SawOsc": "#e2d5f1",
    "PulseOsc": "#e2d5f1",
    # Utility
    "Peek": "#d4edda",
    # Subgraph
    "Subgraph": "#cce5ff",
}


def _build_op_colors() -> dict[str, str]:
    """Map every op string in the Node union to its node class's color.

    Walks the Node discriminated-union JSON schema, reading each member's ``op``
    literal (``const`` for single-op classes, ``enum`` for multi-op classes such
    as BinOp/UnaryOp/Compare/NamedConstant), and assigns the color registered for
    that class in :data:`CLASS_COLORS`. A node class present in the union but
    missing from CLASS_COLORS raises ``RuntimeError`` at import so a gen-dsp
    addition cannot silently fall through to the default color.
    """
    schema = TypeAdapter(Node).json_schema()
    defs = schema.get("$defs", {})
    colors: dict[str, str] = {}
    for class_name, defn in defs.items():
        op_prop = defn.get("properties", {}).get("op")
        if op_prop is None:
            # Not a node member (AudioInput/AudioOutput/Param/Graph).
            continue
        if class_name not in CLASS_COLORS:
            raise RuntimeError(
                f"gen_dsp node class {class_name!r} has no entry in "
                "dsp_graph.convert.CLASS_COLORS; add one to track gen-dsp."
            )
        color = CLASS_COLORS[class_name]
        ops = op_prop.get("enum") or ([op_prop["const"]] if "const" in op_prop else [])
        for op in ops:
            colors[op] = color
    return colors


OP_COLORS: dict[str, str] = _build_op_colors()

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
    # Name of the target node's input field this edge feeds (the React Flow
    # target handle id). None for output-node edges, which have a single input.
    target_handle: str | None = None
    animated: bool = False
    label: str | None = None


class ReactFlowGraph(BaseModel):
    """Complete ReactFlow graph representation."""

    nodes: list[RFNode]
    edges: list[RFEdge]
    name: str = ""
    sample_rate: int = 44100
    control_interval: int = 0
    # Names of all graphs defined in the source (for multi-graph .gdsp files).
    # Empty/[name] for single-graph sources; populated by the gdsp load endpoint.
    graph_names: list[str] = []


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
                                target_handle=field_name,
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
                                target_handle=field_name,
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
                    target_handle=field_name,
                    animated=feedback,
                    label="z^-1" if feedback else None,
                )
            )
            edge_id += 1

    # Edges from outputs
    for out in graph.outputs:
        if isinstance(out.source, str) and out.source in all_ids:
            rf_edges.append(RFEdge(id=f"e{edge_id}", source=out.source, target=out.id))
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


# ---------------------------------------------------------------------------
# Graph -> .gdsp source
# ---------------------------------------------------------------------------

# Re-export graph_to_gdsp from gen-dsp for backward compatibility
__all__ = ["graph_to_gdsp"]
