from __future__ import annotations

from collections import defaultdict

from dsp_graph._deps import build_forward_deps
from dsp_graph.models import (
    Buffer,
    BufRead,
    BufSize,
    BufWrite,
    DelayLine,
    DelayRead,
    DelayWrite,
    Graph,
)


def _collect_refs(node: object) -> list[str]:
    """Return all string references from a node's input fields (excluding 'id' and 'op')."""
    refs: list[str] = []
    for field_name, value in node.__dict__.items():
        if field_name in ("id", "op"):
            continue
        if isinstance(value, str):
            refs.append(value)
    return refs


def validate_graph(graph: Graph) -> list[str]:
    """Validate a DSP graph and return a list of error strings (empty = valid)."""
    errors: list[str] = []

    # Build ID sets
    node_ids: dict[str, int] = {}
    input_ids = {inp.id for inp in graph.inputs}
    param_names = {p.name for p in graph.params}
    all_sources = input_ids | param_names  # valid non-node sources

    # 1. Unique IDs -- no duplicate node IDs, no collision with inputs/params
    for node in graph.nodes:
        nid = node.id
        if nid in node_ids:
            errors.append(f"Duplicate node ID: '{nid}'")
        node_ids[nid] = 0  # just tracking existence

        if nid in input_ids:
            errors.append(f"Node ID '{nid}' collides with audio input ID")
        if nid in param_names:
            errors.append(f"Node ID '{nid}' collides with param name")

    all_ids = set(node_ids) | all_sources

    # String fields that are enum selectors, not node references
    _NON_REF_FIELDS = {"id", "op", "interp", "mode"}

    # 2. Reference resolution -- every str input resolves to a known ID
    for node in graph.nodes:
        for field_name, value in node.__dict__.items():
            if field_name in _NON_REF_FIELDS:
                continue
            if isinstance(value, str):
                if value not in all_ids:
                    nid = node.id
                    errors.append(
                        f"Node '{nid}' field '{field_name}' references unknown ID '{value}'"
                    )

    # 3. Output resolution -- every output source resolves to a node ID
    for out in graph.outputs:
        if out.source not in node_ids:
            errors.append(f"Output '{out.id}' source '{out.source}' does not reference a node")

    # 4. Delay consistency -- DelayRead/DelayWrite must reference a DelayLine
    delay_line_ids = {node.id for node in graph.nodes if isinstance(node, DelayLine)}
    for node in graph.nodes:
        if isinstance(node, DelayRead) and node.delay not in delay_line_ids:
            errors.append(
                f"DelayRead '{node.id}' references non-existent delay line '{node.delay}'"
            )
        if isinstance(node, DelayWrite) and node.delay not in delay_line_ids:
            errors.append(
                f"DelayWrite '{node.id}' references non-existent delay line '{node.delay}'"
            )

    # 4b. Buffer consistency -- BufRead/BufWrite/BufSize must reference a Buffer
    buffer_ids = {node.id for node in graph.nodes if isinstance(node, Buffer)}
    for node in graph.nodes:
        if isinstance(node, BufRead) and node.buffer not in buffer_ids:
            errors.append(f"BufRead '{node.id}' references non-existent buffer '{node.buffer}'")
        if isinstance(node, BufWrite) and node.buffer not in buffer_ids:
            errors.append(f"BufWrite '{node.id}' references non-existent buffer '{node.buffer}'")
        if isinstance(node, BufSize) and node.buffer not in buffer_ids:
            errors.append(f"BufSize '{node.id}' references non-existent buffer '{node.buffer}'")

    # 5. No pure cycles -- topo sort on non-feedback edges must succeed
    deps = build_forward_deps(graph)

    # Kahn's algorithm
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    reverse: dict[str, list[str]] = defaultdict(list)
    for nid, dep_set in deps.items():
        for dep in dep_set:
            if dep in in_degree:
                in_degree[nid] += 1
                reverse[dep].append(nid)

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for dependent in reverse[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if visited < len(node_ids):
        cycle_nodes = [nid for nid, deg in in_degree.items() if deg > 0]
        errors.append(f"Graph contains a cycle through nodes: {', '.join(sorted(cycle_nodes))}")

    return errors
