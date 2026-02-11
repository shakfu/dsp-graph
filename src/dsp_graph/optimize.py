"""Optimization passes for DSP graphs."""

from __future__ import annotations

import math
from typing import Union

from dsp_graph.models import (
    SVF,
    Accum,
    Allpass,
    BinOp,
    Biquad,
    Change,
    Clamp,
    Compare,
    Constant,
    Counter,
    DCBlock,
    DelayLine,
    DelayRead,
    DelayWrite,
    Delta,
    Fold,
    Graph,
    History,
    Latch,
    Mix,
    Node,
    Noise,
    OnePole,
    Phasor,
    PulseOsc,
    SampleHold,
    SawOsc,
    Select,
    SinOsc,
    TriOsc,
    UnaryOp,
    Wrap,
)

# Types that are stateful and must never be constant-folded.
_STATEFUL_TYPES = (
    History,
    DelayLine,
    DelayRead,
    DelayWrite,
    Phasor,
    Noise,
    Delta,
    Change,
    Biquad,
    SVF,
    OnePole,
    DCBlock,
    Allpass,
    SinOsc,
    TriOsc,
    SawOsc,
    PulseOsc,
    SampleHold,
    Latch,
    Accum,
    Counter,
)


def _resolve_ref(ref: Union[str, float], constants: dict[str, float]) -> float | None:
    """Resolve a Ref to a float if it is a literal or a known constant node."""
    if isinstance(ref, float):
        return ref
    return constants.get(ref)


_BINOP_EVAL: dict[str, object] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if b != 0.0 else float("inf"),
    "min": lambda a, b: min(a, b),
    "max": lambda a, b: max(a, b),
    "mod": lambda a, b: math.fmod(a, b) if b != 0.0 else 0.0,
    "pow": lambda a, b: a**b,
}

_UNARYOP_EVAL: dict[str, object] = {
    "sin": math.sin,
    "cos": math.cos,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": lambda x: math.log(x) if x > 0 else float("-inf"),
    "abs": abs,
    "sqrt": lambda x: math.sqrt(x) if x >= 0 else 0.0,
    "neg": lambda x: -x,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "sign": lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0),
    "atan": math.atan,
    "asin": lambda x: math.asin(x) if -1 <= x <= 1 else 0.0,
    "acos": lambda x: math.acos(x) if -1 <= x <= 1 else 0.0,
}

_COMPARE_EVAL: dict[str, object] = {
    "gt": lambda a, b: 1.0 if a > b else 0.0,
    "lt": lambda a, b: 1.0 if a < b else 0.0,
    "gte": lambda a, b: 1.0 if a >= b else 0.0,
    "lte": lambda a, b: 1.0 if a <= b else 0.0,
    "eq": lambda a, b: 1.0 if a == b else 0.0,
}


def _try_fold(node: Node, constants: dict[str, float]) -> float | None:
    """Try to evaluate a node with all-constant inputs. Returns value or None."""
    if isinstance(node, _STATEFUL_TYPES):
        return None

    if isinstance(node, Constant):
        return node.value

    if isinstance(node, BinOp):
        a = _resolve_ref(node.a, constants)
        b = _resolve_ref(node.b, constants)
        if a is not None and b is not None:
            fn = _BINOP_EVAL[node.op]
            return float(fn(a, b))  # type: ignore[operator]
        return None

    if isinstance(node, UnaryOp):
        a = _resolve_ref(node.a, constants)
        if a is not None:
            fn = _UNARYOP_EVAL[node.op]
            return float(fn(a))  # type: ignore[operator]
        return None

    if isinstance(node, Compare):
        a = _resolve_ref(node.a, constants)
        b = _resolve_ref(node.b, constants)
        if a is not None and b is not None:
            fn = _COMPARE_EVAL[node.op]
            return float(fn(a, b))  # type: ignore[operator]
        return None

    if isinstance(node, Select):
        c = _resolve_ref(node.cond, constants)
        a = _resolve_ref(node.a, constants)
        b = _resolve_ref(node.b, constants)
        if c is not None and a is not None and b is not None:
            return a if c > 0.0 else b
        return None

    if isinstance(node, Clamp):
        a = _resolve_ref(node.a, constants)
        lo = _resolve_ref(node.lo, constants)
        hi = _resolve_ref(node.hi, constants)
        if a is not None and lo is not None and hi is not None:
            return min(max(a, lo), hi)
        return None

    if isinstance(node, Wrap):
        a = _resolve_ref(node.a, constants)
        lo = _resolve_ref(node.lo, constants)
        hi = _resolve_ref(node.hi, constants)
        if a is not None and lo is not None and hi is not None:
            r = hi - lo
            if r == 0.0:
                return lo
            raw = math.fmod(a - lo, r)
            return lo + (raw + r if raw < 0.0 else raw)
        return None

    if isinstance(node, Fold):
        a = _resolve_ref(node.a, constants)
        lo = _resolve_ref(node.lo, constants)
        hi = _resolve_ref(node.hi, constants)
        if a is not None and lo is not None and hi is not None:
            r = hi - lo
            if r == 0.0:
                return lo
            t = math.fmod(a - lo, 2.0 * r)
            if t < 0.0:
                t += 2.0 * r
            return lo + t if t <= r else hi - (t - r)
        return None

    if isinstance(node, Mix):
        a = _resolve_ref(node.a, constants)
        b = _resolve_ref(node.b, constants)
        mix_t = _resolve_ref(node.t, constants)
        if a is not None and b is not None and mix_t is not None:
            return a + (b - a) * mix_t
        return None

    return None


def constant_fold(graph: Graph) -> Graph:
    """Replace pure nodes with all-constant inputs by Constant nodes.

    Returns a new Graph (immutable transform). Stateful nodes are never folded.
    """
    from dsp_graph.toposort import toposort

    sorted_nodes = toposort(graph)
    constants: dict[str, float] = {}
    new_nodes: list[Node] = []

    for node in sorted_nodes:
        val = _try_fold(node, constants)
        if val is not None and not isinstance(node, Constant):
            constants[node.id] = val
            new_nodes.append(Constant(id=node.id, value=val))
        else:
            if isinstance(node, Constant):
                constants[node.id] = node.value
            new_nodes.append(node)

    return graph.model_copy(update={"nodes": new_nodes})


def eliminate_dead_nodes(graph: Graph) -> Graph:
    """Remove nodes not reachable from any output.

    Walks backward from output sources, following ALL string fields
    (including feedback edges).  When a DelayRead is reachable, the
    DelayWrite nodes that feed the same delay line are also treated
    as reachable (side-effecting nodes).

    Returns a new Graph with dead nodes removed.
    """
    node_ids = {node.id for node in graph.nodes}
    node_map = {node.id: node for node in graph.nodes}

    # Map delay-line ID -> DelayWrite node IDs that write to it
    delay_writers: dict[str, list[str]] = {}
    for node in graph.nodes:
        if isinstance(node, DelayWrite):
            delay_writers.setdefault(node.delay, []).append(node.id)

    # Seed with output sources
    reachable: set[str] = set()
    worklist: list[str] = [out.source for out in graph.outputs if out.source in node_ids]

    while worklist:
        nid = worklist.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        if nid not in node_map:
            continue
        node = node_map[nid]
        # Follow all string fields
        for field_name, value in node.__dict__.items():
            if field_name in ("id", "op"):
                continue
            if isinstance(value, str) and value in node_ids:
                worklist.append(value)
        # If this is a DelayRead, also mark the corresponding writers
        if isinstance(node, DelayRead):
            for writer_id in delay_writers.get(node.delay, []):
                worklist.append(writer_id)

    new_nodes = [node for node in graph.nodes if node.id in reachable]
    return graph.model_copy(update={"nodes": new_nodes})


def optimize_graph(graph: Graph) -> Graph:
    """Apply all optimization passes: constant folding, then dead node elimination."""
    result = constant_fold(graph)
    result = eliminate_dead_nodes(result)
    return result
