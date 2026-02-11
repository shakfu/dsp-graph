"""C++ code generation from DSP graphs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from dsp_graph.models import (
    BinOp,
    Change,
    Clamp,
    Compare,
    Constant,
    DelayLine,
    DelayRead,
    DelayWrite,
    Delta,
    Fold,
    Graph,
    History,
    Mix,
    Node,
    Noise,
    Param,
    Phasor,
    Select,
    UnaryOp,
    Wrap,
)
from dsp_graph.toposort import toposort
from dsp_graph.validate import validate_graph

_Writer = Callable[[str], None]

_C_ID_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_BINOP_SYMBOLS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}

_BINOP_FUNCS: dict[str, str] = {
    "min": "fminf",
    "max": "fmaxf",
    "mod": "fmodf",
    "pow": "powf",
}

_UNARYOP_FUNCS: dict[str, str] = {
    "sin": "sinf",
    "cos": "cosf",
    "tanh": "tanhf",
    "exp": "expf",
    "log": "logf",
    "abs": "fabsf",
    "sqrt": "sqrtf",
    "floor": "floorf",
    "ceil": "ceilf",
    "round": "roundf",
}

_COMPARE_SYMBOLS: dict[str, str] = {
    "gt": ">",
    "lt": "<",
    "gte": ">=",
    "lte": "<=",
    "eq": "==",
}


def _to_pascal(name: str) -> str:
    """Convert underscore_name to PascalCase."""
    return "".join(part.capitalize() for part in name.split("_"))


def _float_lit(v: float) -> str:
    """Format a float as a C literal with 'f' suffix."""
    s = repr(v)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def _emit_ref(ref: str | float, input_ids: set[str], param_names: set[str]) -> str:
    """Emit a C expression for a Ref value."""
    if isinstance(ref, float):
        return _float_lit(ref)
    if ref in input_ids:
        return ref + "[i]"
    # param names and node IDs are both local C variables
    return ref


def compile_graph(graph: Graph) -> str:
    """Compile a DSP graph to standalone C++ source code.

    Raises ValueError if the graph is invalid or contains IDs that are
    not valid C identifiers.
    """
    errors = validate_graph(graph)
    if errors:
        raise ValueError("Invalid graph: " + "; ".join(errors))

    # Validate all IDs are valid C identifiers
    all_ids: list[str] = []
    all_ids.extend(inp.id for inp in graph.inputs)
    all_ids.extend(out.id for out in graph.outputs)
    all_ids.extend(p.name for p in graph.params)
    all_ids.extend(node.id for node in graph.nodes)
    for ident in all_ids:
        if not _C_ID_RE.match(ident):
            raise ValueError(f"ID '{ident}' is not a valid C identifier")

    sorted_nodes = toposort(graph)
    input_ids = {inp.id for inp in graph.inputs}
    param_names = {p.name for p in graph.params}

    name = graph.name
    pascal = _to_pascal(name)
    struct_name = pascal + "State"

    lines: list[str] = []
    w = lines.append

    # -- Includes
    w("#include <cmath>")
    w("#include <cstdlib>")
    w("#include <cstdint>")
    w("")

    # -- Struct
    w(f"struct {struct_name} {{")
    w("    float sr;")
    # Params
    for p in graph.params:
        w(f"    float p_{p.name};")
    # State fields from nodes
    for node in sorted_nodes:
        _emit_state_fields(node, w)
    w("};")
    w("")

    # -- create()
    w(f"{struct_name}* {name}_create(float sr) {{")
    w(f"    {struct_name}* self = ({struct_name}*)calloc(1, sizeof({struct_name}));")
    w("    if (!self) return nullptr;")
    w("    self->sr = sr;")
    for p in graph.params:
        w(f"    self->p_{p.name} = {_float_lit(p.default)};")
    for node in sorted_nodes:
        _emit_state_init(node, w)
    w("    return self;")
    w("}")
    w("")

    # -- destroy()
    w(f"void {name}_destroy({struct_name}* self) {{")
    for node in sorted_nodes:
        if isinstance(node, DelayLine):
            w(f"    free(self->m_{node.id}_buf);")
    w("    free(self);")
    w("}")
    w("")

    # -- perform()
    _emit_perform(graph, sorted_nodes, input_ids, param_names, name, struct_name, w)
    w("")

    # -- Introspection
    w(f"int {name}_num_inputs(void) {{ return {len(graph.inputs)}; }}")
    w(f"int {name}_num_outputs(void) {{ return {len(graph.outputs)}; }}")
    w(f"int {name}_num_params(void) {{ return {len(graph.params)}; }}")
    w("")

    # -- param_name
    _emit_param_name(graph.params, name, struct_name, w)
    w("")

    # -- param_min / param_max
    _emit_param_minmax(graph.params, name, struct_name, "min", w)
    w("")
    _emit_param_minmax(graph.params, name, struct_name, "max", w)
    w("")

    # -- set_param / get_param
    _emit_param_set(graph.params, name, struct_name, w)
    w("")
    _emit_param_get(graph.params, name, struct_name, w)

    return "\n".join(lines) + "\n"


def compile_graph_to_file(graph: Graph, output_dir: str | Path) -> Path:
    """Compile a DSP graph and write {name}.cpp to output_dir.

    Creates the output directory if it doesn't exist.
    Returns the path to the written file.
    """
    code = compile_graph(graph)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{graph.name}.cpp"
    path.write_text(code)
    return path


# ---------------------------------------------------------------------------
# Struct field emission
# ---------------------------------------------------------------------------


def _emit_state_fields(node: Node, w: _Writer) -> None:
    if isinstance(node, History):
        w(f"    float m_{node.id};")
    elif isinstance(node, DelayLine):
        w(f"    float* m_{node.id}_buf;")
        w(f"    int m_{node.id}_len;")
        w(f"    int m_{node.id}_wr;")
    elif isinstance(node, Phasor):
        w(f"    float m_{node.id}_phase;")
    elif isinstance(node, Noise):
        w(f"    uint32_t m_{node.id}_seed;")
    elif isinstance(node, (Delta, Change)):
        w(f"    float m_{node.id}_prev;")


# ---------------------------------------------------------------------------
# State initialization
# ---------------------------------------------------------------------------


def _emit_state_init(node: Node, w: _Writer) -> None:
    if isinstance(node, History):
        w(f"    self->m_{node.id} = {_float_lit(node.init)};")
    elif isinstance(node, DelayLine):
        w(f"    self->m_{node.id}_len = {node.max_samples};")
        w(f"    self->m_{node.id}_buf = (float*)calloc({node.max_samples}, sizeof(float));")
        w(f"    self->m_{node.id}_wr = 0;")
    elif isinstance(node, Noise):
        w(f"    self->m_{node.id}_seed = 123456789u;")
    elif isinstance(node, (Delta, Change)):
        w(f"    self->m_{node.id}_prev = 0.0f;")


# ---------------------------------------------------------------------------
# perform() body
# ---------------------------------------------------------------------------


def _emit_perform(
    graph: Graph,
    sorted_nodes: list[Node],
    input_ids: set[str],
    param_names: set[str],
    name: str,
    struct_name: str,
    w: _Writer,
) -> None:
    w(f"void {name}_perform({struct_name}* self, float** ins, float** outs, int n) {{")

    # Unpack I/O pointers
    for idx, inp in enumerate(graph.inputs):
        w(f"    float* {inp.id} = ins[{idx}];")
    for idx, out in enumerate(graph.outputs):
        w(f"    float* {out.id} = outs[{idx}];")

    # Load params to locals
    for p in graph.params:
        w(f"    float {p.name} = self->p_{p.name};")

    # Load state to locals
    for node in sorted_nodes:
        _emit_state_load(node, w)

    w("    float sr = self->sr;")
    w("    for (int i = 0; i < n; i++) {")

    # Topo-sorted node computations
    history_nodes: list[History] = []
    delay_write_nodes: list[DelayWrite] = []
    for node in sorted_nodes:
        _emit_node_compute(node, input_ids, param_names, w, history_nodes, delay_write_nodes)

    # History write-backs
    for h in history_nodes:
        ref = _emit_ref(h.input, input_ids, param_names)
        w(f"        {h.id} = {ref};")

    # Output assignments
    for out in graph.outputs:
        w(f"        {out.id}[i] = {out.source};")

    w("    }")

    # Save state back
    for node in sorted_nodes:
        _emit_state_save(node, w)

    w("}")


def _emit_state_load(node: Node, w: _Writer) -> None:
    if isinstance(node, History):
        w(f"    float {node.id} = self->m_{node.id};")
    elif isinstance(node, DelayLine):
        w(f"    float* {node.id}_buf = self->m_{node.id}_buf;")
        w(f"    int {node.id}_len = self->m_{node.id}_len;")
        w(f"    int {node.id}_wr = self->m_{node.id}_wr;")
    elif isinstance(node, Phasor):
        w(f"    float {node.id}_phase = self->m_{node.id}_phase;")
    elif isinstance(node, Noise):
        w(f"    uint32_t {node.id}_seed = self->m_{node.id}_seed;")
    elif isinstance(node, (Delta, Change)):
        w(f"    float {node.id}_prev = self->m_{node.id}_prev;")


def _emit_state_save(node: Node, w: _Writer) -> None:
    if isinstance(node, History):
        w(f"    self->m_{node.id} = {node.id};")
    elif isinstance(node, DelayLine):
        w(f"    self->m_{node.id}_wr = {node.id}_wr;")
    elif isinstance(node, Phasor):
        w(f"    self->m_{node.id}_phase = {node.id}_phase;")
    elif isinstance(node, Noise):
        w(f"    self->m_{node.id}_seed = {node.id}_seed;")
    elif isinstance(node, (Delta, Change)):
        w(f"    self->m_{node.id}_prev = {node.id}_prev;")


def _emit_node_compute(
    node: Node,
    input_ids: set[str],
    param_names: set[str],
    w: _Writer,
    history_nodes: list[History],
    delay_write_nodes: list[DelayWrite],
) -> None:
    def ref(r: str | float) -> str:
        return _emit_ref(r, input_ids, param_names)

    if isinstance(node, BinOp):
        if node.op in _BINOP_FUNCS:
            func = _BINOP_FUNCS[node.op]
            w(f"        float {node.id} = {func}({ref(node.a)}, {ref(node.b)});")
        else:
            sym = _BINOP_SYMBOLS[node.op]
            w(f"        float {node.id} = {ref(node.a)} {sym} {ref(node.b)};")

    elif isinstance(node, UnaryOp):
        if node.op == "neg":
            w(f"        float {node.id} = -{ref(node.a)};")
        elif node.op == "sign":
            a = ref(node.a)
            w(f"        float {node.id} = ({a} > 0.0f ? 1.0f : ({a} < 0.0f ? -1.0f : 0.0f));")
        else:
            func = _UNARYOP_FUNCS[node.op]
            w(f"        float {node.id} = {func}({ref(node.a)});")

    elif isinstance(node, Clamp):
        a, lo, hi = ref(node.a), ref(node.lo), ref(node.hi)
        w(f"        float {node.id} = fminf(fmaxf({a}, {lo}), {hi});")

    elif isinstance(node, Constant):
        w(f"        float {node.id} = {_float_lit(node.value)};")

    elif isinstance(node, History):
        # Value already loaded pre-loop; track for write-back
        history_nodes.append(node)

    elif isinstance(node, DelayLine):
        # State-only node, no per-sample computation
        pass

    elif isinstance(node, DelayRead):
        dl = node.delay
        tap = ref(node.tap)
        if node.interp == "none":
            w(
                f"        int {node.id}_pos = "
                f"(({dl}_wr - (int)({tap})) % {dl}_len + {dl}_len) % {dl}_len;"
            )
            w(f"        float {node.id} = {dl}_buf[{node.id}_pos];")
        elif node.interp == "linear":
            nid = node.id
            _emit_interp_linear(nid, dl, tap, w)
        elif node.interp == "cubic":
            nid = node.id
            _emit_interp_cubic(nid, dl, tap, w)

    elif isinstance(node, DelayWrite):
        delay_write_nodes.append(node)
        val = ref(node.value)
        w(f"        {node.delay}_buf[{node.delay}_wr] = {val};")
        w(f"        {node.delay}_wr = ({node.delay}_wr + 1) % {node.delay}_len;")

    elif isinstance(node, Phasor):
        freq = ref(node.freq)
        w(f"        float {node.id} = {node.id}_phase;")
        w(f"        {node.id}_phase += {freq} / sr;")
        w(f"        if ({node.id}_phase >= 1.0f) {node.id}_phase -= 1.0f;")

    elif isinstance(node, Noise):
        w(f"        {node.id}_seed = {node.id}_seed * 1664525u + 1013904223u;")
        w(f"        float {node.id} = (float)(int32_t){node.id}_seed / 2147483648.0f;")

    elif isinstance(node, Compare):
        sym = _COMPARE_SYMBOLS[node.op]
        w(f"        float {node.id} = (float)({ref(node.a)} {sym} {ref(node.b)});")

    elif isinstance(node, Select):
        w(f"        float {node.id} = {ref(node.cond)} > 0.0f ? {ref(node.a)} : {ref(node.b)};")

    elif isinstance(node, Wrap):
        nid = node.id
        a, lo, hi = ref(node.a), ref(node.lo), ref(node.hi)
        w(f"        float {nid}_range = {hi} - {lo};")
        w(f"        float {nid}_raw = fmodf({a} - {lo}, {nid}_range);")
        raw_expr = f"{nid}_raw < 0.0f ? {nid}_raw + {nid}_range : {nid}_raw"
        w(f"        float {nid} = {lo} + ({raw_expr});")

    elif isinstance(node, Fold):
        nid = node.id
        a, lo, hi = ref(node.a), ref(node.lo), ref(node.hi)
        w(f"        float {nid}_range = {hi} - {lo};")
        w(f"        float {nid}_t = fmodf({a} - {lo}, 2.0f * {nid}_range);")
        w(f"        if ({nid}_t < 0.0f) {nid}_t += 2.0f * {nid}_range;")
        lo_branch = f"{lo} + {nid}_t"
        hi_branch = f"{hi} - ({nid}_t - {nid}_range)"
        w(f"        float {nid} = {nid}_t <= {nid}_range ? {lo_branch} : {hi_branch};")

    elif isinstance(node, Mix):
        a_r, b_r, t_r = ref(node.a), ref(node.b), ref(node.t)
        w(f"        float {node.id} = {a_r} + ({b_r} - {a_r}) * {t_r};")

    elif isinstance(node, Delta):
        nid = node.id
        a = ref(node.a)
        w(f"        float {nid}_cur = {a};")
        w(f"        float {nid} = {nid}_cur - {nid}_prev;")
        w(f"        {nid}_prev = {nid}_cur;")

    elif isinstance(node, Change):
        nid = node.id
        a = ref(node.a)
        w(f"        float {nid}_cur = {a};")
        w(f"        float {nid} = ({nid}_cur != {nid}_prev) ? 1.0f : 0.0f;")
        w(f"        {nid}_prev = {nid}_cur;")


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------


def _wrap_idx(expr: str, dl: str) -> str:
    """Wrap a delay index expression with positive modulo."""
    return f"(({expr}) % {dl}_len + {dl}_len) % {dl}_len"


def _emit_interp_linear(nid: str, dl: str, tap: str, w: _Writer) -> None:
    w(f"        float {nid}_ftap = {tap};")
    w(f"        int {nid}_itap = (int){nid}_ftap;")
    w(f"        float {nid}_frac = {nid}_ftap - (float){nid}_itap;")
    i0 = _wrap_idx(f"{dl}_wr - {nid}_itap", dl)
    i1 = _wrap_idx(f"{dl}_wr - {nid}_itap - 1", dl)
    w(f"        int {nid}_i0 = {i0};")
    w(f"        int {nid}_i1 = {i1};")
    s0 = f"{dl}_buf[{nid}_i0]"
    s1 = f"{dl}_buf[{nid}_i1]"
    w(f"        float {nid} = {s0} + {nid}_frac * ({s1} - {s0});")


def _emit_interp_cubic(nid: str, dl: str, tap: str, w: _Writer) -> None:
    w(f"        float {nid}_ftap = {tap};")
    w(f"        int {nid}_itap = (int){nid}_ftap;")
    w(f"        float {nid}_frac = {nid}_ftap - (float){nid}_itap;")
    i0 = _wrap_idx(f"{dl}_wr - {nid}_itap", dl)
    w(f"        int {nid}_i0 = {i0};")
    w(f"        int {nid}_im1 = ({nid}_i0 + 1) % {dl}_len;")
    i1 = _wrap_idx(f"{dl}_wr - {nid}_itap - 1", dl)
    i2 = _wrap_idx(f"{dl}_wr - {nid}_itap - 2", dl)
    w(f"        int {nid}_i1 = {i1};")
    w(f"        int {nid}_i2 = {i2};")
    w(f"        float {nid}_ym1 = {dl}_buf[{nid}_im1];")
    w(f"        float {nid}_y0 = {dl}_buf[{nid}_i0];")
    w(f"        float {nid}_y1 = {dl}_buf[{nid}_i1];")
    w(f"        float {nid}_y2 = {dl}_buf[{nid}_i2];")
    w(f"        float {nid}_c0 = {nid}_y0;")
    w(f"        float {nid}_c1 = 0.5f * ({nid}_y1 - {nid}_ym1);")
    c2a = f"{nid}_ym1 - 2.5f * {nid}_y0"
    c2b = f"2.0f * {nid}_y1 - 0.5f * {nid}_y2"
    w(f"        float {nid}_c2 = {c2a} + {c2b};")
    c3a = f"0.5f * ({nid}_y2 - {nid}_ym1)"
    c3b = f"1.5f * ({nid}_y0 - {nid}_y1)"
    w(f"        float {nid}_c3 = {c3a} + {c3b};")
    horner = (
        f"(({nid}_c3 * {nid}_frac + {nid}_c2) * {nid}_frac + {nid}_c1) * {nid}_frac + {nid}_c0"
    )
    w(f"        float {nid} = {horner};")


# ---------------------------------------------------------------------------
# Param introspection
# ---------------------------------------------------------------------------


def _emit_param_name(params: list[Param], name: str, struct_name: str, w: _Writer) -> None:
    w(f"const char* {name}_param_name(int index) {{")
    w("    switch (index) {")
    for idx, p in enumerate(params):
        w(f'    case {idx}: return "{p.name}";')
    w('    default: return "";')
    w("    }")
    w("}")


def _emit_param_minmax(
    params: list[Param], name: str, struct_name: str, which: str, w: _Writer
) -> None:
    w(f"float {name}_param_{which}(int index) {{")
    w("    switch (index) {")
    for idx, p in enumerate(params):
        val = p.min if which == "min" else p.max
        w(f"    case {idx}: return {_float_lit(val)};")
    w("    default: return 0.0f;")
    w("    }")
    w("}")


def _emit_param_set(params: list[Param], name: str, struct_name: str, w: _Writer) -> None:
    w(f"void {name}_set_param({struct_name}* self, int index, float value) {{")
    w("    switch (index) {")
    for idx, p in enumerate(params):
        w(f"    case {idx}: self->p_{p.name} = value; break;")
    w("    default: break;")
    w("    }")
    w("}")


def _emit_param_get(params: list[Param], name: str, struct_name: str, w: _Writer) -> None:
    w(f"float {name}_get_param({struct_name}* self, int index) {{")
    w("    switch (index) {")
    for idx, p in enumerate(params):
        w(f"    case {idx}: return self->p_{p.name};")
    w("    default: return 0.0f;")
    w("    }")
    w("}")
