# Pydantic DSP Graph Vocabulary

A sketch of how Pydantic models could define a DSP graph that serializes to JSON and compiles to C++.

## Architecture

```
Python (Pydantic models)  -->  JSON (intermediate)  -->  C++ (compiled output)
      build graph              .model_dump_json()         codegen pass
```

## The Vocabulary

### Core Models

```python
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Param & I/O declarations
# ---------------------------------------------------------------------------

class Param(BaseModel):
    name: str
    min: float = 0.0
    max: float = 1.0
    default: float = 0.0


class AudioInput(BaseModel):
    id: str                     # e.g. "in1"


class AudioOutput(BaseModel):
    id: str                     # e.g. "out1"
    source: str                 # node ID that feeds this output


# ---------------------------------------------------------------------------
# Node types (discriminated union on "op")
# ---------------------------------------------------------------------------

class BinOp(BaseModel):
    """Arithmetic binary operator."""
    id: str
    op: Literal["add", "sub", "mul", "div"]
    a: str | float              # node ID or literal
    b: str | float


class UnaryOp(BaseModel):
    """Math function applied to a single input."""
    id: str
    op: Literal["sin", "cos", "tanh", "exp", "log", "abs", "sqrt"]
    a: str | float


class Clamp(BaseModel):
    """Clamp a signal to [lo, hi]."""
    id: str
    op: Literal["clamp"] = "clamp"
    a: str | float
    lo: str | float = 0.0
    hi: str | float = 1.0


class History(BaseModel):
    """Single-sample delay (z^-1). Breaks feedback loops."""
    id: str
    op: Literal["history"] = "history"
    init: float = 0.0
    input: str                  # node ID whose value is stored for next sample


class DelayLine(BaseModel):
    """Multi-sample circular buffer declaration."""
    id: str
    op: Literal["delay"] = "delay"
    max_samples: int = 48000


class DelayRead(BaseModel):
    """Read from a delay line at a tap position (in samples)."""
    id: str
    op: Literal["delay_read"] = "delay_read"
    delay: str                  # delay line ID
    tap: str | float            # tap position node ID or literal


class DelayWrite(BaseModel):
    """Write a value into a delay line."""
    id: str
    op: Literal["delay_write"] = "delay_write"
    delay: str                  # delay line ID
    value: str | float          # node ID or literal to write


class Phasor(BaseModel):
    """Ramp oscillator 0..1 at given frequency."""
    id: str
    op: Literal["phasor"] = "phasor"
    freq: str | float


class Noise(BaseModel):
    """White noise source."""
    id: str
    op: Literal["noise"] = "noise"


# Discriminated union of all node types
Node = Annotated[
    Union[
        BinOp, UnaryOp, Clamp, History,
        DelayLine, DelayRead, DelayWrite,
        Phasor, Noise,
    ],
    Field(discriminator="op"),
]


# ---------------------------------------------------------------------------
# Top-level graph
# ---------------------------------------------------------------------------

class Graph(BaseModel):
    name: str
    sample_rate: float = 44100.0
    inputs: list[AudioInput] = []
    outputs: list[AudioOutput] = []
    params: list[Param] = []
    nodes: list[Node] = []
```

### What Each Node Type Covers

| Node | State | Purpose |
|------|-------|---------|
| `BinOp` | none | `+`, `-`, `*`, `/` on two inputs |
| `UnaryOp` | none | `sin`, `cos`, `tanh`, `exp`, etc. |
| `Clamp` | none | Saturate a signal to a range |
| `History` | 1 sample | `z^-1` operator; breaks feedback loops |
| `DelayLine` | N samples | Declares a circular buffer |
| `DelayRead` | -- | Reads from a delay line at a tap |
| `DelayWrite` | -- | Writes into a delay line |
| `Phasor` | 1 sample | Accumulating ramp oscillator |
| `Noise` | RNG seed | White noise generator |

This is a minimal vocabulary. A real implementation would add: `Biquad`, `Latch`, `SampleHold`, `BufferRead`/`BufferWrite` (for sample playback), `Select`, `Gate`, comparison ops, etc.

---

## Example: One-Pole Lowpass

### Building the Graph in Python

```python
graph = Graph(
    name="onepole",
    inputs=[AudioInput(id="in1")],
    outputs=[AudioOutput(id="out1", source="result")],
    params=[Param(name="coeff", min=0.0, max=0.999, default=0.5)],
    nodes=[
        # (1 - coeff)
        BinOp(id="inv_coeff", op="sub", a=1.0, b="coeff"),
        # in1 * (1 - coeff)
        BinOp(id="dry", op="mul", a="in1", b="inv_coeff"),
        # previous output
        History(id="prev", init=0.0, input="result"),
        # prev * coeff
        BinOp(id="wet", op="mul", a="prev", b="coeff"),
        # dry + wet
        BinOp(id="result", op="add", a="dry", b="wet"),
    ],
)
```

### JSON Output (`graph.model_dump_json(indent=2)`)

```json
{
  "name": "onepole",
  "sample_rate": 44100.0,
  "inputs": [
    { "id": "in1" }
  ],
  "outputs": [
    { "id": "out1", "source": "result" }
  ],
  "params": [
    { "name": "coeff", "min": 0.0, "max": 0.999, "default": 0.5 }
  ],
  "nodes": [
    { "id": "inv_coeff", "op": "sub", "a": 1.0, "b": "coeff" },
    { "id": "dry",       "op": "mul", "a": "in1", "b": "inv_coeff" },
    { "id": "prev",      "op": "history", "init": 0.0, "input": "result" },
    { "id": "wet",       "op": "mul", "a": "prev", "b": "coeff" },
    { "id": "result",    "op": "add", "a": "dry", "b": "wet" }
  ]
}
```

### C++ Output (from codegen pass)

```cpp
struct State {
    float p_coeff;
    float m_prev;       // history: prev
};

State* create(float sr) {
    State* self = (State*)calloc(1, sizeof(State));
    self->p_coeff = 0.5f;
    self->m_prev = 0.0f;
    return self;
}

void perform(State* self, float** ins, float** outs, int n) {
    float* in1  = ins[0];
    float* out1 = outs[0];
    float coeff = self->p_coeff;
    float prev  = self->m_prev;

    for (int i = 0; i < n; i++) {
        float inv_coeff = 1.0f - coeff;
        float dry       = in1[i] * inv_coeff;
        float wet       = prev * coeff;
        float result    = dry + wet;

        out1[i] = result;
        prev = result;      // history feedback
    }

    self->m_prev = prev;
}
```

---

## Example: Feedback Delay

### Python

```python
graph = Graph(
    name="fbdelay",
    inputs=[AudioInput(id="in1")],
    outputs=[AudioOutput(id="out1", source="mix_out")],
    params=[
        Param(name="delay_ms", min=1.0, max=1000.0, default=250.0),
        Param(name="feedback", min=0.0, max=0.95, default=0.5),
        Param(name="mix", min=0.0, max=1.0, default=0.5),
    ],
    nodes=[
        # delay time: ms -> samples
        BinOp(id="sr_ms", op="div", a=44100.0, b=1000.0),
        BinOp(id="tap", op="mul", a="delay_ms", b="sr_ms"),

        # delay line
        DelayLine(id="dline", max_samples=48000),
        DelayRead(id="delayed", delay="dline", tap="tap"),

        # feedback path: delayed * feedback + input
        BinOp(id="fb_scaled", op="mul", a="delayed", b="feedback"),
        BinOp(id="write_val", op="add", a="in1", b="fb_scaled"),
        DelayWrite(id="dwrite", delay="dline", value="write_val"),

        # dry/wet mix
        BinOp(id="inv_mix", op="sub", a=1.0, b="mix"),
        BinOp(id="dry", op="mul", a="in1", b="inv_mix"),
        BinOp(id="wet", op="mul", a="delayed", b="mix"),
        BinOp(id="mix_out", op="add", a="dry", b="wet"),
    ],
)
```

### JSON Output

```json
{
  "name": "fbdelay",
  "sample_rate": 44100.0,
  "inputs": [{ "id": "in1" }],
  "outputs": [{ "id": "out1", "source": "mix_out" }],
  "params": [
    { "name": "delay_ms", "min": 1.0, "max": 1000.0, "default": 250.0 },
    { "name": "feedback", "min": 0.0, "max": 0.95, "default": 0.5 },
    { "name": "mix",      "min": 0.0, "max": 1.0,  "default": 0.5 }
  ],
  "nodes": [
    { "id": "sr_ms",     "op": "div",         "a": 44100.0, "b": 1000.0 },
    { "id": "tap",       "op": "mul",         "a": "delay_ms", "b": "sr_ms" },
    { "id": "dline",     "op": "delay",       "max_samples": 48000 },
    { "id": "delayed",   "op": "delay_read",  "delay": "dline", "tap": "tap" },
    { "id": "fb_scaled", "op": "mul",         "a": "delayed", "b": "feedback" },
    { "id": "write_val", "op": "add",         "a": "in1", "b": "fb_scaled" },
    { "id": "dwrite",    "op": "delay_write", "delay": "dline", "value": "write_val" },
    { "id": "inv_mix",   "op": "sub",         "a": 1.0, "b": "mix" },
    { "id": "dry",       "op": "mul",         "a": "in1", "b": "inv_mix" },
    { "id": "wet",       "op": "mul",         "a": "delayed", "b": "mix" },
    { "id": "mix_out",   "op": "add",         "a": "dry", "b": "wet" }
  ]
}
```

### C++ Output

```cpp
struct State {
    float  p_delay_ms;
    float  p_feedback;
    float  p_mix;
    float* m_dline_buf;
    int    m_dline_len;
    int    m_dline_wr;
};

State* create(float sr) {
    State* self = (State*)calloc(1, sizeof(State));
    self->p_delay_ms = 250.0f;
    self->p_feedback = 0.5f;
    self->p_mix = 0.5f;
    self->m_dline_len = 48000;
    self->m_dline_buf = (float*)calloc(48000, sizeof(float));
    self->m_dline_wr = 0;
    return self;
}

void perform(State* self, float** ins, float** outs, int n) {
    float* in1  = ins[0];
    float* out1 = outs[0];
    float delay_ms = self->p_delay_ms;
    float feedback = self->p_feedback;
    float mix      = self->p_mix;
    float* buf     = self->m_dline_buf;
    int    len     = self->m_dline_len;
    int    wr      = self->m_dline_wr;

    float sr_ms = 44100.0f / 1000.0f;
    float tap   = delay_ms * sr_ms;

    for (int i = 0; i < n; i++) {
        // delay_read
        int rd = wr - (int)tap;
        if (rd < 0) rd += len;
        float delayed = buf[rd];

        // feedback + write
        float fb_scaled = delayed * feedback;
        float write_val = in1[i] + fb_scaled;
        buf[wr] = write_val;
        wr = (wr + 1) % len;

        // dry/wet mix
        float inv_mix = 1.0f - mix;
        float dry     = in1[i] * inv_mix;
        float wet     = delayed * mix;
        out1[i]       = dry + wet;
    }

    self->m_dline_wr = wr;
}
```

---

## The Codegen Pass

The compiler that transforms `Graph` -> C++ needs roughly three stages:

### 1. Topological Sort

Resolve evaluation order. Nodes reference each other by ID; the compiler builds a dependency graph and sorts it. Feedback edges (`History.input`, `DelayWrite.value` -> `DelayRead`) are back-edges that break cycles.

```python
def toposort(graph: Graph) -> list[Node]:
    """Sort nodes so each node's inputs are computed before it.

    History and DelayRead nodes read from the *previous* sample,
    so their feedback inputs are not dependencies for ordering.
    """
    ...
```

### 2. State Layout

Walk sorted nodes and collect stateful elements:

```python
@dataclass
class StateField:
    c_type: str         # "float", "float*", "int"
    name: str           # "m_prev", "m_dline_buf"
    init: str           # "0.0f", "calloc(48000, sizeof(float))"

def collect_state(graph: Graph) -> list[StateField]:
    fields = []
    for p in graph.params:
        fields.append(StateField("float", f"p_{p.name}", f"{p.default}f"))
    for node in graph.nodes:
        if isinstance(node, History):
            fields.append(StateField("float", f"m_{node.id}", f"{node.init}f"))
        elif isinstance(node, DelayLine):
            fields.append(StateField("float*", f"m_{node.id}_buf", ...))
            fields.append(StateField("int", f"m_{node.id}_len", ...))
            fields.append(StateField("int", f"m_{node.id}_wr", "0"))
    return fields
```

### 3. Code Emission

Walk sorted nodes and emit one C++ statement per node:

```python
def emit_node(node: Node) -> str:
    match node:
        case BinOp(id=id, op="add", a=a, b=b):
            return f"float {id} = {ref(a)} + {ref(b)};"
        case BinOp(id=id, op="mul", a=a, b=b):
            return f"float {id} = {ref(a)} * {ref(b)};"
        case History(id=id):
            return f"float {id} = self->m_{id};"  # read previous
        case DelayRead(id=id, delay=d, tap=t):
            return (
                f"int _rd_{id} = wr_{d} - (int){ref(t)};\n"
                f"if (_rd_{id} < 0) _rd_{id} += len_{d};\n"
                f"float {id} = buf_{d}[_rd_{id}];"
            )
        ...

def ref(x: str | float) -> str:
    """Resolve a node reference or literal to a C expression."""
    if isinstance(x, float):
        return f"{x}f"
    return x  # node ID doubles as C variable name
```

---

## What This Buys You

| Concern | gen~ export path | Pydantic graph path |
|---------|-----------------|---------------------|
| DSP definition | Max/MSP GUI | Python code |
| IR format | C++ (opaque) | JSON (inspectable, transformable) |
| Compiler | gen~ (closed source) | Your own (open, extensible) |
| Host wrappers | gen-dsp scaffolder | Same gen-dsp scaffolder |
| Operator set | gen~ vocabulary | Whatever you define |

The JSON IR is the key artifact: it's diffable, version-controllable, and machine-transformable. You could lint it, optimize it, visualize it, or target backends that C++ can't reach (WASM, FPGA HDL, shader code).

## Open Questions

1. **Sub-graphs**: Gen~ has `gen` (sub-patchers). Represent as nested `Graph` nodes or flatten?
2. **Multi-rate**: Some nodes run at control rate vs. audio rate. Add a `rate` field?
3. **Interpolation**: `DelayRead` with fractional taps needs interpolation mode (none, linear, cubic). Add to the model?
4. **SIMD**: The codegen could emit explicit SIMD intrinsics for vectorizable subgraphs. Worth the complexity?
5. **Validation**: Pydantic validators can enforce that all node ID references actually exist, catch cycles outside of History/Delay, and verify parameter ranges at graph construction time.
