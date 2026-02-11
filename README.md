# dsp-graph

A Python DSL for defining DSP signal graphs with JSON serialization.

Define audio processing graphs using Pydantic models, validate them, and serialize to/from JSON.

## Install

```bash
uv pip install -e .
```

## Usage

```python
from dsp_graph import (
    AudioInput, AudioOutput, BinOp, Graph, History, Param, validate_graph,
)

graph = Graph(
    name="onepole",
    inputs=[AudioInput(id="in1")],
    outputs=[AudioOutput(id="out1", source="result")],
    params=[Param(name="coeff", min=0.0, max=0.999, default=0.5)],
    nodes=[
        BinOp(id="inv_coeff", op="sub", a=1.0, b="coeff"),
        BinOp(id="dry", op="mul", a="in1", b="inv_coeff"),
        History(id="prev", init=0.0, input="result"),
        BinOp(id="wet", op="mul", a="prev", b="coeff"),
        BinOp(id="result", op="add", a="dry", b="wet"),
    ],
)

errors = validate_graph(graph)
assert errors == []

print(graph.model_dump_json(indent=2))
```

## Node Types

| Node | `op` | Purpose |
|------|------|---------|
| `BinOp` | `add`, `sub`, `mul`, `div` | Arithmetic on two inputs |
| `UnaryOp` | `sin`, `cos`, `tanh`, `exp`, `log`, `abs`, `sqrt`, `neg` | Math function on one input |
| `Clamp` | `clamp` | Saturate to `[lo, hi]` |
| `Constant` | `constant` | Literal float value |
| `History` | `history` | Single-sample delay (z^-1) |
| `DelayLine` | `delay` | Circular buffer declaration |
| `DelayRead` | `delay_read` | Read from delay line |
| `DelayWrite` | `delay_write` | Write to delay line |
| `Phasor` | `phasor` | Ramp oscillator 0..1 |
| `Noise` | `noise` | White noise source |

## Validation

`validate_graph()` checks:

1. Unique node IDs (no collisions with inputs or params)
2. All string references resolve to existing IDs
3. Output sources reference existing nodes
4. DelayRead/DelayWrite reference existing DelayLine nodes
5. No pure cycles (cycles must pass through History or delay)

## Development

```bash
make install-dev  # install with dev deps
make test         # run tests
make lint         # ruff check
make typecheck    # mypy --strict
make qa           # all of the above
```
