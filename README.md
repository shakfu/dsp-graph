# dsp-graph

A Python DSL for defining DSP signal graphs, compiling them to standalone C++, and optimizing the result.

Define audio processing graphs using Pydantic models, validate them, compile to C++, and serialize to/from JSON. Zero runtime dependencies beyond Pydantic.

## Install

```bash
pip install dsp-graph
```

For development:

```bash
git clone https://github.com/shakfu/dsp-graph.git
cd dsp-graph
make install-dev
```

## Quick Start

```python
from dsp_graph import (
    AudioInput, AudioOutput, BinOp, Graph, History, Param,
    compile_graph, validate_graph,
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

code = compile_graph(graph)  # standalone C++ string
print(graph.model_dump_json(indent=2))
```

## Node Types (34)

### Arithmetic / Math

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `BinOp` | `add`, `sub`, `mul`, `div`, `min`, `max`, `mod`, `pow` | `a`, `b` | Binary arithmetic |
| `UnaryOp` | `sin`, `cos`, `tanh`, `exp`, `log`, `abs`, `sqrt`, `neg`, `floor`, `ceil`, `round`, `sign`, `atan`, `asin`, `acos` | `a` | Unary math functions |
| `Clamp` | `clamp` | `a`, `lo`, `hi` | Saturate to `[lo, hi]` |
| `Constant` | `constant` | `value` | Literal float value |
| `Compare` | `gt`, `lt`, `gte`, `lte`, `eq` | `a`, `b` | Comparison (returns 0.0 or 1.0) |
| `Select` | `select` | `cond`, `a`, `b` | Conditional: `a` if `cond > 0`, else `b` |
| `Wrap` | `wrap` | `a`, `lo`, `hi` | Wrap value into range |
| `Fold` | `fold` | `a`, `lo`, `hi` | Fold (reflect) value into range |
| `Mix` | `mix` | `a`, `b`, `t` | Linear interpolation: `a + (b - a) * t` |

### Delay

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `DelayLine` | `delay` | `max_samples` | Circular buffer declaration |
| `DelayRead` | `delay_read` | `delay`, `tap`, `interp` | Read from delay line (none/linear/cubic) |
| `DelayWrite` | `delay_write` | `delay`, `value` | Write to delay line |
| `History` | `history` | `input`, `init` | Single-sample delay (z^-1 feedback) |

### Buffer / Table

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `Buffer` | `buffer` | `size` | Random-access data buffer |
| `BufRead` | `buf_read` | `buffer`, `index`, `interp` | Read from buffer (none/linear/cubic, clamped) |
| `BufWrite` | `buf_write` | `buffer`, `index`, `value` | Write to buffer at index |
| `BufSize` | `buf_size` | `buffer` | Returns buffer length as float |

### Filters

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `Biquad` | `biquad` | `a`, `b0`, `b1`, `b2`, `a1`, `a2` | Generic biquad (user supplies coefficients) |
| `SVF` | `svf` | `a`, `freq`, `q`, `mode` | State-variable filter (lp/hp/bp/notch) |
| `OnePole` | `onepole` | `a`, `coeff` | One-pole lowpass |
| `DCBlock` | `dcblock` | `a` | DC blocking filter |
| `Allpass` | `allpass` | `a`, `coeff` | First-order allpass |

### Oscillators / Sources

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `Phasor` | `phasor` | `freq` | Ramp oscillator 0..1 |
| `SinOsc` | `sinosc` | `freq` | Sine oscillator |
| `TriOsc` | `triosc` | `freq` | Triangle wave |
| `SawOsc` | `sawosc` | `freq` | Bipolar saw (-1..1) |
| `PulseOsc` | `pulseosc` | `freq`, `width` | Pulse/square with variable duty cycle |
| `Noise` | `noise` | -- | White noise source |

### State / Timing

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `Delta` | `delta` | `a` | Sample-to-sample difference |
| `Change` | `change` | `a` | 1.0 if value changed, else 0.0 |
| `SampleHold` | `sample_hold` | `a`, `trig` | Latch on any zero crossing |
| `Latch` | `latch` | `a`, `trig` | Latch on rising edge only |
| `Accum` | `accum` | `incr`, `reset` | Running sum, resets when `reset > 0` |
| `Counter` | `counter` | `trig`, `max` | Integer counter, wraps at max |

## C++ Compilation

`compile_graph()` generates a single self-contained `.cpp` file with:

- A state struct (`{Name}State`)
- `create(sr)` / `destroy(self)` / `reset(self)` lifecycle
- `perform(self, ins, outs, n)` sample-processing loop
- Param introspection: `num_params`, `param_name`, `param_min`, `param_max`, `set_param`, `get_param`
- Buffer introspection: `num_buffers`, `buffer_name`, `buffer_size`, `get_buffer`, `set_buffer`

```python
from dsp_graph import compile_graph, compile_graph_to_file

code = compile_graph(graph)           # returns C++ string
path = compile_graph_to_file(graph, "build/")  # writes build/{name}.cpp
```

## gen-dsp Integration

dsp-graph graphs can be compiled into buildable audio plugin projects via [gen-dsp](https://github.com/shakfu/gen-dsp), which supports 11 platforms: ChucK, CLAP, AudioUnit, VST3, LV2, SuperCollider, VCV Rack, Daisy, and more.

`compile_for_gen_dsp()` generates the three files needed to drop into any gen-dsp platform backend:

```python
from dsp_graph import compile_for_gen_dsp

# Generates: test_synth.cpp, _ext_chuck.cpp, manifest.json
compile_for_gen_dsp(graph, "build/", platform="chuck")
```

The adapter replaces gen-dsp's genlib-side code while reusing its platform-side code unchanged. The generated `manifest.json` is compatible with `gen_dsp.core.manifest.Manifest`.

For a fully assembled project (requires gen-dsp installed):

```python
from dsp_graph.gen_dsp_adapter import assemble_project

# Copies platform templates + generates adapter + manifest
assemble_project(graph, "build/chuck_project", platform="chuck")
```

## Optimization

```python
from dsp_graph import optimize_graph, constant_fold, eliminate_cse, eliminate_dead_nodes

optimized = optimize_graph(graph)  # constant folding + CSE + dead node elimination
```

- **Constant folding**: pure nodes with all-constant inputs are replaced by `Constant` nodes
- **Common subexpression elimination**: duplicate pure nodes with identical inputs are merged
- **Dead node elimination**: nodes not reachable from any output are removed (respects side-effecting writers for delay lines and buffers)
- **Loop-invariant code motion**: param-only expressions are hoisted before the sample loop
- **SIMD hints**: `__restrict` on I/O pointers; vectorization pragmas for pure-only graphs

## Validation

`validate_graph()` checks:

1. Unique node IDs (no collisions with inputs or params)
2. All string references resolve to existing IDs
3. Output sources reference existing nodes
4. DelayRead/DelayWrite reference existing DelayLine nodes
5. BufRead/BufWrite/BufSize reference existing Buffer nodes
6. No pure cycles (cycles must pass through History or delay)

## Visualization

```python
from dsp_graph import graph_to_dot, graph_to_dot_file

dot_str = graph_to_dot(graph)                  # DOT string
dot_path = graph_to_dot_file(graph, "build/")  # writes .dot, renders .pdf if `dot` is on PATH
```

## Examples

See `examples/` for complete working graphs:

- `stereo_gain.py` -- stateless stereo gain
- `onepole.py` -- one-pole lowpass with History feedback
- `fbdelay.py` -- feedback delay with dry/wet mix
- `wavetable.py` -- wavetable oscillator using Buffer + Phasor + BufRead

## Development

```bash
make install-dev  # install with dev deps
make test         # run tests
make lint         # ruff check
make typecheck    # mypy --strict
make qa           # all of the above
```
