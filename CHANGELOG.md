# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5]

### Added

- `eliminate_cse()` optimization pass: graph-level common subexpression elimination merges duplicate pure nodes (same type, op, and resolved refs) and rewrites all references to the canonical node. Commutative ops (`add`, `mul`, `min`, `max`) detected regardless of operand order. Stateful nodes never merged. Integrated into `optimize_graph()` between constant folding and dead node elimination.
- Loop-invariant code motion (LICM): pure nodes whose inputs are all params, literals, or other invariant nodes are hoisted before the `for` loop, reducing per-sample overhead.
- SIMD vectorization hints: I/O buffer pointers declared `float* __restrict` to inform the compiler they don't alias. For pure-only graphs (no stateful nodes), `#pragma clang loop vectorize(enable)` / `#pragma GCC ivdep` emitted before the sample loop.

## [0.1.4]

### Added

- `compile_for_gen_dsp(graph, output_dir, platform)` generates adapter files bridging dsp-graph output to gen-dsp platform backends (ChucK, CLAP, AU, VST3, LV2, SC, VCV Rack, Daisy, etc.).
- `generate_adapter_cpp(graph, platform)` generates `_ext_{platform}.cpp` implementing gen-dsp's wrapper interface by delegating to dsp-graph's compiled API.
- `generate_manifest(graph)` produces `manifest.json` compatible with `gen_dsp.core.manifest.Manifest.from_json()`.
- `assemble_project(graph, output_dir, platform)` assembles a complete buildable project by combining dsp-graph output with gen-dsp platform templates (requires gen-dsp installed).
- `{name}_reset()` function: reinitializes all state to creation defaults without reallocating memory. Covers all stateful node types.
- Generated code now includes `<cstring>` for `memset`.

## [0.1.3]

### Added

- `Buffer` / `BufRead` / `BufWrite` / `BufSize` node family (4 new node types, 34 total).
- `BufRead` supports `none`, `linear`, and `cubic` interpolation (clamped indices).
- `BufWrite` is side-effect only (no output value); OOB writes silently ignored.
- External buffer loading API: `num_buffers`, `buffer_name`, `buffer_size`, `get_buffer`, `set_buffer`.
- Buffer consistency validation (BufRead/BufWrite/BufSize must reference existing Buffer).
- Dead node elimination tracks buffer writers (BufWrite kept alive when BufRead/BufSize is reachable).
- Graphviz visualization for buffer nodes.
- Wavetable oscillator example (`examples/wavetable.py`).

## [0.1.2]

### Added

- Filter nodes: `Biquad`, `SVF` (lp/hp/bp/notch), `OnePole`, `DCBlock`, `Allpass`.
- Oscillator nodes: `SinOsc`, `TriOsc`, `SawOsc`, `PulseOsc`.
- State/timing nodes: `SampleHold`, `Latch`, `Accum`, `Counter`.
- `UnaryOp` inverse trig: `atan`, `asin`, `acos`.
- Graphviz visualization for all new node types.

## [0.1.1]

### Added

- `Compare` (`gt`, `lt`, `gte`, `lte`, `eq`) and `Select` nodes.
- `Wrap` and `Fold` boundary nodes.
- `BinOp` ops: `min`, `max`, `mod`, `pow`.
- `UnaryOp` ops: `floor`, `ceil`, `round`, `sign`.
- `Mix` node (linear interpolation between two signals).
- `Delta` and `Change` nodes (signal edge detection).
- Constant folding optimization (pure nodes with all-constant inputs collapsed).
- Dead node elimination (unreachable nodes removed).
- Interpolated delay reads: `linear` and `cubic` modes on `DelayRead`.

## [0.1.0]

### Added

- Core graph model with Pydantic: `Graph`, `Param`, `AudioInput`, `AudioOutput`.
- Node types: `BinOp` (add/sub/mul/div), `UnaryOp` (sin/cos/tanh/exp/log/abs/sqrt/neg), `Clamp`, `Constant`.
- Feedback: `History` (z^-1), `DelayLine`/`DelayRead`/`DelayWrite`.
- Sources: `Phasor`, `Noise`.
- Graph validation: unique IDs, reference resolution, delay consistency, cycle detection.
- C++ code generation: struct-based state, `create`/`destroy`/`perform`, param API.
- Topological sort with feedback edge handling.
- Graphviz DOT visualization with PDF rendering.
- JSON serialization via Pydantic discriminated unions.
