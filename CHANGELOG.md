# Changelog

## 0.6.0

### Compiler optimizations

- New `eliminate_cse()` optimization pass: graph-level common subexpression elimination merges duplicate pure nodes (same type, op, and resolved refs) and rewrites all references to the canonical node. Commutative ops (`add`, `mul`, `min`, `max`) are detected regardless of operand order. Stateful nodes are never merged. Integrated into `optimize_graph()` between constant folding and dead node elimination.
- Loop-invariant code motion (LICM): pure nodes whose inputs are all params, literals, or other invariant nodes are hoisted before the `for` loop, reducing per-sample overhead.
- SIMD vectorization hints: I/O buffer pointers are now declared `float* __restrict` to inform the compiler they don't alias. For pure-only graphs (no stateful nodes), a `#pragma clang loop vectorize(enable)` / `#pragma GCC ivdep` is emitted before the sample loop.

### Other

- Version bumped to 0.6.0

## 0.5.0

### gen-dsp integration

- New `compile_for_gen_dsp(graph, output_dir, platform)` generates adapter files that bridge dsp-graph output to gen-dsp's 11 platform backends (ChucK, CLAP, AU, VST3, LV2, SC, VCV Rack, Daisy, etc.)
- New `generate_adapter_cpp(graph, platform)` generates `_ext_{platform}.cpp` implementing gen-dsp's wrapper interface (`wrapper_create`, `wrapper_perform`, params, buffers) by delegating to dsp-graph's compiled API
- New `generate_manifest(graph)` produces `manifest.json` compatible with `gen_dsp.core.manifest.Manifest.from_json()`
- New `assemble_project(graph, output_dir, platform)` assembles a complete buildable project by combining dsp-graph output with gen-dsp platform templates (requires gen-dsp installed)

### Reset function

- Generated C++ now includes `{name}_reset()` -- reinitializes all state to creation defaults without reallocating memory
- Covers all stateful node types: History (to init value), DelayLine/Buffer (zeroed via `memset`), oscillators/Phasor (phase reset), Noise (seed reset), filters (state zeroed), SampleHold/Latch/Accum/Counter (zeroed)
- Generated code now includes `<cstring>` for `memset`

### Other

- Version bumped to 0.5.0
- New `gen_dsp_graph` test fixture exercising oscillators, filters, delays, buffers, and params

## 0.1.4

- Added `Buffer` / `BufRead` / `BufWrite` / `BufSize` node family (4 new node types, 34 total)
- `BufRead` supports `none`, `linear`, and `cubic` interpolation (clamped indices)
- `BufWrite` is side-effect only (no output value); OOB writes silently ignored
- External buffer loading API: `num_buffers`, `buffer_name`, `buffer_size`, `get_buffer`, `set_buffer`
- `set_buffer` copies data in (min of len and capacity), zero-fills remainder
- Buffer consistency validation (BufRead/BufWrite/BufSize must reference existing Buffer)
- Dead node elimination tracks buffer writers (BufWrite kept alive when BufRead/BufSize is reachable)
- All 4 buffer types added to stateful set (never constant-folded)
- Graphviz visualization for buffer nodes (peach fill, box3d for Buffer)
- Wavetable oscillator example (`examples/wavetable.py`)

## 0.1.3

- Added filter nodes: `Biquad`, `SVF` (lp/hp/bp/notch), `OnePole`, `DCBlock`, `Allpass`
- Added oscillator nodes: `SinOsc`, `TriOsc`, `SawOsc`, `PulseOsc`
- Added state/timing nodes: `SampleHold`, `Latch`, `Accum`, `Counter`
- Extended `UnaryOp` with inverse trig: `atan`, `asin`, `acos`
- Graphviz visualization for all new node types

## 0.1.1

- Added `Compare` (`gt`, `lt`, `gte`, `lte`, `eq`) and `Select` nodes
- Added `Wrap` and `Fold` boundary nodes
- Extended `BinOp` with `min`, `max`, `mod`, `pow`
- Extended `UnaryOp` with `floor`, `ceil`, `round`, `sign`
- Added `Mix` node (linear interpolation between two signals)
- Added `Delta` and `Change` nodes (signal edge detection)
- Optimization: constant folding (pure nodes with all-constant inputs collapsed)
- Optimization: dead node elimination (unreachable nodes removed)
- Interpolated delay reads: `linear` and `cubic` modes on `DelayRead`

## 0.1.0

- Core graph model with Pydantic: `Graph`, `Param`, `AudioInput`, `AudioOutput`
- Node types: `BinOp` (add/sub/mul/div), `UnaryOp` (sin/cos/tanh/exp/log/abs/sqrt/neg), `Clamp`, `Constant`
- Feedback: `History` (z^-1), `DelayLine`/`DelayRead`/`DelayWrite`
- Sources: `Phasor`, `Noise`
- Graph validation: unique IDs, reference resolution, delay consistency, cycle detection
- C++ code generation: struct-based state, `create`/`destroy`/`perform`, param API
- Topological sort with feedback edge handling
- Graphviz DOT visualization with PDF rendering
- JSON serialization via Pydantic discriminated unions
