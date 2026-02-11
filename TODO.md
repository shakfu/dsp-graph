# TODO

## Planned Node Types

All originally planned node types have been implemented (38 total). See README.md for the full list.


## Roadmap

### Core Completeness

- [x] Add `Compare` and `Select` nodes (conditional logic)
- [x] Add `Wrap` and `Fold` nodes (waveshaping boundaries)
- [x] Extend `BinOp` with `min`, `max`, `mod`, `pow`
- [x] Extend `UnaryOp` with `floor`, `ceil`, `round`, `sign`
- [x] Add `Mix` node (very common DSP pattern)
- [x] Add `Delta` and `Change` nodes (signal edge detection)
- [x] Optimization pass: constant folding (e.g. `44100.0 / 1000.0` -> `44.1f`)
- [x] Optimization pass: dead node elimination (unreachable from outputs)
- [x] Interpolated delay reads (linear/cubic) -- currently integer-only

### Filters and Oscillators

- [x] `Biquad` node with coefficient inputs
- [x] `SVF` node (state-variable filter: lp/hp/bp/notch)
- [x] `OnePole` and `DCBlock` as built-in compound nodes
- [x] `Allpass` node
- [x] Oscillator primitives: `SinOsc`, `TriOsc`, `SawOsc`, `PulseOsc`
- [x] `SampleHold`, `Latch`, `Accum`, `Counter` state nodes
- [x] Extend `UnaryOp` with `atan`, `asin`, `acos` (inverse trig)

### Buffers and Tables

- [x] `Buffer` / `BufRead` / `BufWrite` / `BufSize` node family
- [x] Interpolation modes for `BufRead` (none, linear, cubic)
- [x] External buffer loading API (fill from Python, pass to C++)
- [x] Wavetable oscillator as `Buffer` + `Phasor` + `BufRead`

### gen-dsp Integration

- [x] Thin C++ adapter: gen-dsp `_ext.h` interface wrapping dsp-graph `create/destroy/reset/perform`
- [x] Generate adapter alongside `.cpp` so dsp-graph outputs drop into gen-dsp platform backends
- [x] Shared param introspection (manifest.json, compatible with `gen_dsp.core.manifest.Manifest`)
- [x] Integration tests: dsp-graph -> gen-dsp -> ChucK/CLAP/AU g++ compilation
- [x] `{name}_reset()` function: reinitialize all state without reallocating

### Advanced Compiler Optimizations

- [x] Common subexpression elimination
- [x] Loop-invariant code motion (hoist param-only expressions out of sample loop)
- [x] SIMD vectorization hints (mark inner loop as vectorizable)

### New Node Types

- [x] `RateDiv` node (output every N-th sample, hold between)
- [x] `Scale` node (linear range mapping: `in_lo..in_hi` -> `out_lo..out_hi`)
- [x] `SmoothParam` node (one-pole smoothing for parameter changes)
- [x] `Peek` node (debug pass-through readable externally)

### CLI

- [x] CLI tool: `dsp-graph compile graph.json -o build/`
- [x] CLI tool: `dsp-graph validate graph.json`
- [x] CLI tool: `dsp-graph dot graph.json -o build/`

### Future

- [x] Subgraph / macro nodes (inline a Graph as a node in another Graph)
- [x] Multi-rate processing (control-rate vs audio-rate distinction)
- [x] Python DSP simulator (interpret graph in Python for prototyping)
- [ ] WebAssembly codegen target
- [x] FAUST-style block diagram algebra (series `>>`, parallel `<<`, split/merge)
