# TODO

## New Node Types

### Math / Logic

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `BinOp` | `min`, `max` | `a`, `b` | Extend existing BinOp with min/max |
| `BinOp` | `mod`, `pow` | `a`, `b` | Extend existing BinOp with modulo and power |
| `UnaryOp` | `floor`, `ceil`, `round` | `a` | Rounding / truncation |
| `UnaryOp` | `sign` | `a` | Sign function (-1, 0, 1) |
| `UnaryOp` | `atan`, `asin`, `acos` | `a` | Inverse trig |
| `Compare` | `gt`, `lt`, `gte`, `lte`, `eq` | `a`, `b` | Returns 1.0 or 0.0 |
| `Select` | `select` | `cond`, `a`, `b` | `cond > 0 ? a : b` -- conditional routing |
| `Wrap` | `wrap` | `a`, `lo`, `hi` | Modular wrap (like phasor but general) |
| `Fold` | `fold` | `a`, `lo`, `hi` | Fold/reflect at boundaries (waveshaping) |

### State / Timing

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `SampleHold` | `sample_hold` | `a`, `trig` | Latch value of `a` when `trig` crosses zero |
| `Accum` | `accum` | `incr`, `reset` | Running sum, resets to 0 when `reset > 0` |
| `Counter` | `counter` | `trig`, `max` | Integer counter, wraps at max |
| `Latch` | `latch` | `a`, `trig` | Like sample_hold but only latches on rising edge |
| `RateDiv` | `rate_div` | `a`, `divisor` | Output every N-th sample, hold between |
| `Change` | `change` | `a` | Outputs 1.0 on sample where `a` changes, else 0.0 |
| `Delta` | `delta` | `a` | Outputs `a[n] - a[n-1]` (first difference) |

### Oscillators / Sources

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `SinOsc` | `sinosc` | `freq` | Sine oscillator (phasor + sinf, but anti-aliased) |
| `TriOsc` | `triosc` | `freq` | Triangle wave from phasor |
| `SawOsc` | `sawosc` | `freq` | Bipolar saw (-1..1) from phasor |
| `PulseOsc` | `pulseosc` | `freq`, `width` | Pulse/square with variable duty cycle |

### Filters (Compound Nodes)

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `Biquad` | `biquad` | `a`, `b0`..`b2`, `a1`, `a2` | Generic biquad (user supplies coefficients) |
| `OnePole` | `onepole` | `a`, `coeff` | Built-in one-pole lowpass (convenience) |
| `SVF` | `svf` | `a`, `freq`, `q`, `mode` | State-variable filter (lp/hp/bp/notch) |
| `Allpass` | `allpass` | `a`, `coeff` | First-order allpass |
| `DCBlock` | `dcblock` | `a` | DC blocking filter (fixed coefficient) |

### Buffer / Table

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `Buffer` | `buffer` | `size` | Named data buffer (like DelayLine but general) |
| `BufRead` | `buf_read` | `buffer`, `index` | Read from buffer at index (with interpolation) |
| `BufWrite` | `buf_write` | `buffer`, `index`, `value` | Write to buffer at index |
| `BufSize` | `buf_size` | `buffer` | Returns buffer length as float |

### Utility

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `Mix` | `mix` | `a`, `b`, `t` | Linear interpolation: `a*(1-t) + b*t` |
| `Scale` | `scale` | `a`, `in_lo`, `in_hi`, `out_lo`, `out_hi` | Linear range mapping |
| `SmoothParam` | `smooth_param` | `a`, `coeff` | One-pole smoothing for parameter changes |
| `Peek` | `peek` | `a` | Debug: pass-through that can be read externally |


## Roadmap

### v0.2 -- Core Completeness

- [x] Add `Compare` and `Select` nodes (conditional logic)
- [x] Add `Wrap` and `Fold` nodes (waveshaping boundaries)
- [x] Extend `BinOp` with `min`, `max`, `mod`, `pow`
- [x] Extend `UnaryOp` with `floor`, `ceil`, `round`, `sign`
- [x] Add `Mix` node (very common DSP pattern)
- [x] Add `Delta` and `Change` nodes (signal edge detection)
- [x] Optimization pass: constant folding (e.g. `44100.0 / 1000.0` -> `44.1f`)
- [x] Optimization pass: dead node elimination (unreachable from outputs)
- [x] Interpolated delay reads (linear/cubic) -- currently integer-only

### v0.3 -- Filters and Oscillators

- [ ] `Biquad` node with coefficient inputs
- [ ] `SVF` node (state-variable filter: lp/hp/bp/notch)
- [ ] `OnePole` and `DCBlock` as built-in compound nodes
- [ ] `Allpass` node
- [ ] Oscillator primitives: `SinOsc`, `TriOsc`, `SawOsc`, `PulseOsc`
- [ ] `SampleHold`, `Latch`, `Accum`, `Counter` state nodes

### v0.4 -- Buffers and Tables

- [ ] `Buffer` / `BufRead` / `BufWrite` / `BufSize` node family
- [ ] Interpolation modes for `BufRead` (none, linear, cubic)
- [ ] External buffer loading API (fill from Python, pass to C++)
- [ ] Wavetable oscillator as `Buffer` + `Phasor` + `BufRead`

### v0.5 -- gen-dsp Integration

- [ ] Thin C++ adapter: gen-dsp `_ext.h` interface wrapping dsp-graph `create/destroy/perform`
- [ ] Generate adapter alongside `.cpp` so dsp-graph outputs drop into gen-dsp platform backends
- [ ] Shared param introspection (gen-dsp parser reads dsp-graph param API)
- [ ] Integration tests: dsp-graph -> gen-dsp -> ChucK/CLAP/AU build

### v0.6 -- Compiler Optimizations

- [ ] Constant folding (evaluate pure subgraphs at compile time)
- [ ] Dead code elimination (prune nodes not reachable from any output)
- [ ] Common subexpression elimination
- [ ] Loop-invariant code motion (hoist param-only expressions out of sample loop)
- [ ] SIMD vectorization hints (mark inner loop as vectorizable)

### Future

- [ ] Subgraph / macro nodes (inline a Graph as a node in another Graph)
- [ ] Multi-rate processing (control-rate vs audio-rate distinction)
- [ ] Graph visualization (DOT/Graphviz export)
- [ ] CLI tool: `dsp-graph compile graph.json -o build/`
- [ ] Python DSP simulator (interpret graph in Python for prototyping)
- [ ] WebAssembly codegen target
- [ ] FAUST-style block diagram algebra (series `>>`, parallel `<<`, split/merge)
