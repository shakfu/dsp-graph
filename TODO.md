# TODO

## Planned Node Types

### Math / Logic

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `UnaryOp` | `atan`, `asin`, `acos` | `a` | Inverse trig |

### State / Timing

| Node | `op` | Fields | Purpose |
|------|------|--------|---------|
| `SampleHold` | `sample_hold` | `a`, `trig` | Latch value of `a` when `trig` crosses zero |
| `Accum` | `accum` | `incr`, `reset` | Running sum, resets to 0 when `reset > 0` |
| `Counter` | `counter` | `trig`, `max` | Integer counter, wraps at max |
| `Latch` | `latch` | `a`, `trig` | Like sample_hold but only latches on rising edge |
| `RateDiv` | `rate_div` | `a`, `divisor` | Output every N-th sample, hold between |

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

- [x] `Biquad` node with coefficient inputs
- [x] `SVF` node (state-variable filter: lp/hp/bp/notch)
- [x] `OnePole` and `DCBlock` as built-in compound nodes
- [x] `Allpass` node
- [x] Oscillator primitives: `SinOsc`, `TriOsc`, `SawOsc`, `PulseOsc`
- [x] `SampleHold`, `Latch`, `Accum`, `Counter` state nodes
- [x] Extend `UnaryOp` with `atan`, `asin`, `acos` (inverse trig)

### v0.4 -- Buffers and Tables

- [x] `Buffer` / `BufRead` / `BufWrite` / `BufSize` node family
- [x] Interpolation modes for `BufRead` (none, linear, cubic)
- [x] External buffer loading API (fill from Python, pass to C++)
- [x] Wavetable oscillator as `Buffer` + `Phasor` + `BufRead`

### v0.5 -- gen-dsp Integration

- [x] Thin C++ adapter: gen-dsp `_ext.h` interface wrapping dsp-graph `create/destroy/reset/perform`
- [x] Generate adapter alongside `.cpp` so dsp-graph outputs drop into gen-dsp platform backends
- [x] Shared param introspection (manifest.json, compatible with `gen_dsp.core.manifest.Manifest`)
- [x] Integration tests: dsp-graph -> gen-dsp -> ChucK/CLAP/AU g++ compilation
- [x] `{name}_reset()` function: reinitialize all state without reallocating

### v0.6 -- Advanced Compiler Optimizations

- [x] Common subexpression elimination
- [x] Loop-invariant code motion (hoist param-only expressions out of sample loop)
- [x] SIMD vectorization hints (mark inner loop as vectorizable)

### Future

- [ ] Subgraph / macro nodes (inline a Graph as a node in another Graph)
- [ ] Multi-rate processing (control-rate vs audio-rate distinction)
- [ ] CLI tool: `dsp-graph compile graph.json -o build/`
- [ ] Python DSP simulator (interpret graph in Python for prototyping)
- [ ] WebAssembly codegen target
- [ ] FAUST-style block diagram algebra (series `>>`, parallel `<<`, split/merge)
