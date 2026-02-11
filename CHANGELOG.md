# Changelog

## 0.4.0 -- Buffers and Tables

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

## 0.3.0 -- Filters and Oscillators

- Added filter nodes: `Biquad`, `SVF` (lp/hp/bp/notch), `OnePole`, `DCBlock`, `Allpass`
- Added oscillator nodes: `SinOsc`, `TriOsc`, `SawOsc`, `PulseOsc`
- Added state/timing nodes: `SampleHold`, `Latch`, `Accum`, `Counter`
- Extended `UnaryOp` with inverse trig: `atan`, `asin`, `acos`
- Graphviz visualization for all new node types

## 0.2.0 -- Core Completeness

- Added `Compare` (`gt`, `lt`, `gte`, `lte`, `eq`) and `Select` nodes
- Added `Wrap` and `Fold` boundary nodes
- Extended `BinOp` with `min`, `max`, `mod`, `pow`
- Extended `UnaryOp` with `floor`, `ceil`, `round`, `sign`
- Added `Mix` node (linear interpolation between two signals)
- Added `Delta` and `Change` nodes (signal edge detection)
- Optimization: constant folding (pure nodes with all-constant inputs collapsed)
- Optimization: dead node elimination (unreachable nodes removed)
- Interpolated delay reads: `linear` and `cubic` modes on `DelayRead`

## 0.1.0 -- Initial Release

- Core graph model with Pydantic: `Graph`, `Param`, `AudioInput`, `AudioOutput`
- Node types: `BinOp` (add/sub/mul/div), `UnaryOp` (sin/cos/tanh/exp/log/abs/sqrt/neg), `Clamp`, `Constant`
- Feedback: `History` (z^-1), `DelayLine`/`DelayRead`/`DelayWrite`
- Sources: `Phasor`, `Noise`
- Graph validation: unique IDs, reference resolution, delay consistency, cycle detection
- C++ code generation: struct-based state, `create`/`destroy`/`perform`, param API
- Topological sort with feedback edge handling
- Graphviz DOT visualization with PDF rendering
- JSON serialization via Pydantic discriminated unions
