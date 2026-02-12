# DSP Graph Representation

This document sketches what a JSON DSP graph might look like and the corresponding C++ code a compiler would emit. Three examples illustrate increasing complexity: a stateless gain, a stateful filter, and a feedback delay.

## 1. Stereo Gain (stateless)

Signal flow: `in -> * gain -> out`

### Graph

```json
{
  "name": "stereo_gain",
  "inputs": ["in1", "in2"],
  "outputs": ["out1", "out2"],
  "params": [
    { "name": "gain", "min": 0.0, "max": 1.0, "default": 0.5 }
  ],
  "nodes": [
    { "id": "mul1", "op": "mul", "ins": ["in1", "gain"] },
    { "id": "mul2", "op": "mul", "ins": ["in2", "gain"] }
  ],
  "edges": [
    { "from": "mul1", "to": "out1" },
    { "from": "mul2", "to": "out2" }
  ]
}
```

### C++

No state, no history. The `perform()` function is pure arithmetic.

```cpp
struct State {
    float p_gain;
};

void perform(State* self, float** ins, float** outs, int n) {
    float* in1  = ins[0];
    float* in2  = ins[1];
    float* out1 = outs[0];
    float* out2 = outs[1];
    float gain = self->p_gain;

    for (int i = 0; i < n; i++) {
        out1[i] = in1[i] * gain;
        out2[i] = in2[i] * gain;
    }
}
```

---

## 2. One-Pole Lowpass (stateful)

Signal flow: `out = (1 - coeff) * in + coeff * prev`

The `history` node stores the previous output sample. This is the simplest form of state in a DSP graph -- a single-sample delay in a feedback path.

### Graph

```json
{
  "name": "onepole",
  "inputs": ["in1"],
  "outputs": ["out1"],
  "params": [
    { "name": "coeff", "min": 0.0, "max": 0.999, "default": 0.5 }
  ],
  "nodes": [
    { "id": "h1",   "op": "history", "init": 0.0 },
    { "id": "sub1", "op": "sub",     "ins": [1.0, "coeff"] },
    { "id": "mul1", "op": "mul",     "ins": ["in1", "sub1"] },
    { "id": "mul2", "op": "mul",     "ins": ["h1", "coeff"] },
    { "id": "add1", "op": "add",     "ins": ["mul1", "mul2"] }
  ],
  "edges": [
    { "from": "add1", "to": "out1" },
    { "from": "add1", "to": "h1" }
  ]
}
```

Note the feedback edge: `add1 -> h1`. The `history` node breaks what would otherwise be a circular dependency by providing the *previous* sample's value while accepting the *current* sample's value for next time.

### C++

The compiler resolves the feedback loop by introducing `m_history` state and scheduling operations in topological order.

```cpp
struct State {
    float p_coeff;
    float m_history;   // h1: previous output
};

State* create(float sr) {
    State* self = (State*)calloc(1, sizeof(State));
    self->p_coeff = 0.5f;
    self->m_history = 0.0f;
    return self;
}

void perform(State* self, float** ins, float** outs, int n) {
    float* in1  = ins[0];
    float* out1 = outs[0];
    float coeff = self->p_coeff;
    float history = self->m_history;

    for (int i = 0; i < n; i++) {
        float dry = in1[i] * (1.0f - coeff);   // sub1, mul1
        float wet = history * coeff;             // mul2
        float y   = dry + wet;                   // add1

        out1[i] = y;       // edge: add1 -> out1
        history = y;        // edge: add1 -> h1 (feedback)
    }

    self->m_history = history;
}
```

---

## 3. Feedback Delay (delay line + feedback loop)

Signal flow:

```text
in --+--> [delay] --+--> out
     ^              |
     |   feedback   |
     +---[* fb]<----+
```

The delay line introduces *multi-sample* state (a circular buffer), versus the single-sample state of `history`.

### Graph

```json
{
  "name": "fbdelay",
  "inputs": ["in1"],
  "outputs": ["out1"],
  "params": [
    { "name": "delay_ms", "min": 1.0,  "max": 1000.0, "default": 250.0 },
    { "name": "feedback", "min": 0.0,  "max": 0.95,   "default": 0.5 },
    { "name": "mix",      "min": 0.0,  "max": 1.0,    "default": 0.5 }
  ],
  "state": [
    { "id": "dline", "type": "delay", "max_samples": 48000 }
  ],
  "nodes": [
    { "id": "ms2samp", "op": "mul",     "ins": ["delay_ms", "@samplerate_ms"] },
    { "id": "read1",   "op": "delay_read",  "delay": "dline", "tap": "ms2samp" },
    { "id": "fb_mul",  "op": "mul",     "ins": ["read1", "feedback"] },
    { "id": "write_v", "op": "add",     "ins": ["in1", "fb_mul"] },
    { "id": "write1",  "op": "delay_write", "delay": "dline", "value": "write_v" },
    { "id": "dry",     "op": "mul",     "ins": ["in1", { "op": "sub", "ins": [1.0, "mix"] }] },
    { "id": "wet",     "op": "mul",     "ins": ["read1", "mix"] },
    { "id": "sum",     "op": "add",     "ins": ["dry", "wet"] }
  ],
  "edges": [
    { "from": "sum", "to": "out1" }
  ]
}
```

`@samplerate_ms` is a built-in constant (`samplerate / 1000`). The `delay_read`/`delay_write` pair operate on a named delay line. The feedback path is implicit: `read1 -> fb_mul -> write_v -> write1 -> dline -> read1`.

### C++

The compiler allocates the circular buffer, converts delay time to samples, and resolves the feedback ordering (read before write).

```cpp
struct State {
    float  p_delay_ms;
    float  p_feedback;
    float  p_mix;
    float  sr_ms;           // samplerate / 1000
    float* m_delay_buf;     // circular buffer
    int    m_delay_len;     // max delay in samples
    int    m_write_idx;     // write head position
};

State* create(float sr) {
    State* self = (State*)calloc(1, sizeof(State));
    self->p_delay_ms = 250.0f;
    self->p_feedback = 0.5f;
    self->p_mix = 0.5f;
    self->sr_ms = sr / 1000.0f;
    self->m_delay_len = 48000;
    self->m_delay_buf = (float*)calloc(48000, sizeof(float));
    self->m_write_idx = 0;
    return self;
}

void perform(State* self, float** ins, float** outs, int n) {
    float* in1  = ins[0];
    float* out1 = outs[0];

    float delay_samp = self->p_delay_ms * self->sr_ms;  // ms2samp
    float feedback   = self->p_feedback;
    float mix        = self->p_mix;
    float* buf       = self->m_delay_buf;
    int    len       = self->m_delay_len;
    int    wr        = self->m_write_idx;

    for (int i = 0; i < n; i++) {
        // delay_read: read from circular buffer
        int rd = wr - (int)delay_samp;
        if (rd < 0) rd += len;
        float delayed = buf[rd];                // read1

        // feedback path
        float fb_val = delayed * feedback;      // fb_mul
        float write_val = in1[i] + fb_val;      // write_v

        // delay_write: write to circular buffer
        buf[wr] = write_val;                    // write1
        wr = (wr + 1) % len;

        // dry/wet mix
        float dry = in1[i] * (1.0f - mix);
        float wet = delayed * mix;
        out1[i] = dry + wet;                    // sum -> out1
    }

    self->m_write_idx = wr;
}
```

---

## What a Compiler Would Need

Going from the JSON graph to the C++ above requires:

1. **Operator vocabulary** -- semantics for each `op` (`add`, `mul`, `history`, `delay_read`, `delay_write`, `cycle`, `biquad`, `phasor`, ...). Gen~ has hundreds.
2. **Topological sort** -- schedule nodes so that inputs are computed before outputs, with feedback edges broken at `history`/`delay` boundaries.
3. **State allocation** -- determine how much memory each `history`/`delay`/`buffer` node needs, lay out the `State` struct.
4. **Type inference** -- scalar vs. signal rate, integer vs. float, buffer index types.
5. **Optimization** -- constant folding, dead code elimination, SIMD vectorization, loop fusion.

This is the domain of DSP compilers like [FAUST](https://faust.grame.fr/), gen~, and RNBO. Gen-dsp currently operates *downstream* of this compilation: it takes the already-emitted C++ and wraps it for different plugin hosts.
