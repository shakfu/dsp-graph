"""Multi-rate synth: control-rate envelope with audio-rate oscillator.

Demonstrates two-tier processing in a generator (no audio inputs):
  - Control-rate: a SmoothParam tracks the amplitude target, producing a
    staircase envelope that updates every 32 samples.
  - Audio-rate: a sine oscillator runs per-sample and is scaled by the
    held envelope value.

The control-rate envelope avoids per-sample smoothing overhead while still
providing glitch-free amplitude transitions.
"""

from dsp_graph import (
    AudioOutput,
    BinOp,
    Graph,
    Param,
    SinOsc,
    SmoothParam,
    compile_graph_to_file,
    graph_to_dot_file,
    validate_graph,
)

graph = Graph(
    name="multirate_synth",
    sample_rate=48000.0,
    control_interval=32,
    control_nodes=["env"],
    outputs=[AudioOutput(id="out1", source="scaled")],
    params=[
        Param(name="freq", min=20.0, max=20000.0, default=440.0),
        Param(name="amp", min=0.0, max=1.0, default=0.0),
    ],
    nodes=[
        # Control-rate: smooth amplitude envelope (updates every 32 samples)
        SmoothParam(id="env", a="amp", coeff=0.995),
        # Audio-rate: sine oscillator and gain
        SinOsc(id="osc", freq="freq"),
        BinOp(id="scaled", op="mul", a="osc", b="env"),
    ],
)

if __name__ == "__main__":
    errors = validate_graph(graph)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Graph is valid.")
    print()
    print(graph.model_dump_json(indent=2))
    path = compile_graph_to_file(graph, "build")
    print(f"\nGenerated: {path}")
    print(path.read_text())
    dot_path = graph_to_dot_file(graph, "build")
    print(f"DOT: {dot_path}")
