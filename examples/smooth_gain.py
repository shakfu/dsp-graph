"""Smoothed gain with control-rate parameter interpolation.

Demonstrates multi-rate processing: the gain parameter is smoothed at
control rate (once per 64-sample block) to eliminate zipper noise, then
applied per-sample in the audio-rate inner loop.
"""

from dsp_graph import (
    AudioInput,
    AudioOutput,
    BinOp,
    Graph,
    Param,
    SmoothParam,
    compile_graph_to_file,
    graph_to_dot_file,
    validate_graph,
)

graph = Graph(
    name="smooth_gain",
    sample_rate=48000.0,
    control_interval=64,
    control_nodes=["smooth_vol"],
    inputs=[AudioInput(id="in1"), AudioInput(id="in2")],
    outputs=[
        AudioOutput(id="out1", source="scaled1"),
        AudioOutput(id="out2", source="scaled2"),
    ],
    params=[Param(name="vol", min=0.0, max=2.0, default=1.0)],
    nodes=[
        # Control-rate: smooth the volume param (updates every 64 samples)
        SmoothParam(id="smooth_vol", a="vol", coeff=0.99),
        # Audio-rate: apply smoothed gain per sample
        BinOp(id="scaled1", op="mul", a="in1", b="smooth_vol"),
        BinOp(id="scaled2", op="mul", a="in2", b="smooth_vol"),
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
