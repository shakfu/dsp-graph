"""Wavetable oscillator: Buffer + Phasor + BufRead with linear interpolation."""

from dsp_graph import (
    AudioOutput,
    BinOp,
    Buffer,
    BufRead,
    BufSize,
    Graph,
    Param,
    Phasor,
    compile_graph_to_file,
    graph_to_dot_file,
    validate_graph,
)

graph = Graph(
    name="wavetable",
    outputs=[AudioOutput(id="out1", source="sample")],
    params=[
        Param(name="freq", min=0.1, max=20000.0, default=440.0),
    ],
    nodes=[
        # Wavetable buffer (1024 samples, filled externally via set_buffer)
        Buffer(id="wt", size=1024),
        BufSize(id="wt_len", buffer="wt"),
        # Phasor produces 0..1 ramp at the desired frequency
        Phasor(id="phase", freq="freq"),
        # Scale phasor to buffer index range: phase * wt_len
        BinOp(id="idx", op="mul", a="phase", b="wt_len"),
        # Read from wavetable with linear interpolation
        BufRead(id="sample", buffer="wt", index="idx", interp="linear"),
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
