"""Feedback delay with delay line, feedback, and dry/wet mix."""

from dsp_graph import (
    AudioInput,
    AudioOutput,
    BinOp,
    DelayLine,
    DelayRead,
    DelayWrite,
    Graph,
    Param,
    compile_graph_to_file,
    graph_to_dot_file,
    validate_graph,
)

graph = Graph(
    name="fbdelay",
    inputs=[AudioInput(id="in1")],
    outputs=[AudioOutput(id="out1", source="mix_out")],
    params=[
        Param(name="delay_ms", min=1.0, max=1000.0, default=250.0),
        Param(name="feedback", min=0.0, max=0.95, default=0.5),
        Param(name="mix", min=0.0, max=1.0, default=0.5),
    ],
    nodes=[
        BinOp(id="sr_ms", op="div", a=44100.0, b=1000.0),
        BinOp(id="tap", op="mul", a="delay_ms", b="sr_ms"),
        DelayLine(id="dline", max_samples=48000),
        DelayRead(id="delayed", delay="dline", tap="tap"),
        BinOp(id="fb_scaled", op="mul", a="delayed", b="feedback"),
        BinOp(id="write_val", op="add", a="in1", b="fb_scaled"),
        DelayWrite(id="dwrite", delay="dline", value="write_val"),
        BinOp(id="inv_mix", op="sub", a=1.0, b="mix"),
        BinOp(id="dry", op="mul", a="in1", b="inv_mix"),
        BinOp(id="wet", op="mul", a="delayed", b="mix"),
        BinOp(id="mix_out", op="add", a="dry", b="wet"),
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
