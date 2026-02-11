"""Stateless stereo gain example."""

from dsp_graph import (
    AudioInput,
    AudioOutput,
    BinOp,
    Graph,
    Param,
    compile_graph_to_file,
    graph_to_dot_file,
    validate_graph,
)

graph = Graph(
    name="stereo_gain",
    inputs=[AudioInput(id="in1"), AudioInput(id="in2")],
    outputs=[
        AudioOutput(id="out1", source="scaled1"),
        AudioOutput(id="out2", source="scaled2"),
    ],
    params=[Param(name="gain", min=0.0, max=2.0, default=1.0)],
    nodes=[
        BinOp(id="scaled1", op="mul", a="in1", b="gain"),
        BinOp(id="scaled2", op="mul", a="in2", b="gain"),
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
