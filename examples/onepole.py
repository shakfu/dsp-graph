"""One-pole lowpass filter with History feedback."""

from dsp_graph import (
    AudioInput,
    AudioOutput,
    BinOp,
    Graph,
    History,
    Param,
    compile_graph_to_file,
    graph_to_dot_file,
    validate_graph,
)

graph = Graph(
    name="onepole",
    inputs=[AudioInput(id="in1")],
    outputs=[AudioOutput(id="out1", source="result")],
    params=[Param(name="coeff", min=0.0, max=0.999, default=0.5)],
    nodes=[
        BinOp(id="inv_coeff", op="sub", a=1.0, b="coeff"),
        BinOp(id="dry", op="mul", a="in1", b="inv_coeff"),
        History(id="prev", init=0.0, input="result"),
        BinOp(id="wet", op="mul", a="prev", b="coeff"),
        BinOp(id="result", op="add", a="dry", b="wet"),
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
