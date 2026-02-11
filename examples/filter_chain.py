"""Filter chain using graph algebra: two one-pole filters in series.

Demonstrates FAUST-style block diagram combinators:
  - Define reusable filter blocks as standalone Graph objects.
  - Compose with series() / the >> operator.
  - Params are automatically namespaced (lpf_coeff, hpf_coeff).
  - expand_subgraphs() flattens the result into a single graph that
    compiles and simulates like any other.
"""

from dsp_graph import (
    AudioInput,
    AudioOutput,
    Graph,
    OnePole,
    Param,
    compile_graph_to_file,
    graph_to_dot_file,
    validate_graph,
)
from dsp_graph.algebra import series

# Reusable filter blocks
lpf = Graph(
    name="lpf",
    inputs=[AudioInput(id="x")],
    outputs=[AudioOutput(id="y", source="filt")],
    params=[Param(name="coeff", min=0.0, max=1.0, default=0.8)],
    nodes=[OnePole(id="filt", a="x", coeff="coeff")],
)

hpf = Graph(
    name="hpf",
    inputs=[AudioInput(id="x")],
    outputs=[AudioOutput(id="y", source="filt")],
    params=[Param(name="coeff", min=0.0, max=1.0, default=0.1)],
    nodes=[OnePole(id="filt", a="x", coeff="coeff")],
)

# Compose: lpf >> hpf (series -- output of lpf feeds input of hpf)
# Resulting graph has: 1 input, 1 output, 2 params (lpf_coeff, hpf_coeff)
graph = series(lpf, hpf)

# Or equivalently using the operator overload:
# graph = lpf >> hpf

if __name__ == "__main__":
    errors = validate_graph(graph)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Graph is valid.")
    print()
    print(f"Name: {graph.name}")
    print(f"Inputs: {[i.id for i in graph.inputs]}")
    print(f"Outputs: {[o.id for o in graph.outputs]}")
    print(f"Params: {[p.name for p in graph.params]}")
    print()
    path = compile_graph_to_file(graph, "build")
    print(f"Generated: {path}")
    print(path.read_text())
    dot_path = graph_to_dot_file(graph, "build")
    print(f"DOT: {dot_path}")
