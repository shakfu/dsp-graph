import { beforeEach, describe, expect, it } from "vitest";
import type { Node } from "@xyflow/react";
import { useGraph } from "./useGraph";
import type { NodeTypeCatalog, RFNodeData } from "../api/types";

const catalog: NodeTypeCatalog = {
  mul: {
    class: "BinOp",
    color: "#fff",
    fields: {
      a: { type: "string|number", required: true },
      b: { type: "string|number", required: true },
    },
  },
};

function dspNode(
  id: string,
  data: Partial<Record<string, unknown>>
): Node<RFNodeData> {
  return {
    id,
    type: "dsp_node",
    position: { x: 0, y: 0 },
    data: {
      label: id,
      op: "mul",
      color: "#fff",
      node_data: { id, op: "mul", a: 0, b: 0, ...data },
    },
  };
}

function nodeData(id: string): Record<string, unknown> {
  const n = useGraph.getState().nodes.find((x) => x.id === id);
  return n!.data.node_data!;
}

describe("useGraph edge editing", () => {
  beforeEach(() => {
    useGraph.setState({
      nodes: [],
      edges: [],
      undoStack: [],
      redoStack: [],
      canUndo: false,
      canRedo: false,
      nodeTypeCatalog: catalog,
    });
  });

  it("addEdge writes the source into the target's node_data field", () => {
    useGraph.setState({ nodes: [dspNode("m1", {})] });
    useGraph.getState().addEdge("in1", "m1", "a");

    expect(nodeData("m1").a).toBe("in1");
    const edges = useGraph.getState().edges;
    expect(edges).toHaveLength(1);
    expect(edges[0]!.targetHandle).toBe("a");
  });

  it("addEdge replaces an existing connection into the same handle", () => {
    useGraph.setState({ nodes: [dspNode("m1", {})] });
    const g = useGraph.getState();
    g.addEdge("in1", "m1", "a");
    g.addEdge("in2", "m1", "a");

    expect(nodeData("m1").a).toBe("in2");
    const intoA = useGraph
      .getState()
      .edges.filter((e) => e.target === "m1" && e.targetHandle === "a");
    expect(intoA).toHaveLength(1);
  });

  it("addEdge into different handles coexist", () => {
    useGraph.setState({ nodes: [dspNode("m1", {})] });
    const g = useGraph.getState();
    g.addEdge("in1", "m1", "a");
    g.addEdge("gain", "m1", "b");

    expect(nodeData("m1").a).toBe("in1");
    expect(nodeData("m1").b).toBe("gain");
    expect(useGraph.getState().edges).toHaveLength(2);
  });

  it("deleteEdges resets the disconnected field to its default", () => {
    useGraph.setState({ nodes: [dspNode("m1", {})] });
    useGraph.getState().addEdge("in1", "m1", "a");
    const edgeId = useGraph.getState().edges[0]!.id;

    useGraph.getState().deleteEdges([edgeId]);

    expect(nodeData("m1").a).toBe(0);
    expect(useGraph.getState().edges).toHaveLength(0);
  });

  it("deleteNodes clears dangling references in surviving nodes", () => {
    useGraph.setState({
      nodes: [dspNode("src", {}), dspNode("m1", { a: "src" })],
      edges: [{ id: "e1", source: "src", target: "m1", targetHandle: "a" }],
    });

    useGraph.getState().deleteNodes(["src"]);

    expect(useGraph.getState().nodes.map((n) => n.id)).toEqual(["m1"]);
    expect(nodeData("m1").a).toBe(0);
    expect(useGraph.getState().edges).toHaveLength(0);
  });

  it("undo restores node_data after an edge edit", () => {
    useGraph.setState({ nodes: [dspNode("m1", {})] });
    useGraph.getState().addEdge("in1", "m1", "a");
    expect(nodeData("m1").a).toBe("in1");

    useGraph.getState().undo();
    expect(nodeData("m1").a).toBe(0);
  });
});
