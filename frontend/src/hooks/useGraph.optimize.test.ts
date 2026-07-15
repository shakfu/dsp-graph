import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Node } from "@xyflow/react";
import type { RFNodeData } from "../api/types";

// Mock only the client functions runSinglePass touches; keep the rest real.
vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    // exportJson round-trips the current canvas to Graph JSON; return a marker.
    exportGraphJson: vi.fn(async () => ({ name: "g", nodes: [], outputs: [] })),
    exportGraphGdsp: vi.fn(async () => ({ source: "graph g {}" })),
    // Each pass returns an "optimized" graph with a single node; the node id
    // encodes the pass so we can see composition apply to the canvas.
    optimizePass: vi.fn(async (_graph: unknown, passName: string) => ({
      original: { nodes: [], edges: [] },
      optimized: {
        nodes: [{ id: `after_${passName}`, type: "dsp_node", position: { x: 0, y: 0 }, data: {} }],
        edges: [],
      },
      stats: { nodes_before: 2, nodes_after: 1, nodes_removed: 1 },
    })),
  };
});

import { useGraph } from "./useGraph";

function seedNode(id: string): Node<RFNodeData> {
  return { id, type: "dsp_node", position: { x: 0, y: 0 }, data: {} as RFNodeData };
}

describe("useGraph.runSinglePass (optimize step-through)", () => {
  beforeEach(() => {
    useGraph.setState({
      nodes: [seedNode("a"), seedNode("b")],
      edges: [],
      graphName: "g",
      passResults: [],
      preOptimizeSnapshot: null,
      error: null,
    });
  });

  it("applies one pass to the canvas and records its result", async () => {
    await useGraph.getState().runSinglePass("constant_fold");
    const s = useGraph.getState();
    expect(s.nodes.map((n) => n.id)).toEqual(["after_constant_fold"]);
    expect(s.passResults).toEqual([
      { passName: "constant_fold", nodesBefore: 2, nodesAfter: 1 },
    ]);
    expect(s.error).toBeNull();
  });

  it("snapshots the original only once across a step-through sequence", async () => {
    await useGraph.getState().runSinglePass("constant_fold");
    const snap1 = useGraph.getState().preOptimizeSnapshot;
    expect(snap1?.nodes.map((n) => n.id)).toEqual(["a", "b"]); // the true original

    await useGraph.getState().runSinglePass("eliminate_cse");
    const snap2 = useGraph.getState().preOptimizeSnapshot;

    // Same snapshot object -> the second pass did NOT overwrite it with the
    // post-first-pass state, so Reset returns to the real original.
    expect(snap2).toBe(snap1);
    expect(useGraph.getState().passResults).toHaveLength(2);
  });

  it("resetOptimize restores the pre-optimize graph and clears state", async () => {
    await useGraph.getState().runSinglePass("constant_fold");
    await useGraph.getState().runSinglePass("eliminate_dead_nodes");

    useGraph.getState().resetOptimize();
    const s = useGraph.getState();
    expect(s.nodes.map((n) => n.id)).toEqual(["a", "b"]);
    expect(s.passResults).toEqual([]);
    expect(s.preOptimizeSnapshot).toBeNull();
  });
});
