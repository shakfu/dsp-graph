import { create } from "zustand";
import type { Node, Edge } from "@xyflow/react";
import type { RFNodeData, SimulateResponse, OptimizeResponse } from "../api/types";
import {
  loadGraphJson,
  loadGraphGdsp,
  exportGraphJson,
  simulateGraph,
  optimizeGraph,
} from "../api/client";
import {
  elkLayout,
  DEFAULT_LAYOUT_OPTIONS,
  type ElkLayoutOptions,
} from "../utils/elkLayout";

interface GraphState {
  nodes: Node<RFNodeData>[];
  edges: Edge[];
  graphName: string;
  selectedNode: Node<RFNodeData> | null;
  simulationResult: SimulateResponse | null;
  optimizeResult: OptimizeResponse | null;
  layoutOptions: ElkLayoutOptions;
  exportSvg: (() => Promise<void>) | null;
  error: string | null;

  setExportSvg: (fn: (() => Promise<void>) | null) => void;
  setNodes: (nodes: Node<RFNodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  selectNode: (node: Node<RFNodeData> | null) => void;
  loadFromJson: (json: Record<string, unknown>) => Promise<void>;
  loadFromGdsp: (source: string) => Promise<void>;
  exportJson: () => Promise<Record<string, unknown> | null>;
  runSimulation: (nSamples: number) => Promise<void>;
  runOptimize: () => Promise<void>;
  setLayoutOptions: (opts: Partial<ElkLayoutOptions>) => void;
  runLayout: () => Promise<void>;
  clearError: () => void;
}

function toFlowNodes(
  rfNodes: Array<{
    id: string;
    type: string;
    position: { x: number; y: number };
    data: RFNodeData;
  }>
): Node<RFNodeData>[] {
  return rfNodes.map((n) => ({
    id: n.id,
    type: n.type,
    position: n.position,
    data: n.data,
  }));
}

function toFlowEdges(
  rfEdges: Array<{
    id: string;
    source: string;
    target: string;
    animated?: boolean;
    label?: string;
  }>
): Edge[] {
  return rfEdges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    animated: e.animated ?? false,
    label: e.label ?? undefined,
  }));
}

export const useGraph = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  graphName: "",
  selectedNode: null,
  simulationResult: null,
  optimizeResult: null,
  layoutOptions: { ...DEFAULT_LAYOUT_OPTIONS },
  exportSvg: null,
  error: null,

  setExportSvg: (fn) => set({ exportSvg: fn }),
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  selectNode: (node) => set({ selectedNode: node }),
  clearError: () => set({ error: null }),

  loadFromJson: async (json) => {
    try {
      const rf = await loadGraphJson(json);
      set({
        nodes: toFlowNodes(rf.nodes),
        edges: toFlowEdges(rf.edges),
        graphName: rf.name,
        selectedNode: null,
        simulationResult: null,
        optimizeResult: null,
        error: null,
      });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  loadFromGdsp: async (source) => {
    try {
      const rf = await loadGraphGdsp(source);
      set({
        nodes: toFlowNodes(rf.nodes),
        edges: toFlowEdges(rf.edges),
        graphName: rf.name,
        selectedNode: null,
        simulationResult: null,
        optimizeResult: null,
        error: null,
      });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  exportJson: async () => {
    const { nodes, edges, graphName } = get();
    try {
      const rf = {
        nodes: nodes.map((n) => ({
          id: n.id,
          type: n.type ?? "dsp_node",
          position: n.position,
          data: n.data as RFNodeData,
        })),
        edges: edges.map((e) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          animated: e.animated ?? false,
          label: typeof e.label === "string" ? e.label : undefined,
        })),
        name: graphName,
        sample_rate: 44100,
        control_interval: 0,
      };
      return await exportGraphJson(rf);
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
      return null;
    }
  },

  runSimulation: async (nSamples) => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const result = await simulateGraph(exported, nSamples);
      set({ simulationResult: result, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  setLayoutOptions: (opts) =>
    set((s) => ({ layoutOptions: { ...s.layoutOptions, ...opts } })),

  runLayout: async () => {
    const { nodes, edges, layoutOptions } = get();
    if (nodes.length === 0) return;
    try {
      const laid = await elkLayout(nodes, edges, layoutOptions);
      set({ nodes: laid, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  runOptimize: async () => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const result = await optimizeGraph(exported);
      set({
        nodes: toFlowNodes(result.optimized.nodes),
        edges: toFlowEdges(result.optimized.edges),
        optimizeResult: result,
        error: null,
      });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },
}));
