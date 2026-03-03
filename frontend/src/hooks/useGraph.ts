import { create } from "zustand";
import type { Node, Edge } from "@xyflow/react";
import type {
  RFNodeData,
  SimulateResponse,
  OptimizeResponse,
  CompileResponse,
  BuildResponse,
  ValidateResponse,
  ValidationErrorDetail,
  InputSignalType,
  ParseError,
  NodeTypeCatalog,
} from "../api/types";
import {
  loadGraphJson,
  loadGraphGdsp,
  exportGraphJson,
  simulateGraph,
  simulateContinue,
  simulateSetParam,
  simulatePeek,
  simulateReset,
  optimizeGraph,
  compileGraph,
  buildGraph,
  buildGraphZip,
  getBuildPlatforms,
  validateGraph,
  getNodeTypes,
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
  simSessionId: string | null;
  peekValues: Record<string, number> | null;
  optimizeResult: OptimizeResponse | null;
  compileResult: CompileResponse | null;
  buildResult: BuildResponse | null;
  buildPlatforms: string[];
  validationResult: ValidateResponse | null;
  validationErrors: ValidationErrorDetail[];
  errorNodeIds: Set<string>;
  warnNodeIds: Set<string>;
  layoutOptions: ElkLayoutOptions;
  exportSvg: (() => Promise<void>) | null;
  inputSignals: Record<string, InputSignalType>;
  error: string | null;

  // Editor state
  gdspSource: string;
  parseError: ParseError | null;
  isLivePreview: boolean;
  showEditor: boolean;
  showGraph: boolean;

  // Node catalog
  nodeTypeCatalog: NodeTypeCatalog | null;

  setExportSvg: (fn: (() => Promise<void>) | null) => void;
  setNodes: (nodes: Node<RFNodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  selectNode: (node: Node<RFNodeData> | null) => void;
  setInputSignal: (inputId: string, signalType: InputSignalType) => void;
  loadFromJson: (json: Record<string, unknown>) => Promise<void>;
  loadFromGdsp: (source: string) => Promise<void>;
  exportJson: () => Promise<Record<string, unknown> | null>;
  runSimulation: (nSamples: number) => Promise<void>;
  continueSimulation: (nSamples: number) => Promise<void>;
  setSimParam: (name: string, value: number) => Promise<void>;
  fetchPeek: () => Promise<void>;
  resetSimulation: () => Promise<void>;
  runOptimize: () => Promise<void>;
  runCompile: () => Promise<void>;
  runBuild: (platform: string) => Promise<void>;
  downloadBuildZip: (platform: string) => Promise<void>;
  fetchBuildPlatforms: () => Promise<void>;
  runValidate: () => Promise<void>;
  setLayoutOptions: (opts: Partial<ElkLayoutOptions>) => void;
  runLayout: () => Promise<void>;
  clearError: () => void;
  setGdspSource: (source: string) => void;
  setParseError: (error: ParseError | null) => void;
  setLivePreview: (on: boolean) => void;
  setShowEditor: (show: boolean) => void;
  setShowGraph: (show: boolean) => void;
  fetchNodeTypes: () => Promise<void>;

  // Graph editing
  addNode: (op: string, position: { x: number; y: number }) => void;
  deleteNodes: (ids: string[]) => void;
  deleteEdges: (ids: string[]) => void;
  addEdge: (source: string, target: string) => void;
  duplicateNodes: (ids: string[]) => void;
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
  simSessionId: null,
  peekValues: null,
  optimizeResult: null,
  compileResult: null,
  buildResult: null,
  buildPlatforms: [],
  validationResult: null,
  validationErrors: [],
  errorNodeIds: new Set<string>(),
  warnNodeIds: new Set<string>(),
  layoutOptions: { ...DEFAULT_LAYOUT_OPTIONS },
  exportSvg: null,
  inputSignals: {},
  error: null,
  gdspSource: "",
  parseError: null,
  isLivePreview: true,
  showEditor: true,
  showGraph: true,
  nodeTypeCatalog: null,

  setExportSvg: (fn) => set({ exportSvg: fn }),
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  selectNode: (node) => set({ selectedNode: node }),
  setInputSignal: (inputId, signalType) =>
    set((s) => ({ inputSignals: { ...s.inputSignals, [inputId]: signalType } })),
  clearError: () => set({ error: null }),
  setGdspSource: (source) => set({ gdspSource: source }),
  setParseError: (error) => set({ parseError: error }),
  setLivePreview: (on) => set({ isLivePreview: on }),
  setShowEditor: (show) => set({ showEditor: show }),
  setShowGraph: (show) => set({ showGraph: show }),

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
        compileResult: null,
        validationResult: null,
        inputSignals: {},
        error: null,
      });
      await get().runLayout();
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
        gdspSource: source,
        selectedNode: null,
        simulationResult: null,
        optimizeResult: null,
        compileResult: null,
        validationResult: null,
        inputSignals: {},
        parseError: null,
        error: null,
      });
      await get().runLayout();
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
      const signals = get().inputSignals;
      const inputs = Object.keys(signals).length > 0 ? signals : undefined;
      const result = await simulateGraph(exported, nSamples, undefined, inputs);
      set({
        simulationResult: result,
        simSessionId: result.session_id,
        peekValues: null,
        error: null,
      });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  continueSimulation: async (nSamples) => {
    const sessionId = get().simSessionId;
    if (!sessionId) return;
    try {
      const signals = get().inputSignals;
      const inputs = Object.keys(signals).length > 0 ? signals : undefined;
      const result = await simulateContinue(sessionId, nSamples, inputs);
      set({ simulationResult: result, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  setSimParam: async (name, value) => {
    const sessionId = get().simSessionId;
    if (!sessionId) return;
    try {
      await simulateSetParam(sessionId, name, value);
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  fetchPeek: async () => {
    const sessionId = get().simSessionId;
    if (!sessionId) return;
    try {
      const result = await simulatePeek(sessionId);
      set({ peekValues: result.values });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  resetSimulation: async () => {
    const sessionId = get().simSessionId;
    if (!sessionId) return;
    try {
      await simulateReset(sessionId);
      set({ simulationResult: null, peekValues: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  runCompile: async () => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const result = await compileGraph(exported);
      set({ compileResult: result, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  runBuild: async (platform) => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const result = await buildGraph(exported, platform);
      set({ buildResult: result, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  downloadBuildZip: async (platform) => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const blob = await buildGraphZip(exported, platform);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${get().graphName || "graph"}_${platform}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  fetchBuildPlatforms: async () => {
    try {
      const platforms = await getBuildPlatforms();
      set({ buildPlatforms: platforms });
    } catch (e) {
      console.error("Failed to fetch build platforms:", e);
    }
  },

  runValidate: async () => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const result = await validateGraph(exported);
      const errors = result.errors;
      const errorIds = new Set<string>();
      const warnIds = new Set<string>();
      for (const e of errors) {
        if (e.node_id) {
          if (e.severity === "warning") {
            warnIds.add(e.node_id);
          } else {
            errorIds.add(e.node_id);
          }
        }
      }
      set({
        validationResult: result,
        validationErrors: errors,
        errorNodeIds: errorIds,
        warnNodeIds: warnIds,
        error: null,
      });
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

  fetchNodeTypes: async () => {
    try {
      const result = await getNodeTypes();
      set({ nodeTypeCatalog: result.catalog });
    } catch (e) {
      console.error("Failed to fetch node types:", e);
    }
  },

  addNode: (op, position) => {
    const catalog = get().nodeTypeCatalog;
    if (!catalog || !catalog[op]) return;
    const info = catalog[op];
    const id = `${op}_${Date.now().toString(36)}`;
    // Build default node_data from catalog fields
    const nodeData: Record<string, unknown> = { id, op };
    for (const [fname, finfo] of Object.entries(info.fields)) {
      if (finfo.default !== undefined) {
        nodeData[fname] = finfo.default;
      }
    }
    const newNode: Node<RFNodeData> = {
      id,
      type: "dsp_node",
      position,
      data: {
        label: id,
        op,
        color: info.color,
        node_data: nodeData,
      },
    };
    set((s) => ({ nodes: [...s.nodes, newNode] }));
  },

  deleteNodes: (ids) => {
    const idSet = new Set(ids);
    set((s) => ({
      nodes: s.nodes.filter((n) => !idSet.has(n.id)),
      edges: s.edges.filter((e) => !idSet.has(e.source) && !idSet.has(e.target)),
      selectedNode: s.selectedNode && idSet.has(s.selectedNode.id) ? null : s.selectedNode,
    }));
  },

  deleteEdges: (ids) => {
    const idSet = new Set(ids);
    set((s) => ({
      edges: s.edges.filter((e) => !idSet.has(e.id)),
    }));
  },

  addEdge: (source, target) => {
    const id = `e_${source}_${target}_${Date.now().toString(36)}`;
    const newEdge: Edge = { id, source, target };
    set((s) => ({ edges: [...s.edges, newEdge] }));
  },

  duplicateNodes: (ids) => {
    const { nodes } = get();
    const toDuplicate = nodes.filter((n) => ids.includes(n.id));
    const newNodes = toDuplicate.map((n) => {
      const newId = `${n.data.op ?? n.type}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
      return {
        ...n,
        id: newId,
        position: { x: n.position.x + 50, y: n.position.y + 50 },
        data: {
          ...n.data,
          label: newId,
          node_data: n.data.node_data
            ? { ...n.data.node_data, id: newId }
            : undefined,
        },
      };
    });
    set((s) => ({ nodes: [...s.nodes, ...newNodes] }));
  },
}));
