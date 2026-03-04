import { create } from "zustand";
import type { Node, Edge } from "@xyflow/react";
import type {
  RFNodeData,
  SimulateResponse,
  OptimizeResponse,
  CompileResponse,
  GenerateResponse,
  BuildResponse,
  ValidateResponse,
  ValidationErrorDetail,
  InputSignalType,
  ParseError,
  NodeTypeCatalog,
  OptimizePassName,
} from "../api/types";
import {
  loadGraphJson,
  loadGraphGdsp,
  exportGraphJson,
  exportGraphGdsp,
  simulateGraph,
  simulateContinue,
  simulateSetParam,
  simulatePeek,
  simulateReset,
  getBuffer,
  optimizePass,
  compileGraph,
  generateGraph,
  generateGraphZip,
  buildGraph,
  downloadBuiltBinary,
  getGeneratePlatforms,
  batchBuild,
  downloadBatchZip,
  validateGraph,
  getNodeTypes,
} from "../api/client";
import {
  elkLayout,
  DEFAULT_LAYOUT_OPTIONS,
  type ElkLayoutOptions,
} from "../utils/elkLayout";

interface UndoSnapshot {
  nodes: Node<RFNodeData>[];
  edges: Edge[];
}

const MAX_UNDO = 50;

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
  generateResult: GenerateResponse | null;
  buildResult: BuildResponse | null;
  buildPlatforms: string[];
  validationResult: ValidateResponse | null;
  validationErrors: ValidationErrorDetail[];
  errorNodeIds: Set<string>;
  warnNodeIds: Set<string>;
  cycleNodeIds: Set<string>;
  accumulatedOutputs: Record<string, number[]>;
  layoutOptions: ElkLayoutOptions;
  exportSvg: (() => Promise<void>) | null;
  inputSignals: Record<string, InputSignalType>;
  error: string | null;

  // Optimization pass results
  passResults: { passName: string; nodesBefore: number; nodesAfter: number }[];
  preOptimizeSnapshot: { nodes: Node<RFNodeData>[]; edges: Edge[]; gdspSource: string } | null;
  resetOptimize: () => void;

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
  exportGdsp: () => Promise<string | null>;
  runSimulation: (nSamples: number) => Promise<void>;
  continueSimulation: (nSamples: number) => Promise<void>;
  setSimParam: (name: string, value: number) => Promise<void>;
  fetchPeek: () => Promise<void>;
  fetchBuffer: (bufferId: string) => Promise<void>;
  bufferData: Record<string, number[]>;
  resetSimulation: () => Promise<void>;
  runOptimize: () => Promise<void>;
  runCompile: () => Promise<void>;
  runGenerate: (platform: string) => Promise<void>;
  runBuild: (platform: string) => Promise<void>;
  downloadGenerateZip: (platform: string) => Promise<void>;
  downloadBuiltBinary: (platform: string) => Promise<void>;
  fetchBuildPlatforms: () => Promise<void>;
  batchBuildResults: BuildResponse[];
  batchBuildId: string | null;
  runBatchBuild: (platforms: string[]) => Promise<void>;
  downloadBatchBuildZip: () => Promise<void>;
  runValidate: () => Promise<void>;
  layoutVersion: number;
  setLayoutOptions: (opts: Partial<ElkLayoutOptions>) => void;
  runLayout: () => Promise<void>;
  clearError: () => void;
  setGdspSource: (source: string) => void;
  setParseError: (error: ParseError | null) => void;
  setLivePreview: (on: boolean) => void;
  setShowEditor: (show: boolean) => void;
  setShowGraph: (show: boolean) => void;
  fetchNodeTypes: () => Promise<void>;

  // Undo/redo
  undoStack: UndoSnapshot[];
  redoStack: UndoSnapshot[];
  pushUndo: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;

  // Graph editing
  addNode: (op: string, position: { x: number; y: number }) => void;
  deleteNodes: (ids: string[]) => void;
  deleteEdges: (ids: string[]) => void;
  addEdge: (source: string, target: string) => void;
  duplicateNodes: (ids: string[]) => void;
  selectNodeById: (id: string) => void;
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

const STORAGE_KEY = "dsp-graph:gdspSource";

function loadSavedSource(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export const useGraph = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  graphName: "",
  selectedNode: null,
  simulationResult: null,
  simSessionId: null,
  peekValues: null,
  bufferData: {},
  optimizeResult: null,
  compileResult: null,
  generateResult: null,
  buildResult: null,
  buildPlatforms: [],
  batchBuildResults: [],
  batchBuildId: null,
  validationResult: null,
  validationErrors: [],
  errorNodeIds: new Set<string>(),
  warnNodeIds: new Set<string>(),
  cycleNodeIds: new Set<string>(),
  accumulatedOutputs: {},
  layoutVersion: 0,
  layoutOptions: { ...DEFAULT_LAYOUT_OPTIONS },
  exportSvg: null,
  inputSignals: {},
  error: null,
  gdspSource: loadSavedSource(),
  parseError: null,
  isLivePreview: true,
  showEditor: true,
  showGraph: true,
  nodeTypeCatalog: null,
  passResults: [],
  preOptimizeSnapshot: null,
  undoStack: [],
  redoStack: [],
  canUndo: false,
  canRedo: false,

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

  exportGdsp: async () => {
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
      const result = await exportGraphGdsp(rf);
      return result.source;
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
        accumulatedOutputs: { ...result.outputs },
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
      const prev = get().accumulatedOutputs;
      const acc: Record<string, number[]> = {};
      for (const [key, samples] of Object.entries(result.outputs)) {
        acc[key] = [...(prev[key] ?? []), ...samples];
      }
      set({ simulationResult: result, accumulatedOutputs: acc, error: null });
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

  fetchBuffer: async (bufferId) => {
    const sessionId = get().simSessionId;
    if (!sessionId) return;
    try {
      const result = await getBuffer(sessionId, bufferId);
      set((s) => ({
        bufferData: { ...s.bufferData, [bufferId]: result.data },
      }));
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  resetSimulation: async () => {
    const sessionId = get().simSessionId;
    if (!sessionId) return;
    try {
      await simulateReset(sessionId);
      set({ simulationResult: null, accumulatedOutputs: {}, peekValues: null });
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

  runGenerate: async (platform) => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const result = await generateGraph(exported, platform);
      set({ generateResult: result, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  downloadGenerateZip: async (platform) => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const blob = await generateGraphZip(exported, platform);
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

  downloadBuiltBinary: async (platform) => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const blob = await downloadBuiltBinary(exported, platform);
      const result = get().buildResult;
      const filename = result?.output_file || `${get().graphName || "graph"}_${platform}`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  runBatchBuild: async (platforms) => {
    const exported = await get().exportJson();
    if (!exported) return;
    try {
      const result = await batchBuild(exported, platforms);
      set({ batchBuildResults: result.results, batchBuildId: result.batch_id, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  downloadBatchBuildZip: async () => {
    const batchId = get().batchBuildId;
    if (!batchId) return;
    try {
      const blob = await downloadBatchZip(batchId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${get().graphName || "graph"}_all_platforms.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  fetchBuildPlatforms: async () => {
    try {
      const platforms = await getGeneratePlatforms();
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
      const cycleIds = new Set<string>();
      for (const e of errors) {
        if (e.node_id) {
          if (e.severity === "warning") {
            warnIds.add(e.node_id);
          } else {
            errorIds.add(e.node_id);
          }
        }
        if (e.cycle_node_ids) {
          for (const cid of e.cycle_node_ids) {
            cycleIds.add(cid);
          }
        }
      }
      set({
        validationResult: result,
        validationErrors: errors,
        errorNodeIds: errorIds,
        warnNodeIds: warnIds,
        cycleNodeIds: cycleIds,
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
      set((s) => ({ nodes: laid, layoutVersion: s.layoutVersion + 1, error: null }));
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },

  resetOptimize: () => {
    const snapshot = get().preOptimizeSnapshot;
    if (!snapshot) return;
    set({
      nodes: snapshot.nodes,
      edges: snapshot.edges,
      gdspSource: snapshot.gdspSource,
      passResults: [],
      preOptimizeSnapshot: null,
      optimizeResult: null,
    });
  },

  runOptimize: async () => {
    const exported = await get().exportJson();
    if (!exported) return;

    // Save snapshot for reset
    set({
      preOptimizeSnapshot: {
        nodes: [...get().nodes],
        edges: [...get().edges],
        gdspSource: get().gdspSource,
      },
      passResults: [],
    });

    const passes: OptimizePassName[] = [
      "constant_fold",
      "eliminate_cse",
      "eliminate_dead_nodes",
      "promote_control_rate",
    ];

    try {
      let currentGraph = exported;
      const results: { passName: string; nodesBefore: number; nodesAfter: number }[] = [];
      let lastOptimized = null;

      for (const passName of passes) {
        const result = await optimizePass(currentGraph, passName);
        results.push({
          passName,
          nodesBefore: result.stats.nodes_before,
          nodesAfter: result.stats.nodes_after,
        });
        lastOptimized = result.optimized;
        // Export optimized ReactFlowGraph back to Graph JSON for next pass
        currentGraph = await exportGraphJson(result.optimized);
      }

      if (lastOptimized) {
        set({
          nodes: toFlowNodes(lastOptimized.nodes),
          edges: toFlowEdges(lastOptimized.edges),
          passResults: results,
          error: null,
        });
        // Sync editor
        const gdspSource = await get().exportGdsp();
        if (gdspSource) {
          set({ gdspSource });
        }
      }
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

  pushUndo: () => {
    const { nodes, edges, undoStack } = get();
    const snapshot: UndoSnapshot = { nodes: [...nodes], edges: [...edges] };
    const newStack = [...undoStack, snapshot].slice(-MAX_UNDO);
    set({ undoStack: newStack, redoStack: [], canUndo: true, canRedo: false });
  },

  undo: () => {
    const { undoStack, nodes, edges } = get();
    if (undoStack.length === 0) return;
    const prev = undoStack[undoStack.length - 1]!;
    const newUndoStack = undoStack.slice(0, -1);
    const currentSnapshot: UndoSnapshot = { nodes: [...nodes], edges: [...edges] };
    set((s) => ({
      nodes: prev.nodes,
      edges: prev.edges,
      undoStack: newUndoStack,
      redoStack: [...s.redoStack, currentSnapshot],
      canUndo: newUndoStack.length > 0,
      canRedo: true,
    }));
  },

  redo: () => {
    const { redoStack, nodes, edges } = get();
    if (redoStack.length === 0) return;
    const next = redoStack[redoStack.length - 1]!;
    const newRedoStack = redoStack.slice(0, -1);
    const currentSnapshot: UndoSnapshot = { nodes: [...nodes], edges: [...edges] };
    set((s) => ({
      nodes: next.nodes,
      edges: next.edges,
      redoStack: newRedoStack,
      undoStack: [...s.undoStack, currentSnapshot],
      canUndo: true,
      canRedo: newRedoStack.length > 0,
    }));
  },

  addNode: (op, position) => {
    const catalog = get().nodeTypeCatalog;
    if (!catalog || !catalog[op]) return;
    get().pushUndo();
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
    get().pushUndo();
    const idSet = new Set(ids);
    set((s) => ({
      nodes: s.nodes.filter((n) => !idSet.has(n.id)),
      edges: s.edges.filter((e) => !idSet.has(e.source) && !idSet.has(e.target)),
      selectedNode: s.selectedNode && idSet.has(s.selectedNode.id) ? null : s.selectedNode,
    }));
  },

  deleteEdges: (ids) => {
    get().pushUndo();
    const idSet = new Set(ids);
    set((s) => ({
      edges: s.edges.filter((e) => !idSet.has(e.id)),
    }));
  },

  addEdge: (source, target) => {
    get().pushUndo();
    const id = `e_${source}_${target}_${Date.now().toString(36)}`;
    const newEdge: Edge = { id, source, target };
    set((s) => ({ edges: [...s.edges, newEdge] }));
  },

  duplicateNodes: (ids) => {
    get().pushUndo();
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

  selectNodeById: (id) => {
    const node = get().nodes.find((n) => n.id === id) ?? null;
    if (node) {
      set({
        selectedNode: node,
        nodes: get().nodes.map((n) => ({
          ...n,
          selected: n.id === id,
        })),
      });
    }
  },
}));

// Auto-save gdspSource to localStorage (debounced)
let saveTimer: ReturnType<typeof setTimeout> | null = null;
let lastSavedSource = useGraph.getState().gdspSource;
useGraph.subscribe((state) => {
  if (state.gdspSource === lastSavedSource) return;
  lastSavedSource = state.gdspSource;
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE_KEY, state.gdspSource);
    } catch {
      // localStorage full or unavailable -- silently ignore
    }
  }, 500);
});
