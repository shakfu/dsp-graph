export interface RFNodeData {
  [key: string]: unknown;
  label: string;
  op?: string;
  color: string;
  node_data?: Record<string, unknown>;
}

export interface RFNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: RFNodeData;
}

export interface RFEdge {
  id: string;
  source: string;
  target: string;
  animated?: boolean;
  label?: string;
}

export interface ReactFlowGraph {
  nodes: RFNode[];
  edges: RFEdge[];
  name: string;
  sample_rate: number;
  control_interval: number;
}

export interface ValidateResponse {
  valid: boolean;
  errors: string[];
}

export interface SimulateResponse {
  outputs: Record<string, number[]>;
}

export interface OptimizeResponse {
  original: ReactFlowGraph;
  optimized: ReactFlowGraph;
  stats: Record<string, unknown>;
}

export interface CompileResponse {
  cpp_source: string;
}
