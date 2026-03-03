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

export interface ValidationErrorDetail {
  message: string;
  kind: string;
  node_id: string | null;
  field_name: string | null;
  severity: string;
}

export interface ValidateResponse {
  valid: boolean;
  errors: ValidationErrorDetail[];
}

export interface SimulateResponse {
  outputs: Record<string, number[]>;
  session_id: string;
}

export interface PeekResponse {
  values: Record<string, number>;
}

export interface OptimizeResponse {
  original: ReactFlowGraph;
  optimized: ReactFlowGraph;
  stats: Record<string, unknown>;
}

export interface CompileResponse {
  cpp_source: string;
}

export interface BuildResponse {
  dsp_cpp: string;
  adapter_cpp: string;
  manifest: string;
  platform: string;
  supported_platforms: string[];
}

export type InputSignalType = "impulse" | "sine" | "noise" | "ones";

export interface ParseError {
  message: string;
  line: number;
  col: number;
}

export interface NodeFieldInfo {
  type: string;
  required: boolean;
  default?: unknown;
}

export interface NodeTypeInfo {
  class: string;
  fields: Record<string, NodeFieldInfo>;
  color: string;
}

export interface CompileBuildResponse {
  success: boolean;
  platform: string;
  stdout: string;
  stderr: string;
  output_file: string | null;
}

export type NodeTypeCatalog = Record<string, NodeTypeInfo>;
