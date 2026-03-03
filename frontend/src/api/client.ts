import type {
  ReactFlowGraph,
  ValidateResponse,
  SimulateResponse,
  OptimizeResponse,
  CompileResponse,
  ParseError,
  NodeTypeCatalog,
} from "./types";

const BASE = "/api";

async function post<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`API error ${resp.status}: ${detail}`);
  }
  return resp.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE}${path}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json() as Promise<T>;
}

export async function loadGraphJson(
  graph: Record<string, unknown>
): Promise<ReactFlowGraph> {
  return post<ReactFlowGraph>("/graph/load/json", { graph });
}

export async function loadGraphGdsp(source: string): Promise<ReactFlowGraph> {
  return post<ReactFlowGraph>("/graph/load/gdsp", { source });
}

export async function loadGraphGdspWithErrors(
  source: string,
  signal?: AbortSignal
): Promise<{ graph?: ReactFlowGraph; error?: ParseError }> {
  const resp = await fetch(`${BASE}/graph/load/gdsp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
    signal,
  });
  if (resp.ok) {
    const graph = (await resp.json()) as ReactFlowGraph;
    return { graph };
  }
  if (resp.status === 422) {
    const body = await resp.json();
    const detail = body.detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      return { error: detail as ParseError };
    }
    return { error: { message: String(detail), line: 0, col: 0 } };
  }
  throw new Error(`API error ${resp.status}`);
}

export async function validateGraph(
  graph: Record<string, unknown>
): Promise<ValidateResponse> {
  return post<ValidateResponse>("/graph/validate", { graph });
}

export async function exportGraphJson(
  rf: ReactFlowGraph
): Promise<Record<string, unknown>> {
  return post<Record<string, unknown>>("/graph/export/json", rf);
}

export async function getNodeTypes(): Promise<{
  colors: Record<string, string>;
  catalog: NodeTypeCatalog;
}> {
  return get<{ colors: Record<string, string>; catalog: NodeTypeCatalog }>(
    "/graph/node-types"
  );
}

export async function simulateGraph(
  graph: Record<string, unknown>,
  nSamples: number,
  params?: Record<string, number>,
  inputs?: Record<string, string>
): Promise<SimulateResponse> {
  return post<SimulateResponse>("/simulate", {
    graph,
    n_samples: nSamples,
    params,
    inputs,
  });
}

export async function optimizeGraph(
  graph: Record<string, unknown>
): Promise<OptimizeResponse> {
  return post<OptimizeResponse>("/optimize", { graph });
}

export async function compileGraph(
  graph: Record<string, unknown>
): Promise<CompileResponse> {
  return post<CompileResponse>("/compile", { graph });
}
