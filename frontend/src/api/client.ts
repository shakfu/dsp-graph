import type {
  ReactFlowGraph,
  ValidateResponse,
  SimulateResponse,
  OptimizeResponse,
  CompileResponse,
  BuildResponse,
  CompileBuildResponse,
  PeekResponse,
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

export async function exportGraphGdsp(
  rf: ReactFlowGraph
): Promise<{ source: string }> {
  return post<{ source: string }>("/graph/export/gdsp", rf);
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
  inputs?: Record<string, string>,
  sessionId?: string
): Promise<SimulateResponse> {
  return post<SimulateResponse>("/simulate", {
    graph,
    n_samples: nSamples,
    params,
    inputs,
    session_id: sessionId,
  });
}

export async function simulateContinue(
  sessionId: string,
  nSamples: number,
  inputs?: Record<string, string>
): Promise<SimulateResponse> {
  return post<SimulateResponse>("/simulate/continue", {
    session_id: sessionId,
    n_samples: nSamples,
    inputs,
  });
}

export async function simulateSetParam(
  sessionId: string,
  name: string,
  value: number
): Promise<void> {
  await post<{ status: string }>("/simulate/param", {
    session_id: sessionId,
    name,
    value,
  });
}

export async function simulatePeek(
  sessionId: string
): Promise<PeekResponse> {
  return post<PeekResponse>("/simulate/peek", {
    session_id: sessionId,
  });
}

export async function simulateReset(
  sessionId: string
): Promise<void> {
  await post<{ status: string }>("/simulate/reset", {
    session_id: sessionId,
  });
}

export async function getBuffer(
  sessionId: string,
  bufferId: string
): Promise<{ data: number[] }> {
  return post<{ data: number[] }>("/simulate/buffer/get", {
    session_id: sessionId,
    buffer_id: bufferId,
  });
}

export async function setBuffer(
  sessionId: string,
  bufferId: string,
  data: number[]
): Promise<void> {
  await post<{ status: string }>("/simulate/buffer/set", {
    session_id: sessionId,
    buffer_id: bufferId,
    data,
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

export async function buildGraph(
  graph: Record<string, unknown>,
  platform: string
): Promise<BuildResponse> {
  return post<BuildResponse>("/build", { graph, platform });
}

export async function buildGraphZip(
  graph: Record<string, unknown>,
  platform: string
): Promise<Blob> {
  const resp = await fetch(`${BASE}/build/zip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph, platform }),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`API error ${resp.status}: ${detail}`);
  }
  return resp.blob();
}

export async function compileBuild(
  graph: Record<string, unknown>,
  platform: string
): Promise<CompileBuildResponse> {
  return post<CompileBuildResponse>("/build/compile", { graph, platform });
}

export async function downloadBuiltBinary(
  graph: Record<string, unknown>,
  platform: string
): Promise<Blob> {
  const resp = await fetch(`${BASE}/build/binary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph, platform }),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`API error ${resp.status}: ${detail}`);
  }
  return resp.blob();
}

export async function getBuildPlatforms(): Promise<string[]> {
  const data = await get<{ platforms: string[] }>("/build/platforms");
  return data.platforms;
}

export async function batchBuild(
  graph: Record<string, unknown>,
  platforms: string[]
): Promise<{ results: CompileBuildResponse[] }> {
  return post<{ results: CompileBuildResponse[] }>("/build/batch", {
    graph,
    platforms,
  });
}
