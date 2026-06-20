import type { NodeTypeCatalog } from "../api/types";

/**
 * Input fields of a dsp node op that can be the target of a connection, i.e.
 * that get a target handle on the canvas. Includes scalar reference fields
 * (type admits "string", e.g. "string" or "string|number") and structured
 * reference fields ("array"/"object", e.g. selector.inputs, subgraph.inputs)
 * so their loaded edges still attach to a handle. Pure scalar fields
 * (e.g. "integer", "number") are excluded.
 *
 * Note: structured fields render a handle for display only; addEdge/deleteEdges
 * write back to node_data only for scalar fields (see useGraph).
 */
export function connectableFields(
  catalog: NodeTypeCatalog | null,
  op: string | undefined
): string[] {
  if (!catalog || !op || !catalog[op]) return [];
  return Object.entries(catalog[op].fields)
    .filter(([, info]) => {
      const t = info.type;
      return t.split("|").includes("string") || t === "array" || t === "object";
    })
    .map(([name]) => name);
}
