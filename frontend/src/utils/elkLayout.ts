import ELK from "elkjs/lib/elk.bundled.js";
import type { Node, Edge } from "@xyflow/react";

export type ElkAlgorithm = "layered" | "stress" | "mrtree" | "radial" | "force";
export type ElkDirection = "RIGHT" | "DOWN" | "LEFT" | "UP";

export interface ElkLayoutOptions {
  algorithm: ElkAlgorithm;
  direction: ElkDirection;
  nodeSpacing: number;
  layerSpacing: number;
}

export const DEFAULT_LAYOUT_OPTIONS: ElkLayoutOptions = {
  algorithm: "layered",
  direction: "RIGHT",
  nodeSpacing: 50,
  layerSpacing: 150,
};

const elk = new ELK();

export async function elkLayout<T extends Record<string, unknown> = Record<string, unknown>>(
  nodes: Node<T>[],
  edges: Edge[],
  options: ElkLayoutOptions
): Promise<Node<T>[]> {
  const graph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": options.algorithm,
      "elk.direction": options.direction,
      "elk.spacing.nodeNode": String(options.nodeSpacing),
      "elk.layered.spacing.nodeNodeBetweenLayers": String(options.layerSpacing),
    },
    children: nodes.map((n) => ({
      id: n.id,
      width: n.measured?.width ?? 150,
      height: n.measured?.height ?? 40,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      sources: [e.source],
      targets: [e.target],
    })),
  };

  const result = await elk.layout(graph);

  const positionMap = new Map<string, { x: number; y: number }>();
  for (const child of result.children ?? []) {
    positionMap.set(child.id, { x: child.x ?? 0, y: child.y ?? 0 });
  }

  return nodes.map((n) => ({
    ...n,
    position: positionMap.get(n.id) ?? n.position,
  }));
}
