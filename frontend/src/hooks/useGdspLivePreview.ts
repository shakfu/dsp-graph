import { useEffect, useRef } from "react";
import { useGraph } from "./useGraph";
import { loadGraphGdspWithErrors } from "../api/client";
import type { Node } from "@xyflow/react";
import type { RFNodeData } from "../api/types";

function nodeIdSet(nodes: Node<RFNodeData>[]): string {
  return nodes
    .map((n) => n.id)
    .sort()
    .join(",");
}

export function useGdspLivePreview() {
  const gdspSource = useGraph((s) => s.gdspSource);
  const isLivePreview = useGraph((s) => s.isLivePreview);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!isLivePreview || !gdspSource.trim()) return;

    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(async () => {
      if (abortRef.current) abortRef.current.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const result = await loadGraphGdspWithErrors(
          gdspSource,
          controller.signal
        );

        if (controller.signal.aborted) return;

        const state = useGraph.getState();

        if (result.error) {
          state.setParseError(result.error);
        } else if (result.graph) {
          const newNodes = result.graph.nodes.map((n) => ({
            id: n.id,
            type: n.type,
            position: n.position,
            data: n.data,
          }));
          const newEdges = result.graph.edges.map((e) => ({
            id: e.id,
            source: e.source,
            target: e.target,
            animated: e.animated ?? false,
            label: e.label ?? undefined,
          }));

          const topologyChanged =
            nodeIdSet(newNodes) !== nodeIdSet(state.nodes);

          if (topologyChanged) {
            // New topology: use backend positions, then run ELK layout
            useGraph.setState({
              nodes: newNodes,
              edges: newEdges,
              graphName: result.graph.name,
              parseError: null,
              error: null,
            });
            await useGraph.getState().runLayout();
          } else {
            // Same topology: preserve existing laid-out positions,
            // only update node data and edges
            const posMap = new Map(
              state.nodes.map((n) => [n.id, n.position])
            );
            const merged = newNodes.map((n) => ({
              ...n,
              position: posMap.get(n.id) ?? n.position,
            }));
            useGraph.setState({
              nodes: merged,
              edges: newEdges,
              graphName: result.graph.name,
              parseError: null,
              error: null,
            });
          }

          // Auto-validate after successful parse
          await useGraph.getState().runValidate();
        }
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        console.error("Live preview error:", e);
      }
    }, 300);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [gdspSource, isLivePreview]);
}
