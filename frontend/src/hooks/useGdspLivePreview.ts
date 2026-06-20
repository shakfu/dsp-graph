import { useEffect, useRef } from "react";
import { useGraph, toFlowNodes, toFlowEdges } from "./useGraph";
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
  const selectedGraphName = useGraph((s) => s.selectedGraphName);
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
          controller.signal,
          selectedGraphName ?? undefined
        );

        if (controller.signal.aborted) return;

        const state = useGraph.getState();

        if (result.error) {
          state.setParseError(result.error);
        } else if (result.graph) {
          const names = result.graph.graph_names ?? [];
          // Drop a stale selection that no longer exists in the edited source.
          const reconciledSel =
            selectedGraphName && names.includes(selectedGraphName)
              ? selectedGraphName
              : null;
          // Reuse the shared mappers so this path can't drift from the
          // load actions (e.g. dropping target_handle / sample_rate fields).
          const newNodes = toFlowNodes(result.graph.nodes);
          const newEdges = toFlowEdges(result.graph.edges);

          const topologyChanged =
            nodeIdSet(newNodes) !== nodeIdSet(state.nodes);

          if (topologyChanged) {
            // New topology: use backend positions, then run ELK layout
            useGraph.setState({
              nodes: newNodes,
              edges: newEdges,
              graphName: result.graph.name,
              graphNames: names,
              selectedGraphName: reconciledSel,
              sampleRate: result.graph.sample_rate,
              controlInterval: result.graph.control_interval,
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
              graphNames: names,
              selectedGraphName: reconciledSel,
              sampleRate: result.graph.sample_rate,
              controlInterval: result.graph.control_interval,
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
  }, [gdspSource, isLivePreview, selectedGraphName]);
}
