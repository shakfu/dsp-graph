import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  useUpdateNodeInternals,
  useReactFlow,
} from "@xyflow/react";
import type { Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useCallback } from "react";
import { toSvg } from "html-to-image";
import { useGraph } from "../hooks/useGraph";
import type { RFNodeData } from "../api/types";
import { DspNode } from "../nodes/DspNode";
import { InputNode } from "../nodes/InputNode";
import { OutputNode } from "../nodes/OutputNode";
import { ParamNode } from "../nodes/ParamNode";
import { DirectionContext } from "../hooks/useHandlePositions";

const nodeTypes = {
  dsp_node: DspNode,
  input: InputNode,
  output: OutputNode,
  param: ParamNode,
};

/** Runs inside <ReactFlow> to access useUpdateNodeInternals. */
function HandlePositionUpdater({ nodeIds, direction }: { nodeIds: string[]; direction: string }) {
  const updateNodeInternals = useUpdateNodeInternals();

  useEffect(() => {
    for (const id of nodeIds) {
      updateNodeInternals(id);
    }
  }, [direction, nodeIds, updateNodeInternals]);

  return null;
}

export function GraphCanvas() {
  const storeNodes = useGraph((s) => s.nodes);
  const storeEdges = useGraph((s) => s.edges);
  const selectNode = useGraph((s) => s.selectNode);
  const setStoreNodes = useGraph((s) => s.setNodes);
  const setStoreEdges = useGraph((s) => s.setEdges);
  const setExportSvg = useGraph((s) => s.setExportSvg);
  const graphName = useGraph((s) => s.graphName);
  const direction = useGraph((s) => s.layoutOptions.direction);

  const { fitView, getViewport } = useReactFlow();
  const [nodes, setNodes, onNodesChange] = useNodesState(storeNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges);

  useEffect(() => {
    setNodes(storeNodes);
    // fitView after a tick so React Flow has measured the new nodes
    requestAnimationFrame(() => fitView());
  }, [storeNodes, setNodes, fitView]);

  useEffect(() => {
    setEdges(storeEdges);
  }, [storeEdges, setEdges]);

  useEffect(() => {
    setStoreNodes(nodes as Node<RFNodeData>[]);
  }, [nodes, setStoreNodes]);

  useEffect(() => {
    setStoreEdges(edges);
  }, [edges, setStoreEdges]);

  const exportSvg = useCallback(async () => {
    const container = document.querySelector<HTMLElement>(".react-flow");
    const viewportEl = document.querySelector<HTMLElement>(".react-flow__viewport");
    if (!container || !viewportEl) return;

    const { x, y, zoom } = getViewport();
    const width = container.clientWidth;
    const height = container.clientHeight;

    const dataUrl = await toSvg(viewportEl, {
      width,
      height,
      style: {
        width: `${width}px`,
        height: `${height}px`,
        transform: `translate(${x}px, ${y}px) scale(${zoom})`,
      },
    });

    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `${graphName || "graph"}.svg`;
    a.click();
  }, [getViewport, graphName]);

  useEffect(() => {
    setExportSvg(exportSvg);
    return () => setExportSvg(null);
  }, [exportSvg, setExportSvg]);

  const nodeIds = nodes.map((n) => n.id);

  return (
    <DirectionContext.Provider value={direction}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={(_event, node) => selectNode(node as Node<RFNodeData>)}
        onPaneClick={() => selectNode(null)}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.1}
        maxZoom={4}
      >
        <HandlePositionUpdater nodeIds={nodeIds} direction={direction} />
        <Controls />
        <MiniMap />
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
      </ReactFlow>
    </DirectionContext.Provider>
  );
}
