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
import type { Node, Connection } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useCallback, useMemo, useState, useRef } from "react";
import { toSvg } from "html-to-image";
import { useGraph } from "../hooks/useGraph";
import type { RFNodeData, NodeTypeCatalog } from "../api/types";
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

// -- Context menu types and component --

interface ContextMenuState {
  x: number;
  y: number;
  screenX: number;
  screenY: number;
  type: "pane" | "node" | "edge";
  nodeId?: string;
  edgeId?: string;
}

function NodePicker({
  catalog,
  onSelect,
  onClose,
}: {
  catalog: NodeTypeCatalog;
  onSelect: (op: string) => void;
  onClose: () => void;
}) {
  const [filter, setFilter] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const ops = useMemo(() => {
    const all = Object.keys(catalog).sort();
    if (!filter) return all;
    const f = filter.toLowerCase();
    return all.filter((op) => op.includes(f) || (catalog[op]?.class ?? "").toLowerCase().includes(f));
  }, [catalog, filter]);

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #ccc",
        borderRadius: 4,
        padding: 4,
        maxHeight: 260,
        width: 180,
        display: "flex",
        flexDirection: "column",
        boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
      }}
    >
      <input
        ref={inputRef}
        type="text"
        placeholder="Filter nodes..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") onClose();
          if (e.key === "Enter" && ops.length > 0 && ops[0] !== undefined) onSelect(ops[0]);
        }}
        style={{
          fontSize: 11,
          padding: "3px 6px",
          border: "1px solid #ddd",
          borderRadius: 3,
          marginBottom: 4,
          outline: "none",
        }}
      />
      <div style={{ overflow: "auto", flex: 1 }}>
        {ops.map((op) => (
          <div
            key={op}
            onClick={() => onSelect(op)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "3px 6px",
              fontSize: 11,
              cursor: "pointer",
              borderRadius: 2,
            }}
            onMouseOver={(e) => {
              (e.currentTarget as HTMLElement).style.background = "#f0f0f0";
            }}
            onMouseOut={(e) => {
              (e.currentTarget as HTMLElement).style.background = "transparent";
            }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: catalog[op]?.color ?? "#fff",
                border: "1px solid #ccc",
                flexShrink: 0,
              }}
            />
            <span>{op}</span>
            <span style={{ fontSize: 9, color: "#999", marginLeft: "auto" }}>
              {catalog[op]?.class ?? ""}
            </span>
          </div>
        ))}
        {ops.length === 0 && (
          <div style={{ fontSize: 11, color: "#999", padding: 4 }}>No matches</div>
        )}
      </div>
    </div>
  );
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
  const errorNodeIds = useGraph((s) => s.errorNodeIds);
  const warnNodeIds = useGraph((s) => s.warnNodeIds);
  const peekValues = useGraph((s) => s.peekValues);
  const nodeTypeCatalog = useGraph((s) => s.nodeTypeCatalog);
  const addNodeToGraph = useGraph((s) => s.addNode);
  const deleteNodes = useGraph((s) => s.deleteNodes);
  const deleteEdges = useGraph((s) => s.deleteEdges);
  const addEdgeToGraph = useGraph((s) => s.addEdge);
  const duplicateNodes = useGraph((s) => s.duplicateNodes);

  const { fitView, getViewport, screenToFlowPosition } = useReactFlow();
  const [nodes, setNodes, onNodesChange] = useNodesState(storeNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges);

  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [showNodePicker, setShowNodePicker] = useState(false);

  // Apply validation error styling and peek badges to nodes
  const styledNodes = useMemo(() => {
    const hasErrors = errorNodeIds.size > 0 || warnNodeIds.size > 0;
    const hasPeek = peekValues && Object.keys(peekValues).length > 0;
    if (!hasErrors && !hasPeek) return nodes;
    return nodes.map((n) => {
      let updated = n;
      if (errorNodeIds.has(n.id)) {
        updated = { ...updated, style: { ...updated.style, outline: "2px solid #dc3545", outlineOffset: 1 } };
      } else if (warnNodeIds.has(n.id)) {
        updated = { ...updated, style: { ...updated.style, outline: "2px solid #ffc107", outlineOffset: 1 } };
      }
      if (hasPeek && n.id in peekValues) {
        updated = {
          ...updated,
          data: { ...updated.data, peekValue: peekValues[n.id] },
        };
      }
      return updated;
    });
  }, [nodes, errorNodeIds, warnNodeIds, peekValues]);

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

  // Edge drawing via onConnect
  const onConnect = useCallback(
    (connection: Connection) => {
      if (connection.source && connection.target) {
        addEdgeToGraph(connection.source, connection.target);
      }
    },
    [addEdgeToGraph]
  );

  // Context menu handlers
  const onPaneContextMenu = useCallback(
    (event: React.MouseEvent | MouseEvent) => {
      event.preventDefault();
      const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      setContextMenu({
        x: flowPos.x,
        y: flowPos.y,
        screenX: event.clientX,
        screenY: event.clientY,
        type: "pane",
      });
      setShowNodePicker(false);
    },
    [screenToFlowPosition]
  );

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();
      const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      setContextMenu({
        x: flowPos.x,
        y: flowPos.y,
        screenX: event.clientX,
        screenY: event.clientY,
        type: "node",
        nodeId: node.id,
      });
      setShowNodePicker(false);
    },
    [screenToFlowPosition]
  );

  const onEdgeContextMenu = useCallback(
    (event: React.MouseEvent, edge: { id: string }) => {
      event.preventDefault();
      const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      setContextMenu({
        x: flowPos.x,
        y: flowPos.y,
        screenX: event.clientX,
        screenY: event.clientY,
        type: "edge",
        edgeId: edge.id,
      });
      setShowNodePicker(false);
    },
    [screenToFlowPosition]
  );

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
    setShowNodePicker(false);
  }, []);

  // Keyboard delete
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "Delete" || event.key === "Backspace") {
        // Don't intercept if focused on an input element
        const tag = (event.target as HTMLElement).tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

        const selectedNodeIds = nodes.filter((n) => n.selected).map((n) => n.id);
        const selectedEdgeIds = edges.filter((e) => e.selected).map((e) => e.id);

        if (selectedNodeIds.length > 0) {
          deleteNodes(selectedNodeIds);
        }
        if (selectedEdgeIds.length > 0) {
          deleteEdges(selectedEdgeIds);
        }
      }
      if (event.key === "Escape") {
        closeContextMenu();
      }
    },
    [nodes, edges, deleteNodes, deleteEdges, closeContextMenu]
  );

  const handleAddNode = useCallback(
    (op: string) => {
      if (!contextMenu) return;
      addNodeToGraph(op, { x: contextMenu.x, y: contextMenu.y });
      closeContextMenu();
    },
    [contextMenu, addNodeToGraph, closeContextMenu]
  );

  const nodeIds = nodes.map((n) => n.id);

  const menuItemStyle: React.CSSProperties = {
    padding: "5px 12px",
    fontSize: 12,
    cursor: "pointer",
    whiteSpace: "nowrap",
  };

  return (
    <DirectionContext.Provider value={direction}>
      {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
      <div style={{ width: "100%", height: "100%" }} onKeyDown={onKeyDown} tabIndex={-1}>
        <ReactFlow
          nodes={styledNodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={(_event, node) => {
            selectNode(node as Node<RFNodeData>);
            closeContextMenu();
          }}
          onPaneClick={() => {
            selectNode(null);
            closeContextMenu();
          }}
          onNodeContextMenu={onNodeContextMenu}
          onEdgeContextMenu={onEdgeContextMenu}
          onPaneContextMenu={onPaneContextMenu}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.1}
          maxZoom={4}
          deleteKeyCode={null}
          selectionKeyCode="Shift"
        >
          <HandlePositionUpdater nodeIds={nodeIds} direction={direction} />
          <Controls />
          <MiniMap />
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        </ReactFlow>

        {/* Context Menu Overlay */}
        {contextMenu && (
          <>
            {/* Click-away backdrop */}
            {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions */}
            <div
              onClick={closeContextMenu}
              style={{
                position: "fixed",
                inset: 0,
                zIndex: 99,
              }}
            />
            <div
              style={{
                position: "fixed",
                left: contextMenu.screenX,
                top: contextMenu.screenY,
                zIndex: 100,
                background: "#fff",
                border: "1px solid #ccc",
                borderRadius: 4,
                boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
                minWidth: 140,
              }}
            >
              {contextMenu.type === "pane" && !showNodePicker && (
                <div>
                  <div
                    style={menuItemStyle}
                    onClick={() => setShowNodePicker(true)}
                    onMouseOver={(e) => { (e.currentTarget as HTMLElement).style.background = "#f0f0f0"; }}
                    onMouseOut={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
                  >
                    Add Node...
                  </div>
                </div>
              )}

              {contextMenu.type === "pane" && showNodePicker && nodeTypeCatalog && (
                <NodePicker
                  catalog={nodeTypeCatalog}
                  onSelect={handleAddNode}
                  onClose={closeContextMenu}
                />
              )}

              {contextMenu.type === "node" && contextMenu.nodeId && (
                <div>
                  <div
                    style={menuItemStyle}
                    onClick={() => {
                      duplicateNodes([contextMenu.nodeId!]);
                      closeContextMenu();
                    }}
                    onMouseOver={(e) => { (e.currentTarget as HTMLElement).style.background = "#f0f0f0"; }}
                    onMouseOut={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
                  >
                    Duplicate
                  </div>
                  <div
                    style={{ ...menuItemStyle, color: "#dc3545" }}
                    onClick={() => {
                      deleteNodes([contextMenu.nodeId!]);
                      closeContextMenu();
                    }}
                    onMouseOver={(e) => { (e.currentTarget as HTMLElement).style.background = "#f0f0f0"; }}
                    onMouseOut={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
                  >
                    Delete
                  </div>
                </div>
              )}

              {contextMenu.type === "edge" && contextMenu.edgeId && (
                <div>
                  <div
                    style={{ ...menuItemStyle, color: "#dc3545" }}
                    onClick={() => {
                      deleteEdges([contextMenu.edgeId!]);
                      closeContextMenu();
                    }}
                    onMouseOver={(e) => { (e.currentTarget as HTMLElement).style.background = "#f0f0f0"; }}
                    onMouseOut={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
                  >
                    Delete Edge
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </DirectionContext.Provider>
  );
}
