import { useState, useCallback, useRef, useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { GraphCanvas } from "./components/GraphCanvas";
import { GraphToolbar } from "./components/GraphToolbar";
import { NodeInspector } from "./components/NodeInspector";
import { SimulationPanel } from "./components/SimulationPanel";
import { OptimizePanel } from "./components/OptimizePanel";
import { CompilePanel } from "./components/CompilePanel";
import { LayoutPanel } from "./components/LayoutPanel";
import { GdspEditor } from "./components/GdspEditor";
import { StatusBar } from "./components/StatusBar";
import { NodeCatalog } from "./components/NodeCatalog";
import { useGraph } from "./hooks/useGraph";

function DividerHandle({
  onDrag,
}: {
  onDrag: (deltaX: number) => void;
}) {
  const dragging = useRef(false);
  const lastX = useRef(0);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;
      lastX.current = e.clientX;

      const onMouseMove = (ev: MouseEvent) => {
        if (!dragging.current) return;
        const dx = ev.clientX - lastX.current;
        lastX.current = ev.clientX;
        onDrag(dx);
      };
      const onMouseUp = () => {
        dragging.current = false;
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
      };
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    },
    [onDrag]
  );

  return (
    <div
      onMouseDown={onMouseDown}
      style={{
        width: 5,
        cursor: "col-resize",
        background: "#e0e0e0",
        flexShrink: 0,
      }}
    />
  );
}

export function App() {
  const selectedNode = useGraph((s) => s.selectedNode);
  const simulationResult = useGraph((s) => s.simulationResult);
  const optimizeResult = useGraph((s) => s.optimizeResult);
  const showEditor = useGraph((s) => s.showEditor);
  const fetchNodeTypes = useGraph((s) => s.fetchNodeTypes);

  const [editorWidth, setEditorWidth] = useState(400);

  useEffect(() => {
    void fetchNodeTypes();
  }, [fetchNodeTypes]);

  const handleDividerDrag = useCallback((dx: number) => {
    setEditorWidth((w) => Math.max(200, Math.min(800, w + dx)));
  }, []);

  return (
    <ReactFlowProvider>
      <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <GraphToolbar />
        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
          {showEditor && (
            <>
              <div style={{ width: editorWidth, flexShrink: 0 }}>
                <GdspEditor />
              </div>
              <DividerHandle onDrag={handleDividerDrag} />
            </>
          )}
          <div style={{ flex: 1, position: "relative" }}>
            <GraphCanvas />
          </div>
          <div
            style={{
              width: 320,
              borderLeft: "1px solid #ddd",
              overflow: "auto",
              background: "#fafafa",
              padding: 12,
              flexShrink: 0,
            }}
          >
            {selectedNode && <NodeInspector node={selectedNode} />}
            <SimulationPanel />
            <OptimizePanel />
            <CompilePanel />
            <LayoutPanel />
            <NodeCatalog />
            {simulationResult && (
              <div style={{ marginTop: 12 }}>
                <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>
                  Simulation Output
                </h4>
                {Object.entries(simulationResult.outputs).map(([key, vals]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <strong>{key}</strong>: {vals.length} samples
                    <br />
                    <span style={{ fontSize: 11, color: "#666" }}>
                      [{vals.slice(0, 5).map((v) => v.toFixed(4)).join(", ")}
                      {vals.length > 5 ? ", ..." : ""}]
                    </span>
                  </div>
                ))}
              </div>
            )}
            {optimizeResult && (
              <div style={{ marginTop: 12 }}>
                <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>
                  Optimization Stats
                </h4>
                <pre style={{ fontSize: 11 }}>
                  {JSON.stringify(optimizeResult.stats, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
        <StatusBar />
      </div>
    </ReactFlowProvider>
  );
}
