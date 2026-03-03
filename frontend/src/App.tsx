import { useState, useCallback, useRef, useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { GraphCanvas } from "./components/GraphCanvas";
import { GraphToolbar } from "./components/GraphToolbar";
import { EditorPane } from "./components/EditorPane";
import { Sidebar } from "./components/Sidebar";
import { StatusBar } from "./components/StatusBar";
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
  const showEditor = useGraph((s) => s.showEditor);
  const showGraph = useGraph((s) => s.showGraph);
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
              <div style={showGraph ? { width: editorWidth, flexShrink: 0 } : { flex: 1 }}>
                <EditorPane />
              </div>
              {showGraph && <DividerHandle onDrag={handleDividerDrag} />}
            </>
          )}
          {showGraph && (
            <div style={{ flex: 1, position: "relative" }}>
              <GraphCanvas />
            </div>
          )}
          <Sidebar />
        </div>
        <StatusBar />
      </div>
    </ReactFlowProvider>
  );
}
