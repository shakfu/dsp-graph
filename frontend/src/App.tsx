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
  const setGdspSource = useGraph((s) => s.setGdspSource);
  const loadFromJson = useGraph((s) => s.loadFromJson);
  const setShowEditor = useGraph((s) => s.setShowEditor);

  const [editorWidth, setEditorWidth] = useState(500);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    void fetchNodeTypes();
  }, [fetchNodeTypes]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);

      const file = e.dataTransfer.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = () => {
        const text = reader.result as string;
        if (file.name.endsWith(".gdsp")) {
          setGdspSource(text);
          setShowEditor(true);
        } else if (file.name.endsWith(".json")) {
          try {
            const json = JSON.parse(text) as Record<string, unknown>;
            void loadFromJson(json);
          } catch {
            console.error("Invalid JSON file");
          }
        }
      };
      reader.readAsText(file);
    },
    [setGdspSource, loadFromJson, setShowEditor]
  );

  const handleDividerDrag = useCallback((dx: number) => {
    setEditorWidth((w) => Math.max(200, Math.min(800, w + dx)));
  }, []);

  return (
    <ReactFlowProvider>
      <div
        style={{ display: "flex", flexDirection: "column", height: "100vh", position: "relative" }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {dragOver && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 9999,
              background: "rgba(152, 195, 121, 0.15)",
              border: "3px dashed #98c379",
              borderRadius: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              pointerEvents: "none",
            }}
          >
            <span
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: "#98c379",
                background: "#282c34",
                padding: "8px 20px",
                borderRadius: 6,
              }}
            >
              Drop .gdsp or .json file
            </span>
          </div>
        )}
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
