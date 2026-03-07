import { useRef } from "react";
import { useGraph } from "../hooks/useGraph";

const buttonStyle: React.CSSProperties = {
  padding: "4px 12px",
  fontSize: 13,
  border: "1px solid #ccc",
  borderRadius: 4,
  background: "#fff",
  cursor: "pointer",
};

const activeButtonStyle: React.CSSProperties = {
  ...buttonStyle,
  background: "#0d6efd",
  color: "#fff",
  borderColor: "#0d6efd",
};

export function GraphToolbar() {
  const loadFromJson = useGraph((s) => s.loadFromJson);
  const loadFromGdsp = useGraph((s) => s.loadFromGdsp);
  const exportJson = useGraph((s) => s.exportJson);
  const exportGdsp = useGraph((s) => s.exportGdsp);
  const exportSvg = useGraph((s) => s.exportSvg);
  const graphName = useGraph((s) => s.graphName);
  const error = useGraph((s) => s.error);
  const clearError = useGraph((s) => s.clearError);
  const showEditor = useGraph((s) => s.showEditor);
  const setShowEditor = useGraph((s) => s.setShowEditor);
  const showGraph = useGraph((s) => s.showGraph);
  const setShowGraph = useGraph((s) => s.setShowGraph);
  const isLivePreview = useGraph((s) => s.isLivePreview);
  const setLivePreview = useGraph((s) => s.setLivePreview);
  const setGdspSource = useGraph((s) => s.setGdspSource);
  const undo = useGraph((s) => s.undo);
  const redo = useGraph((s) => s.redo);
  const canUndo = useGraph((s) => s.canUndo);
  const canRedo = useGraph((s) => s.canRedo);
  const fileInput = useRef<HTMLInputElement>(null);

  const handleFileLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      if (file.name.endsWith(".gdsp")) {
        setGdspSource(text);
        setShowEditor(true);
        await loadFromGdsp(text);
      } else {
        const json = JSON.parse(text);
        await loadFromJson(json);
      }
    } catch (err) {
      console.error("Failed to load file:", err);
    }
    if (fileInput.current) fileInput.current.value = "";
  };

  const handleExport = async () => {
    const json = await exportJson();
    if (!json) return;
    const blob = new Blob([JSON.stringify(json, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${graphName || "graph"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportGdsp = async () => {
    const source = await exportGdsp();
    if (!source) return;
    const blob = new Blob([source], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${graphName || "graph"}.gdsp`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleManualParse = async () => {
    const source = useGraph.getState().gdspSource;
    if (source.trim()) {
      await loadFromGdsp(source);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 12px",
        borderBottom: "1px solid #ddd",
        background: "#f8f9fa",
        flexShrink: 0,
      }}
    >
      <strong style={{ fontSize: 14 }}>dsp-graph</strong>
      {graphName && (
        <span style={{ fontSize: 12, color: "#666" }}>/ {graphName}</span>
      )}
      <div style={{ flex: 1 }} />
      <button
        style={showEditor ? activeButtonStyle : buttonStyle}
        onClick={() => setShowEditor(!showEditor)}
      >
        Editor
      </button>
      <button
        style={showGraph ? activeButtonStyle : buttonStyle}
        onClick={() => setShowGraph(!showGraph)}
      >
        Graph
      </button>
      <button
        style={isLivePreview ? activeButtonStyle : buttonStyle}
        onClick={() => setLivePreview(!isLivePreview)}
      >
        Live
      </button>
      {!isLivePreview && (
        <button style={buttonStyle} onClick={() => void handleManualParse()}>
          Parse
        </button>
      )}
      <div
        style={{ width: 1, height: 20, background: "#ddd", margin: "0 4px" }}
      />
      <button
        style={{ ...buttonStyle, opacity: canUndo ? 1 : 0.4 }}
        onClick={undo}
        disabled={!canUndo}
        title="Undo (Cmd+Z)"
      >
        Undo
      </button>
      <button
        style={{ ...buttonStyle, opacity: canRedo ? 1 : 0.4 }}
        onClick={redo}
        disabled={!canRedo}
        title="Redo (Cmd+Shift+Z)"
      >
        Redo
      </button>
      <div
        style={{ width: 1, height: 20, background: "#ddd", margin: "0 4px" }}
      />
      <input
        ref={fileInput}
        type="file"
        accept=".json,.gdsp"
        style={{ display: "none" }}
        onChange={handleFileLoad}
      />
      <button style={buttonStyle} onClick={() => fileInput.current?.click()}>
        Load File
      </button>
      <button style={buttonStyle} onClick={handleExport}>
        Export JSON
      </button>
      <button style={buttonStyle} onClick={handleExportGdsp}>
        Export GDSP
      </button>
      <button
        style={buttonStyle}
        onClick={() => exportSvg?.()}
        disabled={!exportSvg}
      >
        Export SVG
      </button>
      {error && (
        <span
          style={{
            fontSize: 12,
            color: "#dc3545",
            maxWidth: 300,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            cursor: "pointer",
          }}
          onClick={clearError}
          title={error}
        >
          {error}
        </span>
      )}
    </div>
  );
}
