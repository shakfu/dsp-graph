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

export function GraphToolbar() {
  const loadFromJson = useGraph((s) => s.loadFromJson);
  const loadFromGdsp = useGraph((s) => s.loadFromGdsp);
  const exportJson = useGraph((s) => s.exportJson);
  const exportSvg = useGraph((s) => s.exportSvg);
  const graphName = useGraph((s) => s.graphName);
  const error = useGraph((s) => s.error);
  const clearError = useGraph((s) => s.clearError);
  const fileInput = useRef<HTMLInputElement>(null);

  const handleFileLoad = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      if (file.name.endsWith(".gdsp")) {
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

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 12px",
        borderBottom: "1px solid #ddd",
        background: "#f8f9fa",
      }}
    >
      <strong style={{ fontSize: 14 }}>dsp-graph</strong>
      {graphName && (
        <span style={{ fontSize: 12, color: "#666" }}>/ {graphName}</span>
      )}
      <div style={{ flex: 1 }} />
      <input
        ref={fileInput}
        type="file"
        accept=".json,.gdsp"
        style={{ display: "none" }}
        onChange={handleFileLoad}
      />
      <button style={buttonStyle} onClick={() => fileInput.current?.click()}>
        Load Graph
      </button>
      <button style={buttonStyle} onClick={handleExport}>
        Export JSON
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
