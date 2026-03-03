import { useState } from "react";
import { NodeInspector } from "./NodeInspector";
import { SimulationPanel } from "./SimulationPanel";
import { OptimizePanel } from "./OptimizePanel";
import { LayoutPanel } from "./LayoutPanel";
import { BuildPanel } from "./BuildPanel";
import { NodeCatalog } from "./NodeCatalog";
import { useGraph } from "../hooks/useGraph";

type Tab = "inspect" | "tools" | "catalog";

const tabStyle = (active: boolean): React.CSSProperties => ({
  flex: 1,
  padding: "6px 0",
  fontSize: 11,
  fontWeight: active ? 600 : 400,
  color: active ? "#0d6efd" : "#666",
  background: active ? "#fff" : "#f0f0f0",
  border: "none",
  borderBottom: active ? "2px solid #0d6efd" : "2px solid transparent",
  cursor: "pointer",
  textTransform: "uppercase",
  letterSpacing: 0.5,
});

export function Sidebar() {
  const selectedNode = useGraph((s) => s.selectedNode);
  const simulationResult = useGraph((s) => s.simulationResult);
  const optimizeResult = useGraph((s) => s.optimizeResult);
  const validationErrors = useGraph((s) => s.validationErrors);
  const selectNode = useGraph((s) => s.selectNode);
  const storeNodes = useGraph((s) => s.nodes);
  const nodes = storeNodes;
  const [activeTab, setActiveTab] = useState<Tab>("tools");

  return (
    <div
      style={{
        width: 320,
        borderLeft: "1px solid #ddd",
        display: "flex",
        flexDirection: "column",
        background: "#fafafa",
        flexShrink: 0,
      }}
    >
      {/* Tab bar */}
      <div
        style={{
          display: "flex",
          borderBottom: "1px solid #ddd",
          flexShrink: 0,
        }}
      >
        <button style={tabStyle(activeTab === "tools")} onClick={() => setActiveTab("tools")}>
          Tools
        </button>
        <button style={tabStyle(activeTab === "inspect")} onClick={() => setActiveTab("inspect")}>
          Inspect
        </button>
        <button style={tabStyle(activeTab === "catalog")} onClick={() => setActiveTab("catalog")}>
          Catalog
        </button>
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: "auto", padding: 12 }}>
        {activeTab === "inspect" && (
          <>
            {selectedNode ? (
              <NodeInspector node={selectedNode} />
            ) : (
              <div style={{ fontSize: 12, color: "#999", padding: 8 }}>
                Click a node in the graph to inspect it.
              </div>
            )}
            {simulationResult && (
              <div style={{ marginTop: 12, borderTop: "1px solid #ddd", paddingTop: 12 }}>
                <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>
                  Simulation Output
                </h4>
                {Object.entries(simulationResult.outputs).map(([key, vals]) => (
                  <div key={key} style={{ marginBottom: 8 }}>
                    <strong style={{ fontSize: 12 }}>{key}</strong>
                    <span style={{ fontSize: 11, color: "#666" }}>
                      {" "}({vals.length} samples)
                    </span>
                    <br />
                    <span style={{ fontSize: 11, color: "#666", fontFamily: "monospace" }}>
                      [{vals.slice(0, 5).map((v) => v.toFixed(4)).join(", ")}
                      {vals.length > 5 ? ", ..." : ""}]
                    </span>
                  </div>
                ))}
              </div>
            )}
            {optimizeResult && (
              <div style={{ marginTop: 12, borderTop: "1px solid #ddd", paddingTop: 12 }}>
                <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>
                  Optimization Stats
                </h4>
                <pre style={{ fontSize: 11, margin: 0 }}>
                  {JSON.stringify(optimizeResult.stats, null, 2)}
                </pre>
              </div>
            )}
            {validationErrors.length > 0 && (
              <div style={{ marginTop: 12, borderTop: "1px solid #ddd", paddingTop: 12 }}>
                <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>
                  Validation Errors ({validationErrors.length})
                </h4>
                {validationErrors.map((err, i) => (
                  <div
                    key={i}
                    onClick={() => {
                      if (err.node_id) {
                        const node = nodes.find((n) => n.id === err.node_id);
                        if (node) selectNode(node);
                      }
                    }}
                    style={{
                      padding: "4px 6px",
                      marginBottom: 4,
                      fontSize: 11,
                      borderRadius: 3,
                      cursor: err.node_id ? "pointer" : "default",
                      background: err.severity === "error" ? "#f8d7da" : "#fff3cd",
                      border: `1px solid ${err.severity === "error" ? "#f5c6cb" : "#ffeeba"}`,
                    }}
                  >
                    <span style={{ fontWeight: 600 }}>[{err.kind}]</span>{" "}
                    {err.message}
                    {err.node_id && (
                      <span style={{ color: "#666" }}> (node: {err.node_id})</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {activeTab === "tools" && (
          <>
            <SimulationPanel />
            <OptimizePanel />
            <BuildPanel />
            <LayoutPanel />
          </>
        )}

        {activeTab === "catalog" && <NodeCatalog />}
      </div>
    </div>
  );
}
