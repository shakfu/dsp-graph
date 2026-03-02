import { ReactFlowProvider } from "@xyflow/react";
import { GraphCanvas } from "./components/GraphCanvas";
import { GraphToolbar } from "./components/GraphToolbar";
import { NodeInspector } from "./components/NodeInspector";
import { SimulationPanel } from "./components/SimulationPanel";
import { OptimizePanel } from "./components/OptimizePanel";
import { LayoutPanel } from "./components/LayoutPanel";
import { useGraph } from "./hooks/useGraph";

export function App() {
  const selectedNode = useGraph((s) => s.selectedNode);
  const simulationResult = useGraph((s) => s.simulationResult);
  const optimizeResult = useGraph((s) => s.optimizeResult);

  return (
    <ReactFlowProvider>
      <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <GraphToolbar />
        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
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
            }}
          >
            {selectedNode && <NodeInspector node={selectedNode} />}
            <SimulationPanel />
            <OptimizePanel />
            <LayoutPanel />
            {simulationResult && (
              <div style={{ marginTop: 12 }}>
                <h4>Simulation Output</h4>
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
                <h4>Optimization Stats</h4>
                <pre style={{ fontSize: 11 }}>
                  {JSON.stringify(optimizeResult.stats, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </ReactFlowProvider>
  );
}
