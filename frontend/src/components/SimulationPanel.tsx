import { useState } from "react";
import { useGraph } from "../hooks/useGraph";

const buttonStyle: React.CSSProperties = {
  padding: "4px 10px",
  fontSize: 12,
  border: "1px solid #ccc",
  borderRadius: 4,
  background: "#fff",
  cursor: "pointer",
};

export function SimulationPanel() {
  const runSimulation = useGraph((s) => s.runSimulation);
  const nodes = useGraph((s) => s.nodes);
  const [nSamples, setNSamples] = useState(64);

  if (nodes.length === 0) return null;

  return (
    <div style={{ marginTop: 16, borderTop: "1px solid #ddd", paddingTop: 12 }}>
      <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>Simulate</h4>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <label style={{ fontSize: 12 }}>Samples:</label>
        <input
          type="number"
          value={nSamples}
          onChange={(e) => setNSamples(Number(e.target.value))}
          style={{ width: 60, fontSize: 12, padding: 2 }}
          min={1}
          max={4096}
        />
        <button style={buttonStyle} onClick={() => void runSimulation(nSamples)}>
          Run
        </button>
      </div>
    </div>
  );
}
