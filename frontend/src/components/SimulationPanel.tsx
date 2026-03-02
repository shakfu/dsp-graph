import { useState } from "react";
import { useGraph } from "../hooks/useGraph";
import type { InputSignalType } from "../api/types";

const buttonStyle: React.CSSProperties = {
  padding: "4px 10px",
  fontSize: 12,
  border: "1px solid #ccc",
  borderRadius: 4,
  background: "#fff",
  cursor: "pointer",
};

const selectStyle: React.CSSProperties = {
  fontSize: 12,
  padding: 2,
  border: "1px solid #ccc",
  borderRadius: 4,
};

const SIGNAL_OPTIONS: { value: InputSignalType; label: string }[] = [
  { value: "impulse", label: "Impulse" },
  { value: "sine", label: "Sine (440 Hz)" },
  { value: "noise", label: "White Noise" },
  { value: "ones", label: "DC (ones)" },
];

export function SimulationPanel() {
  const runSimulation = useGraph((s) => s.runSimulation);
  const nodes = useGraph((s) => s.nodes);
  const inputSignals = useGraph((s) => s.inputSignals);
  const setInputSignal = useGraph((s) => s.setInputSignal);
  const [nSamples, setNSamples] = useState(64);

  if (nodes.length === 0) return null;

  const inputNodes = nodes.filter((n) => n.type === "input");

  return (
    <div style={{ marginTop: 16, borderTop: "1px solid #ddd", paddingTop: 12 }}>
      <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>Simulate</h4>
      {inputNodes.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>Input signals:</div>
          {inputNodes.map((node) => (
            <div
              key={node.id}
              style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 4 }}
            >
              <span style={{ fontSize: 12, minWidth: 40 }}>{node.id}</span>
              <select
                style={selectStyle}
                value={inputSignals[node.id] ?? "sine"}
                onChange={(e) =>
                  setInputSignal(node.id, e.target.value as InputSignalType)
                }
              >
                {SIGNAL_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>
      )}
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
