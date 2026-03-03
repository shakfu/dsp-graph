import { useState } from "react";
import { useGraph } from "../hooks/useGraph";
import { WaveformDisplay } from "./WaveformDisplay";
import { SpectrumDisplay } from "./SpectrumDisplay";
import type { InputSignalType } from "../api/types";

const buttonStyle: React.CSSProperties = {
  padding: "4px 10px",
  fontSize: 12,
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

function ParamSlider({
  name,
  defaultValue,
  min,
  max,
  onChangeCommitted,
}: {
  name: string;
  defaultValue: number;
  min: number;
  max: number;
  onChangeCommitted: (value: number) => void;
}) {
  const [value, setValue] = useState(defaultValue);

  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 4 }}>
      <span style={{ fontSize: 11, minWidth: 50 }}>{name}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={(max - min) / 200}
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          setValue(v);
        }}
        onMouseUp={() => onChangeCommitted(value)}
        onTouchEnd={() => onChangeCommitted(value)}
        style={{ flex: 1 }}
      />
      <span style={{ fontSize: 10, fontFamily: "monospace", minWidth: 40, textAlign: "right" }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

type DisplayMode = "none" | "time" | "freq";

export function SimulationPanel() {
  const runSimulation = useGraph((s) => s.runSimulation);
  const continueSimulation = useGraph((s) => s.continueSimulation);
  const resetSimulation = useGraph((s) => s.resetSimulation);
  const setSimParam = useGraph((s) => s.setSimParam);
  const fetchPeek = useGraph((s) => s.fetchPeek);
  const nodes = useGraph((s) => s.nodes);
  const simSessionId = useGraph((s) => s.simSessionId);
  const simulationResult = useGraph((s) => s.simulationResult);
  const accumulatedOutputs = useGraph((s) => s.accumulatedOutputs);
  const peekValues = useGraph((s) => s.peekValues);
  const bufferData = useGraph((s) => s.bufferData);
  const fetchBuffer = useGraph((s) => s.fetchBuffer);
  const inputSignals = useGraph((s) => s.inputSignals);
  const setInputSignal = useGraph((s) => s.setInputSignal);
  const [nSamples, setNSamples] = useState(64);
  const [displayMode, setDisplayMode] = useState<DisplayMode>("time");

  if (nodes.length === 0) return null;

  const inputNodes = nodes.filter((n) => n.type === "input");
  const paramNodes = nodes.filter((n) => n.type === "param");
  const bufferNodes = nodes.filter((n) => n.data.op === "buffer");

  const outputKeys = simulationResult
    ? Object.keys(simulationResult.outputs)
    : [];

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

      {simSessionId && (
        <div style={{ marginTop: 8 }}>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            <button
              style={buttonStyle}
              onClick={() => void continueSimulation(nSamples)}
            >
              Continue ({nSamples})
            </button>
            <button style={buttonStyle} onClick={() => void resetSimulation()}>
              Reset
            </button>
            <button style={buttonStyle} onClick={() => void fetchPeek()}>
              Peek
            </button>
          </div>

          {paramNodes.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>
                Parameters:
              </div>
              {paramNodes.map((node) => {
                const nd = node.data.node_data ?? {};
                const name = (nd.name as string) ?? node.id;
                const min = (nd.min as number) ?? 0;
                const max = (nd.max as number) ?? 1;
                const def = (nd.default as number) ?? (min + max) / 2;
                return (
                  <ParamSlider
                    key={name}
                    name={name}
                    defaultValue={def}
                    min={min}
                    max={max}
                    onChangeCommitted={(v) => {
                      void setSimParam(name, v);
                      void continueSimulation(nSamples);
                    }}
                  />
                );
              })}
            </div>
          )}

          {peekValues && Object.keys(peekValues).length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>
                Peek Values:
              </div>
              {Object.entries(peekValues).map(([id, val]) => (
                <div key={id} style={{ fontSize: 11, fontFamily: "monospace" }}>
                  {id}: {val.toFixed(6)}
                </div>
              ))}
            </div>
          )}

          {bufferNodes.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>
                Buffers:
              </div>
              {bufferNodes.map((node) => (
                <div key={node.id} style={{ marginBottom: 4 }}>
                  <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    <span style={{ fontSize: 11, fontFamily: "monospace" }}>{node.id}</span>
                    <button
                      style={{ ...buttonStyle, fontSize: 10, padding: "2px 6px" }}
                      onClick={() => void fetchBuffer(node.id)}
                    >
                      View
                    </button>
                  </div>
                  {bufferData[node.id] != null && (
                    <WaveformDisplay
                      data={bufferData[node.id]!}
                      label={`${node.id} buffer`}
                      width={260}
                      height={60}
                      color="#ff9800"
                    />
                  )}
                </div>
              ))}
            </div>
          )}

          {outputKeys.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>
                Output display:
              </div>
              <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
                <button
                  style={displayMode === "time" ? activeButtonStyle : buttonStyle}
                  onClick={() => setDisplayMode(displayMode === "time" ? "none" : "time")}
                >
                  Time
                </button>
                <button
                  style={displayMode === "freq" ? activeButtonStyle : buttonStyle}
                  onClick={() => setDisplayMode(displayMode === "freq" ? "none" : "freq")}
                >
                  Freq
                </button>
              </div>
              {displayMode === "time" &&
                outputKeys.map((key) => (
                  <WaveformDisplay
                    key={key}
                    data={accumulatedOutputs[key] ?? []}
                    label={key}
                    width={260}
                    height={80}
                  />
                ))}
              {displayMode === "freq" &&
                outputKeys.map((key) => (
                  <SpectrumDisplay
                    key={key}
                    data={accumulatedOutputs[key] ?? []}
                    label={key}
                    width={260}
                    height={80}
                  />
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
