import type { OptimizePassName } from "../api/types";
import { useGraph } from "../hooks/useGraph";

const PASSES: { name: OptimizePassName; label: string }[] = [
  { name: "constant_fold", label: "Constant fold" },
  { name: "eliminate_cse", label: "Common subexpression elimination" },
  { name: "eliminate_dead_nodes", label: "Dead node elimination" },
  { name: "promote_control_rate", label: "Control-rate promotion" },
];

const PASS_LABELS: Record<string, string> = Object.fromEntries(
  PASSES.map((p) => [p.name, p.label])
);

const buttonStyle: React.CSSProperties = {
  padding: "4px 10px",
  fontSize: 12,
  border: "1px solid #ccc",
  borderRadius: 4,
  background: "#fff",
  cursor: "pointer",
};

const passButtonStyle: React.CSSProperties = {
  ...buttonStyle,
  padding: "3px 8px",
  fontSize: 11,
  textAlign: "left",
};

export function OptimizePanel() {
  const runOptimize = useGraph((s) => s.runOptimize);
  const runSinglePass = useGraph((s) => s.runSinglePass);
  const resetOptimize = useGraph((s) => s.resetOptimize);
  const passResults = useGraph((s) => s.passResults);
  const preOptimizeSnapshot = useGraph((s) => s.preOptimizeSnapshot);
  const nodes = useGraph((s) => s.nodes);

  if (nodes.length === 0) return null;

  return (
    <div style={{ marginTop: 12, borderTop: "1px solid #ddd", paddingTop: 12 }}>
      <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>Optimize</h4>
      <div style={{ display: "flex", gap: 4 }}>
        <button style={buttonStyle} onClick={() => void runOptimize()}>
          Run All
        </button>
        {preOptimizeSnapshot && (
          <button style={buttonStyle} onClick={resetOptimize}>
            Reset
          </button>
        )}
      </div>

      <div style={{ marginTop: 8, fontSize: 11, color: "#888" }}>
        Step through one pass at a time:
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 4 }}>
        {PASSES.map((p) => (
          <button
            key={p.name}
            style={passButtonStyle}
            title={`Apply ${p.label} to the current graph`}
            onClick={() => void runSinglePass(p.name)}
          >
            {p.label}
          </button>
        ))}
      </div>

      {passResults.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, lineHeight: 1.6 }}>
          {passResults.map((r, i) => {
            const label = PASS_LABELS[r.passName] ?? r.passName;
            const diff = r.nodesBefore - r.nodesAfter;
            return (
              <div key={i} style={{ color: diff > 0 ? "#2a7" : "#888" }}>
                {i + 1}. {label}: {r.nodesBefore} &rarr; {r.nodesAfter} nodes{" "}
                {diff > 0 ? `(${diff} removed)` : "(no change)"}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
