import { useGraph } from "../hooks/useGraph";

const PASS_LABELS: Record<string, string> = {
  constant_fold: "Constant fold",
  eliminate_cse: "Common subexpression elimination",
  eliminate_dead_nodes: "Dead node elimination",
  promote_control_rate: "Control-rate promotion",
};

const buttonStyle: React.CSSProperties = {
  padding: "4px 10px",
  fontSize: 12,
  border: "1px solid #ccc",
  borderRadius: 4,
  background: "#fff",
  cursor: "pointer",
};

export function OptimizePanel() {
  const runOptimize = useGraph((s) => s.runOptimize);
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
      {passResults.length > 0 && (
        <div style={{ marginTop: 8, fontSize: 11, lineHeight: 1.6 }}>
          {passResults.map((r, i) => {
            const label = PASS_LABELS[r.passName] ?? r.passName;
            const diff = r.nodesBefore - r.nodesAfter;
            return (
              <div key={i} style={{ color: diff > 0 ? "#2a7" : "#888" }}>
                {label}: {diff > 0 ? `${diff} node${diff !== 1 ? "s" : ""} removed` : "no change"}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
