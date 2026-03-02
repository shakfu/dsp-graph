import { useGraph } from "../hooks/useGraph";

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
  const nodes = useGraph((s) => s.nodes);

  if (nodes.length === 0) return null;

  return (
    <div style={{ marginTop: 12, borderTop: "1px solid #ddd", paddingTop: 12 }}>
      <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>Optimize</h4>
      <button style={buttonStyle} onClick={() => void runOptimize()}>
        Run Optimizer
      </button>
    </div>
  );
}
