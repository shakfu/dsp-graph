import { useGraph } from "../hooks/useGraph";
import type { ElkAlgorithm, ElkDirection } from "../utils/elkLayout";

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
};

const inputStyle: React.CSSProperties = {
  width: 60,
  fontSize: 12,
  padding: 2,
};

const ALGORITHMS: { value: ElkAlgorithm; label: string }[] = [
  { value: "layered", label: "Layered" },
  { value: "stress", label: "Stress" },
  { value: "mrtree", label: "Mr. Tree" },
  { value: "radial", label: "Radial" },
  { value: "force", label: "Force" },
];

const DIRECTIONS: { value: ElkDirection; label: string }[] = [
  { value: "RIGHT", label: "Right" },
  { value: "DOWN", label: "Down" },
  { value: "LEFT", label: "Left" },
  { value: "UP", label: "Up" },
];

const DIRECTIONAL_ALGORITHMS: ElkAlgorithm[] = ["layered", "mrtree"];

export function LayoutPanel() {
  const nodes = useGraph((s) => s.nodes);
  const layoutOptions = useGraph((s) => s.layoutOptions);
  const setLayoutOptions = useGraph((s) => s.setLayoutOptions);
  const runLayout = useGraph((s) => s.runLayout);

  if (nodes.length === 0) return null;

  const showDirection = DIRECTIONAL_ALGORITHMS.includes(layoutOptions.algorithm);

  return (
    <div style={{ marginTop: 16, borderTop: "1px solid #ddd", paddingTop: 12 }}>
      <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>Layout</h4>

      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <label style={{ fontSize: 12 }}>Algorithm:</label>
          <select
            style={selectStyle}
            value={layoutOptions.algorithm}
            onChange={(e) =>
              setLayoutOptions({ algorithm: e.target.value as ElkAlgorithm })
            }
          >
            {ALGORITHMS.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>

        {showDirection && (
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <label style={{ fontSize: 12 }}>Direction:</label>
            <select
              style={selectStyle}
              value={layoutOptions.direction}
              onChange={(e) =>
                setLayoutOptions({ direction: e.target.value as ElkDirection })
              }
            >
              {DIRECTIONS.map((d) => (
                <option key={d.value} value={d.value}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
        )}

        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <label style={{ fontSize: 12 }}>Node spacing:</label>
          <input
            type="number"
            style={inputStyle}
            value={layoutOptions.nodeSpacing}
            onChange={(e) => setLayoutOptions({ nodeSpacing: Number(e.target.value) })}
            min={0}
            max={500}
          />
        </div>

        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <label style={{ fontSize: 12 }}>Layer spacing:</label>
          <input
            type="number"
            style={inputStyle}
            value={layoutOptions.layerSpacing}
            onChange={(e) => setLayoutOptions({ layerSpacing: Number(e.target.value) })}
            min={0}
            max={500}
          />
        </div>

        <button style={buttonStyle} onClick={() => void runLayout()}>
          Apply Layout
        </button>
      </div>
    </div>
  );
}
