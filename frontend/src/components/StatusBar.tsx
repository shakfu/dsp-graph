import { useGraph } from "../hooks/useGraph";

const statusStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "3px 12px",
  fontSize: 11,
  color: "#666",
  borderTop: "1px solid #ddd",
  background: "#f8f9fa",
  flexShrink: 0,
};

const dotStyle = (color: string): React.CSSProperties => ({
  display: "inline-block",
  width: 7,
  height: 7,
  borderRadius: "50%",
  background: color,
  marginRight: 4,
});

export function StatusBar() {
  const nodes = useGraph((s) => s.nodes);
  const edges = useGraph((s) => s.edges);
  const parseError = useGraph((s) => s.parseError);
  const validationResult = useGraph((s) => s.validationResult);
  const isLivePreview = useGraph((s) => s.isLivePreview);
  const graphName = useGraph((s) => s.graphName);

  const parseOk = !parseError;
  const validOk = validationResult?.valid === true;

  return (
    <div style={statusStyle}>
      {isLivePreview && (
        <span>
          <span style={dotStyle(parseOk ? "#28a745" : "#dc3545")} />
          Parse: {parseOk ? "OK" : "Error"}
        </span>
      )}
      {validationResult && (
        <span>
          <span style={dotStyle(validOk ? "#28a745" : "#ffc107")} />
          Valid: {validOk ? "Yes" : `${validationResult.errors.length} error(s)`}
        </span>
      )}
      {nodes.length > 0 && (
        <>
          <span>Nodes: {nodes.length}</span>
          <span>Edges: {edges.length}</span>
        </>
      )}
      {graphName && <span style={{ marginLeft: "auto" }}>{graphName}</span>}
    </div>
  );
}
