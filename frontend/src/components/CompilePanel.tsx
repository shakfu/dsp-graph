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

export function CompilePanel() {
  const runCompile = useGraph((s) => s.runCompile);
  const compileResult = useGraph((s) => s.compileResult);
  const nodes = useGraph((s) => s.nodes);
  const [expanded, setExpanded] = useState(false);

  if (nodes.length === 0) return null;

  const handleCopy = () => {
    if (compileResult?.cpp_source) {
      void navigator.clipboard.writeText(compileResult.cpp_source);
    }
  };

  return (
    <div style={{ marginTop: 12, borderTop: "1px solid #ddd", paddingTop: 12 }}>
      <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>Compile</h4>
      <div style={{ display: "flex", gap: 6 }}>
        <button style={buttonStyle} onClick={() => void runCompile()}>
          Compile to C++
        </button>
        {compileResult && (
          <>
            <button
              style={buttonStyle}
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? "Hide" : "Show"}
            </button>
            <button style={buttonStyle} onClick={handleCopy}>
              Copy
            </button>
          </>
        )}
      </div>
      {compileResult && expanded && (
        <pre
          style={{
            marginTop: 8,
            padding: 8,
            background: "#1e1e1e",
            color: "#d4d4d4",
            fontSize: 11,
            borderRadius: 4,
            overflow: "auto",
            maxHeight: 300,
            whiteSpace: "pre-wrap",
            wordBreak: "break-all",
          }}
        >
          {compileResult.cpp_source}
        </pre>
      )}
    </div>
  );
}
