import CodeMirror from "@uiw/react-codemirror";
import { javascript } from "@codemirror/lang-javascript";
import { oneDark } from "@codemirror/theme-one-dark";
import { useGraph } from "../hooks/useGraph";
import { useGdspLivePreview } from "../hooks/useGdspLivePreview";

const extensions = [javascript()];

export function GdspEditor() {
  const gdspSource = useGraph((s) => s.gdspSource);
  const setGdspSource = useGraph((s) => s.setGdspSource);
  const parseError = useGraph((s) => s.parseError);
  const isLivePreview = useGraph((s) => s.isLivePreview);

  useGdspLivePreview();

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "#282c34",
      }}
    >
      <div
        style={{
          padding: "6px 12px",
          fontSize: 12,
          color: "#abb2bf",
          borderBottom: "1px solid #3e4451",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span style={{ fontWeight: 600 }}>.gdsp</span>
        {isLivePreview && (
          <span
            style={{
              fontSize: 10,
              padding: "1px 6px",
              borderRadius: 3,
              background: parseError ? "#e06c75" : "#98c379",
              color: "#282c34",
              fontWeight: 600,
            }}
          >
            {parseError ? "ERROR" : "OK"}
          </span>
        )}
      </div>
      <div style={{ flex: 1, overflow: "auto" }}>
        <CodeMirror
          value={gdspSource}
          onChange={setGdspSource}
          theme={oneDark}
          extensions={extensions}
          height="100%"
          style={{ height: "100%" }}
          basicSetup={{
            lineNumbers: true,
            foldGutter: false,
            highlightActiveLine: true,
          }}
        />
      </div>
      {parseError && (
        <div
          style={{
            padding: "4px 12px",
            fontSize: 11,
            color: "#e06c75",
            background: "#2c2c2c",
            borderTop: "1px solid #3e4451",
            fontFamily: "monospace",
          }}
        >
          Line {parseError.line}:{parseError.col} - {parseError.message}
        </div>
      )}
    </div>
  );
}
