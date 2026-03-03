import { useState, useMemo } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { cpp } from "@codemirror/lang-cpp";
import { oneDark } from "@codemirror/theme-one-dark";
import { EditorState } from "@codemirror/state";
import { useGraph } from "../hooks/useGraph";
import { useGdspLivePreview } from "../hooks/useGdspLivePreview";
import { gdspLanguage } from "../utils/gdspLang";

const gdspExtensions = [gdspLanguage];
const cppExtensions = [cpp(), EditorState.readOnly.of(true)];

type Tab = "gdsp" | "cpp";

const tabStyle = (active: boolean): React.CSSProperties => ({
  padding: "5px 14px",
  fontSize: 12,
  fontWeight: active ? 600 : 400,
  color: active ? "#e5c07b" : "#636d83",
  background: active ? "#282c34" : "#21252b",
  border: "none",
  borderBottom: active ? "2px solid #e5c07b" : "2px solid transparent",
  cursor: "pointer",
});

export function EditorPane() {
  const gdspSource = useGraph((s) => s.gdspSource);
  const setGdspSource = useGraph((s) => s.setGdspSource);
  const parseError = useGraph((s) => s.parseError);
  const isLivePreview = useGraph((s) => s.isLivePreview);
  const compileResult = useGraph((s) => s.compileResult);
  const runCompile = useGraph((s) => s.runCompile);
  const nodes = useGraph((s) => s.nodes);

  const [activeTab, setActiveTab] = useState<Tab>("gdsp");

  useGdspLivePreview();

  const handleCopy = () => {
    if (compileResult?.cpp_source) {
      void navigator.clipboard.writeText(compileResult.cpp_source);
    }
  };

  const cppSource = useMemo(
    () => compileResult?.cpp_source ?? "",
    [compileResult]
  );

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "#282c34",
      }}
    >
      {/* Tab bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          background: "#21252b",
          borderBottom: "1px solid #181a1f",
          flexShrink: 0,
        }}
      >
        <button style={tabStyle(activeTab === "gdsp")} onClick={() => setActiveTab("gdsp")}>
          .gdsp
          {isLivePreview && (
            <span
              style={{
                fontSize: 9,
                marginLeft: 6,
                padding: "1px 5px",
                borderRadius: 3,
                background: parseError ? "#e06c75" : "#98c379",
                color: "#282c34",
                fontWeight: 600,
              }}
            >
              {parseError ? "ERR" : "OK"}
            </span>
          )}
        </button>
        <button style={tabStyle(activeTab === "cpp")} onClick={() => setActiveTab("cpp")}>
          C++
        </button>
        <div style={{ flex: 1 }} />
        {activeTab === "cpp" && (
          <div style={{ display: "flex", gap: 4, paddingRight: 8 }}>
            <button
              onClick={() => void runCompile()}
              disabled={nodes.length === 0}
              style={{
                padding: "2px 10px",
                fontSize: 11,
                background: "#98c379",
                color: "#282c34",
                border: "none",
                borderRadius: 3,
                cursor: nodes.length === 0 ? "default" : "pointer",
                fontWeight: 600,
                opacity: nodes.length === 0 ? 0.4 : 1,
              }}
            >
              Compile
            </button>
            {compileResult && (
              <button
                onClick={handleCopy}
                style={{
                  padding: "2px 10px",
                  fontSize: 11,
                  background: "transparent",
                  color: "#abb2bf",
                  border: "1px solid #3e4451",
                  borderRadius: 3,
                  cursor: "pointer",
                }}
              >
                Copy
              </button>
            )}
          </div>
        )}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {activeTab === "gdsp" && (
          <CodeMirror
            value={gdspSource}
            onChange={setGdspSource}
            theme={oneDark}
            extensions={gdspExtensions}
            height="100%"
            style={{ height: "100%" }}
            basicSetup={{
              lineNumbers: true,
              foldGutter: false,
              highlightActiveLine: true,
            }}
          />
        )}
        {activeTab === "cpp" && (
          <>
            {cppSource ? (
              <CodeMirror
                value={cppSource}
                theme={oneDark}
                extensions={cppExtensions}
                height="100%"
                style={{ height: "100%" }}
                editable={false}
                basicSetup={{
                  lineNumbers: true,
                  foldGutter: true,
                  highlightActiveLine: false,
                }}
              />
            ) : (
              <div
                style={{
                  padding: 24,
                  color: "#636d83",
                  fontSize: 13,
                  textAlign: "center",
                }}
              >
                Click "Compile" to generate C++ source.
              </div>
            )}
          </>
        )}
      </div>

      {/* Error bar (gdsp tab only) */}
      {activeTab === "gdsp" && parseError && (
        <div
          style={{
            padding: "4px 12px",
            fontSize: 11,
            color: "#e06c75",
            background: "#2c2c2c",
            borderTop: "1px solid #3e4451",
            fontFamily: "monospace",
            flexShrink: 0,
          }}
        >
          Line {parseError.line}:{parseError.col} - {parseError.message}
        </div>
      )}
    </div>
  );
}
