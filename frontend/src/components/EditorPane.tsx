import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import CodeMirror, { type ReactCodeMirrorRef } from "@uiw/react-codemirror";
import { cpp } from "@codemirror/lang-cpp";
import { oneDark } from "@codemirror/theme-one-dark";
import { EditorState } from "@codemirror/state";
import { bracketMatching } from "@codemirror/language";
import { autocompletion } from "@codemirror/autocomplete";
import { setDiagnostics, type Diagnostic } from "@codemirror/lint";
import { useGraph } from "../hooks/useGraph";
import { useGdspLivePreview } from "../hooks/useGdspLivePreview";
import { gdspLanguage } from "../utils/gdspLang";
import { gdspCompletionSource } from "../utils/gdspCompletions";
import { gdspGoToDef } from "../utils/gdspGoToDef";

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
  const selectNodeById = useGraph((s) => s.selectNodeById);

  const editorRef = useRef<ReactCodeMirrorRef>(null);
  const [activeTab, setActiveTab] = useState<Tab>("gdsp");

  useGdspLivePreview();

  // Build gdsp extensions with dynamic completions and go-to-def
  const gdspExtensions = useMemo(() => {
    const getNodeIds = () => useGraph.getState().nodes.map((n) => n.id);
    const getNodeIdSet = () => new Set(getNodeIds());

    const completionExt = autocompletion({
      override: [gdspCompletionSource(getNodeIds)],
    });

    const goToDefExt = gdspGoToDef(getNodeIdSet, selectNodeById);

    return [
      gdspLanguage,
      bracketMatching(),
      completionExt,
      goToDefExt,
    ];
  }, [selectNodeById]);

  // Push diagnostics imperatively when parseError changes
  useEffect(() => {
    const view = editorRef.current?.view;
    if (!view) return;

    const diagnostics: Diagnostic[] = [];
    if (parseError && parseError.line > 0) {
      const doc = view.state.doc;
      const lineCount = doc.lines;
      const lineNum = Math.min(parseError.line, lineCount);
      const line = doc.line(lineNum);
      const col = Math.max(0, parseError.col - 1);
      const from = Math.min(line.from + col, line.to);
      const to = line.to;
      if (from <= doc.length) {
        diagnostics.push({
          from,
          to: Math.max(to, from + 1),
          severity: "error",
          message: parseError.message,
        });
      }
    }
    view.dispatch(setDiagnostics(view.state, diagnostics));
  }, [parseError]);

  const graphName = useGraph((s) => s.graphName);

  const handleCopy = () => {
    if (compileResult?.cpp_source) {
      void navigator.clipboard.writeText(compileResult.cpp_source);
    }
  };

  const handleDownloadCpp = () => {
    if (!compileResult?.cpp_source) return;
    const filename = `${graphName || "graph"}.h`;
    const blob = new Blob([compileResult.cpp_source], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const cppSource = useMemo(
    () => compileResult?.cpp_source ?? "",
    [compileResult]
  );

  const onCreateEditor = useCallback(() => {
    // EditorView is accessible via ref after creation -- no extra work needed
  }, []);

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
        {activeTab === "gdsp" && (
          <div style={{ paddingRight: 8 }}>
            <button
              onClick={() => {
                void runCompile().then(() => setActiveTab("cpp"));
              }}
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
          </div>
        )}
        {activeTab === "cpp" && compileResult && (
          <div style={{ paddingRight: 8, display: "flex", gap: 4 }}>
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
            <button
              onClick={handleDownloadCpp}
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
              Download
            </button>
          </div>
        )}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {activeTab === "gdsp" && (
          <CodeMirror
            ref={editorRef}
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
              bracketMatching: false, // we provide our own
              closeBrackets: true,
              autocompletion: false, // we provide our own
            }}
            onCreateEditor={onCreateEditor}
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
