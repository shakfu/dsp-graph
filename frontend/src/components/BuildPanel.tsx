import { useState, useEffect } from "react";
import { useGraph } from "../hooks/useGraph";

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
  border: "1px solid #ccc",
  borderRadius: 4,
};

const codeBlockStyle: React.CSSProperties = {
  fontSize: 10,
  fontFamily: "monospace",
  background: "#f5f5f5",
  border: "1px solid #ddd",
  borderRadius: 3,
  padding: 6,
  maxHeight: 120,
  overflow: "auto",
  whiteSpace: "pre",
  marginTop: 4,
};

export function BuildPanel() {
  const nodes = useGraph((s) => s.nodes);
  const buildResult = useGraph((s) => s.buildResult);
  const compileBuildResult = useGraph((s) => s.compileBuildResult);
  const buildPlatforms = useGraph((s) => s.buildPlatforms);
  const runBuild = useGraph((s) => s.runBuild);
  const runCompileBuild = useGraph((s) => s.runCompileBuild);
  const downloadBuildZip = useGraph((s) => s.downloadBuildZip);
  const downloadBuiltBinary = useGraph((s) => s.downloadBuiltBinary);
  const fetchBuildPlatforms = useGraph((s) => s.fetchBuildPlatforms);

  const [platform, setPlatform] = useState("");
  const [showOutput, setShowOutput] = useState<"dsp" | "adapter" | "manifest" | null>(null);
  const [showBuildLog, setShowBuildLog] = useState(false);

  useEffect(() => {
    if (buildPlatforms.length === 0) {
      void fetchBuildPlatforms();
    }
  }, [buildPlatforms.length, fetchBuildPlatforms]);

  // Default to first available platform once fetched
  useEffect(() => {
    if (buildPlatforms.length > 0 && !platform) {
      setPlatform(buildPlatforms[0] ?? "");
    }
  }, [buildPlatforms, platform]);

  if (nodes.length === 0) return null;

  return (
    <div>
      <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>Plugin Target</h4>
      <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 8 }}>
        <label style={{ fontSize: 12 }}>Platform:</label>
        <select
          style={selectStyle}
          value={platform}
          onChange={(e) => setPlatform(e.target.value)}
        >
          {buildPlatforms.map((p) => (
            <option key={p} value={p}>{p.toUpperCase()}</option>
          ))}
        </select>
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <button style={buttonStyle} onClick={() => void runBuild(platform)}>
          Generate
        </button>
        <button style={buttonStyle} onClick={() => void downloadBuildZip(platform)}>
          Download Zip
        </button>
        <button style={buttonStyle} onClick={() => void runCompileBuild(platform)}>
          Build
        </button>
      </div>
      {buildResult && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>
            Generated for {buildResult.platform.toUpperCase()}
          </div>
          <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
            <button
              style={{
                ...buttonStyle,
                fontSize: 10,
                padding: "2px 6px",
                fontWeight: showOutput === "dsp" ? 600 : 400,
              }}
              onClick={() => setShowOutput(showOutput === "dsp" ? null : "dsp")}
            >
              DSP C++
            </button>
            <button
              style={{
                ...buttonStyle,
                fontSize: 10,
                padding: "2px 6px",
                fontWeight: showOutput === "adapter" ? 600 : 400,
              }}
              onClick={() => setShowOutput(showOutput === "adapter" ? null : "adapter")}
            >
              Adapter
            </button>
            <button
              style={{
                ...buttonStyle,
                fontSize: 10,
                padding: "2px 6px",
                fontWeight: showOutput === "manifest" ? 600 : 400,
              }}
              onClick={() => setShowOutput(showOutput === "manifest" ? null : "manifest")}
            >
              Manifest
            </button>
          </div>
          {showOutput === "dsp" && (
            <div style={codeBlockStyle}>{buildResult.dsp_cpp}</div>
          )}
          {showOutput === "adapter" && (
            <div style={codeBlockStyle}>{buildResult.adapter_cpp}</div>
          )}
          {showOutput === "manifest" && (
            <div style={codeBlockStyle}>{buildResult.manifest}</div>
          )}
        </div>
      )}
      {compileBuildResult && (
        <div style={{ marginTop: 8 }}>
          <div
            style={{
              fontSize: 11,
              color: compileBuildResult.success ? "#2e7d32" : "#c62828",
              marginBottom: 4,
              fontWeight: 600,
            }}
          >
            {compileBuildResult.success ? "Build succeeded" : "Build failed"}
            {" -- "}
            {compileBuildResult.platform.toUpperCase()}
          </div>
          {(compileBuildResult.stdout || compileBuildResult.stderr) && (
            <div>
              <button
                style={{
                  ...buttonStyle,
                  fontSize: 10,
                  padding: "2px 6px",
                  fontWeight: showBuildLog ? 600 : 400,
                  marginBottom: 4,
                }}
                onClick={() => setShowBuildLog(!showBuildLog)}
              >
                {showBuildLog ? "Hide Log" : "Show Log"}
              </button>
              {showBuildLog && (
                <div style={codeBlockStyle}>
                  {compileBuildResult.stdout}
                  {compileBuildResult.stderr && (
                    <>
                      {compileBuildResult.stdout ? "\n--- stderr ---\n" : ""}
                      {compileBuildResult.stderr}
                    </>
                  )}
                </div>
              )}
            </div>
          )}
          {compileBuildResult.success && compileBuildResult.output_file && (
            <button
              style={{ ...buttonStyle, marginTop: 4 }}
              onClick={() => void downloadBuiltBinary(platform)}
            >
              Download Binary
            </button>
          )}
        </div>
      )}
    </div>
  );
}
