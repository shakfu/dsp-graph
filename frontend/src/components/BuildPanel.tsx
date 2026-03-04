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
  const generateResult = useGraph((s) => s.generateResult);
  const buildResult = useGraph((s) => s.buildResult);
  const buildPlatforms = useGraph((s) => s.buildPlatforms);
  const batchBuildResults = useGraph((s) => s.batchBuildResults);
  const batchBuildId = useGraph((s) => s.batchBuildId);
  const runGenerate = useGraph((s) => s.runGenerate);
  const runBuild = useGraph((s) => s.runBuild);
  const runBatchBuild = useGraph((s) => s.runBatchBuild);
  const downloadGenerateZip = useGraph((s) => s.downloadGenerateZip);
  const downloadBuiltBinary = useGraph((s) => s.downloadBuiltBinary);
  const downloadBatchBuildZip = useGraph((s) => s.downloadBatchBuildZip);
  const fetchBuildPlatforms = useGraph((s) => s.fetchBuildPlatforms);

  const [platform, setPlatform] = useState("");
  const [showOutput, setShowOutput] = useState<"dsp" | "adapter" | "manifest" | null>(null);
  const [showBuildLog, setShowBuildLog] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);

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

  const handleBatchBuild = async () => {
    setBatchLoading(true);
    try {
      await runBatchBuild(buildPlatforms);
    } finally {
      setBatchLoading(false);
    }
  };

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
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        <button style={buttonStyle} onClick={() => void runGenerate(platform)}>
          Generate
        </button>
        <button style={buttonStyle} onClick={() => void downloadGenerateZip(platform)}>
          Download Zip
        </button>
        <button style={buttonStyle} onClick={() => void runBuild(platform)}>
          Build
        </button>
        <button
          style={buttonStyle}
          onClick={() => void handleBatchBuild()}
          disabled={batchLoading}
        >
          {batchLoading ? "Building..." : "Build All"}
        </button>
      </div>
      {generateResult && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>
            Generated for {generateResult.platform.toUpperCase()}
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
            <div style={codeBlockStyle}>{generateResult.dsp_cpp}</div>
          )}
          {showOutput === "adapter" && (
            <div style={codeBlockStyle}>{generateResult.adapter_cpp}</div>
          )}
          {showOutput === "manifest" && (
            <div style={codeBlockStyle}>{generateResult.manifest}</div>
          )}
        </div>
      )}
      {buildResult && (
        <div style={{ marginTop: 8 }}>
          <div
            style={{
              fontSize: 11,
              color: buildResult.success ? "#2e7d32" : "#c62828",
              marginBottom: 4,
              fontWeight: 600,
            }}
          >
            {buildResult.success ? "Build succeeded" : "Build failed"}
            {" -- "}
            {buildResult.platform.toUpperCase()}
          </div>
          {(buildResult.stdout || buildResult.stderr) && (
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
                  {buildResult.stdout}
                  {buildResult.stderr && (
                    <>
                      {buildResult.stdout ? "\n--- stderr ---\n" : ""}
                      {buildResult.stderr}
                    </>
                  )}
                </div>
              )}
            </div>
          )}
          {buildResult.success && buildResult.output_file && (
            <button
              style={{ ...buttonStyle, marginTop: 4 }}
              onClick={() => void downloadBuiltBinary(platform)}
            >
              Download Binary
            </button>
          )}
        </div>
      )}
      {batchBuildResults.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: "#666", marginBottom: 4, fontWeight: 600 }}>
            Batch Build Results
          </div>
          <table style={{ fontSize: 11, borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "2px 6px", borderBottom: "1px solid #ddd" }}>
                  Platform
                </th>
                <th style={{ textAlign: "left", padding: "2px 6px", borderBottom: "1px solid #ddd" }}>
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {batchBuildResults.map((r) => (
                <tr key={r.platform}>
                  <td style={{ padding: "2px 6px", fontFamily: "monospace" }}>
                    {r.platform.toUpperCase()}
                  </td>
                  <td
                    style={{
                      padding: "2px 6px",
                      color: r.success ? "#2e7d32" : "#c62828",
                      fontWeight: 600,
                    }}
                  >
                    {r.success ? "OK" : "FAIL"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {batchBuildId && batchBuildResults.some((r) => r.success) && (
            <button
              style={{ ...buttonStyle, marginTop: 6 }}
              onClick={() => void downloadBatchBuildZip()}
            >
              Download All
            </button>
          )}
        </div>
      )}
    </div>
  );
}
