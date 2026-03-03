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
  const buildPlatforms = useGraph((s) => s.buildPlatforms);
  const runBuild = useGraph((s) => s.runBuild);
  const downloadBuildZip = useGraph((s) => s.downloadBuildZip);
  const fetchBuildPlatforms = useGraph((s) => s.fetchBuildPlatforms);

  const [platform, setPlatform] = useState("clap");
  const [showOutput, setShowOutput] = useState<"dsp" | "adapter" | "manifest" | null>(null);

  useEffect(() => {
    if (buildPlatforms.length === 0) {
      void fetchBuildPlatforms();
    }
  }, [buildPlatforms.length, fetchBuildPlatforms]);

  if (nodes.length === 0) return null;

  const platformList = buildPlatforms.length > 0
    ? buildPlatforms
    : ["au", "chuck", "circle", "clap", "daisy", "lv2", "max", "pd", "sc", "vcvrack", "vst3"];

  return (
    <div style={{ marginTop: 12, borderTop: "1px solid #ddd", paddingTop: 12 }}>
      <h4 style={{ margin: "0 0 8px", fontSize: 13 }}>Build Plugin</h4>
      <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 8 }}>
        <label style={{ fontSize: 12 }}>Platform:</label>
        <select
          style={selectStyle}
          value={platform}
          onChange={(e) => setPlatform(e.target.value)}
        >
          {platformList.map((p) => (
            <option key={p} value={p}>{p.toUpperCase()}</option>
          ))}
        </select>
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <button style={buttonStyle} onClick={() => void runBuild(platform)}>
          Build
        </button>
        <button style={buttonStyle} onClick={() => void downloadBuildZip(platform)}>
          Download Zip
        </button>
      </div>
      {buildResult && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>
            Built for {buildResult.platform.toUpperCase()}
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
    </div>
  );
}
