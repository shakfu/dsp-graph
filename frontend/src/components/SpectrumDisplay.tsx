import { useRef, useEffect } from "react";
import { computeSpectrum } from "../utils/fft";

interface SpectrumDisplayProps {
  data: number[];
  sampleRate?: number;
  width?: number;
  height?: number;
  color?: string;
  label?: string;
}

export function SpectrumDisplay({
  data,
  sampleRate = 44100,
  width = 300,
  height = 100,
  color = "#e91e63",
  label,
}: SpectrumDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, width, height);

    const mag = computeSpectrum(data);
    const bins = mag.length;

    // dB range
    const dbMin = -80;
    const dbMax = 0;
    const dbRange = dbMax - dbMin;

    // Draw spectrum
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const step = width / Math.max(bins - 1, 1);
    for (let i = 0; i < bins; i++) {
      const x = i * step;
      const db = Math.max(dbMin, Math.min(dbMax, mag[i]!));
      const y = height - ((db - dbMin) / dbRange) * height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw dB grid lines
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 4]);
    for (const db of [-20, -40, -60]) {
      const y = height - ((db - dbMin) / dbRange) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = "#888";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("0 dB", 2, 10);
    ctx.fillText(`${dbMin} dB`, 2, height - 3);
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(sampleRate / 2)} Hz`, width - 2, height - 3);
  }, [data, sampleRate, width, height, color]);

  return (
    <div style={{ marginTop: 4 }}>
      {label && (
        <div style={{ fontSize: 10, color: "#888", marginBottom: 2 }}>{label}</div>
      )}
      <canvas
        ref={canvasRef}
        style={{ width, height, borderRadius: 3, display: "block" }}
      />
    </div>
  );
}
