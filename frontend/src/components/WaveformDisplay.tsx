import { useRef, useEffect } from "react";

interface WaveformDisplayProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  label?: string;
}

export function WaveformDisplay({
  data,
  width = 300,
  height = 100,
  color = "#1976d2",
  label,
}: WaveformDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, width, height);

    // Find amplitude range
    let min = data[0] ?? 0;
    let max = data[0] ?? 0;
    for (const v of data) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    // Ensure symmetric range around zero, with minimum range
    const absMax = Math.max(Math.abs(min), Math.abs(max), 0.001);
    const rangeMin = -absMax;
    const rangeMax = absMax;
    const range = rangeMax - rangeMin;

    // Draw center line
    const centerY = height / 2;
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw waveform
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const step = width / Math.max(data.length - 1, 1);
    for (let i = 0; i < data.length; i++) {
      const x = i * step;
      const y = height - ((((data[i] ?? 0) - rangeMin) / range) * height);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw amplitude labels
    ctx.fillStyle = "#888";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`+${absMax.toFixed(2)}`, 2, 10);
    ctx.fillText(`-${absMax.toFixed(2)}`, 2, height - 3);
  }, [data, width, height, color]);

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
