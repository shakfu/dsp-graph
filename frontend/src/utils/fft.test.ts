import { describe, expect, it } from "vitest";
import { computeSpectrum } from "./fft";

/** Index of the maximum-magnitude bin. */
function argmax(mag: Float64Array): number {
  let best = 0;
  for (let i = 1; i < mag.length; i++) {
    if (mag[i]! > mag[best]!) best = i;
  }
  return best;
}

describe("computeSpectrum", () => {
  it("returns n/2 + 1 bins (DC through Nyquist inclusive)", () => {
    const n = 64;
    const data = Array.from({ length: n }, () => 0);
    expect(computeSpectrum(data).length).toBe(n / 2 + 1);
  });

  it("peaks at the bin of a bin-aligned sinusoid, near -6 dB", () => {
    const n = 64;
    const k = 8; // 8 cycles across the frame -> energy in bin 8
    const data = Array.from({ length: n }, (_, i) => Math.cos((2 * Math.PI * k * i) / n));
    const mag = computeSpectrum(data);
    // The exact bin captures ~0.5 * coherentGain regardless of the window, so a
    // full-scale bin-aligned sinusoid reads about -6 dB (single-sided, no x2).
    expect(argmax(mag)).toBe(k);
    expect(Math.abs(mag[k]! - -6.02)).toBeLessThan(0.6);
  });

  it("places a Nyquist-frequency signal in the last (Nyquist) bin", () => {
    const n = 64;
    // Alternating +/-1 is the Nyquist frequency (sr/2).
    const data = Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 1 : -1));
    const mag = computeSpectrum(data);
    expect(argmax(mag)).toBe(mag.length - 1); // == n/2, the Nyquist bin
  });

  it("zero-pads non-power-of-2 input to the next power of 2", () => {
    // len 48 -> n 64 -> 33 bins
    const data = Array.from({ length: 48 }, (_, i) => Math.sin(i));
    expect(computeSpectrum(data).length).toBe(64 / 2 + 1);
  });

  it("windowing suppresses leakage: an off-bin tone is more concentrated than rectangular", () => {
    const n = 256;
    // 10.5 cycles: deliberately between bins 10 and 11 to provoke leakage.
    const data = Array.from({ length: n }, (_, i) => Math.cos((2 * Math.PI * 10.5 * i) / n));
    const mag = computeSpectrum(data);
    // Energy well away from the tone (bin 40+) should be strongly attenuated;
    // with a rectangular window the 1/f leakage skirt would keep these bins high.
    const peak = Math.max(mag[10]!, mag[11]!);
    let farMax = -Infinity;
    for (let i = 40; i < mag.length; i++) farMax = Math.max(farMax, mag[i]!);
    expect(peak - farMax).toBeGreaterThan(40); // >40 dB down far from the tone
  });
});
