/** Radix-2 FFT and magnitude-spectrum helpers for the spectrum display. */

/** Radix-2 FFT (in-place, iterative). Expects arrays of length 2^n. */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      [re[i]!, re[j]!] = [re[j]!, re[i]!];
      [im[i]!, im[j]!] = [im[j]!, im[i]!];
    }
  }
  // FFT butterfly
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < half; j++) {
        const tRe = curRe * re[i + j + half]! - curIm * im[i + j + half]!;
        const tIm = curRe * im[i + j + half]! + curIm * re[i + j + half]!;
        re[i + j + half] = re[i + j]! - tRe;
        im[i + j + half] = im[i + j]! - tIm;
        re[i + j] = re[i + j]! + tRe;
        im[i + j] = im[i + j]! + tIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/** Hann window of a given length (w[0]=0 .. w[N-1]=0). */
export function hannWindow(length: number): Float64Array {
  const w = new Float64Array(length);
  if (length === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < length; i++) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
  }
  return w;
}

/**
 * Single-sided magnitude spectrum in dB.
 *
 * Applies a Hann window over the real samples before the (zero-padded, radix-2)
 * FFT to suppress the spectral leakage a rectangular window produces when the
 * frame is not periodic. Normalizes by the window's coherent gain (sum of its
 * coefficients) so windowing does not shift the displayed levels -- a full-scale
 * bin-aligned sinusoid still reads about -6 dB, as with a rectangular window.
 * Returns `n/2 + 1` bins, i.e. DC through Nyquist inclusive, so the last bin
 * maps exactly to sampleRate/2 and the frequency axis label is correct even for
 * zero-padded / short frames.
 */
export function computeSpectrum(data: number[]): Float64Array {
  const len = data.length;
  // Zero-pad to the next power of 2 for the radix-2 FFT.
  let n = 1;
  while (n < len) n <<= 1;
  const re = new Float64Array(n);
  const im = new Float64Array(n);

  const w = hannWindow(len);
  let windowSum = 0;
  for (let i = 0; i < len && i < n; i++) {
    re[i] = data[i]! * w[i]!;
    windowSum += w[i]!;
  }
  const norm = windowSum > 0 ? windowSum : n;

  fft(re, im);

  const half = n >> 1;
  const mag = new Float64Array(half + 1); // include the Nyquist bin (index n/2)
  for (let i = 0; i <= half; i++) {
    const m = Math.sqrt(re[i]! * re[i]! + im[i]! * im[i]!) / norm;
    mag[i] = 20 * Math.log10(Math.max(m, 1e-10));
  }
  return mag;
}
