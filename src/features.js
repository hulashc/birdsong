// features.js — browser-side audio feature extraction matching process_birds.py
// Uses Web Audio API + manual DSP. No external deps.

const N_MFCC = 40;
const N_FFT = 2048;
const HOP_LENGTH = 512;
const N_CHROMA = 12;
const SR = 22050;
const DURATION = 10.0;

// ── DCT-II (for MFCC) ─────────────────────────────────────────────────────────
function dct2(arr) {
  const N = arr.length;
  const out = new Float32Array(N);
  for (let k = 0; k < N; k++) {
    let s = 0;
    for (let n = 0; n < N; n++) s += arr[n] * Math.cos((Math.PI / N) * (n + 0.5) * k);
    out[k] = s;
  }
  return out;
}

// ── Mel filterbank ────────────────────────────────────────────────────────────
function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1); }

function melFilterbank(nFilters, nFft, sr) {
  const fMin = 0, fMax = sr / 2;
  const melMin = hzToMel(fMin), melMax = hzToMel(fMax);
  const melPoints = new Float32Array(nFilters + 2);
  for (let i = 0; i < nFilters + 2; i++)
    melPoints[i] = melToHz(melMin + (i / (nFilters + 1)) * (melMax - melMin));

  const freqBins = Math.floor(nFft / 2) + 1;
  const filters = [];
  for (let m = 1; m <= nFilters; m++) {
    const f = new Float32Array(freqBins);
    const fLow = melPoints[m - 1], fCenter = melPoints[m], fHigh = melPoints[m + 1];
    for (let k = 0; k < freqBins; k++) {
      const freq = (k / (freqBins - 1)) * (sr / 2);
      if (freq >= fLow && freq <= fCenter) f[k] = (freq - fLow) / (fCenter - fLow + 1e-10);
      else if (freq > fCenter && freq <= fHigh) f[k] = (fHigh - freq) / (fHigh - fCenter + 1e-10);
    }
    filters.push(f);
  }
  return filters;
}

// ── Chroma filterbank ─────────────────────────────────────────────────────────
function chromaFilterbank(nChroma, nFft, sr) {
  const freqBins = Math.floor(nFft / 2) + 1;
  const filters = Array.from({ length: nChroma }, () => new Float32Array(freqBins));
  for (let k = 1; k < freqBins; k++) {
    const freq = (k / (freqBins - 1)) * (sr / 2);
    if (freq <= 0) continue;
    const pitchClass = ((12 * Math.log2(freq / 440)) % 12 + 12) % 12;
    const ci = Math.floor(pitchClass);
    const frac = pitchClass - ci;
    filters[ci % nChroma][k] += 1 - frac;
    filters[(ci + 1) % nChroma][k] += frac;
  }
  return filters;
}

// ── Hanning window ────────────────────────────────────────────────────────────
function hanningWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

// ── Simple FFT (radix-2 Cooley-Tukey) ────────────────────────────────────────
function fft(re, im) {
  const n = re.length;
  if (n <= 1) return;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI) / len;
    const wRe = Math.cos(ang), wIm = -Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
        const vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;
        re[i + j] = uRe + vRe; im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe; im[i + j + len / 2] = uIm - vIm;
        const newRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe; curRe = newRe;
      }
    }
  }
}

// ── Power spectrum for one frame ──────────────────────────────────────────────
function powerSpectrum(frame, window) {
  const n = N_FFT;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < frame.length && i < n; i++) re[i] = frame[i] * window[i];
  fft(re, im);
  const bins = Math.floor(n / 2) + 1;
  const ps = new Float32Array(bins);
  for (let i = 0; i < bins; i++) ps[i] = re[i] * re[i] + im[i] * im[i];
  return ps;
}

// ── StandardScaler (fit on all frames, transform) ────────────────────────────
function standardScale(matrix) {
  const nFrames = matrix.length, nFeatures = matrix[0].length;
  const mean = new Float32Array(nFeatures);
  const std = new Float32Array(nFeatures);

  for (const row of matrix) for (let j = 0; j < nFeatures; j++) mean[j] += row[j];
  for (let j = 0; j < nFeatures; j++) mean[j] /= nFrames;

  for (const row of matrix) for (let j = 0; j < nFeatures; j++) std[j] += (row[j] - mean[j]) ** 2;
  for (let j = 0; j < nFeatures; j++) std[j] = Math.sqrt(std[j] / nFrames) || 1;

  return matrix.map(row => {
    const r = new Float32Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) r[j] = (row[j] - mean[j]) / std[j];
    return r;
  });
}

// ── PCA (3 components via power iteration) ────────────────────────────────────
function pca3(matrix) {
  const nFrames = matrix.length, nFeatures = matrix[0].length;

  const cov = [];
  for (let i = 0; i < nFeatures; i++) {
    cov.push(new Float32Array(nFeatures));
    for (let j = 0; j < nFeatures; j++) {
      let s = 0;
      for (let k = 0; k < nFrames; k++) s += matrix[k][i] * matrix[k][j];
      cov[i][j] = s / nFrames;
    }
  }

  function powerIter(C, deflated) {
    let v = new Float32Array(nFeatures).fill(0);
    v[0] = 1;
    for (let iter = 0; iter < 200; iter++) {
      const Cv = new Float32Array(nFeatures);
      for (let i = 0; i < nFeatures; i++)
        for (let j = 0; j < nFeatures; j++) Cv[i] += C[i][j] * v[j];
      for (const prev of deflated) {
        let dot = 0;
        for (let i = 0; i < nFeatures; i++) dot += Cv[i] * prev[i];
        for (let i = 0; i < nFeatures; i++) Cv[i] -= dot * prev[i];
      }
      let norm = 0;
      for (let i = 0; i < nFeatures; i++) norm += Cv[i] * Cv[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < nFeatures; i++) v[i] = Cv[i] / norm;
    }
    return v;
  }

  const evec1 = powerIter(cov, []);
  const evec2 = powerIter(cov, [evec1]);
  const evec3 = powerIter(cov, [evec1, evec2]);

  return matrix.map(row => {
    let p1 = 0, p2 = 0, p3 = 0;
    for (let j = 0; j < nFeatures; j++) {
      p1 += row[j] * evec1[j];
      p2 += row[j] * evec2[j];
      p3 += row[j] * evec3[j];
    }
    return [p1, p2, p3];
  });
}

// ── Main extraction function ──────────────────────────────────────────────────
export async function extractManifold(audioBuffer, speciesName = 'upload') {
  let samples;
  if (audioBuffer.sampleRate !== SR) {
    const ctx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * SR), SR);
    const src = ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(ctx.destination);
    src.start(0);
    const rendered = await ctx.startRendering();
    samples = rendered.getChannelData(0);
  } else {
    samples = audioBuffer.getChannelData(0);
  }

  const maxSamples = Math.floor(SR * DURATION);
  if (samples.length > maxSamples) samples = samples.slice(0, maxSamples);
  const duration_s = samples.length / SR;

  const window = hanningWindow(N_FFT);
  const melFilters = melFilterbank(128, N_FFT, SR);
  const chromaFilters = chromaFilterbank(N_CHROMA, N_FFT, SR);
  const freqBins = Math.floor(N_FFT / 2) + 1;

  const frameCount = Math.floor((samples.length - N_FFT) / HOP_LENGTH) + 1;
  if (frameCount < 3) throw new Error('Audio too short — need at least 3 frames');

  const featureMatrix = [];
  const energyRaw = [];
  const times = [];

  let prevMelSpec = new Float32Array(128);

  for (let f = 0; f < frameCount; f++) {
    const start = f * HOP_LENGTH;
    const frame = samples.slice(start, start + N_FFT);
    const ps = powerSpectrum(frame, window);

    const melSpec = new Float32Array(128);
    for (let m = 0; m < 128; m++) {
      for (let k = 0; k < freqBins; k++) melSpec[m] += ps[k] * melFilters[m][k];
      melSpec[m] = Math.log(melSpec[m] + 1e-10);
    }

    const mfcc = dct2(melSpec).slice(0, N_MFCC);

    const chroma = new Float32Array(N_CHROMA);
    for (let c = 0; c < N_CHROMA; c++)
      for (let k = 0; k < freqBins; k++) chroma[c] += ps[k] * chromaFilters[c][k];
    const chromaMax = Math.max(...chroma) || 1;
    for (let c = 0; c < N_CHROMA; c++) chroma[c] /= chromaMax;

    let num = 0, den = 0;
    for (let k = 0; k < freqBins; k++) {
      const freq = (k / (freqBins - 1)) * (SR / 2);
      num += freq * ps[k]; den += ps[k];
    }
    const centroid = den > 0 ? num / den : 0;

    let bwNum = 0;
    for (let k = 0; k < freqBins; k++) {
      const freq = (k / (freqBins - 1)) * (SR / 2);
      bwNum += ps[k] * (freq - centroid) ** 2;
    }
    const bandwidth = den > 0 ? Math.sqrt(bwNum / den) : 0;

    const totalEnergy = ps.reduce((a, b) => a + b, 0);
    let cumEnergy = 0, rolloffBin = freqBins - 1;
    for (let k = 0; k < freqBins; k++) {
      cumEnergy += ps[k];
      if (cumEnergy >= 0.85 * totalEnergy) { rolloffBin = k; break; }
    }
    const rolloff = (rolloffBin / (freqBins - 1)) * (SR / 2);

    let zcr = 0;
    for (let i = 1; i < frame.length; i++)
      if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) zcr++;
    zcr /= frame.length;

    let onset = 0;
    for (let m = 0; m < 128; m++) onset += Math.max(0, melSpec[m] - prevMelSpec[m]);
    prevMelSpec = melSpec.slice();

    let rms = 0;
    for (let i = 0; i < frame.length; i++) rms += frame[i] ** 2;
    rms = Math.sqrt(rms / frame.length);

    const feat = new Float32Array(57);
    feat.set(mfcc, 0);
    feat.set(chroma, 40);
    feat[52] = centroid;
    feat[53] = bandwidth;
    feat[54] = rolloff;
    feat[55] = zcr;
    feat[56] = onset;

    featureMatrix.push(feat);
    energyRaw.push(rms);
    times.push((start + N_FFT / 2) / SR);
  }

  const scaled = standardScale(featureMatrix);
  const xyz = pca3(scaled);

  let maxAbs = 0;
  for (const [x, y, z] of xyz) maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z));
  if (maxAbs > 0) for (const p of xyz) { p[0] /= maxAbs; p[1] /= maxAbs; p[2] /= maxAbs; }

  const sorted = [...energyRaw].sort((a, b) => a - b);
  const p95 = sorted[Math.floor(sorted.length * 0.95)] || 1e-9;
  const energy = energyRaw.map(e => Math.min(1, e / (p95 + 1e-9)));

  return {
    sr: SR,
    hop_length: HOP_LENGTH,
    duration_s,
    features_used: ['mfcc_40', 'chroma_12', 'centroid', 'bandwidth', 'rolloff', 'zcr', 'onset'],
    t: times,
    xyz,
    energy,
    species: speciesName,
  };
}
