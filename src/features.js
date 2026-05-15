/**
 * features.js
 * Browser-side audio feature extraction using Web Audio API.
 * Extracts MFCCs, chroma, spectral centroid/bandwidth/rolloff,
 * zero-crossing rate, onset strength, and RMS energy — matching
 * the Python process_birds.py pipeline output schema.
 */

// ── Constants ─────────────────────────────────────
const N_MFCC      = 13;
const N_CHROMA    = 12;
const N_FFT       = 2048;
const HOP_LENGTH  = 512;

// ── Mel filterbank ────────────────────────────────
function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1); }

function buildMelFilterbank(nFilters, nFft, sampleRate) {
  const nBins = nFft / 2 + 1;
  const melMin = hzToMel(0);
  const melMax = hzToMel(sampleRate / 2);
  const melPoints = [];
  for (let i = 0; i <= nFilters + 1; i++) {
    melPoints.push(melToHz(melMin + (i / (nFilters + 1)) * (melMax - melMin)));
  }
  const binPoints = melPoints.map(hz => Math.floor((nFft + 1) * hz / sampleRate));
  const fb = [];
  for (let m = 1; m <= nFilters; m++) {
    const row = new Float32Array(nBins);
    for (let k = 0; k < nBins; k++) {
      if (k >= binPoints[m - 1] && k <= binPoints[m]) {
        row[k] = (k - binPoints[m - 1]) / (binPoints[m] - binPoints[m - 1] + 1e-10);
      } else if (k >= binPoints[m] && k <= binPoints[m + 1]) {
        row[k] = (binPoints[m + 1] - k) / (binPoints[m + 1] - binPoints[m] + 1e-10);
      }
    }
    fb.push(row);
  }
  return fb;
}

// ── DCT-II for MFCCs ──────────────────────────────
function dct2(input) {
  const N = input.length;
  const out = new Float32Array(N_MFCC);
  for (let k = 0; k < N_MFCC; k++) {
    let s = 0;
    for (let n = 0; n < N; n++) {
      s += input[n] * Math.cos((Math.PI / N) * (n + 0.5) * k);
    }
    out[k] = s;
  }
  return out;
}

// ── Chroma ────────────────────────────────────────
function chromaFromMag(mag, sampleRate) {
  const nBins = mag.length;
  const chroma = new Float32Array(N_CHROMA);
  for (let k = 1; k < nBins; k++) {
    const hz = k * sampleRate / N_FFT;
    if (hz < 27.5 || hz > 4186) continue;
    const midi = 12 * Math.log2(hz / 440) + 69;
    const pc = ((Math.round(midi) % 12) + 12) % 12;
    chroma[pc] += mag[k];
  }
  const sum = chroma.reduce((a, b) => a + b, 0) + 1e-10;
  return chroma.map(v => v / sum);
}

// ── Spectral features ─────────────────────────────
function spectralCentroid(mag, sampleRate) {
  let num = 0, den = 0;
  for (let k = 0; k < mag.length; k++) {
    const hz = k * sampleRate / N_FFT;
    num += hz * mag[k]; den += mag[k];
  }
  return den > 1e-10 ? num / den : 0;
}

function spectralBandwidth(mag, sampleRate, centroid) {
  let num = 0, den = 0;
  for (let k = 0; k < mag.length; k++) {
    const hz = k * sampleRate / N_FFT;
    num += Math.pow(hz - centroid, 2) * mag[k]; den += mag[k];
  }
  return den > 1e-10 ? Math.sqrt(num / den) : 0;
}

function spectralRolloff(mag, sampleRate, rollPct = 0.85) {
  const total = mag.reduce((a, b) => a + b, 0);
  const thresh = total * rollPct;
  let cum = 0;
  for (let k = 0; k < mag.length; k++) {
    cum += mag[k];
    if (cum >= thresh) return k * sampleRate / N_FFT;
  }
  return sampleRate / 2;
}

function onsetStrength(prevMag, mag) {
  let flux = 0;
  for (let k = 0; k < mag.length; k++) {
    const d = mag[k] - (prevMag ? prevMag[k] : 0);
    if (d > 0) flux += d;
  }
  return flux;
}

function zcr(frame) {
  let crossings = 0;
  for (let i = 1; i < frame.length; i++) {
    if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) crossings++;
  }
  return crossings / frame.length;
}

function rms(frame) {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  return Math.sqrt(sum / frame.length);
}

function makeHann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
  return w;
}

function magnitudeSpectrum(frame, hann) {
  const N = N_FFT;
  const nBins = N / 2 + 1;
  const mag = new Float32Array(nBins);
  const windowed = new Float32Array(N);
  for (let i = 0; i < Math.min(frame.length, N); i++) windowed[i] = frame[i] * hann[i];
  for (let k = 0; k < nBins; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += windowed[n] * Math.cos(angle);
      im -= windowed[n] * Math.sin(angle);
    }
    mag[k] = Math.sqrt(re * re + im * im);
  }
  return mag;
}

async function magnitudeSpectraFast(pcm, sampleRate, onProgress) {
  const N = N_FFT;
  const hop = HOP_LENGTH;
  const nFrames = Math.floor((pcm.length - N) / hop) + 1;
  const hann = makeHann(N);
  const allMag = [];
  const allZcr = [];
  const allRms = [];

  for (let f = 0; f < nFrames; f++) {
    const start = f * hop;
    const frame = pcm.slice(start, start + N);
    const mag = magnitudeSpectrum(frame, hann);
    allMag.push(mag);
    allZcr.push(zcr(frame));
    allRms.push(rms(frame));
    if (f % 50 === 0) onProgress(f / nFrames * 0.7);
    if (f % 100 === 0) await new Promise(r => setTimeout(r, 0));
  }
  return { allMag, allZcr, allRms, nFrames };
}

function normalise(arr) {
  let mn = Infinity, mx = -Infinity;
  for (const v of arr) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const range = mx - mn + 1e-10;
  return arr.map(v => (v - mn) / range);
}

/**
 * PCA via power iteration.
 * FIX: seed vector is deterministic (all-ones normalised) so the eigenvector
 * always converges to the same direction — no random noise baked into xyz.
 * Iterations raised from 80 → 200 for reliable convergence on 57-dim input.
 */
function pcaProject(matrix, nComponents = 3) {
  const nFrames = matrix.length;
  const nFeats  = matrix[0].length;

  // Centre
  const mean = new Float64Array(nFeats);
  for (const row of matrix) for (let j = 0; j < nFeats; j++) mean[j] += row[j];
  for (let j = 0; j < nFeats; j++) mean[j] /= nFrames;

  const centred = matrix.map(row => row.map((v, j) => v - mean[j]));

  // Covariance
  const cov = [];
  for (let i = 0; i < nFeats; i++) {
    cov.push(new Float64Array(nFeats));
    for (let j = 0; j < nFeats; j++) {
      let s = 0;
      for (const row of centred) s += row[i] * row[j];
      cov[i][j] = s / (nFrames - 1);
    }
  }

  // Power iteration — deterministic seed: uniform vector, then normalise
  const vecs = [];
  const deflated = cov.map(r => Array.from(r));
  for (let k = 0; k < nComponents; k++) {
    // Deterministic seed: all-ones normalised
    const seedVal = 1 / Math.sqrt(nFeats);
    let v = new Array(nFeats).fill(seedVal);

    for (let iter = 0; iter < 200; iter++) {
      const nv = new Array(nFeats).fill(0);
      for (let i = 0; i < nFeats; i++)
        for (let j = 0; j < nFeats; j++)
          nv[i] += deflated[i][j] * v[j];
      const norm = Math.sqrt(nv.reduce((s, x) => s + x * x, 0)) + 1e-10;
      v = nv.map(x => x / norm);
    }
    vecs.push(v);

    // Deflate
    const eigval = v.reduce((s, vi, i) => {
      let Av = 0;
      for (let j = 0; j < nFeats; j++) Av += deflated[i][j] * v[j];
      return s + vi * Av;
    }, 0);
    for (let i = 0; i < nFeats; i++)
      for (let j = 0; j < nFeats; j++)
        deflated[i][j] -= eigval * v[i] * v[j];
  }

  // Project
  const projected = centred.map(row =>
    vecs.map(vec => row.reduce((s, v, j) => s + v * vec[j], 0))
  );

  // Scale to [-1, 1] per axis
  for (let k = 0; k < nComponents; k++) {
    let mn = Infinity, mx = -Infinity;
    for (const row of projected) { if (row[k] < mn) mn = row[k]; if (row[k] > mx) mx = row[k]; }
    const range = mx - mn + 1e-10;
    for (const row of projected) row[k] = ((row[k] - mn) / range) * 2 - 1;
  }

  return projected;
}

/**
 * extractManifold(file, onProgress)
 *
 * FIX: Downsampling now uses linear interpolation over evenly-spaced time
 * positions instead of integer frame stepping. This produces a perfectly
 * uniform t[] array so fracNodeForTime's binary search never stutters.
 */
export async function extractManifold(file, onProgress = () => {}) {
  onProgress(0.02);

  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
  let decoded;
  try {
    decoded = await audioCtx.decodeAudioData(arrayBuffer);
  } finally {
    audioCtx.close();
  }

  const sampleRate = decoded.sampleRate;
  const pcm = decoded.numberOfChannels > 1
    ? (() => {
        const out = new Float32Array(decoded.length);
        for (let c = 0; c < decoded.numberOfChannels; c++) {
          const ch = decoded.getChannelData(c);
          for (let i = 0; i < out.length; i++) out[i] += ch[i];
        }
        for (let i = 0; i < out.length; i++) out[i] /= decoded.numberOfChannels;
        return out;
      })()
    : decoded.getChannelData(0);

  const duration_s = decoded.duration;
  onProgress(0.08);

  const N = N_FFT, hop = HOP_LENGTH;
  const nBins = N / 2 + 1;
  const melFB = buildMelFilterbank(40, N, sampleRate);

  const { allMag, allZcr, allRms, nFrames } = await magnitudeSpectraFast(pcm, sampleRate, onProgress);

  const featureMatrix = [];
  const energyArr     = [];
  const timeArr       = [];
  let prevMag = null;

  for (let f = 0; f < nFrames; f++) {
    const mag = allMag[f];
    const t   = (f * hop) / sampleRate;

    const melE = melFB.map(row => {
      let s = 0;
      for (let k = 0; k < nBins; k++) s += row[k] * mag[k];
      return Math.log(s + 1e-10);
    });

    const mfcc   = dct2(melE);
    const chroma = chromaFromMag(mag, sampleRate);
    const cent   = spectralCentroid(mag, sampleRate);
    const bw     = spectralBandwidth(mag, sampleRate, cent);
    const roll   = spectralRolloff(mag, sampleRate);
    const onset  = onsetStrength(prevMag, mag);
    const z      = allZcr[f];
    const e      = allRms[f];

    prevMag = mag;

    featureMatrix.push([
      ...Array.from(mfcc),
      ...Array.from(chroma),
      cent / (sampleRate / 2),
      bw   / (sampleRate / 2),
      roll / (sampleRate / 2),
      z,
      Math.min(onset / 10, 1),
    ]);
    energyArr.push(e);
    timeArr.push(t);

    if (f % 50 === 0) onProgress(0.70 + (f / nFrames) * 0.15);
  }

  onProgress(0.85);

  const normEnergy = normalise(energyArr);
  const xyz3d      = pcaProject(featureMatrix, 3);

  onProgress(0.97);

  // ── Downsample via linear interpolation over evenly-spaced time positions
  // This guarantees a perfectly uniform t[] so the binary search in
  // fracNodeForTime never encounters uneven gaps that cause stutter.
  const maxFrames   = 500;
  const totalFrames = Math.min(nFrames, maxFrames);
  const t_out   = [];
  const xyz_out = [];
  const e_out   = [];

  for (let s = 0; s < totalFrames; s++) {
    // Exact fractional position in the original frame array
    const frac = s / (totalFrames - 1) * (nFrames - 1);
    const lo   = Math.floor(frac);
    const hi   = Math.min(lo + 1, nFrames - 1);
    const w    = frac - lo; // interpolation weight

    const t_interp = timeArr[lo] * (1 - w) + timeArr[hi] * w;
    const e_interp = normEnergy[lo] * (1 - w) + normEnergy[hi] * w;
    const x_interp = xyz3d[lo][0] * (1 - w) + xyz3d[hi][0] * w;
    const y_interp = xyz3d[lo][1] * (1 - w) + xyz3d[hi][1] * w;
    const z_interp = xyz3d[lo][2] * (1 - w) + xyz3d[hi][2] * w;

    t_out.push(+t_interp.toFixed(4));
    xyz_out.push([+x_interp.toFixed(5), +y_interp.toFixed(5), +z_interp.toFixed(5)]);
    e_out.push(+e_interp.toFixed(4));
  }

  onProgress(1.0);

  return {
    species:    file.name.replace(/\.[^.]+$/, '').replace(/[_\s]+/g, '_'),
    t:          t_out,
    xyz:        xyz_out,
    energy:     e_out,
    duration_s: +duration_s.toFixed(3),
  };
}
