// upload.js — drag-and-drop / file picker UI that calls extractManifold()
// and returns the manifold to a callback.

import { extractManifold } from './features.js';

export function initUpload({ onManifold, onError, onProgress }) {
  const panel = document.getElementById('upload-panel');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('audioFileInput');
  const statusEl = document.getElementById('upload-status');
  const progressBar = document.getElementById('upload-progress-bar');

  if (!panel) return;

  function showPanel() { panel.style.display = 'block'; }

  document.getElementById('overlay')?.addEventListener('click', () => {
    setTimeout(showPanel, 1000);
  }, { once: true });

  function setStatus(msg, isError = false) {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.style.color = isError ? 'rgba(255,100,80,0.9)' : 'rgba(255,255,255,0.5)';
  }

  function setProgress(pct) {
    if (!progressBar) return;
    progressBar.style.width = pct + '%';
    progressBar.style.opacity = pct > 0 && pct < 100 ? '1' : '0';
  }

  async function processFile(file) {
    if (!file.type.startsWith('audio/') && !/\.(mp3|wav|ogg|flac|aac|m4a)$/i.test(file.name)) {
      setStatus('Please upload an audio file (.mp3, .wav, .ogg, .flac)', true);
      return;
    }

    const speciesName = file.name.replace(/\.[^.]+$/, '').toLowerCase().replace(/\s+/g, '_');
    setStatus('Reading file…');
    setProgress(10);
    onProgress?.('reading');

    try {
      const arrayBuffer = await file.arrayBuffer();
      setStatus('Decoding audio…');
      setProgress(25);

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      audioCtx.close();

      setStatus('Extracting features (MFCC, Chroma, PCA)…');
      setProgress(40);
      onProgress?.('extracting');

      await new Promise(r => setTimeout(r, 20));

      const manifold = await extractManifold(audioBuffer, speciesName);

      setProgress(100);
      setStatus(`Loaded: ${speciesName.replace(/_/g, ' ')}`);
      onProgress?.('done');

      onManifold?.(manifold, file);
    } catch (err) {
      console.error(err);
      setStatus('Failed: ' + err.message, true);
      setProgress(0);
      onError?.(err);
    }
  }

  fileInput?.addEventListener('change', e => {
    if (e.target.files[0]) processFile(e.target.files[0]);
  });

  dropzone?.addEventListener('click', () => fileInput?.click());

  dropzone?.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
  dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone?.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  });
}
