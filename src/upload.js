/**
 * upload.js
 * Dropzone UI + file-picker wired to features.js extractManifold().
 * Calls callbacks: onManifold(manifold, file), onError(err), onProgress(ratio).
 */

import { extractManifold } from './features.js';

export function initUpload({ onManifold, onError, onProgress }) {
  const dropzone   = document.getElementById('dropzone');
  const fileInput  = document.getElementById('audioFileInput');
  const statusEl   = document.getElementById('upload-status');
  const progressBg = document.getElementById('upload-progress-bg');
  const progressBar= document.getElementById('upload-progress-bar');

  if (!dropzone || !fileInput) return;

  // ── Status helpers
  function setStatus(msg, isError = false) {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.style.color = isError
      ? 'rgba(255, 80, 80, 0.75)'
      : 'rgba(218, 218, 218, 0.35)';
  }

  function setProgress(ratio) {
    if (!progressBar) return;
    if (ratio <= 0) {
      progressBar.style.opacity = '0';
      progressBar.style.width   = '0%';
      return;
    }
    progressBar.style.opacity = '1';
    progressBar.style.width   = `${Math.round(ratio * 100)}%`;
    if (ratio >= 1) {
      setTimeout(() => {
        progressBar.style.opacity = '0';
        progressBar.style.width   = '0%';
      }, 900);
    }
  }

  // ── Process a File object
  async function processFile(file) {
    if (!file || !file.type.startsWith('audio/')) {
      setStatus('Not an audio file.', true);
      if (onError) onError(new Error('Not audio'));
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setStatus('File too large (max 50 MB).', true);
      if (onError) onError(new Error('File too large'));
      return;
    }

    dropzone.classList.remove('drag-over');
    setStatus('Decoding audio…');
    setProgress(0.02);

    try {
      const manifold = await extractManifold(file, (ratio) => {
        setProgress(ratio);
        if (ratio < 0.75) setStatus('Extracting features…');
        else if (ratio < 0.9) setStatus('Running PCA…');
        else setStatus('Building manifold…');
        if (onProgress) onProgress(ratio);
      });

      setProgress(1.0);
      setStatus(`Done — ${manifold.t.length} frames`);

      if (onManifold) onManifold(manifold, file);
    } catch (err) {
      console.error('[upload]', err);
      setStatus('Failed: ' + (err.message || 'unknown error'), true);
      setProgress(0);
      if (onError) onError(err);
    }
  }

  // ── Click to open file picker
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) processFile(fileInput.files[0]);
    fileInput.value = '';
  });

  // ── Drag-and-drop
  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
  });
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  });
}
