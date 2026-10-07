/** Audio Detector — AASIST integration for dashboard */
document.addEventListener('DOMContentLoaded', () => {
  const dropZone    = document.getElementById('audioDropZone');
  const fileInput   = document.getElementById('audioFileInput');
  const previewZone = document.getElementById('audioPreviewZone');
  const audioPlayer = document.getElementById('audioPlayer');
  const waveCanvas  = document.getElementById('audioWaveform');
  const resetBtn    = document.getElementById('audioReset');
  const statusChip  = document.getElementById('audioStatusChip');
  const statCards   = document.getElementById('audioStatCards');

  if (!dropZone || !fileInput) return;

  setupDropZone(dropZone, fileInput, handleFile);
  if (resetBtn) resetBtn.addEventListener('click', reset);

  async function handleFile(file) {
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    dropZone.style.display      = 'none';
    previewZone.style.display   = 'flex';
    if (resetBtn) resetBtn.style.display = 'inline-block';
    statusChip.textContent = 'Analyzing…';
    statusChip.style.color = 'var(--status-warn)';

    drawWaveform(file);

    if (statCards) statCards.style.display = 'none';

    // Show loading on main reset button too
    const mainReset = document.getElementById('mediaResetBtn');
    if (mainReset) mainReset.style.display = 'inline-block';

    try {
      const fd = new FormData();
      fd.append('file', file);
      const res  = await fetch(`${API_BASE}/api/detect/audio`, { method: 'POST', body: fd });
      const data = await res.json();

      const prob = data.fake_probability ?? 0.5;
      const modelUsed = data.model_used || 'classical_only';

      // Status chip
      statusChip.textContent = data.verdict || (prob < 0.35 ? 'Real' : prob < 0.65 ? 'Uncertain' : 'Fake');
      statusChip.style.color = prob < 0.35 ? 'var(--status-real)' : prob < 0.65 ? 'var(--status-warn)' : 'var(--status-fake)';

      // Stat cards
      if (statCards) statCards.style.display = 'grid';
      const det = data.details || {};

      // AASIST/Overall score card
      updateStatCard('acScore', 'acBadge', prob);

      // Spectral flatness
      const flatnessEl = document.getElementById('acFlatness');
      if (flatnessEl) {
        flatnessEl.textContent = det.spectral_flatness !== undefined
          ? (det.spectral_flatness * 100).toFixed(1) + '%' : '—';
      }

      // Pitch variability
      const pitchEl = document.getElementById('acPitch');
      if (pitchEl) {
        pitchEl.textContent = det.pitch_variability !== undefined
          ? (det.pitch_variability * 100).toFixed(1) + '%'
          : det.pitch_std !== undefined
            ? det.pitch_std.toFixed(1) + 'Hz' : '—';
      }

      // Duration
      const durEl = document.getElementById('acDuration');
      if (durEl) {
        durEl.textContent = det.duration_seconds ? `${det.duration_seconds}s` : '—';
      }

      // --- Update top metric cards ---
      rpSetVerdict(prob, modelUsed, true);

      // Faces card → show duration for audio mode
      const cardFaces = document.getElementById('cardFaces');
      if (cardFaces) cardFaces.textContent = det.duration_seconds ? `${det.duration_seconds}s` : '--';

      // Model card
      const cardEffNet = document.getElementById('cardEffNet');
      if (cardEffNet) {
        if (modelUsed === 'aasist') {
          cardEffNet.textContent = 'AASIST';
        } else {
          cardEffNet.textContent = 'Classical';
        }
      }

      // Classical CV → show classical score
      const cardClassical = document.getElementById('cardClassical');
      if (cardClassical) {
        cardClassical.textContent = det.classical_score !== undefined
          ? Math.round(det.classical_score * 100) + '%' : '--';
      }

      // Audio-specific justification
      rpSetAudioJustification(prob, modelUsed);

    } catch (err) {
      statusChip.textContent = 'Error';
      statusChip.style.color = 'var(--status-fake)';
      rpSetVerdict(0.5, 'error', true);

      const justText = document.getElementById('justificationText');
      if (justText) justText.textContent = 'Backend offline — start the server with: python -m uvicorn main:app --host 0.0.0.0 --port 8000';
    }
  }

  async function drawWaveform(file) {
    if (!waveCanvas) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const actx  = new AudioCtx();
      const buf   = await file.arrayBuffer();
      const audio = await actx.decodeAudioData(buf);
      const data  = audio.getChannelData(0);

      const dpr = window.devicePixelRatio || 1;
      waveCanvas.width  = waveCanvas.offsetWidth  * dpr;
      waveCanvas.height = waveCanvas.offsetHeight * dpr;
      const w = waveCanvas.width, h = waveCanvas.height;
      const ctx = waveCanvas.getContext('2d');
      const step = Math.ceil(data.length / w);
      const mid  = h / 2;

      ctx.clearRect(0, 0, w, h);

      // Background gradient
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, '#34d399');
      grad.addColorStop(0.5, '#CCFF00');
      grad.addColorStop(1, '#f87171');

      ctx.strokeStyle = grad;
      ctx.lineWidth   = 1.5;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        let mn = 1, mx = -1;
        for (let j = 0; j < step; j++) {
          const d = data[x * step + j] || 0;
          if (d < mn) mn = d; if (d > mx) mx = d;
        }
        ctx.moveTo(x, mid + mn * mid);
        ctx.lineTo(x, mid + mx * mid);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      actx.close();
    } catch { /* waveform optional */ }
  }

  function reset() {
    dropZone.style.display   = 'flex';
    previewZone.style.display = 'none';
    if (resetBtn) resetBtn.style.display = 'none';
    if (statCards) statCards.style.display = 'none';
    statusChip.textContent = 'Ready';
    statusChip.style.color = '';
    audioPlayer.src = '';
    fileInput.value = '';
    if (waveCanvas) {
      const ctx = waveCanvas.getContext('2d');
      ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
    }
    resetApp();
  }
});
