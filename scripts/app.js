/**
 * DeepScan AI — App Controller (v3: Image + Video + Audio + Webcam)
 */

const API_BASE = window.location.origin;
const WS_BASE = window.location.protocol === 'https:' ? `wss://${window.location.host}` : `ws://${window.location.host}`;
const ROUTE_HEALTH = '/api/health';
const ROUTE_IMAGE = '/api/detect/image';
const ROUTE_VIDEO = '/api/detect/video';
const ROUTE_AUDIO = '/api/detect/audio';

document.addEventListener('DOMContentLoaded', () => {
  // ─── Theme Management ───
  const savedTheme = localStorage.getItem('deepscan-theme') || 'light';
  applyTheme(savedTheme);

  const themeToggleBtns = [
    document.getElementById('floatingThemeBtn'),
    document.getElementById('themeToggleBtn')
  ];

  themeToggleBtns.forEach(btn => {
    if (!btn) return;
    btn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(current);
      localStorage.setItem('deepscan-theme', current);
    });
  });

  function applyTheme(theme) {
    if (theme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    // Re-draw gauge track color
    drawGauge(window.lastFakeProb ?? 0, window.lastDisplayPct ?? 0);
  }

  // ─── View Controller (About Landing vs Dashboard) ───
  const aboutSection = document.getElementById('aboutSection');
  const dashboardSection = document.getElementById('dashboardSection');
  const pillLinks = document.querySelectorAll('.pill-links .pill-link');
  const pillAbout = document.getElementById('pillLinkAbout');
  const pillEngine = document.getElementById('pillLinkEngine');
  const pillModalities = document.getElementById('pillLinkModalities');
  const pillDashboard = document.getElementById('pillLinkDashboard');

  function setPillActive(targetBtn) {
    pillLinks.forEach(b => b.classList.remove('active'));
    if (targetBtn) targetBtn.classList.add('active');
  }

  window.showDashboard = function(targetTab = null) {
    if (aboutSection) aboutSection.style.display = 'none';
    if (dashboardSection) dashboardSection.style.display = 'block';
    setPillActive(pillDashboard);

    if (targetTab) {
      const tabBtn = document.querySelector(`.nav-item[data-tab="${targetTab}"]`);
      if (tabBtn) tabBtn.click();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  window.showAbout = function(targetAnchor = null) {
    if (dashboardSection) dashboardSection.style.display = 'none';
    if (aboutSection) aboutSection.style.display = 'block';

    if (targetAnchor) {
      const targetEl = document.querySelector(targetAnchor);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth' });
        if (targetAnchor.includes('engine')) setPillActive(pillEngine);
        else if (targetAnchor.includes('modalities')) setPillActive(pillModalities);
        return;
      }
    }

    setPillActive(pillAbout);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // ─── Navigation Buttons ───
  // Get Started buttons -> Show Dashboard
  const getStartedBtns = [
    document.getElementById('floatingGetStartedBtn'),
    document.getElementById('heroGetStartedBtn'),
    document.getElementById('ctaGetStartedBtn')
  ];
  getStartedBtns.forEach(btn => {
    if (btn) btn.addEventListener('click', () => showDashboard());
  });

  // Floating Nav Pill links
  const brandPill = document.getElementById('brandPillLink');
  if (brandPill) {
    brandPill.addEventListener('click', (e) => {
      e.preventDefault();
      showAbout();
    });
  }
  if (pillAbout) pillAbout.addEventListener('click', () => showAbout());
  if (pillEngine) pillEngine.addEventListener('click', () => showAbout('#engineSection'));
  if (pillModalities) pillModalities.addEventListener('click', () => showAbout('#modalitiesSection'));
  if (pillDashboard) pillDashboard.addEventListener('click', () => showDashboard());

  // Hero Explore button -> scroll to engine
  const heroExploreBtn = document.getElementById('heroExploreBtn');
  if (heroExploreBtn) {
    heroExploreBtn.addEventListener('click', () => showAbout('#engineSection'));
  }

  // Modality launch buttons (Inside About page cards)
  const launchBtns = document.querySelectorAll('.btn-launch-modality[data-launch]');
  launchBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.launch;
      showDashboard(mode);
    });
  });

  // Sidebar back to About
  const navBackToAbout = document.getElementById('navBackToAbout');
  if (navBackToAbout) {
    navBackToAbout.addEventListener('click', () => showAbout());
  }

  // Top header button in dashboard -> back to About
  const headerAboutBtn = document.getElementById('headerAboutBtn');
  if (headerAboutBtn) {
    headerAboutBtn.addEventListener('click', () => showAbout());
  }

  // ─── Tab Navigation (Sidebar) ───
  const navBtns = document.querySelectorAll('.nav-item[data-tab]');
  const imgUpload = document.getElementById('imageUploadContainer');
  const vidUpload = document.getElementById('videoUploadContainer');
  const audUpload = document.getElementById('audioUploadContainer');
  const wcContainer = document.getElementById('webcamContainer');

  const containers = { image: imgUpload, video: vidUpload, audio: audUpload, webcam: wcContainer };

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const layoutEl = document.querySelector('.layout');
      if (layoutEl) {
        layoutEl.classList.remove('verification-mode');
        layoutEl.classList.remove('layout-full-screen');
      }

      // Reset UI state
      resetApp();

      // Show/hide containers
      Object.keys(containers).forEach(key => {
        if (containers[key]) {
          containers[key].style.display = key === btn.dataset.tab ? 'flex' : 'none';
        }
      });

      // Update metric card labels based on mode
      updateMetricLabels(btn.dataset.tab);
    });
  });

  // Reset Button
  const resetBtn = document.getElementById('mediaResetBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', resetApp);
  }

  // Health Check Polling
  checkHealth();
  setInterval(checkHealth, 15000);

  // Check URL hash routing on initial page load
  const currentHash = window.location.hash.toLowerCase();
  if (currentHash === '#dashboard' || currentHash === '#image') {
    showDashboard('image');
  } else if (currentHash === '#video') {
    showDashboard('video');
  } else if (currentHash === '#audio') {
    showDashboard('audio');
  } else if (currentHash === '#webcam') {
    showDashboard('webcam');
  } else if (currentHash.includes('engine')) {
    showAbout('#engineSection');
  } else if (currentHash.includes('modalities')) {
    showAbout('#modalitiesSection');
  }
});

// ─── Update metric card labels per mode ───
function updateMetricLabels(mode) {
  const cardFacesTitle = document.querySelector('#cardFaces')?.closest('.metric-card')?.querySelector('.metric-title');
  const cardModelTitle = document.querySelector('#cardEffNet')?.closest('.metric-card')?.querySelector('.metric-title');
  const cardClassicalTitle = document.querySelector('#cardClassical')?.closest('.metric-card')?.querySelector('.metric-title');

  const cardFacesTrend = document.querySelector('#cardFaces')?.closest('.metric-card')?.querySelector('.metric-trend');
  const cardClassicalTrend = document.querySelector('#cardClassical')?.closest('.metric-card')?.querySelector('.metric-trend');

  if (mode === 'audio') {
    if (cardFacesTitle) cardFacesTitle.textContent = 'Duration';
    if (cardModelTitle) cardModelTitle.textContent = 'Active AI Model';
    if (cardClassicalTitle) cardClassicalTitle.textContent = 'Classical Analysis';
    if (cardFacesTrend) cardFacesTrend.innerHTML = '<span class="trend-icon green">⏱</span> Total Length';
    if (cardClassicalTrend) cardClassicalTrend.innerHTML = '<span class="trend-icon green">↑</span> Acoustic Spectral';
  } else if (mode === 'webcam') {
    if (cardFacesTitle) cardFacesTitle.textContent = 'Faces Detected';
    if (cardModelTitle) cardModelTitle.textContent = 'Active AI Model';
    if (cardClassicalTitle) cardClassicalTitle.textContent = 'Moiré Analysis';
    if (cardFacesTrend) cardFacesTrend.innerHTML = '<span class="trend-icon green">↑</span> MTCNN Algorithm';
    if (cardClassicalTrend) cardClassicalTrend.innerHTML = '<span class="trend-icon green">↑</span> Moiré Pattern Scan';
  } else {
    if (cardFacesTitle) cardFacesTitle.textContent = 'Faces Detected';
    if (cardModelTitle) cardModelTitle.textContent = 'Active AI Model';
    if (cardClassicalTitle) cardClassicalTitle.textContent = 'Classical CV';
    if (cardFacesTrend) cardFacesTrend.innerHTML = '<span class="trend-icon green">↑</span> MTCNN Algorithm';
    if (cardClassicalTrend) cardClassicalTrend.innerHTML = '<span class="trend-icon green">↑</span> Error Level Analysis';
  }
}

// ─── Reset App State ───
function resetApp() {
  // Metrics Cards
  const cards = ['cardAiScore', 'cardVerdictText', 'cardFaces', 'cardEffNet', 'cardClassical', 'gaugeScoreText', 'gaugeLabelText'];
  cards.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '--';
  });
  const primaryCardTitle = document.querySelector('.metric-card.card-dark .metric-title');
  if (primaryCardTitle) primaryCardTitle.textContent = 'Overall Fake Prob.';
  const trendIconProb = document.getElementById('trendIconProb');
  if (trendIconProb) {
    trendIconProb.textContent = '↑';
    trendIconProb.className = 'trend-icon';
    trendIconProb.style.color = '';
  }

  // Dropzones
  const dzList = ['imageDropZone', 'videoDropZone', 'audioDropZone'];
  const pzList = ['imagePreviewZone', 'videoPreviewZone', 'audioPreviewZone'];
  dzList.forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'flex'; });
  pzList.forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'none'; });

  const resetBtn = document.getElementById('mediaResetBtn');
  if (resetBtn) resetBtn.style.display = 'none';

  // Audio reset
  const audioReset = document.getElementById('audioReset');
  if (audioReset) audioReset.style.display = 'none';
  const audioPlayer = document.getElementById('audioPlayer');
  if (audioPlayer) audioPlayer.src = '';
  const audioStatCards = document.getElementById('audioStatCards');
  if (audioStatCards) audioStatCards.style.display = 'none';
  const audioStatusChip = document.getElementById('audioStatusChip');
  if (audioStatusChip) { audioStatusChip.textContent = 'Ready'; audioStatusChip.style.color = ''; }

  // Gauge
  drawGauge(0);

  // Conf Bar
  const fill = document.getElementById('rpConfFill');
  const thumb = document.getElementById('rpConfThumb');
  if (fill) { fill.style.width = '0%'; fill.style.background = 'var(--primary-color)'; }
  if (thumb) { thumb.style.left = '0%'; }

  // Justification
  const justBox = document.getElementById('justificationBox');
  const justText = document.getElementById('justificationText');
  if (justBox) { justBox.style.display = 'block'; justBox.style.borderLeftColor = 'var(--primary-color)'; }
  if (justText) { justText.textContent = "Upload media to see the model's reasoning here."; }

  // Canvas Clears
  const faceCanvas = document.getElementById('imageFaceCanvas');
  if (faceCanvas) {
    const ctx = faceCanvas.getContext('2d');
    ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
  }

  const waveCanvas = document.getElementById('audioWaveform');
  if (waveCanvas) {
    const ctx = waveCanvas.getContext('2d');
    ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
  }
}

// ─── Verdict gauge ───
function drawGauge(fakeProb, displayPct = null) {
  const canvas = document.getElementById('verdictGauge');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 15;
  const R = 100;
  const startAngle = Math.PI;
  const endAngle = 0;

  const style = getComputedStyle(document.documentElement);
  let trackColor = style.getPropertyValue('--gauge-track').trim() || '#f1f5f9';

  // Track (Thick light gray or dark)
  ctx.beginPath();
  ctx.arc(cx, cy, R, startAngle, endAngle);
  ctx.strokeStyle = trackColor;
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();

  if (fakeProb > 0 || (displayPct && displayPct > 0)) {
    let color, ratio;
    if (fakeProb < 0.35) {
      color = style.getPropertyValue('--status-real').trim() || '#34d399';
      ratio = displayPct ? (displayPct / 100) : (1 - fakeProb);
    } else if (fakeProb < 0.65) {
      color = style.getPropertyValue('--status-warn').trim() || '#fbbf24';
      ratio = displayPct ? (displayPct / 100) : Math.max(fakeProb, 1 - fakeProb);
    } else {
      color = style.getPropertyValue('--status-fake').trim() || '#f87171';
      ratio = displayPct ? (displayPct / 100) : fakeProb;
    }

    ratio = Math.min(Math.max(ratio, 0.05), 1.0);
    const fillEnd = startAngle + (Math.PI * ratio);
    ctx.beginPath();
    ctx.arc(cx, cy, R, startAngle, fillEnd);
    ctx.strokeStyle = color;
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';

    // Glowing effect
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.stroke();

    // Reset shadow
    ctx.shadowBlur = 0;
  }
}

// ─── Shared UI Updates for Results ───
function rpSetVerdict(fakeProb, modelStr, signals) {
  const pctSuspicious = Math.round(fakeProb * 100);

  // Authenticity Score Logic:
  // - If real (< 35% suspicious): show realness percentage (100 - pctSuspicious)
  //   e.g. 13% suspicious -> 87% Real
  // - If fake (>= 65% suspicious): show fake percentage (pctSuspicious)
  //   e.g. 95% suspicious -> 95% Fake
  // - If uncertain: show leading confidence
  let color, labelText, displayPct;
  if (fakeProb < 0.35) {
    color = 'var(--status-real)';
    labelText = 'Real';
    displayPct = 100 - pctSuspicious;
  } else if (fakeProb < 0.65) {
    color = 'var(--status-warn)';
    labelText = 'Uncertain';
    displayPct = Math.max(pctSuspicious, 100 - pctSuspicious);
  } else {
    color = 'var(--status-fake)';
    labelText = 'Fake';
    displayPct = pctSuspicious;
  }

  window.lastFakeProb = fakeProb;
  window.lastDisplayPct = displayPct;

  // Top Primary Card
  const cardAiScore = document.getElementById('cardAiScore');
  const cardVerdictText = document.getElementById('cardVerdictText');
  const primaryCardTitle = document.querySelector('.metric-card.card-dark .metric-title');
  if (primaryCardTitle) {
    primaryCardTitle.textContent = (labelText === 'Real') ? 'Authenticity Score' : 'Overall Fake Prob.';
  }

  // Gauge Texts
  const gaugeScoreText = document.getElementById('gaugeScoreText');
  const gaugeLabelText = document.getElementById('gaugeLabelText');

  // Gauge Arc Drawing
  drawGauge(fakeProb, displayPct);

  if (cardAiScore) {
    cardAiScore.textContent = displayPct + '%';
    cardAiScore.style.color = (labelText === 'Real') ? 'var(--status-real)' : (labelText === 'Fake' ? 'var(--status-fake)' : 'var(--status-warn)');
  }
  if (cardVerdictText) {
    cardVerdictText.textContent = labelText;
    cardVerdictText.style.color = (labelText === 'Real') ? 'var(--status-real)' : (labelText === 'Fake' ? 'var(--status-fake)' : 'var(--status-warn)');
  }

  const trendIconProb = document.getElementById('trendIconProb');
  if (trendIconProb) {
    if (labelText === 'Real') {
      trendIconProb.textContent = '✓';
      trendIconProb.className = 'trend-icon green';
      trendIconProb.style.color = 'var(--status-real)';
    } else if (labelText === 'Fake') {
      trendIconProb.textContent = '↑';
      trendIconProb.className = 'trend-icon red';
      trendIconProb.style.color = 'var(--status-fake)';
    } else {
      trendIconProb.textContent = '~';
      trendIconProb.className = 'trend-icon yellow';
      trendIconProb.style.color = 'var(--status-warn)';
    }
  }

  if (gaugeScoreText) {
    gaugeScoreText.textContent = displayPct + '%';
    gaugeScoreText.style.color = color;
  }
  if (gaugeLabelText) {
    gaugeLabelText.textContent = labelText;
    gaugeLabelText.style.color = color;
  }

  // Confidence thresholds bar
  const fill = document.getElementById('rpConfFill');
  const thumb = document.getElementById('rpConfThumb');
  if (fill) {
    fill.style.width = pctSuspicious + '%';
    fill.style.background = (labelText === 'Real') ? 'var(--status-real)' : (labelText === 'Fake' ? 'var(--status-fake)' : 'var(--status-warn)');
  }
  if (thumb) {
    thumb.style.left = `calc(${pctSuspicious}% - 12px)`;
  }

  // Justification Box
  const justificationBox = document.getElementById('justificationBox');
  const justificationText = document.getElementById('justificationText');

  if (justificationBox && justificationText && signals) {
    justificationBox.style.display = 'block';
    justificationBox.style.borderLeftColor = color;
    if (fakeProb < 0.35) {
      justificationText.textContent = `The media analyzed demonstrates ${displayPct}% authenticity confidence. Spatial, frequency, and facial landmark signals confirm high fidelity with minimal manipulation artifacts (${pctSuspicious}%).`;
    } else if (fakeProb < 0.65) {
      justificationText.textContent = `The models detected subtle irregularities (${pctSuspicious}% suspicious), resulting in an uncertain classification (${displayPct}% confidence). This may be due to compression noise, low resolution, or lighting variations.`;
    } else {
      justificationText.textContent = `Strong manipulation artifacts were confirmed with ${displayPct}% fake probability. Facial boundary blending, frequency noise, and generative patterns strongly indicate AI generation or tampering.`;
    }
  } else if (justificationBox) {
    justificationBox.style.display = 'none';
  }
}

// ─── Audio-specific justification ───
function rpSetAudioJustification(fakeProb, modelUsed) {
  const justificationBox = document.getElementById('justificationBox');
  const justificationText = document.getElementById('justificationText');
  if (!justificationBox || !justificationText) return;

  justificationBox.style.display = 'block';
  justificationBox.style.borderLeftColor = (fakeProb < 0.35) ? 'var(--status-real)' : (fakeProb < 0.65 ? 'var(--status-warn)' : 'var(--status-fake)');

  const modelLabel = modelUsed === 'aasist' ? 'AASIST (Graph Attention Network)' : 'Classical Spectral Analysis';

  if (fakeProb < 0.35) {
    justificationText.textContent = `${modelLabel} analysis indicates authentic audio. The spectral characteristics, pitch stability, and temporal patterns are consistent with natural human speech. No TTS or voice cloning artifacts detected.`;
  } else if (fakeProb < 0.65) {
    justificationText.textContent = `${modelLabel} detected some anomalies in the audio signal. The spectral flatness or pitch regularity shows patterns that could indicate either synthetic generation or unusual recording conditions (compression, noise).`;
  } else {
    justificationText.textContent = `${modelLabel} strongly indicates AI-generated audio. Detected artifacts include unnatural spectral smoothness, overly stable pitch contours, and temporal patterns consistent with TTS/voice cloning systems.`;
  }
}

// ─── Drop zone setup ───
function setupDropZone(zoneEl, inputEl, onFile) {
  if (!zoneEl || !inputEl) return;
  zoneEl.addEventListener('click', e => {
    if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;
    inputEl.click();
  });
  inputEl.addEventListener('change', e => {
    if (e.target.files.length > 0) onFile(e.target.files[0]);
  });
  zoneEl.addEventListener('dragover', e => {
    e.preventDefault();
    zoneEl.style.borderColor = 'var(--bg-dark-green)';
    zoneEl.style.background = 'var(--dropzone-hover)';
  });
  zoneEl.addEventListener('dragleave', e => {
    if (!zoneEl.contains(e.relatedTarget)) {
      zoneEl.style.borderColor = '';
      zoneEl.style.background = '';
    }
  });
  zoneEl.addEventListener('drop', e => {
    e.preventDefault();
    zoneEl.style.borderColor = '';
    zoneEl.style.background = '';
    if (e.dataTransfer.files.length > 0) onFile(e.dataTransfer.files[0]);
  });
}

// ─── Face bounding boxes ───
function drawFaceBboxes(canvas, imgEl, faces) {
  if (!faces || !faces.length) return;
  const rect = imgEl.getBoundingClientRect();
  const nw = imgEl.naturalWidth, nh = imgEl.naturalHeight;
  const rw = rect.width, rh = rect.height;

  canvas.width = nw;
  canvas.height = nh;
  canvas.style.width = rw + 'px';
  canvas.style.height = rh + 'px';

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, nw, nh);

  faces.forEach(face => {
    const bbox = face.bbox || face.box || [];
    if (bbox.length < 4) return;
    let [x1, y1, x2, y2] = bbox;
    if (x2 < x1) { x2 = x1 + x2; y2 = y1 + y2; }

    const bw = x2 - x1, bh = y2 - y1;
    const score = face.fake_score ?? 0;
    const color = score < 0.35 ? '#34d399' : score < 0.65 ? '#fbbf24' : '#f87171';

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, bw, bh);

    // Label
    const pctSuspicious = Math.round(score * 100);
    const isReal = score < 0.35;
    const isFake = score >= 0.65;
    const displayPct = isReal ? (100 - pctSuspicious) : (isFake ? pctSuspicious : Math.max(pctSuspicious, 100 - pctSuspicious));
    const label = isReal ? `Real · ${displayPct}%` : (isFake ? `Fake · ${displayPct}%` : `Uncertain · ${displayPct}%`);
    ctx.font = `600 ${Math.max(12, bw * 0.1)}px Inter, sans-serif`;
    const tw = ctx.measureText(label).width + 12;
    const lh = Math.max(24, bw * 0.15);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x1, y1 - lh - 4, tw, lh, 6);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 6, y1 - 8);
  });
}

// ─── Stat card updater (used by audio) ───
function updateStatCard(valueId, badgeId, prob) {
  const valueEl = document.getElementById(valueId);
  const badgeEl = document.getElementById(badgeId);
  const isReal = prob < 0.35;
  const isFake = prob >= 0.65;
  const pctSuspicious = Math.round(prob * 100);
  const displayVal = isReal ? (100 - pctSuspicious) : (isFake ? pctSuspicious : Math.max(pctSuspicious, 100 - pctSuspicious));

  if (valueEl) {
    valueEl.textContent = displayVal + '%';
    valueEl.style.color = isReal ? 'var(--status-real)' : (isFake ? 'var(--status-fake)' : 'var(--status-warn)');
  }
  if (badgeEl) {
    badgeEl.textContent = isReal ? 'REAL' : (isFake ? 'FAKE' : 'UNCERTAIN');
    badgeEl.style.background = isReal ? 'var(--status-real)' : (isFake ? 'var(--status-fake)' : 'var(--status-warn)');
    badgeEl.style.color = '#fff';
  }
}

// ─── Health Check ───
async function checkHealth() {
  // Silent health check
  try {
    await fetch(`${API_BASE}${ROUTE_HEALTH}`, { signal: AbortSignal.timeout(4000) });
  } catch { }
}
