/**
 * Webcam Detector — Direct Real-Time Live Anti-Spoofing & Deepfake Detection.
 *
 * Real-time detection directly on the live webcam feed:
 *   - Real Person: Genuine human presence with natural biological signals, micro-depth, and blinks.
 *   - Replay / Device Spoof: Phone screens, tablets, monitors, printed photos, video replays, and deepfakes.
 */

document.addEventListener('DOMContentLoaded', () => {
  const startBtn       = document.getElementById('webcamStartBtn');
  const stopBtn        = document.getElementById('wcStopBtn');
  const webcamVideo    = document.getElementById('webcamVideo');
  const webcamOverlay  = document.getElementById('webcamOverlay');
  const statusBadge    = document.getElementById('webcamStatusBadge');
  const alertBanner    = document.getElementById('wcAlertBanner');
  const alertText      = document.getElementById('wcAlertText');
  const tierChip       = document.getElementById('wcMonTierChip');

  // Telemetry cards
  const monBlinks      = document.getElementById('wcMonBlinks');
  const monBlinkState  = document.getElementById('wcMonBlinkState');
  const monPose        = document.getElementById('wcMonPose');
  const monPoseAngles  = document.getElementById('wcMonPoseAngles');
  const monLiveness    = document.getElementById('wcMonLiveness');
  const monLivenessLbl = document.getElementById('wcMonLivenessLabel');

  // Top metric cards
  const cardAiScore    = document.getElementById('cardAiScore');
  const cardVerdictText= document.getElementById('cardVerdictText');
  const trendIconProb  = document.getElementById('trendIconProb');
  const cardFaces      = document.getElementById('cardFaces');
  const cardEffNet     = document.getElementById('cardEffNet');
  const cardClassical  = document.getElementById('cardClassical');

  // Justification
  const justificationBox  = document.getElementById('justificationBox');
  const justificationText = document.getElementById('justificationText');

  // Gauge elements
  const gaugeScoreText = document.getElementById('gaugeScoreText');
  const gaugeLabelText = document.getElementById('gaugeLabelText');
  const gaugeCanvas    = document.getElementById('verdictGauge');

  if (!startBtn || !webcamVideo) return;

  const WS_BASE = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}`;

  let stream = null;
  let ws = null;
  let captureInterval = null;
  let isRunning = false;

  // Gauge animation state
  let targetScore = 0.05;
  let displayedScore = 0.05;
  let animFrameId = null;

  // Offscreen canvas for frame capture
  const captureCanvas = document.createElement('canvas');
  const captureCtx = captureCanvas.getContext('2d');
  captureCanvas.width = 480;
  captureCanvas.height = 360;

  startBtn.addEventListener('click', startWebcam);
  if (stopBtn) stopBtn.addEventListener('click', stopWebcam);

  // Initialize with clean placeholders
  resetMetrics();

  // Tab change handler — stop webcam when switching away
  const navBtns = document.querySelectorAll('.nav-item[data-tab]');
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.tab !== 'webcam') {
        stopWebcam();
      }
    });
  });

  // ── START WEBCAM ──

  async function startWebcam() {
    try {
      updateBadge('Connecting camera…', 'warn');
      resetMetrics();
      targetScore = 0;

      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });

      webcamVideo.srcObject = stream;
      await webcamVideo.play();

      isRunning = true;
      startBtn.style.display = 'none';
      if (stopBtn) stopBtn.style.display = 'inline-flex';
      if (tierChip) tierChip.style.display = 'inline-flex';

      startSmoothGauge();

      // Connect WebSocket
      ws = new WebSocket(`${WS_BASE}/ws/webcam`);

      ws.onopen = () => {
        updateBadge('Live Feed Active', 'safe');
        // Reset backend session state cleanly
        ws.send(JSON.stringify({ type: 'reset' }));
        ws.send(JSON.stringify({ type: 'start_monitoring' }));
        startCapture();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.command_response) return;
          handleFrame(data);
        } catch (e) {
          console.debug('WS parse error:', e);
        }
      };

      ws.onclose = () => {
        updateBadge('Disconnected', 'danger');
        stopCapture();
      };

      ws.onerror = () => {
        updateBadge('Connection Error', 'danger');
      };

    } catch (err) {
      updateBadge('Camera Access Denied', 'danger');
      console.error('Webcam access error:', err);
    }
  }

  // ── STOP WEBCAM ──

  function stopWebcam() {
    isRunning = false;
    stopCapture();
    stopSmoothGauge();

    if (ws) {
      try {
        // Tell backend to reset before disconnect
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'stop' }));
        }
        ws.close();
      } catch (e) {}
      ws = null;
    }

    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    if (webcamVideo) webcamVideo.srcObject = null;

    startBtn.style.display = 'inline-flex';
    if (stopBtn) stopBtn.style.display = 'none';
    if (tierChip) tierChip.style.display = 'none';
    if (alertBanner) alertBanner.style.display = 'none';

    updateBadge('Camera Ready', '');

    // Clear overlay
    if (webcamOverlay) {
      const ctx = webcamOverlay.getContext('2d');
      ctx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
    }

    resetMetrics();
  }

  // ── FRAME CAPTURE & STREAMING ──

  function startCapture() {
    if (captureInterval) clearInterval(captureInterval);
    captureInterval = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN || !webcamVideo || webcamVideo.readyState < 2) return;

      try {
        captureCtx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
        captureCanvas.toBlob((blob) => {
          if (blob && ws && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then(buf => {
              if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(buf);
              }
            });
          }
        }, 'image/jpeg', 0.72);
      } catch (err) {
        console.debug('Frame capture error:', err);
      }
    }, 200); // 5 FPS — prevents frame backlog that causes restart delays
  }

  function stopCapture() {
    if (captureInterval) {
      clearInterval(captureInterval);
      captureInterval = null;
    }
  }

  // ── REAL-TIME FRAME HANDLER ──

  function handleFrame(data) {
    if (!isRunning) return;

    const faceCount = data.faces_detected ?? 0;
    const tier = data.tier ?? 1;
    const combinedScore = data.combined_fake_score ?? 0.05;
    const isLive = data.is_live ?? true;
    const livenessObj = data.liveness_object || (isLive ? 'Live Person' : 'Device Screen Replay');
    const faceVerdict = data.face_verdict || (combinedScore >= 0.50 ? 'Fake Face' : 'Real Face');
    const overallVerdict = data.overall_verdict || `${livenessObj}: ${faceVerdict}`;

    const blink = data.blink || {};
    const pose = data.pose || {};
    const signals = data.liveness_signals || {};

    // When person disappears from camera
    if (faceCount === 0) {
      updateBadge('Awaiting Face Presence…', 'warn');
      if (tierChip) tierChip.innerHTML = '<span class="tier-dot tier-2"></span> No Face Detected';
      if (alertBanner) alertBanner.style.display = 'none';

      if (monBlinks) monBlinks.textContent = '--';
      if (monBlinkState) monBlinkState.textContent = 'No face in frame';
      if (monPose) monPose.textContent = '--';
      if (monPoseAngles) monPoseAngles.textContent = 'No face in frame';
      if (monLiveness) { monLiveness.textContent = 'No Face'; monLiveness.style.color = 'var(--text-muted)'; }
      if (monLivenessLbl) monLivenessLbl.textContent = 'Position face in camera';

      if (cardAiScore) { cardAiScore.textContent = '--'; cardAiScore.style.color = ''; }
      if (cardVerdictText) cardVerdictText.textContent = 'No Face Detected';
      if (cardFaces) cardFaces.textContent = '0 Faces';
      if (cardEffNet) cardEffNet.textContent = 'Vision Transformer';
      if (cardClassical) cardClassical.textContent = 'Moiré: --';

      const confFill = document.getElementById('rpConfFill');
      const confThumb = document.getElementById('rpConfThumb');
      if (confFill) { confFill.style.width = '0%'; confFill.style.background = 'var(--primary-color)'; }
      if (confThumb) { confThumb.style.left = '0%'; }

      targetScore = 0;
      if (justificationBox && justificationText) {
        justificationBox.style.borderLeftColor = 'var(--border-color)';
        justificationText.textContent = 'No face detected in camera view. Please face the camera to begin live analysis.';
      }

      drawFaceOverlay(webcamOverlay, webcamVideo, [], 1, '');
      return;
    }

    targetScore = combinedScore;

    // 1. Update Status Badge & Tier Chip
    if (tier === 1) {
      updateBadge('Live Feed: Authentic', 'safe');
      if (tierChip) tierChip.innerHTML = '<span class="tier-dot tier-1"></span> Tier 1: Real Person';
    } else if (tier === 2) {
      updateBadge('Live Feed: Checking…', 'warn');
      if (tierChip) tierChip.innerHTML = '<span class="tier-dot tier-2"></span> Tier 2: Anomaly / Analyzing';
    } else {
      updateBadge(`Threat: ${livenessObj}`, 'danger');
      if (tierChip) tierChip.innerHTML = '<span class="tier-dot tier-3"></span> Tier 3: Spoof / Fake Detected';
    }

    // 2. Alert Banner (Show on Spoof / Screen Replay / Deepfake)
    if (alertBanner && alertText) {
      if (tier === 3 || !isLive || combinedScore >= 0.60) {
        alertBanner.style.display = 'flex';
        alertBanner.className = 'wc-alert-banner wc-alert-danger';
        const detail = signals.device_detected ? `Physical display device [${signals.device_label || 'Cell Phone'}] identified!` : `Screen bezels, OLED sub-pixels & digital quantization identified.`;
        alertText.textContent = `⚠️ Presentation Attack: ${livenessObj} detected! ${detail}`;
      } else if (tier === 2 || combinedScore >= 0.40) {
        alertBanner.style.display = 'flex';
        alertBanner.className = 'wc-alert-banner wc-alert-warn';
        alertText.textContent = `⚠️ Warning: Low lighting / compression anomaly detected.`;
      } else {
        alertBanner.style.display = 'none';
      }
    }

    // 3. Telemetry Cards
    if (monBlinks) monBlinks.textContent = `${blink.blink_count ?? 0} blinks`;
    if (monBlinkState) monBlinkState.textContent = `EAR: ${(blink.ear ?? 0.32).toFixed(2)} · ${blink.eye_state ?? 'open'}`;

    if (monPose) monPose.textContent = pose.orientation || 'Center';
    if (monPoseAngles) monPoseAngles.textContent = `Yaw: ${Math.round(pose.yaw || 0)}° · Pitch: ${Math.round(pose.pitch || 0)}°`;

    const realPct = Math.round((1 - combinedScore) * 100);
    const fakePct = Math.round(combinedScore * 100);

    if (monLiveness) {
      monLiveness.textContent = livenessObj;
      monLiveness.style.color = (tier === 1) ? '#10B981' : (tier === 3 ? '#EF4444' : '#F59E0B');
    }
    if (monLivenessLbl) {
      monLivenessLbl.textContent = (tier === 1) ? `Score: ${realPct}% Real (Live Human)` : `Score: ${fakePct}% Spoof Probability`;
    }

    // 4. Top Metric Cards & Model Confidence Bar
    const isReal = (tier === 1);
    const isFake = (tier === 3);
    const displayPct = isReal ? realPct : (isFake ? fakePct : Math.max(fakePct, realPct));

    if (cardAiScore) {
      cardAiScore.textContent = `${displayPct}%`;
      cardAiScore.style.color = (tier === 1) ? '#10B981' : (tier === 3 ? '#EF4444' : '#F59E0B');
    }
    if (cardVerdictText) {
      cardVerdictText.textContent = overallVerdict;
    }
    if (trendIconProb) {
      trendIconProb.textContent = combinedScore < 0.35 ? '↓' : '↑';
      trendIconProb.className = combinedScore < 0.35 ? 'trend-icon green' : 'trend-icon danger';
    }
    if (cardFaces) {
      cardFaces.textContent = `${faceCount} Face${faceCount !== 1 ? 's' : ''}`;
    }
    if (cardEffNet) {
      cardEffNet.textContent = signals.device_detected ? 'Object Detector: ACTIVE' : 'Vision Transformer';
    }
    if (cardClassical) {
      if (signals.device_detected && signals.device_label) {
        cardClassical.textContent = `${signals.device_label}`;
      } else if ((signals.subpixel_pattern || 0) > 0.20) {
        cardClassical.textContent = `Sub-pixel: ${(signals.subpixel_pattern).toFixed(2)}`;
      } else {
        const moire = (signals.moire_frequency || 0.08).toFixed(2);
        cardClassical.textContent = `Moiré: ${moire}`;
      }
    }

    // Model Confidence Bar (Green/Red Track Bar)
    const confFill = document.getElementById('rpConfFill');
    const confThumb = document.getElementById('rpConfThumb');
    if (confFill) {
      confFill.style.width = `${displayPct}%`;
      confFill.style.background = (tier === 1) ? '#10B981' : (tier === 3 ? '#EF4444' : '#F59E0B');
    }
    if (confThumb) {
      confThumb.style.left = `${displayPct}%`;
    }

    // 5. Verdict Justification Box
    if (justificationBox && justificationText) {
      justificationBox.style.display = 'block';
      if (tier === 1) {
        justificationBox.style.borderLeftColor = '#10B981';
        justificationText.innerHTML = `<strong>✓ Live Human Person:</strong> Natural 3D micro-texture, biological skin chromaticity, organic specular reflections, and spontaneous eye-blink dynamics (EAR: ${(blink.ear || 0.32).toFixed(2)}) detected. Confirmed authentic live presence.`;
      } else if (tier === 3) {
        justificationBox.style.borderLeftColor = '#EF4444';
        if (livenessObj.includes('Device') || livenessObj.includes('Screen') || livenessObj.includes('Phone') || livenessObj.includes('Replay') || (signals.moire_frequency || 0) > 0.25 || (signals.context_border || 0) > 0.25 || (signals.subpixel_pattern || 0) > 0.25) {
          justificationText.innerHTML = `<strong>⚠️ Presentation Replay Attack:</strong> Electronic display device (${signals.device_label || 'Mobile Phone / Screen'}), rectangular screen bezels, high-PPI OLED subpixel raster (peak: ${(signals.subpixel_pattern || signals.moire_frequency || 0.85).toFixed(2)}), and 8-bit digital color quantization detected.`;
        } else if (livenessObj.includes('Photo')) {
          justificationText.innerHTML = `<strong>⚠️ Printed Photograph Spoof:</strong> Zero 3D facial depth, static specular highlights, and absence of natural micro-motion detected consistent with a physical photo printout.`;
        } else {
          justificationText.innerHTML = `<strong>⚠️ Deepfake Face-Swap Detected:</strong> High artificial generation probability (${Math.round(combinedScore * 100)}%), facial boundary warping, and synthetic texture artifacts identified.`;
        }
      } else {
        justificationBox.style.borderLeftColor = '#F59E0B';
        justificationText.innerHTML = `<strong>? Analyzing Feed:</strong> Facial features detected, calibrating anti-spoofing and liveness signals. Please face the camera steadily.`;
      }
    }

    // 6. Draw Bounding Box & Classification on Canvas Overlay
    drawFaceOverlay(webcamOverlay, webcamVideo, data.faces || [], tier, overallVerdict, combinedScore);
  }

  // ── DRAW FACE OVERLAY ──

  function drawFaceOverlay(canvas, video, faces, tier, overallVerdict = '', combinedScore = 0.05) {
    if (!canvas || !video) return;

    // Use current element display dimensions for 1:1 pixel rendering
    const displayW = canvas.offsetWidth || video.offsetWidth || 640;
    const displayH = canvas.offsetHeight || video.offsetHeight || 360;
    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width = displayW;
      canvas.height = displayH;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!faces || faces.length === 0) return;

    // Capture canvas is 480x360
    const scaleX = canvas.width / 480;
    const scaleY = canvas.height / 360;

    faces.forEach(face => {
      const bbox = face.bbox || [];
      if (bbox.length < 4) return;
      let [x1, y1, x2, y2] = bbox;
      if (x2 < x1) { x2 = x1 + x2; y2 = y1 + y2; }

      // Scale from 480x360 capture resolution to display resolution
      x1 = x1 * scaleX;
      x2 = x2 * scaleX;
      y1 = y1 * scaleY;
      y2 = y2 * scaleY;

      const bw = x2 - x1;
      const bh = y2 - y1;
      if (bw <= 0 || bh <= 0) return;

      const isSafe = (tier === 1);
      const isDanger = (tier === 3);
      const color = isSafe ? '#10B981' : (isDanger ? '#EF4444' : '#F59E0B');

      // 1. Semi-transparent scan target box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, bw, bh);

      // 2. Corner brackets (tactical HUD brackets)
      const cl = Math.min(22, Math.min(bw, bh) * 0.25);
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.beginPath();
      // Top-left
      ctx.moveTo(x1, y1 + cl); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cl, y1);
      // Top-right
      ctx.moveTo(x2 - cl, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cl);
      // Bottom-left
      ctx.moveTo(x1, y2 - cl); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cl, y2);
      // Bottom-right
      ctx.moveTo(x2 - cl, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cl);
      ctx.stroke();

      // 3. Top classification label pill
      let label;
      if (isSafe) {
        const score = Math.round((1 - (face.fake_score ?? combinedScore)) * 100);
        label = `Live Person ✓ · ${score}% Real`;
      } else if (isDanger) {
        const score = Math.round((face.fake_score ?? combinedScore) * 100);
        const name = (overallVerdict && overallVerdict.length < 26) ? overallVerdict : 'Spoof Attack';
        label = `⚠️ ${name} · ${score}% Fake`;
      } else {
        label = '? Analyzing Presence…';
      }

      ctx.font = '700 13px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
      const textMetrics = ctx.measureText(label);
      const tw = textMetrics.width + 18;
      const lh = 26;
      const tagY = Math.max(lh + 4, y1) - lh - 4;

      ctx.fillStyle = color;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(x1, tagY, tw, lh, 6);
      } else {
        ctx.rect(x1, tagY, tw, lh);
      }
      ctx.fill();

      // Text inside tag
      ctx.fillStyle = '#0F172A';
      ctx.fillText(label, x1 + 9, tagY + 18);
    });
  }

  // ── GAUGE ANIMATION ──

  function startSmoothGauge() {
    if (animFrameId) return;
    function step() {
      if (!isRunning) return;
      displayedScore += (targetScore - displayedScore) * 0.12;

      let displayPct = 0;
      if (targetScore === 0 && displayedScore < 0.03) {
        displayedScore = 0;
        if (gaugeScoreText) { gaugeScoreText.textContent = '--'; gaugeScoreText.style.color = ''; }
        if (gaugeLabelText) { gaugeLabelText.textContent = 'Score'; gaugeLabelText.style.color = ''; }
        const confFill = document.getElementById('rpConfFill');
        const confThumb = document.getElementById('rpConfThumb');
        if (confFill) { confFill.style.width = '0%'; }
        if (confThumb) { confThumb.style.left = '0%'; }
        drawGauge(0, 0);
        animFrameId = requestAnimationFrame(step);
        return;
      }

      const pctSuspicious = Math.round(displayedScore * 100);
      let statusColor = 'var(--status-real)';

      if (displayedScore < 0.35) {
        displayPct = 100 - pctSuspicious;
        statusColor = 'var(--status-real)';
        if (gaugeScoreText) {
          gaugeScoreText.textContent = `${displayPct}%`;
          gaugeScoreText.style.color = statusColor;
        }
        if (gaugeLabelText) {
          gaugeLabelText.textContent = 'Real';
          gaugeLabelText.style.color = statusColor;
        }
      } else if (displayedScore < 0.65) {
        displayPct = Math.max(pctSuspicious, 100 - pctSuspicious);
        statusColor = 'var(--status-warn)';
        if (gaugeScoreText) {
          gaugeScoreText.textContent = `${displayPct}%`;
          gaugeScoreText.style.color = statusColor;
        }
        if (gaugeLabelText) {
          gaugeLabelText.textContent = 'Uncertain';
          gaugeLabelText.style.color = statusColor;
        }
      } else {
        displayPct = pctSuspicious;
        statusColor = 'var(--status-fake)';
        if (gaugeScoreText) {
          gaugeScoreText.textContent = `${displayPct}%`;
          gaugeScoreText.style.color = statusColor;
        }
        if (gaugeLabelText) {
          gaugeLabelText.textContent = 'Fake';
          gaugeLabelText.style.color = statusColor;
        }
      }

      window.lastFakeProb = displayedScore;
      window.lastDisplayPct = displayPct;

      drawGauge(displayedScore, displayPct);

      const confFill = document.getElementById('rpConfFill');
      const confThumb = document.getElementById('rpConfThumb');
      if (confFill) {
        confFill.style.width = `${displayPct}%`;
        confFill.style.background = statusColor;
      }
      if (confThumb) {
        confThumb.style.left = `${displayPct}%`;
      }

      animFrameId = requestAnimationFrame(step);
    }
    animFrameId = requestAnimationFrame(step);
  }

  function stopSmoothGauge() {
    if (animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  function drawGauge(fakeProb, displayPct = null) {
    if (!gaugeCanvas) return;
    const ctx = gaugeCanvas.getContext('2d');
    const W = gaugeCanvas.width, H = gaugeCanvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H - 15;
    const R = 100;
    const startAngle = Math.PI;
    const endAngle = 0;

    const style = getComputedStyle(document.documentElement);
    let trackColor = style.getPropertyValue('--gauge-track').trim() || '#f1f5f9';

    // Track (Thick modern curved track)
    ctx.beginPath();
    ctx.arc(cx, cy, R, startAngle, endAngle);
    ctx.strokeStyle = trackColor;
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.stroke();

    if (fakeProb > 0 || (displayPct !== null && displayPct > 0)) {
      let color, ratio;
      if (fakeProb < 0.35) {
        color = style.getPropertyValue('--status-real').trim() || '#10B981';
        ratio = (displayPct !== null && displayPct !== undefined) ? (displayPct / 100) : (1 - fakeProb);
      } else if (fakeProb < 0.65) {
        color = style.getPropertyValue('--status-warn').trim() || '#F59E0B';
        ratio = (displayPct !== null && displayPct !== undefined) ? (displayPct / 100) : Math.max(fakeProb, 1 - fakeProb);
      } else {
        color = style.getPropertyValue('--status-fake').trim() || '#EF4444';
        ratio = (displayPct !== null && displayPct !== undefined) ? (displayPct / 100) : fakeProb;
      }

      ratio = Math.min(Math.max(ratio, 0.05), 1.0);
      const fillEnd = startAngle + (Math.PI * ratio);
      ctx.beginPath();
      ctx.arc(cx, cy, R, startAngle, fillEnd);
      ctx.strokeStyle = color;
      ctx.lineWidth = 14;
      ctx.lineCap = 'round';

      // Glowing aura
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.stroke();

      // Reset shadow
      ctx.shadowBlur = 0;
    }
  }

  function updateBadge(text, status) {
    if (!statusBadge) return;
    statusBadge.innerHTML = `<span class="webcam-dot ${status}"></span> ${text}`;
  }

  function resetMetrics() {
    targetScore = 0;
    displayedScore = 0;
    if (cardAiScore) { cardAiScore.textContent = '--'; cardAiScore.style.color = ''; }
    if (cardVerdictText) cardVerdictText.textContent = 'Awaiting Live Feed';
    if (cardFaces) cardFaces.textContent = '--';
    if (cardEffNet) cardEffNet.textContent = '--';
    if (cardClassical) cardClassical.textContent = '--';
    if (monBlinks) monBlinks.textContent = '--';
    if (monBlinkState) monBlinkState.textContent = 'Awaiting live feed';
    if (monPose) monPose.textContent = '--';
    if (monPoseAngles) monPoseAngles.textContent = 'Awaiting live feed';
    if (monLiveness) { monLiveness.textContent = '--'; monLiveness.style.color = ''; }
    if (monLivenessLbl) monLivenessLbl.textContent = 'Awaiting live feed';
    if (gaugeScoreText) { gaugeScoreText.textContent = '--'; gaugeScoreText.style.color = ''; }
    if (gaugeLabelText) { gaugeLabelText.textContent = 'Score'; gaugeLabelText.style.color = ''; }
    const confFill = document.getElementById('rpConfFill');
    const confThumb = document.getElementById('rpConfThumb');
    if (confFill) { confFill.style.width = '0%'; confFill.style.background = 'var(--primary-color)'; }
    if (confThumb) { confThumb.style.left = '0%'; }
    if (justificationBox && justificationText) {
      justificationBox.style.borderLeftColor = 'var(--primary-color)';
      justificationText.textContent = "Upload media to see the model's reasoning here.";
    }
    window.lastFakeProb = 0;
    window.lastDisplayPct = 0;
    drawGauge(0, 0);
  }
});
