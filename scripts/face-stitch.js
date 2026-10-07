/**
 * DeepScan AI — Live Face-Stitching Simulation Engine (Universal Edition)
 * ======================================================================
 * Educational simulation demonstrating real-time face swap mechanics.
 * Inspired by Deep-Live-Cam's architecture:
 *   Face Detection → Landmark Extraction → Affine Alignment
 *   → Neural Morph (Simulated) → Poisson Seamless Clone Blending.
 *
 * Capabilities:
 *  - Native FaceDetector API with seamless Adaptive Skin-Centroid Tracker fallback
 *  - 4 High-Resolution Presets (CEO, Anchor, Professor, Athlete) + Custom Photo Upload
 *  - Live Camera input OR Synthetic Animated Demo Feed (runs even without webcam)
 *  - Real-time Split Screen comparison (Real vs. Stitched)
 *  - Forensic Seam Boundary Heatmap mode
 *  - Landmark debug overlay & real-time pipeline step telemetry
 *
 * 100% Client-Side. Zero external network calls. Zero data retention.
 */

(function () {
  'use strict';

  // ─── Persona Definitions ───
  const PERSONAS = [
    {
      id: 'tech_ceo',
      name: 'Alex Chen',
      role: 'Tech CEO',
      img: 'assets/persona_tech_ceo.jpg',
      eyeLeft: { x: 0.42, y: 0.33 },
      eyeRight: { x: 0.58, y: 0.33 },
      noseTip: { x: 0.50, y: 0.45 },
      mouth: { x: 0.50, y: 0.53 },
      faceTop: 0.18,
      faceBottom: 0.62,
      faceLeft: 0.28,
      faceRight: 0.72,
    },
    {
      id: 'news_anchor',
      name: 'Priya Sharma',
      role: 'News Anchor',
      img: 'assets/persona_news_anchor.jpg',
      eyeLeft: { x: 0.41, y: 0.33 },
      eyeRight: { x: 0.59, y: 0.33 },
      noseTip: { x: 0.50, y: 0.45 },
      mouth: { x: 0.50, y: 0.53 },
      faceTop: 0.20,
      faceBottom: 0.64,
      faceLeft: 0.28,
      faceRight: 0.72,
    },
    {
      id: 'professor',
      name: 'Dr. James Ward',
      role: 'Academic Dean',
      img: 'assets/persona_professor.jpg',
      eyeLeft: { x: 0.42, y: 0.32 },
      eyeRight: { x: 0.58, y: 0.32 },
      noseTip: { x: 0.50, y: 0.44 },
      mouth: { x: 0.50, y: 0.52 },
      faceTop: 0.18,
      faceBottom: 0.60,
      faceLeft: 0.30,
      faceRight: 0.70,
    },
    {
      id: 'athlete',
      name: 'Maya Torres',
      role: 'Pro Athlete',
      img: 'assets/persona_athlete.jpg',
      eyeLeft: { x: 0.41, y: 0.33 },
      eyeRight: { x: 0.59, y: 0.33 },
      noseTip: { x: 0.50, y: 0.45 },
      mouth: { x: 0.50, y: 0.53 },
      faceTop: 0.20,
      faceBottom: 0.64,
      faceLeft: 0.28,
      faceRight: 0.72,
    },
  ];

  // ─── Adaptive Fallback Face Tracker ───
  // Operates directly in Canvas 2D without requiring experimental browser flags
  class AdaptiveFaceTracker {
    constructor() {
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
      this.canvas.width = 160;
      this.canvas.height = 120;
      this.prevBox = null;
    }

    detect(videoOrCanvas) {
      const vw = videoOrCanvas.videoWidth || videoOrCanvas.width || 640;
      const vh = videoOrCanvas.videoHeight || videoOrCanvas.height || 480;

      // Draw downsampled frame for sub-millisecond skin tracking
      this.ctx.drawImage(videoOrCanvas, 0, 0, 160, 120);
      let imgData;
      try {
        imgData = this.ctx.getImageData(0, 0, 160, 120);
      } catch (err) {
        return this.getDefaultBox(vw, vh);
      }

      const d = imgData.data;
      let totalSkin = 0;
      let sumX = 0;
      let sumY = 0;
      let minX = 160, maxX = 0, minY = 120, maxY = 0;

      // Search in upper 75% region (face zone)
      for (let y = 12; y < 105; y += 2) {
        for (let x = 16; x < 144; x += 2) {
          const idx = (y * 160 + x) * 4;
          const r = d[idx];
          const g = d[idx + 1];
          const b = d[idx + 2];

          // Normalized skin chrominance heuristic
          const maxVal = Math.max(r, g, b);
          const minVal = Math.min(r, g, b);
          const isSkin = (r > 65 && g > 35 && b > 20 && (maxVal - minVal) > 10 && r > g && r > b && (r - g) >= 8);

          if (isSkin) {
            totalSkin++;
            sumX += x;
            sumY += y;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      const scaleX = vw / 160;
      const scaleY = vh / 120;

      if (totalSkin > 80 && (maxX - minX) > 12 && (maxY - minY) > 12) {
        const cx = (sumX / totalSkin) * scaleX;
        const cy = (sumY / totalSkin) * scaleY;
        const bw = Math.max((maxX - minX) * scaleX * 1.15, vw * 0.30);
        const bh = Math.max((maxY - minY) * scaleY * 1.25, vh * 0.40);

        const targetBox = {
          x: Math.max(10, Math.min(vw - bw - 10, cx - bw / 2)),
          y: Math.max(10, Math.min(vh - bh - 10, cy - bh / 2.2)),
          width: bw,
          height: bh,
        };

        if (this.prevBox) {
          this.prevBox.x += (targetBox.x - this.prevBox.x) * 0.35;
          this.prevBox.y += (targetBox.y - this.prevBox.y) * 0.35;
          this.prevBox.width += (targetBox.width - this.prevBox.width) * 0.35;
          this.prevBox.height += (targetBox.height - this.prevBox.height) * 0.35;
        } else {
          this.prevBox = targetBox;
        }

        return this.prevBox;
      }

      return this.getDefaultBox(vw, vh);
    }

    getDefaultBox(vw, vh) {
      const bw = vw * 0.40;
      const bh = vh * 0.52;
      const def = {
        x: (vw - bw) / 2,
        y: (vh - bh) / 2.5,
        width: bw,
        height: bh,
      };
      if (this.prevBox) {
        this.prevBox.x += (def.x - this.prevBox.x) * 0.1;
        this.prevBox.y += (def.y - this.prevBox.y) * 0.1;
        this.prevBox.width += (def.width - this.prevBox.width) * 0.1;
        this.prevBox.height += (def.height - this.prevBox.height) * 0.1;
        return this.prevBox;
      }
      this.prevBox = def;
      return def;
    }
  }

  // ─── Synthetic Demo Feed Generator ───
  // Generates a realistic animated human subject for live testing without a camera
  class SyntheticVideoFeed {
    constructor(width, height) {
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d');
      this.width = width || 640;
      this.height = height || 480;
      this.canvas.width = this.width;
      this.canvas.height = this.height;
      this.startTime = performance.now();
    }

    renderFrame() {
      const t = (performance.now() - this.startTime) * 0.001;
      const ctx = this.ctx;
      const w = this.width;
      const h = this.height;

      // Realistic background (modern office / room blur)
      const bgGrad = ctx.createLinearGradient(0, 0, w, h);
      bgGrad.addColorStop(0, '#1c2234');
      bgGrad.addColorStop(0.5, '#161928');
      bgGrad.addColorStop(1, '#0f121d');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Office elements in background
      ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
      ctx.fillRect(w * 0.08, h * 0.12, w * 0.25, h * 0.45);
      ctx.fillRect(w * 0.72, h * 0.20, w * 0.20, h * 0.60);

      // Animated breathing & sway
      const swayX = Math.sin(t * 0.9) * 16 + Math.cos(t * 1.7) * 6;
      const swayY = Math.sin(t * 1.5) * 8;
      const headTilt = Math.sin(t * 0.7) * 0.05;

      const cx = w * 0.5 + swayX;
      const cy = h * 0.48 + swayY;

      // Torso & Shoulders
      ctx.fillStyle = '#202636';
      ctx.beginPath();
      ctx.ellipse(cx, h * 0.92, w * 0.34, h * 0.24, 0, 0, Math.PI * 2);
      ctx.fill();

      // Shirt collar
      ctx.fillStyle = '#2c3349';
      ctx.beginPath();
      ctx.moveTo(cx - 38, h * 0.74);
      ctx.lineTo(cx, h * 0.85);
      ctx.lineTo(cx + 38, h * 0.74);
      ctx.lineTo(cx + 26, h * 0.70);
      ctx.lineTo(cx, h * 0.78);
      ctx.lineTo(cx - 26, h * 0.70);
      ctx.closePath();
      ctx.fill();

      // Neck
      ctx.fillStyle = '#dca786';
      ctx.fillRect(cx - 24, cy + 50, 48, 65);

      // Head with subtle rotation tilt
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(headTilt);

      // Face silhouette & skin tone
      ctx.fillStyle = '#ebba99';
      ctx.beginPath();
      ctx.ellipse(0, 0, 72, 96, 0, 0, Math.PI * 2);
      ctx.fill();

      // Hair
      ctx.fillStyle = '#2b2320';
      ctx.beginPath();
      ctx.arc(0, -28, 78, Math.PI, Math.PI * 2);
      ctx.lineTo(76, 10);
      ctx.lineTo(58, -10);
      ctx.lineTo(-58, -10);
      ctx.lineTo(-76, 10);
      ctx.closePath();
      ctx.fill();

      // Eyebrows
      ctx.strokeStyle = '#362b25';
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(-44, -24);
      ctx.quadraticCurveTo(-26, -30, -10, -24);
      ctx.moveTo(10, -24);
      ctx.quadraticCurveTo(26, -30, 44, -24);
      ctx.stroke();

      // Eyes with periodic natural blinks
      const blinkCycle = t % 3.8;
      const isBlinking = blinkCycle > 3.65;

      if (isBlinking) {
        ctx.strokeStyle = '#362b25';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(-36, -12);
        ctx.lineTo(-16, -12);
        ctx.moveTo(16, -12);
        ctx.lineTo(36, -12);
        ctx.stroke();
      } else {
        // Eye whites
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.ellipse(-26, -12, 12, 7, 0, 0, Math.PI * 2);
        ctx.ellipse(26, -12, 12, 7, 0, 0, Math.PI * 2);
        ctx.fill();

        // Irises & pupils
        const gazeX = Math.sin(t * 0.5) * 2;
        ctx.fillStyle = '#3a2e28';
        ctx.beginPath();
        ctx.arc(-26 + gazeX, -12, 5, 0, Math.PI * 2);
        ctx.arc(26 + gazeX, -12, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#111';
        ctx.beginPath();
        ctx.arc(-26 + gazeX, -12, 2.5, 0, Math.PI * 2);
        ctx.arc(26 + gazeX, -12, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Nose
      ctx.strokeStyle = '#c48f70';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, -14);
      ctx.lineTo(4, 14);
      ctx.lineTo(-4, 20);
      ctx.lineTo(4, 20);
      ctx.stroke();

      // Mouth with subtle speaking movement
      const mouthOpen = Math.sin(t * 3.5) > 0.6 ? 4 : 1;
      ctx.fillStyle = '#be6866';
      ctx.beginPath();
      ctx.ellipse(0, 44, 18, 4 + mouthOpen, 0, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();

      return this.canvas;
    }
  }

  // ─── State ───
  let selectedPersona = PERSONAS[0];
  let personaImg = null;
  let customImgUrl = null;
  let stream = null;
  let animFrameId = null;
  let faceDetector = null;
  let adaptiveTracker = new AdaptiveFaceTracker();
  let syntheticFeed = new SyntheticVideoFeed(640, 480);
  let isRunning = false;
  let useSyntheticFeed = false;
  let blendIntensity = 0.85;
  let showDebugLandmarks = false;
  let showSplitView = false;
  let showSeamHeatmap = false;
  let trackerMode = 'adaptive'; // 'native' or 'adaptive'

  // Geometric smoothing
  let smoothedBox = null;
  let smoothedAngle = 0;
  let detectionCounter = 0;
  const DETECTION_INTERVAL = 2; // Frame stride for detection
  const SMOOTHING = 0.40;

  // DOM elements
  let video, outputCanvas, outputCtx, tempCanvas, tempCtx, maskCanvas, maskCtx;
  let startBtn, stopBtn, intensitySlider, intensityValue;
  let statusBadge, statusText, fpsDisplay;
  let sourceWebcamBtn, sourceDemoBtn;
  let debugToggle, splitToggle, seamToggle;
  let trackerNameEl;
  let pipelineSteps;
  let customCard, customInput, customNameEl;
  let placeholderEl;

  // Performance telemetry
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let currentFps = 0;

  // ─── Initialize ───
  document.addEventListener('DOMContentLoaded', init);

  function init() {
    video = document.getElementById('fsWebcamVideo');
    outputCanvas = document.getElementById('fsOutputCanvas');
    startBtn = document.getElementById('fsStartBtn');
    stopBtn = document.getElementById('fsStopBtn');
    intensitySlider = document.getElementById('fsIntensitySlider');
    intensityValue = document.getElementById('fsIntensityValue');
    statusBadge = document.getElementById('fsStatusBadge');
    statusText = document.getElementById('fsStatusText');
    fpsDisplay = document.getElementById('fsFpsDisplay');
    trackerNameEl = document.getElementById('fsTrackerName');

    sourceWebcamBtn = document.getElementById('fsSourceWebcam');
    sourceDemoBtn = document.getElementById('fsSourceDemo');
    debugToggle = document.getElementById('fsDebugToggle');
    splitToggle = document.getElementById('fsSplitToggle');
    seamToggle = document.getElementById('fsSeamToggle');

    customCard = document.getElementById('card_custom');
    customInput = document.getElementById('fsCustomInput');
    customNameEl = document.getElementById('fsCustomName');
    placeholderEl = document.getElementById('fsPlaceholder');

    pipelineSteps = document.querySelectorAll('.pipeline-step');

    if (outputCanvas) {
      outputCanvas.width = 640;
      outputCanvas.height = 480;
      outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true });

      tempCanvas = document.createElement('canvas');
      tempCanvas.width = 640;
      tempCanvas.height = 480;
      tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });

      maskCanvas = document.createElement('canvas');
      maskCanvas.width = 640;
      maskCanvas.height = 480;
      maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
    }

    // Initialize detectors & personas
    checkDetectionCapabilities();
    setupPersonaSelection();
    setupControls();
    selectPersona('tech_ceo');

    // Theme sync
    const savedTheme = localStorage.getItem('deepscan-theme') || 'dark';
    if (savedTheme === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
    const themeBtn = document.getElementById('fsThemeBtn');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const next = isDark ? 'light' : 'dark';
        if (next === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
        else document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('deepscan-theme', next);
      });
    }
  }

  // ─── Detection Capabilities ───
  function checkDetectionCapabilities() {
    if ('FaceDetector' in window) {
      try {
        faceDetector = new FaceDetector({ maxDetectedFaces: 1, fastMode: true });
        trackerMode = 'native';
        if (trackerNameEl) trackerNameEl.textContent = 'Native FaceDetector';
        updateStatus('ready', 'FaceDetector Ready — Select Persona');
        return;
      } catch (e) {
        console.warn('FaceDetector init failed, using Adaptive Tracker:', e);
      }
    }
    trackerMode = 'adaptive';
    if (trackerNameEl) trackerNameEl.textContent = 'Universal Tracker';
    updateStatus('ready', 'Universal Tracker Ready — Select Persona');
  }

  // ─── Persona Setup ───
  function setupPersonaSelection() {
    // Preset cards
    PERSONAS.forEach((p) => {
      const card = document.getElementById(`card_${p.id}`);
      if (card) {
        card.addEventListener('click', () => {
          selectPersona(p.id);
        });
      }
    });

    // Custom upload card
    if (customCard && customInput) {
      customCard.addEventListener('click', () => {
        customInput.click();
      });

      customInput.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (loadEvent) => {
          customImgUrl = loadEvent.target.result;
          const customPersona = {
            id: 'custom',
            name: file.name.replace(/\.[^/.]+$/, '').slice(0, 15) || 'Custom Persona',
            role: 'User Upload',
            img: customImgUrl,
            eyeLeft: { x: 0.42, y: 0.33 },
            eyeRight: { x: 0.58, y: 0.33 },
            noseTip: { x: 0.50, y: 0.45 },
            mouth: { x: 0.50, y: 0.53 },
            faceTop: 0.18,
            faceBottom: 0.62,
            faceLeft: 0.28,
            faceRight: 0.72,
          };

          // Register & select
          selectedPersona = customPersona;
          loadPersonaImage(customPersona);

          // Update UI
          document.querySelectorAll('.persona-card').forEach((c) => c.classList.remove('active'));
          customCard.classList.add('active');
          if (customNameEl) customNameEl.textContent = customPersona.name;
          updateStatus('ready', `Loaded: ${customPersona.name}`);
        };
        reader.readAsDataURL(file);
      });
    }
  }

  function selectPersona(id) {
    const p = PERSONAS.find((item) => item.id === id);
    if (!p) return;

    selectedPersona = p;
    document.querySelectorAll('.persona-card').forEach((c) => c.classList.remove('active'));
    const targetCard = document.getElementById(`card_${p.id}`);
    if (targetCard) targetCard.classList.add('active');

    loadPersonaImage(p);
  }

  function loadPersonaImage(persona) {
    personaImg = new Image();
    personaImg.crossOrigin = 'anonymous';
    personaImg.onload = async () => {
      // If native FaceDetector is available, refine source landmarks
      if (faceDetector) {
        try {
          const detections = await faceDetector.detect(personaImg);
          if (detections && detections.length > 0) {
            const f = detections[0];
            const iw = personaImg.naturalWidth;
            const ih = personaImg.naturalHeight;
            if (f.landmarks && f.landmarks.length >= 2) {
              const eyes = f.landmarks.filter((l) => l.type === 'eye');
              if (eyes.length >= 2) {
                const el = eyes[0].locations[0];
                const er = eyes[1].locations[0];
                persona.eyeLeft = { x: el.x / iw, y: el.y / ih };
                persona.eyeRight = { x: er.x / iw, y: er.y / ih };
              }
            }
          }
        } catch (e) {
          // Use default anthropometric landmarks
        }
      }
      updateStatus('ready', `Active: ${persona.name}`);
    };
    personaImg.src = persona.img;
  }

  // ─── Controls Setup ───
  function setupControls() {
    if (startBtn) startBtn.addEventListener('click', startSimulation);
    if (stopBtn) stopBtn.addEventListener('click', stopSimulation);

    // Feed Source Selector
    if (sourceWebcamBtn && sourceDemoBtn) {
      sourceWebcamBtn.addEventListener('click', () => {
        sourceWebcamBtn.classList.add('active');
        sourceDemoBtn.classList.remove('active');
        useSyntheticFeed = false;
        if (isRunning) {
          stopSimulation();
          startSimulation();
        }
      });

      sourceDemoBtn.addEventListener('click', () => {
        sourceDemoBtn.classList.add('active');
        sourceWebcamBtn.classList.remove('active');
        useSyntheticFeed = true;
        if (isRunning) {
          stopSimulation();
          startSimulation();
        }
      });
    }

    // Blend Slider
    if (intensitySlider) {
      intensitySlider.addEventListener('input', (e) => {
        blendIntensity = parseFloat(e.target.value);
        if (intensityValue) intensityValue.textContent = Math.round(blendIntensity * 100) + '%';
      });
    }

    // Toggles
    if (debugToggle) {
      debugToggle.addEventListener('change', (e) => {
        showDebugLandmarks = e.target.checked;
      });
    }

    if (splitToggle) {
      splitToggle.addEventListener('change', (e) => {
        showSplitView = e.target.checked;
      });
    }

    if (seamToggle) {
      seamToggle.addEventListener('change', (e) => {
        showSeamHeatmap = e.target.checked;
      });
    }
  }

  // ─── Start / Stop Simulation ───
  async function startSimulation() {
    if (!selectedPersona || !personaImg) {
      updateStatus('warn', 'Select a persona first');
      return;
    }

    // Hide placeholder
    if (placeholderEl) placeholderEl.style.display = 'none';

    if (useSyntheticFeed) {
      // Synthetic Demo Feed mode
      isRunning = true;
      if (startBtn) startBtn.style.display = 'none';
      if (stopBtn) stopBtn.style.display = 'inline-flex';
      updateStatus('live', `DEMO LIVE — Stitching ${selectedPersona.name}`);
      activatePipelineStep(0);

      frameCount = 0;
      lastFpsTime = performance.now();
      renderLoop();
      return;
    }

    // Physical Camera mode
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });

      video.srcObject = stream;
      await video.play();

      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      outputCanvas.width = vw;
      outputCanvas.height = vh;
      tempCanvas.width = vw;
      tempCanvas.height = vh;
      maskCanvas.width = vw;
      maskCanvas.height = vh;

      isRunning = true;
      if (startBtn) startBtn.style.display = 'none';
      if (stopBtn) stopBtn.style.display = 'inline-flex';
      updateStatus('live', `LIVE CAMERA — Stitching ${selectedPersona.name}`);
      activatePipelineStep(0);

      frameCount = 0;
      lastFpsTime = performance.now();
      renderLoop();
    } catch (err) {
      console.warn('Camera access denied or unavailable. Falling back to Demo Feed:', err);
      // Fallback seamlessly to Demo Feed
      useSyntheticFeed = true;
      if (sourceDemoBtn) sourceDemoBtn.classList.add('active');
      if (sourceWebcamBtn) sourceWebcamBtn.classList.remove('active');

      isRunning = true;
      if (startBtn) startBtn.style.display = 'none';
      if (stopBtn) stopBtn.style.display = 'inline-flex';
      updateStatus('live', `DEMO FEED (Camera Unavailable) — Stitching ${selectedPersona.name}`);
      activatePipelineStep(0);

      frameCount = 0;
      lastFpsTime = performance.now();
      renderLoop();
    }
  }

  function stopSimulation() {
    isRunning = false;
    if (animFrameId) cancelAnimationFrame(animFrameId);
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    if (video) video.srcObject = null;

    if (startBtn) startBtn.style.display = 'inline-flex';
    if (stopBtn) stopBtn.style.display = 'none';
    if (placeholderEl) placeholderEl.style.display = 'flex';

    smoothedBox = null;
    updateStatus('ready', 'Simulation Paused');
    deactivateAllPipelineSteps();

    if (outputCtx) {
      outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    }
  }

  // ─── Main Render Loop ───
  function renderLoop() {
    if (!isRunning) return;
    animFrameId = requestAnimationFrame(renderLoop);

    // Calculate FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      currentFps = frameCount;
      frameCount = 0;
      lastFpsTime = now;
      if (fpsDisplay) fpsDisplay.textContent = currentFps + ' FPS';
    }

    const cw = outputCanvas.width;
    const ch = outputCanvas.height;

    // 1. Render base video frame
    if (useSyntheticFeed) {
      const synFrame = syntheticFeed.renderFrame();
      outputCtx.drawImage(synFrame, 0, 0, cw, ch);
    } else if (video && video.readyState >= 2) {
      outputCtx.drawImage(video, 0, 0, cw, ch);
    }

    // Keep clean copy for split-screen comparison if enabled
    let cleanFrame = null;
    if (showSplitView) {
      cleanFrame = outputCtx.getImageData(0, 0, cw, ch);
    }

    // 2. Face Detection & Overlay
    detectionCounter++;
    if (detectionCounter >= DETECTION_INTERVAL || !smoothedBox) {
      detectionCounter = 0;
      performDetectionAndStitch(cleanFrame);
    } else if (smoothedBox) {
      // Intermediate frame: apply smoothed cached geometry for ultra-high FPS
      applyStitching(smoothedBox, cleanFrame);
    }
  }

  // ─── Detection & Stitching ───
  async function performDetectionAndStitch(cleanFrame) {
    const inputSource = useSyntheticFeed ? syntheticFeed.canvas : video;
    if (!inputSource) return;

    activatePipelineStep(1); // Step 1: Detect

    let detectedBox = null;

    // A. Native FaceDetector
    if (trackerMode === 'native' && faceDetector && !useSyntheticFeed) {
      try {
        const detections = await faceDetector.detect(video);
        if (detections && detections.length > 0) {
          const b = detections[0].boundingBox;
          detectedBox = { x: b.x, y: b.y, width: b.width, height: b.height };
        }
      } catch (e) {
        // Fall back to adaptive
      }
    }

    // B. Adaptive Universal Tracker
    if (!detectedBox) {
      detectedBox = adaptiveTracker.detect(inputSource);
    }

    if (detectedBox && selectedPersona && personaImg) {
      // Exponential smoothing (EMA) to prevent jitter
      if (!smoothedBox) {
        smoothedBox = { ...detectedBox };
      } else {
        smoothedBox.x += (detectedBox.x - smoothedBox.x) * SMOOTHING;
        smoothedBox.y += (detectedBox.y - smoothedBox.y) * SMOOTHING;
        smoothedBox.width += (detectedBox.width - smoothedBox.width) * SMOOTHING;
        smoothedBox.height += (detectedBox.height - smoothedBox.height) * SMOOTHING;
      }

      activatePipelineStep(2); // Step 2: Landmarks
      activatePipelineStep(3); // Step 3: Affine Transform
      applyStitching(smoothedBox, cleanFrame);
      activatePipelineStep(4); // Step 4: Compositing
    } else {
      deactivatePipelineSteps([2, 3, 4]);
    }
  }

  // ─── Face Stitching Pipeline (Affine + Poisson Blend) ───
  function applyStitching(box, cleanFrame) {
    if (!selectedPersona || !personaImg || !box) return;

    const cw = outputCanvas.width;
    const ch = outputCanvas.height;

    // Target anthropometric landmarks
    const targetEyeL = { x: box.x + box.width * 0.32, y: box.y + box.height * 0.35 };
    const targetEyeR = { x: box.x + box.width * 0.68, y: box.y + box.height * 0.35 };
    const targetNose = { x: box.x + box.width * 0.50, y: box.y + box.height * 0.54 };
    const targetMouth = { x: box.x + box.width * 0.50, y: box.y + box.height * 0.72 };

    const targetEyeDist = targetEyeR.x - targetEyeL.x;
    const targetCenter = { x: (targetEyeL.x + targetEyeR.x) / 2, y: (targetEyeL.y + targetEyeR.y) / 2 };

    // Source landmarks
    const srcW = personaImg.naturalWidth || 400;
    const srcH = personaImg.naturalHeight || 500;
    const srcEyeL = { x: selectedPersona.eyeLeft.x * srcW, y: selectedPersona.eyeLeft.y * srcH };
    const srcEyeR = { x: selectedPersona.eyeRight.x * srcW, y: selectedPersona.eyeRight.y * srcH };
    const srcEyeDist = Math.max(20, srcEyeR.x - srcEyeL.x);
    const srcEyeCenter = { x: (srcEyeL.x + srcEyeR.x) / 2, y: (srcEyeL.y + srcEyeR.y) / 2 };

    const scale = (targetEyeDist / srcEyeDist) * 1.82;

    // ── STEP 1: Affine Transform of Source Face onto Temp Canvas ──
    tempCtx.clearRect(0, 0, cw, ch);
    tempCtx.save();
    tempCtx.translate(targetCenter.x, targetCenter.y);
    tempCtx.rotate(smoothedAngle);
    tempCtx.scale(scale, scale);
    tempCtx.translate(-srcEyeCenter.x, -srcEyeCenter.y);
    tempCtx.drawImage(personaImg, 0, 0, srcW, srcH);
    tempCtx.restore();

    // ── STEP 2: Create Multi-Stop Radial Feather Mask ──
    const faceW = box.width * 0.92;
    const faceH = box.height * 1.02;
    const faceCx = targetCenter.x;
    const faceCy = targetCenter.y + box.height * 0.12;

    maskCtx.clearRect(0, 0, cw, ch);
    maskCtx.save();
    maskCtx.translate(faceCx, faceCy);
    maskCtx.rotate(smoothedAngle);

    const grad = maskCtx.createRadialGradient(0, 0, 0, 0, 0, Math.max(faceW, faceH) * 0.5);
    grad.addColorStop(0, `rgba(255,255,255,${blendIntensity})`);
    grad.addColorStop(0.55, `rgba(255,255,255,${blendIntensity * 0.95})`);
    grad.addColorStop(0.78, `rgba(255,255,255,${blendIntensity * 0.55})`);
    grad.addColorStop(0.90, `rgba(255,255,255,${blendIntensity * 0.18})`);
    grad.addColorStop(1.0, 'rgba(255,255,255,0)');

    maskCtx.fillStyle = grad;
    maskCtx.scale(1, faceH / faceW);
    maskCtx.beginPath();
    maskCtx.arc(0, 0, faceW / 2, 0, Math.PI * 2);
    maskCtx.fill();
    maskCtx.restore();

    // ── STEP 3: Poisson-Style Alpha Compositing & Tone Transfer ──
    const tempImgData = tempCtx.getImageData(0, 0, cw, ch);
    const maskImgData = maskCtx.getImageData(0, 0, cw, ch);
    const outImgData = outputCtx.getImageData(0, 0, cw, ch);

    const src = tempImgData.data;
    const msk = maskImgData.data;
    const out = outImgData.data;

    // Sub-region optimization: only compute pixels within the bounding box
    const startY = Math.max(0, Math.floor(faceCy - faceH / 2 - 20));
    const endY = Math.min(ch, Math.ceil(faceCy + faceH / 2 + 20));
    const startX = Math.max(0, Math.floor(faceCx - faceW / 2 - 20));
    const endX = Math.min(cw, Math.ceil(faceCx + faceW / 2 + 20));

    for (let y = startY; y < endY; y++) {
      for (let x = startX; x < endX; x++) {
        const i = (y * cw + x) * 4;
        let alpha = msk[i + 3] / 255;
        if (alpha > 0.01) {
          const srcLum = (src[i] + src[i + 1] + src[i + 2]) / 3;
          if (srcLum > 10) {
            // Color tone harmonization: 15% blend toward ambient target light
            const toneWeight = 0.15;
            const r = src[i] * (1 - toneWeight) + out[i] * toneWeight;
            const g = src[i + 1] * (1 - toneWeight) + out[i + 1] * toneWeight;
            const b = src[i + 2] * (1 - toneWeight) + out[i + 2] * toneWeight;

            // Seam Heatmap mode
            if (showSeamHeatmap && alpha > 0.15 && alpha < 0.75) {
              // Highlight the seam feathering ring in vivid forensic cyan
              out[i] = 0;
              out[i + 1] = 255;
              out[i + 2] = 220;
            } else {
              out[i] = out[i] * (1 - alpha) + r * alpha;
              out[i + 1] = out[i + 1] * (1 - alpha) + g * alpha;
              out[i + 2] = out[i + 2] * (1 - alpha) + b * alpha;
            }
          }
        }
      }
    }

    outputCtx.putImageData(outImgData, 0, 0);

    // ── STEP 4: Split View Comparison (Real vs Stitched) ──
    if (showSplitView && cleanFrame) {
      const splitX = Math.floor(cw * 0.5);
      // Restore clean frame on left half
      outputCtx.putImageData(cleanFrame, 0, 0, 0, 0, splitX, ch);

      // Draw cyber dividing line
      outputCtx.save();
      outputCtx.strokeStyle = '#00ff88';
      outputCtx.lineWidth = 2;
      outputCtx.beginPath();
      outputCtx.moveTo(splitX, 0);
      outputCtx.lineTo(splitX, ch);
      outputCtx.stroke();

      // Split labels
      outputCtx.font = 'bold 11px monospace';
      outputCtx.fillStyle = '#ffffff';
      outputCtx.fillText('REAL CAMERA', 16, 26);
      outputCtx.fillStyle = '#ff4466';
      outputCtx.fillText('STITCHED DEEPFAKE', splitX + 16, 26);
      outputCtx.restore();
    }

    // ── STEP 5: Debug & HUD Overlays ──
    if (showDebugLandmarks) {
      drawDebugOverlays(box, targetEyeL, targetEyeR, targetNose, targetMouth, faceCx, faceCy, faceW, faceH);
    }

    drawHudTelemetry(box);
  }

  // ─── Overlays & Telemetry ───
  function drawDebugOverlays(box, eyeL, eyeR, nose, mouth, cx, cy, fw, fh) {
    outputCtx.save();
    // Bounding Box
    outputCtx.strokeStyle = '#00ff88';
    outputCtx.lineWidth = 2;
    outputCtx.setLineDash([4, 4]);
    outputCtx.strokeRect(box.x, box.y, box.width, box.height);
    outputCtx.setLineDash([]);

    // Landmarks
    const landmarks = [
      { p: eyeL, label: 'L-Pupil', c: '#ff4488' },
      { p: eyeR, label: 'R-Pupil', c: '#ff4488' },
      { p: nose, label: 'Nose-Tip', c: '#44aaff' },
      { p: mouth, label: 'Mouth', c: '#ffaa44' },
    ];

    landmarks.forEach(({ p, label, c }) => {
      outputCtx.fillStyle = c;
      outputCtx.beginPath();
      outputCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      outputCtx.fill();
      outputCtx.fillStyle = '#ffffff';
      outputCtx.font = '9px monospace';
      outputCtx.fillText(label, p.x + 6, p.y - 4);
    });

    // Feathering Perimeter Ellipse
    outputCtx.strokeStyle = 'rgba(255, 68, 102, 0.6)';
    outputCtx.lineWidth = 1.5;
    outputCtx.beginPath();
    outputCtx.ellipse(cx, cy, fw / 2, fh / 2, smoothedAngle, 0, Math.PI * 2);
    outputCtx.stroke();

    outputCtx.restore();
  }

  function drawHudTelemetry(box) {
    outputCtx.save();

    // Corner brackets
    const bLen = 14;
    outputCtx.strokeStyle = 'rgba(0, 255, 136, 0.7)';
    outputCtx.lineWidth = 2;

    // Top-Left
    outputCtx.beginPath();
    outputCtx.moveTo(box.x, box.y + bLen);
    outputCtx.lineTo(box.x, box.y);
    outputCtx.lineTo(box.x + bLen, box.y);
    outputCtx.stroke();

    // Top-Right
    outputCtx.beginPath();
    outputCtx.moveTo(box.x + box.width - bLen, box.y);
    outputCtx.lineTo(box.x + box.width, box.y);
    outputCtx.lineTo(box.x + box.width, box.y + bLen);
    outputCtx.stroke();

    // Bottom-Left
    outputCtx.beginPath();
    outputCtx.moveTo(box.x, box.y + box.height - bLen);
    outputCtx.lineTo(box.x, box.y + box.height);
    outputCtx.lineTo(box.x + bLen, box.y + box.height);
    outputCtx.stroke();

    // Bottom-Right
    outputCtx.beginPath();
    outputCtx.moveTo(box.x + box.width - bLen, box.y + box.height);
    outputCtx.lineTo(box.x + box.width, box.y + box.height);
    outputCtx.lineTo(box.x + box.width, box.y + box.height - bLen);
    outputCtx.stroke();

    // Morph identity badge
    outputCtx.font = '10px "Space Grotesk", monospace';
    outputCtx.fillStyle = 'rgba(0, 255, 136, 0.9)';
    outputCtx.fillText(`STITCH: ${selectedPersona.name.toUpperCase()}`, box.x, box.y - 8);

    outputCtx.restore();
  }

  // ─── Pipeline Stepper Telemetry ───
  function activatePipelineStep(index) {
    if (!pipelineSteps || !pipelineSteps[index]) return;
    pipelineSteps[index].classList.add('active');
  }

  function deactivatePipelineSteps(indices) {
    indices.forEach((i) => {
      if (pipelineSteps && pipelineSteps[i]) {
        pipelineSteps[i].classList.remove('active');
      }
    });
  }

  function deactivateAllPipelineSteps() {
    if (!pipelineSteps) return;
    pipelineSteps.forEach((s) => s.classList.remove('active'));
  }

  // ─── Status Updates ───
  function updateStatus(type, text) {
    if (!statusBadge || !statusText) return;
    statusBadge.className = 'fs-status-badge fs-status-' + type;
    statusText.textContent = text;
  }

  // ─── Global Navigation ───
  window.goBackToMain = function () {
    window.location.href = 'index.html';
  };
})();
