/**
 * Video Detector Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('videoDropZone');
  const fileInput = document.getElementById('videoFileInput');
  const previewZone = document.getElementById('videoPreviewZone');
  const previewVideo = document.getElementById('videoPreviewEl');
  const overlay = document.getElementById('videoOverlay');
  const chartCanvas = document.getElementById('timelineChart');
  const resetBtn = document.getElementById('mediaResetBtn');
  
  let chartInstance = null;

  setupDropZone(dropZone, fileInput, handleVideoUpload);



  async function handleVideoUpload(file) {
    if (!file.type.startsWith('video/')) {
      alert('Please upload a video file');
      return;
    }

    const objUrl = URL.createObjectURL(file);
    previewVideo.src = objUrl;
    dropZone.style.display = 'none';
    previewZone.style.display = 'flex';
    if (resetBtn) resetBtn.style.display = 'inline-block';
    
    overlay.style.display = 'flex';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('sample_frames', 5);

    try {
      const res = await fetch(`${API_BASE}${ROUTE_VIDEO}`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      
      overlay.style.display = 'none';

      const prob = data.fake_probability;
      const model = data.model_used;
      const signals = data.details || {};
      const frames = data.frame_timeline || [];

      // Calculate total faces tracked across all analyzed frames
      let maxFaces = 0;
      let totalFaceDetections = 0;
      frames.forEach(f => {
        totalFaceDetections += (f.faces_detected || 0);
        if ((f.faces_detected || 0) > maxFaces) maxFaces = f.faces_detected;
      });

      // 4 Metric Cards
      rpSetVerdict(prob, model, signals);
      
      // Faces card: show max concurrent faces detected
      document.getElementById('cardFaces').textContent = maxFaces || 0;
      
      // Model card
      let modelDisplayName = 'ViT Deepfake';
      if (model === 'classical_only') modelDisplayName = 'Classical CV';
      document.getElementById('cardEffNet').textContent = modelDisplayName;
      
      // Temporal penalty card
      document.getElementById('cardClassical').textContent = 
        signals.temporal_penalty !== undefined ? '+' + Math.round(signals.temporal_penalty * 100) + '%' : 'N/A';

      // Log analysis summary to console for debugging
      console.log(`[DeepScan] Video analyzed: ${data.frames_analyzed} frames, ` +
        `${signals.video_duration}s duration, ${maxFaces} max faces, ` +
        `score=${prob}, decoder=${signals.decoder}`);

    } catch (err) {
      console.error(err);
      overlay.style.display = 'none';
      alert('Failed to analyze video. Ensure backend is running.');
    }
  }
});
