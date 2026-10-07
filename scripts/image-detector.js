/**
 * Image Detector Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('imageDropZone');
  const fileInput = document.getElementById('imageFileInput');
  const previewZone = document.getElementById('imagePreviewZone');
  const previewImg = document.getElementById('imagePreviewImg');
  const faceCanvas = document.getElementById('imageFaceCanvas');
  const scanSweep = document.getElementById('imageScanSweep');
  const resetBtn = document.getElementById('mediaResetBtn');

  setupDropZone(dropZone, fileInput, handleImageUpload);


  async function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    const objUrl = URL.createObjectURL(file);
    previewImg.src = objUrl;
    dropZone.style.display = 'none';
    previewZone.style.display = 'flex';
    if (resetBtn) resetBtn.style.display = 'inline-block';

    scanSweep.style.animation = 'sweep 2s infinite linear';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}${ROUTE_IMAGE}`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      
      scanSweep.style.animation = 'none';
      scanSweep.style.display = 'none';

      const prob = data.fake_probability;
      const model = data.model_used;
      const signals = data.details || {};
      const faces = data.faces || [];

      // 4 Metric Cards
      rpSetVerdict(prob, model, signals);
      
      document.getElementById('cardFaces').textContent = faces.length;
      
      // Show which model is active
      let modelDisplayName = 'ViT Deepfake';
      if (model === 'classical_only') modelDisplayName = 'Classical CV';
      document.getElementById('cardEffNet').textContent = modelDisplayName;
      
      document.getElementById('cardClassical').textContent = 
        signals.classical_score !== undefined ? Math.round(signals.classical_score * 100) + '%' : 'N/A';
      
      // Wait for image to load to get accurate dimensions for canvas
      if (previewImg.complete) {
        drawFaceBboxes(faceCanvas, previewImg, faces);
      } else {
        previewImg.onload = () => drawFaceBboxes(faceCanvas, previewImg, faces);
      }

    } catch (err) {
      console.error(err);
      scanSweep.style.animation = 'none';
      alert('Failed to analyze image. Please ensure backend is running.');
    }
  }
});
