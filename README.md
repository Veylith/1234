---
title: Deepscanai
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.20.0
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
---

# DeepScan AI — Multi-Modal Deepfake & Liveness Detection

DeepScan AI is a real-time, state-of-the-art multi-modal deepfake detection and authenticity verification platform for **Images**, **Videos**, **Audio**, and **Live Webcam Streams**.

---

## 📸 Platform Previews & Forensic Detection Detections

### 1. Image Deepfake Detection
DeepScan AI utilizes Vision Transformers (ViT) and MTCNN face detection paired with Error Level Analysis (ELA) to detect generative artifacts, facial boundary anomalies, and deepfake manipulations.

| Authentic Human Face (Authenticity: 95%) | AI-Generated Deepfake Face (Fake: 90%) |
| :---: | :---: |
| ![Real Image Scan](assets/screenshots/image_scan_real.png) | ![Fake Image Scan](assets/screenshots/image_scan_fake.png) |

---

### 2. Video Temporal Deepfake Verification
Frame-by-frame temporal analysis tracks facial landmarks across video sequences, identifying frame-to-frame jitter, blinking anomalies, compression boundaries, and neural reenactment artifacts.

| Authentic Video Sequence (Authenticity: 78%) | Deepfake Video Sequence (Fake: 84%) |
| :---: | :---: |
| ![Real Video Scan](assets/screenshots/video_scan_real.png) | ![Fake Video Scan](assets/screenshots/video_scan_fake.png) |

---

### 3. Audio Voice Clone & Synthetic Speech Detection
Powered by **AASIST** (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*) with SincNet filterbanks and acoustic spectral analysis (Wiener entropy, pitch variability, formant resonance).

| Authentic Human Speech (Authenticity: 98% Real) | AI Voice Clone / Synthetic Speech (Fake: 75% Fake) |
| :---: | :---: |
| ![Real Audio Scan](assets/screenshots/audio_scan_real.png) | ![Fake Audio Scan](assets/screenshots/audio_scan_fake.png) |

---

### 4. Real-Time Webcam Liveness & Anti-Spoofing
Active and passive anti-spoofing pipeline detecting live presence, 3D head pose movements, eye dynamics, and presentation attacks (screen replays, OLED sub-pixel grids, and printed photos).

| Live Authentic Human (Tier 1: Real Person) | Screen Replay Presentation Attack (Tier 3: Spoof Detected) |
| :---: | :---: |
| ![Live Webcam Real](assets/screenshots/webcam_scan_real.png) | ![Live Webcam Spoof](assets/screenshots/webcam_scan_spoof.png) |

---

## ✨ Key Features
- **🖼️ Multi-Modal Forensics**: Instant analysis of Images (JPEG, PNG, WebP), Videos (MP4, WebM, AVI), Audio (WAV, MP3, OGG, FLAC), and real-time Webcam.
- **🧠 Advanced Neural Backbones**:
  - **Vision Transformer (ViT)** fine-tuned for facial manipulation detection.
  - **AASIST** graph attention networks for voice-cloning and synthetic TTS identification.
  - **MTCNN** high-accuracy face extraction and alignment.
- **🔬 Classical Signal & Spectral Forensics**:
  - Error Level Analysis (ELA) compression discrepancy mapping.
  - Frequency domain FFT analysis and Wiener spectral entropy.
  - Pitch prosody contour variability and Moiré pattern scanning.
- **🛡️ Multi-Tier Liveness Anti-Spoofing**:
  - Real-time 3D head pose estimation (Yaw/Pitch/Roll).
  - Eye blink dynamic tracking (Eye Aspect Ratio - EAR).
  - High-frequency OLED/LCD raster & bezel boundary detection.
- **🎨 Modern Obsidian Glassmorphism UI**:
  - Interactive glowing gauge, dynamic authenticity score calculations, and real-time confidence tracking.
  - Dark/Light theme switching and smooth tabbed modal workflows.

---

## 🚀 Quick Start Guide (How to Run)

### Prerequisites
- **Python 3.9 – 3.12** installed on your system.
- Ensure `pip` is updated (`python -m pip install --upgrade pip`).

---

### Step 1: Open Terminal / PowerShell in the Project Folder
```bash
cd deepscan-ai
```

---

### Step 2: Create & Activate Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
*(If PowerShell gives a script execution policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again)*

#### On Windows (Command Prompt `cmd`):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Step 4: Start the Server
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Step 5: Open the Application
Open your web browser and navigate to:
👉 **`http://localhost:8000`**

The web app is served directly from the root with all frontend detection tabs (Image, Video, Audio, Webcam) fully operational!

---

## 📡 API Endpoints
- `GET  /api/health` — Health check and loaded model diagnostics
- `POST /api/detect/image` — Deepfake image detection (multipart file upload)
- `POST /api/detect/video` — Frame-by-frame temporal video detection
- `POST /api/detect/audio` — Synthetic speech & voice-clone detection
- `POST /api/detect/webcam` — Real-time frame & liveness anti-spoofing
