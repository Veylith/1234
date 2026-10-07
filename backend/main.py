"""
DeepScan AI — FastAPI Backend

Endpoints:
  POST /api/detect/image   → Image deepfake detection (EfficientNet-B4 + MTCNN)
  POST /api/detect/video   → Video deepfake detection (frame-by-frame)
  POST /api/detect/audio   → Audio deepfake detection (Wav2Vec2)
  WS   /ws/webcam          → Real-time webcam analysis (dual-signal)
  GET  /api/health          → Model status and health check

Models are loaded LAZILY on first request to keep startup fast.
"""

import sys
import os
import logging
import json
import threading

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("deepscan.api")

app = FastAPI(
    title="DeepScan AI",
    description="Multimodal deepfake detection API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model registry ──
_models = {}
_model_lock = threading.Lock()


def get_face_detector():
    if "face_detector" not in _models:
        with _model_lock:
            if "face_detector" not in _models:
                from models.face_detector import FaceDetector
                _models["face_detector"] = FaceDetector()
                logger.info(f"✓ FaceDetector [{_models['face_detector'].backend}]")
    return _models["face_detector"]


def get_image_classifier():
    if "image_classifier" not in _models:
        with _model_lock:
            if "image_classifier" not in _models:
                from models.image_classifier import ImageClassifier
                _models["image_classifier"] = ImageClassifier()
                logger.info(f"✓ ImageClassifier [{_models['image_classifier'].mode}]")
    return _models["image_classifier"]


def get_deepfake_detector():
    if "deepfake_detector" not in _models:
        with _model_lock:
            if "deepfake_detector" not in _models:
                from models.deepfake_detector import DeepfakeDetector
                _models["deepfake_detector"] = DeepfakeDetector(image_classifier=get_image_classifier())
                logger.info("✓ DeepfakeDetector")
    return _models["deepfake_detector"]


def get_liveness_detector():
    if "liveness_detector" not in _models:
        with _model_lock:
            if "liveness_detector" not in _models:
                from models.liveness_detector import LivenessDetector
                _models["liveness_detector"] = LivenessDetector()
                logger.info("✓ LivenessDetector")
    return _models["liveness_detector"]


def get_video_detector():
    if "video_detector" not in _models:
        with _model_lock:
            if "video_detector" not in _models:
                from models.video_detector import VideoDetector
                _models["video_detector"] = VideoDetector(
                    get_face_detector(), get_image_classifier(), get_deepfake_detector()
                )
                logger.info("✓ VideoDetector")
    return _models["video_detector"]


def get_audio_detector():
    if "audio_detector" not in _models:
        with _model_lock:
            if "audio_detector" not in _models:
                from models.audio_detector import AudioDetector
                _models["audio_detector"] = AudioDetector()
                logger.info(f"✓ AudioDetector [{_models['audio_detector'].mode}]")
    return _models["audio_detector"]


def get_webcam_pipeline():
    if "webcam_pipeline" not in _models:
        with _model_lock:
            if "webcam_pipeline" not in _models:
                from models.webcam_pipeline import WebcamPipeline
                _models["webcam_pipeline"] = WebcamPipeline(
                    get_face_detector(), get_liveness_detector(), get_deepfake_detector()
                )
                logger.info("✓ WebcamPipeline")
    return _models["webcam_pipeline"]


# ── Startup: pre-warm models in background thread & open browser ──
@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("DeepScan AI — pre-warming models in background...")
    logger.info("=" * 60)
    import threading
    threading.Thread(target=_prewarm, daemon=True).start()
    threading.Thread(target=_open_browser, daemon=True).start()


def _prewarm():
    """Load models in background so first requests are instant."""
    # In cloud environments with strict RAM limits (e.g. Render Free 512MB),
    # skip pre-warming all 7 heavy models simultaneously to prevent OOM crash.
    if os.environ.get("RENDER") or os.environ.get("LOW_MEMORY") or os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE"):
        logger.info("Cloud environment detected. Skipping bulk pre-warming; models will load on demand under GPU context.")
        try:
            get_face_detector()
            logger.info("✓ Lightweight FaceDetector initialized.")
        except Exception as e:
            logger.warning(f"FaceDetector init note: {e}")
        return

    try:
        logger.info("Pre-warm: loading FaceDetector...")
        get_face_detector()
        logger.info("Pre-warm: loading ImageClassifier...")
        get_image_classifier()
        logger.info("Pre-warm: loading DeepfakeDetector...")
        get_deepfake_detector()
        logger.info("Pre-warm: loading LivenessDetector...")
        get_liveness_detector()
        logger.info("Pre-warm: loading VideoDetector...")
        get_video_detector()
        logger.info("Pre-warm: loading AudioDetector...")
        get_audio_detector()
        logger.info("Pre-warm: loading WebcamPipeline...")
        get_webcam_pipeline()
        logger.info("✅ All models pre-warmed and ready!")
    except Exception as e:
        logger.error(f"Pre-warm error (non-fatal): {e}")


def _open_browser():
    """Wait for server to become responsive, then open Chrome or default browser."""
    if os.environ.get("RENDER") or os.environ.get("HEADLESS") or os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE"):
        return

    import time
    import urllib.request
    import webbrowser
    import subprocess

    url = "http://localhost:8000"

    # Avoid opening duplicate tabs during quick uvicorn reloads
    flag_file = os.path.join(os.path.dirname(__file__), ".browser_opened")
    if os.path.exists(flag_file):
        try:
            mtime = os.path.getmtime(flag_file)
            if time.time() - mtime < 12:
                return
        except Exception:
            pass

    # Wait for the server to be ready and responding
    for _ in range(35):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass

    try:
        with open(flag_file, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass

    print("\n" + "=" * 62)
    print("  🚀 DeepScan AI Server Active & Ready!")
    print(f"  👉 Clickable Link: http://localhost:8000")
    print(f"  🌐 Localhost:     http://127.0.0.1:8000")
    print("  Opening automatically in your Chrome browser...")
    print("=" * 62 + "\n")

    # Try Google Chrome first on Windows
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
    ]
    opened = False
    for cp in chrome_paths:
        if os.path.exists(cp):
            try:
                subprocess.Popen([cp, url])
                opened = True
                break
            except Exception:
                pass

    if not opened:
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not automatically open browser: {e}")


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    loaded = list(_models.keys())
    return {
        "status": "ok",
        "loaded_models": loaded,
        "models": {
            "face_detector": _models["face_detector"].backend if "face_detector" in _models else "not loaded",
            "image_classifier": _models["image_classifier"].mode if "image_classifier" in _models else "not loaded",
            "deepfake_detector": "loaded" if "deepfake_detector" in _models else "not loaded",
            "liveness_detector": "loaded" if "liveness_detector" in _models else "not loaded",
            "video_detector": "loaded" if "video_detector" in _models else "not loaded",
            "audio_detector": _models["audio_detector"].mode if "audio_detector" in _models else "not loaded",
            "webcam_pipeline": "loaded" if "webcam_pipeline" in _models else "not loaded",
        },
    }


# ──────────────────────────────────────────────
# Image Detection
# ──────────────────────────────────────────────

def _process_image(contents, filename):
    from PIL import Image
    import io
    import numpy as np
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(pil_image)

    fd = get_face_detector()
    ic = get_image_classifier()

    faces = fd.detect(img_array, is_bgr=False)
    result = ic.classify(pil_image, faces=faces)
    result["filename"] = filename
    return result


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        import asyncio
        result = await asyncio.to_thread(_process_image, contents, file.filename)
        return result
    except Exception as e:
        logger.error(f"Image detection failed: {e}", exc_info=True)
        return {"error": str(e), "fake_probability": 0.5}


# ──────────────────────────────────────────────
# Video Detection
# ──────────────────────────────────────────────

def _process_video(contents, filename):
    vd = get_video_detector()
    result = vd.analyze(contents, filename=filename)
    result["filename"] = filename
    return result


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        import asyncio
        result = await asyncio.to_thread(_process_video, contents, file.filename)
        return result
    except Exception as e:
        logger.error(f"Video detection failed: {e}", exc_info=True)
        return {"error": str(e), "fake_probability": 0.5}


# ──────────────────────────────────────────────
# Audio Detection
# ──────────────────────────────────────────────

def _process_audio(contents, filename):
    ad = get_audio_detector()
    import asyncio
    # audio detector analyze may be async or sync
    import inspect
    if inspect.iscoroutinefunction(ad.analyze):
        import asyncio
        return asyncio.run(ad.analyze(contents, filename=filename))
    else:
        return ad.analyze(contents, filename=filename)


@app.post("/api/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        import asyncio
        ad = get_audio_detector()
        result = await ad.analyze(contents, filename=file.filename)
        result["filename"] = file.filename
        return result
    except Exception as e:
        logger.error(f"Audio detection failed: {e}", exc_info=True)
        return {"error": str(e), "fake_probability": 0.5}


# ──────────────────────────────────────────────
# Webcam WebSocket
# ──────────────────────────────────────────────

@app.websocket("/ws/webcam")
async def websocket_webcam(ws: WebSocket):
    await ws.accept()
    logger.info("Webcam WebSocket connected")

    wp = get_webcam_pipeline()
    wp.reset_session()

    try:
        while True:
            message = await ws.receive()
            if "bytes" in message and message["bytes"]:
                frame_bytes = message["bytes"]
                result = wp.analyze_frame(frame_bytes)
                await ws.send_text(json.dumps(result))
            elif "text" in message and message["text"]:
                try:
                    cmd = json.loads(message["text"])
                    cmd_type = cmd.get("type", "")
                    res = wp.handle_command(cmd_type, cmd)
                    await ws.send_text(json.dumps({"command_response": True, "type": cmd_type, "data": res}))
                except Exception as ex:
                    logger.warning(f"Error handling WebSocket text cmd: {ex}")
            elif message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.info("Webcam WebSocket disconnected")
        wp.reset_session()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.close()
        except Exception:
            pass

import os
from fastapi.staticfiles import StaticFiles

# Serve the static frontend at the root so we can share it easily over a single port/tunnel
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
