"""
DeepScan AI — Hugging Face Space Application
Serves the custom Cyberpunk Obsidian Web Dashboard at root `/`
with ZeroGPU acceleration and full FastAPI REST endpoints.
"""

import os
import sys
import time
import io
import json
import logging
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deepscan.app")

# Path configuration
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Hugging Face ZeroGPU safe compatibility
try:
    import spaces
    _HAS_SPACES = True
    logger.info("✓ Hugging Face Spaces ZeroGPU module detected.")
except ImportError:
    _HAS_SPACES = False
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60):
            def decorator(f):
                return f
            if func is not None and callable(func):
                return func
            return decorator

try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError:
    gr = None
    _HAS_GRADIO = False

from fastapi.staticfiles import StaticFiles

# Import FastAPI application and model accessors
from backend.main import (
    app as fastapi_app,
    get_face_detector,
    get_image_classifier,
    get_deepfake_detector,
    get_audio_detector,
    get_video_detector,
)

# Ensure the custom Cyberpunk Obsidian web frontend is mounted at root `/`
frontend_dir = ROOT_DIR
# Note: backend/main.py already mounted frontend_dir at "/", but we ensure it here as well
try:
    fastapi_app.mount("/web", StaticFiles(directory=frontend_dir, html=True), name="web_dashboard")
except Exception:
    pass


# ─────────────────────────────────────────────────────────────
# Optional Secondary Gradio Interface (Mounted at /gradio)
# ─────────────────────────────────────────────────────────────

def build_gradio_app():
    if not _HAS_GRADIO:
        return None

    @spaces.GPU
    def analyze_image_gradio(input_img):
        if input_img is None:
            return "⚠️ Please upload an image or take a photo.", None, None, {}
        try:
            pil_img = Image.fromarray(input_img) if isinstance(input_img, np.ndarray) else input_img
            face_detector = get_face_detector()
            faces = face_detector.detect_faces(pil_img)
            deepfake_detector = get_deepfake_detector()
            image_classifier = get_image_classifier()
            np_frame = np.array(pil_img)
            df_result = deepfake_detector.analyze(np_frame, faces)
            ela_img = image_classifier.compute_ela(pil_img)
            face_crop = None
            if faces and len(faces) > 0 and faces[0].get("crop") is not None:
                face_crop = Image.fromarray(faces[0]["crop"])
            score = df_result.get("authenticity_score", 0.0)
            verdict = df_result.get("authenticity_verdict", "Analysis Complete")
            is_fake = score >= 0.5
            conf_pct = round(score * 100, 1) if is_fake else round((1 - score) * 100, 1)
            status_color = "#ef4444" if is_fake else "#10b981"
            badge = "🚨 DEEPFAKE DETECTED" if is_fake else "✅ AUTHENTIC HUMAN MEDIA"
            summary_md = f"### <span style='color: {status_color};'>{badge}</span>\n**Verdict**: {verdict}\n**Confidence**: {conf_pct}%"
            signals = df_result.get("authenticity_signals", {})
            return summary_md, face_crop, ela_img, json.dumps(signals, indent=2)
        except Exception as e:
            return f"❌ Error: {str(e)}", None, None, {"error": str(e)}

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), title="DeepScan AI") as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 16px; background: #0f172a; border-radius: 12px; margin-bottom: 12px;">
            <h2 style="color: #38bdf8; margin: 0 0 6px 0;">🛡️ DeepScan AI — Neural Inspector</h2>
            <p style="color: #94a3b8; margin: 0 0 10px 0;">Backup Gradio Inspector Interface</p>
            <a href="/" target="_self" style="color: #38bdf8; font-weight: bold; text-decoration: underline; font-size: 1.1rem;">👉 Return to Full Cyberpunk Web Dashboard</a>
        </div>
        """)
        with gr.Tabs():
            with gr.TabItem("📸 Image Forensic"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(type="pil", label="Upload Image")
                        btn = gr.Button("Analyze", variant="primary")
                    with gr.Column():
                        res_md = gr.Markdown()
                        with gr.Row():
                            crop_out = gr.Image(label="Face Crop", type="pil")
                            ela_out = gr.Image(label="ELA Heatmap", type="pil")
                        sig_out = gr.JSON(label="Signals")
                btn.click(fn=analyze_image_gradio, inputs=[img_in], outputs=[res_md, crop_out, ela_out, sig_out])
    return demo


# ─────────────────────────────────────────────────────────────
# Server Execution: Root `/` Serves the User's Web Dashboard!
# ─────────────────────────────────────────────────────────────

demo = build_gradio_app()
if demo is not None:
    # Mount Gradio at /gradio so root `/` serves YOUR custom Cyberpunk Web Dashboard (index.html)
    app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
else:
    app = fastapi_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
