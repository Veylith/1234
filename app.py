"""
DeepScan AI — Hugging Face Space Application
Serves the custom Cyberpunk Obsidian Web Dashboard at root `/`
with full FastAPI REST endpoints and WebSockets on port 7860.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deepscan.app")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.main import app

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 60)
    logger.info("Starting DeepScan AI on port 7860...")
    logger.info("Serving custom Obsidian Cyberpunk Web Dashboard at root `/`")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=7860)
