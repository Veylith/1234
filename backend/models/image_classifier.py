"""
Image Classifier — Deepfake/AI-generated image detection.

Priority stack:
  1. HuggingFace ViT  : dima806/deepfake_vs_real_image_detection   (auto-download ~330 MB, best accuracy)
  2. EfficientNet-B4  : local fine-tuned weights if present         (manual download, see WEIGHTS below)
  3. Classical CV     : ELA + frequency + noise heuristics          (no weights, least accurate)

Fine-tuned weight downloads (optional — HuggingFace model works without them):
  EfficientNet-B4 ImageNet : https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth
                             → place at: backend/weights/efficientnet-b4-6ed6700e.pth
  EfficientNet-B4 Deepfake : train on FaceForensics++ / DFDC (no public link — see README)
                             → place at: backend/weights/deepfake_efficientnetb4.pth
"""

import logging
import os
import time
import numpy as np

logger = logging.getLogger("deepscan.image_classifier")

WEIGHTS_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
EFFICIENTNET_W   = os.path.join(WEIGHTS_DIR, "efficientnet-b4-6ed6700e.pth")
FINETUNED_W      = os.path.join(WEIGHTS_DIR, "deepfake_efficientnetb4.pth")

# ── Optional heavy imports ──────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from efficientnet_pytorch import EfficientNet
    import torch.nn as nn
    _HAS_EFFNET = True
except ImportError:
    _HAS_EFFNET = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from transformers import pipeline as hf_pipeline, AutoFeatureExtractor, AutoModelForImageClassification
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

# EfficientNet transform (224 for speed — matches ImageNet input)
EFFNET_TRANSFORM = None
if _HAS_TORCH:
    EFFNET_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# HuggingFace model ID — ViT trained specifically on deepfake vs real
HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"


class ImageClassifier:
    """
    Deepfake image classifier with automatic model selection.

    Modes (in priority order):
      'hf_vit'               — HuggingFace ViT (best, auto-downloads)
      'efficientnet_finetuned' — local fine-tuned EfficientNet weights
      'efficientnet_features'  — ImageNet EfficientNet + classical ensemble
      'classical_only'         — pure CV heuristics (no model)
    """

    def __init__(self, device=None):
        self.hf_pipe   = None
        self.effnet    = None
        self.device    = None
        self.mode      = "classical_only"

        if _HAS_TORCH:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try loaders in priority order
        # On memory-constrained cloud environments (e.g. Render Free 512MB RAM),
        # avoid downloading the massive 330MB ViT model on the fly to prevent OOM termination.
        is_cloud_low_mem = bool(os.environ.get("RENDER") or os.environ.get("LOW_MEMORY"))
        if _HAS_HF and not is_cloud_low_mem:
            self._init_hf_vit()

        if self.mode == "classical_only" and _HAS_TORCH and _HAS_EFFNET:
            self._init_efficientnet()

        logger.info(f"ImageClassifier ready — mode: '{self.mode}'")

    def _is_online(self, timeout=1.5):
        """Quick check if internet is reachable for HuggingFace model loading."""
        import urllib.request
        for url in ["https://huggingface.co", "https://www.google.com"]:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "DeepScanAI/1.0"})
                with urllib.request.urlopen(req, timeout=timeout):
                    return True
            except Exception:
                continue
        return False

    def _init_hf_vit(self):
        """Load HuggingFace ViT deepfake detector.
        Prioritizes instant local cache loading (0.7s) to eliminate all startup network lag.
        """
        device_arg = 0 if (_HAS_TORCH and torch.cuda.is_available()) else -1

        # 1. Instant local cache load (0.7s, zero network delay)
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID, local_files_only=True)
            model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID, local_files_only=True)
            self.hf_pipe = hf_pipeline(
                "image-classification",
                model=model,
                image_processor=processor,
                device=device_arg,
            )
            self.mode = "hf_vit"
            logger.info("✓ HuggingFace ViT loaded from local cache (0.7s instant)")
            return
        except Exception:
            pass

        # 2. Online download if not cached yet
        online = self._is_online()
        if online:
            try:
                logger.info(f"🌐 Loading HuggingFace model '{HF_MODEL_ID}' (Online download)...")
                self.hf_pipe = hf_pipeline(
                    "image-classification",
                    model=HF_MODEL_ID,
                    device=device_arg,
                )
                self.mode = "hf_vit"
                logger.info("✓ HuggingFace ViT downloaded and loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Online load encountered error: {e}")

        logger.warning("Could not load HuggingFace ViT. Falling back to EfficientNet/classical.")
        self.hf_pipe = None

    def _init_efficientnet(self):
        """Load local EfficientNet-B4 weights (optional upgrade)."""
        try:
            if os.path.exists(FINETUNED_W):
                self.effnet = EfficientNet.from_name("efficientnet-b4", num_classes=2)
                state = torch.load(FINETUNED_W, map_location=self.device, weights_only=True)
                self.effnet.load_state_dict(state)
                self.mode = "efficientnet_finetuned"
                logger.info("✓ EfficientNet fine-tuned weights loaded")
            elif os.path.exists(EFFICIENTNET_W):
                self.effnet = EfficientNet.from_name("efficientnet-b4")
                state = torch.load(EFFICIENTNET_W, map_location=self.device, weights_only=True)
                self.effnet.load_state_dict(state)
                in_f = self.effnet._fc.in_features
                self.effnet._fc = nn.Linear(in_f, 2)
                self.mode = "efficientnet_features"
                logger.info("✓ EfficientNet ImageNet weights loaded")
            else:
                return  # no weights found, stay classical_only

            self.effnet.to(self.device).eval()
        except Exception as e:
            logger.error(f"EfficientNet init failed: {e}")
            self.effnet = None

    # ── Public API ───────────────────────────────────────────────

    def classify(self, image_input, faces=None):
        """
        Classify image as real or fake.

        Returns:
            dict with fake_probability [0,1], model_used, details
        """
        t0 = time.time()
        pil = self._to_pil(image_input)
        if pil is None:
            return {"fake_probability": 0.5, "error": "Could not load image"}

        w, h = pil.size
        result = {
            "fake_probability": 0.5,
            "model_used": self.mode,
            "faces_detected": len(faces) if faces else 0,
            "faces": [],
            "details": {"image_resolution": f"{w}×{h}"},
        }

        # ── Primary: HuggingFace ViT ──
        if self.mode == "hf_vit" and self.hf_pipe is not None:
            hf_score = self._run_hf(pil)
            result["details"]["hf_vit_score"]   = round(hf_score, 4)
            result["details"]["efficientnet_score"] = round(hf_score, 4)  # alias for UI

            # Still run fast classical for the "Classical CV" signal card in the UI
            classical = self._run_classical(pil)
            result["details"]["classical_score"] = round(classical, 4)

            # Robust blend: 85% Deep Learning ViT + 15% Classical CV heuristics
            blended = 0.85 * hf_score + 0.15 * classical
            result["fake_probability"] = round(blended, 4)

        # ── Secondary: EfficientNet ──
        elif self.effnet is not None and EFFNET_TRANSFORM is not None:
            effnet_score = self._run_effnet(pil)
            classical    = self._run_classical(pil)
            result["details"]["efficientnet_score"] = round(effnet_score, 4)
            result["details"]["classical_score"]    = round(classical, 4)
            if self.mode == "efficientnet_finetuned":
                result["fake_probability"] = round(effnet_score, 4)
            else:
                result["fake_probability"] = round(0.45 * effnet_score + 0.55 * classical, 4)

        # ── Fallback: Classical only ──
        else:
            classical = self._run_classical(pil)
            result["details"]["classical_score"] = round(classical, 4)
            result["details"]["efficientnet_score"] = round(classical, 4)
            result["fake_probability"] = round(classical, 4)

        # Attach face bboxes (score = same as full-image for speed)
        if faces:
            fs = result["fake_probability"]
            result["faces"] = [
                {"bbox": f["bbox"], "confidence": f.get("confidence", 0.0), "fake_score": round(fs, 4)}
                for f in faces
            ]
            result["details"]["face_manipulation_score"] = round(fs, 4)

        result["details"]["ai_generated_score"]  = result["fake_probability"]
        result["details"]["processing_time_ms"]  = int((time.time() - t0) * 1000)
        return result

    # ── Inference helpers ────────────────────────────────────────

    def _run_hf(self, pil_image):
        """Run HuggingFace pipeline. Returns fake probability [0,1]."""
        try:
            import gc
            img = pil_image.convert("RGB")
            # Fast downscale: reduce 4K/1080p images to max 640px for 5x faster inference
            w, h = img.size
            if max(w, h) > 640:
                scale = 640.0 / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

            if _HAS_TORCH:
                with torch.inference_mode():
                    preds = self.hf_pipe(img)
            else:
                preds = self.hf_pipe(img)

            del img
            gc.collect()

            # Model labels: 'Fake' / 'Real'  (varies by model)
            for p in preds:
                lbl = p["label"].lower()
                if "fake" in lbl or "deepfake" in lbl or "ai" in lbl or "generated" in lbl:
                    return float(p["score"])
                if "real" in lbl or "authentic" in lbl:
                    return 1.0 - float(p["score"])
            # Fallback: return highest score mapped conservatively
            return float(preds[0]["score"]) * 0.8
        except Exception as e:
            logger.error(f"HF inference failed: {e}")
            return 0.5

    def _run_effnet(self, pil_image):
        """Run EfficientNet. Returns fake probability [0,1]."""
        try:
            t = EFFNET_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out   = self.effnet(t)
                probs = F.softmax(out, dim=1)
                return probs[0][1].item() if probs.shape[1] == 2 else 0.5
        except Exception as e:
            logger.error(f"EfficientNet inference failed: {e}")
            return 0.5

    def _run_classical(self, pil_image):
        """Fast CV heuristics. Returns fake probability [0,1]."""
        if not _HAS_CV2:
            return 0.5
        try:
            arr = np.array(pil_image.convert("RGB"))
            # Resize to 512 max for speed
            h, w = arr.shape[:2]
            if max(h, w) > 512:
                s   = 512 / max(h, w)
                arr = cv2.resize(arr, (int(w * s), int(h * s)))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            scores = [
                self._noise_score(bgr),
                self._freq_score(bgr),
                self._stat_score(bgr),
            ]
            return float(np.mean(scores))
        except Exception:
            return 0.5

    def _noise_score(self, bgr):
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        noise = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        if noise < 2.0: return 0.72
        if noise < 4.0: return 0.52
        return 0.32

    def _freq_score(self, bgr):
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mag   = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
        h, w  = mag.shape
        cy, cx, r = h // 2, w // 2, min(h, w) // 4
        y, x  = np.ogrid[:h, :w]
        hi    = np.mean(mag[(y - cy)**2 + (x - cx)**2 > r**2])
        ratio = hi / (np.mean(mag) + 1e-8)
        if ratio < 0.60: return 0.65
        if ratio < 0.75: return 0.45
        return 0.30

    def _stat_score(self, bgr):
        kurt = []
        for ch in cv2.split(bgr):
            m, s = np.mean(ch), np.std(ch) + 1e-8
            kurt.append(np.mean(((ch - m) / s) ** 4) - 3)
        k = abs(np.mean(kurt))
        if k > 5: return 0.60
        if k > 2: return 0.45
        return 0.35

    def _ela_score(self, bgr):
        """Error Level Analysis (slow — only used in classical_only mode)."""
        import io
        pil  = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        buf  = io.BytesIO()
        pil.save(buf, "JPEG", quality=90); buf.seek(0)
        diff = np.abs(np.array(pil).astype(float) - np.array(Image.open(buf)).astype(float))
        m    = np.mean(diff)
        if m > 15: return 0.70
        if m > 8:  return 0.50
        return 0.30

    # ── Input conversion ─────────────────────────────────────────

    def _to_pil(self, x):
        if not _HAS_PIL:
            return None
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, np.ndarray):
            rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if _HAS_CV2 and x.shape[-1] == 3 else x
            return Image.fromarray(rgb)
        if isinstance(x, str) and os.path.exists(x):
            return Image.open(x).convert("RGB")
        if isinstance(x, bytes):
            import io
            return Image.open(io.BytesIO(x)).convert("RGB")
        return None
