"""
Face Detector — MTCNN-based face detection.

Uses facenet-pytorch's MTCNN for robust face detection.
Falls back to OpenCV's Haar cascade if facenet-pytorch is unavailable.
Returns bounding boxes, confidence scores, and aligned face crops.
"""

import logging
import os
import numpy as np

logger = logging.getLogger("deepscan.face_detector")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from facenet_pytorch import MTCNN
    import torch
    _HAS_MTCNN = True
except ImportError:
    _HAS_MTCNN = False
    logger.warning("facenet-pytorch not installed. Install with: pip install facenet-pytorch")

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class FaceDetector:
    """Detect faces using MTCNN (primary) or OpenCV Haar cascade (fallback)."""

    def __init__(self, device=None):
        self.backend = "none"
        self.detector = None
        self.device = device

        if _HAS_MTCNN:
            self._init_mtcnn()
        elif _HAS_CV2:
            self._init_opencv()
        else:
            logger.error("No face detection backend available!")

    def _init_mtcnn(self):
        """Initialize MTCNN face detector."""
        if self.device is None:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector = MTCNN(
            image_size=224,
            margin=40,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            select_largest=False,
            keep_all=True,
            device=self.device,
        )
        self.backend = "mtcnn"
        logger.info(f"FaceDetector initialized with MTCNN on {self.device}")

    def _init_opencv(self):
        """Fallback: OpenCV Haar cascade."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.backend = "opencv_haar"
            logger.info("FaceDetector initialized with OpenCV Haar cascade (fallback)")
        else:
            logger.error("OpenCV cascade file not found")

    def detect(self, image_input, is_bgr=False):
        """
        Detect faces in an image.

        Args:
            image_input: numpy array (BGR/RGB), PIL Image, or file path
            is_bgr: True if numpy array is in BGR format (e.g. directly from cv2.imread/VideoCapture)

        Returns:
            list of dicts: [{bbox: [x1,y1,x2,y2], confidence: float, crop: np.array}, ...]
        """
        # Convert input to RGB numpy array and PIL Image
        rgb_array, pil_image = self._prepare_input(image_input, is_bgr=is_bgr)

        if rgb_array is None:
            return []

        if self.backend == "mtcnn":
            return self._detect_mtcnn(pil_image, rgb_array)
        elif self.backend == "opencv_haar":
            return self._detect_opencv(rgb_array)
        else:
            return []

    def _prepare_input(self, image_input, is_bgr=False):
        """Convert various input types to RGB numpy array and PIL Image."""
        rgb_array = None
        pil_image = None

        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                if is_bgr:
                    rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if _HAS_CV2 else image_input
                else:
                    rgb_array = image_input
            else:
                rgb_array = image_input
            if _HAS_PIL:
                pil_image = Image.fromarray(rgb_array)

        elif _HAS_PIL and isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            rgb_array = np.array(pil_image)

        elif isinstance(image_input, str) and os.path.exists(image_input):
            if _HAS_PIL:
                pil_image = Image.open(image_input).convert("RGB")
                rgb_array = np.array(pil_image)
            elif _HAS_CV2:
                bgr = cv2.imread(image_input)
                rgb_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        return rgb_array, pil_image

    def _detect_mtcnn(self, pil_image, rgb_array):
        """Detect faces using MTCNN with automatic fast downscaling for high-res frames."""
        try:
            orig_w, orig_h = pil_image.size
            scale = 1.0
            
            # Fast downscaling: If image is large (>720px), scale down for detection to keep it blazing fast (0.05s vs 5s)
            max_dim = max(orig_w, orig_h)
            if max_dim > 720:
                scale = 720.0 / max_dim
                detect_w = int(orig_w * scale)
                detect_h = int(orig_h * scale)
                detect_img = pil_image.resize((detect_w, detect_h), Image.Resampling.BILINEAR)
            else:
                detect_img = pil_image

            boxes, probs, landmarks = self.detector.detect(detect_img, landmarks=True)

            if boxes is None:
                return []

            results = []
            h, w = rgb_array.shape[:2]

            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.5:
                    continue

                # Scale boxes back to original dimensions
                x1, y1, x2, y2 = [int(c / scale) for c in box]
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Extract natural face crop with 20% optimal margin (capturing full facial features and boundary)
                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = int(bw * 0.20), int(bh * 0.20)
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
                crop = rgb_array[cy1:cy2, cx1:cx2]

                scaled_landmarks = None
                if landmarks is not None and i < len(landmarks) and landmarks[i] is not None:
                    scaled_landmarks = (np.array(landmarks[i]) / scale).tolist()

                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(prob),
                    "crop": crop,
                    "landmark": scaled_landmarks,
                })

            return results

        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return []

    def _detect_opencv(self, rgb_array):
        """Detect faces using OpenCV Haar cascade (fallback)."""
        try:
            gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

            results = []
            for (x, y, w, h) in faces:
                crop = rgb_array[y:y+h, x:x+w]
                results.append({
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "confidence": 0.85,  # Haar doesn't give confidence
                    "crop": crop,
                    "landmark": None,
                })

            return results

        except Exception as e:
            logger.error(f"OpenCV detection failed: {e}")
            return []
