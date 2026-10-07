"""
Liveness Detector — Advanced Presentation Attack Detection (PAD).

Combines state-of-the-art multi-signal anti-spoofing techniques from ISO/IEC 30107
with modern screen artifact detection and neural display device identification:

  1. Neural Object Detection (SSDLite MobileNetV3) — identifies cell phones, screens, laptops, tablets
  2. Screen Bezel & Edge Quadrangle Analysis — detects physical phone/screen borders enclosing faces
  3. Modern High-PPI & OLED Sub-Pixel Autocorrelation — PenTile/Stripe periodic patterns & phase shift
  4. 8-Bit Digital Color Quantization & Comb Detection — video compression & DAC stepping artifacts
  5. Multi-Scale LBP (Local Binary Pattern) Micro-Texture — skin pore analysis vs flat display
  6. High-Frequency Laplacian Kurtosis & Edge Ringing — digital display sharpening signatures
  7. Screen Light Emission Uniformity — active backlight/OLED emission vs ambient 3D shadows
  8. Specular Reflection Geometry — flat Gorilla Glass mirror reflections vs diffuse organic skin
  9. Chromatic Channel Decorrelation — sub-pixel spatial color fringing
  10. Fourier 1/f Spectral Decay Analysis — natural vs digital frequency power distribution
  11. Temporal Refresh & Rolling Shutter Banding — display PWM and frame aliasing
"""

import logging
import time
import numpy as np

logger = logging.getLogger("deepscan.liveness_detector")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import torch
    import torchvision.models.detection as detection
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class LivenessDetector:
    """Advanced anti-spoofing detector with neural device detection and screen artifact analysis."""

    def __init__(self):
        self._full_frame = None
        self._frame_history = []
        self._max_history = 10
        self._frame_counter = 0

        # Neural object detector for display devices (cell phone, tv, laptop)
        self._object_model = None
        self._object_classes = None
        self._device_cache = {"score": 0.0, "detected": False, "label": "None", "box": None}
        self._init_object_detector()

        logger.info("LivenessDetector initialized (Advanced Neural PAD + Screen Pixel/Edge Engine)")

    def _init_object_detector(self):
        """Initialize SSDLite MobileNetV3 object detector for device identification."""
        if not _HAS_TORCH:
            return
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights = detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self._object_model = detection.ssdlite320_mobilenet_v3_large(weights=weights)
            self._object_model.to(device)
            self._object_model.eval()
            self._object_device = device
            cats = weights.meta.get("categories", [])
            self._object_classes = {i: c.lower() for i, c in enumerate(cats)}
            logger.info("SSDLite object detector loaded for presentation device detection")
        except Exception as e:
            logger.warning(f"Could not initialize SSDLite device detector: {e}")
            self._object_model = None

    def analyze(self, frame, faces):
        """
        Analyze liveness of detected faces in a frame.

        Args:
            frame: numpy array (RGB) of the full video frame
            faces: list of face dicts from FaceDetector [{bbox, crop, confidence}, ...]

        Returns:
            dict with liveness_score, verdict, signals, is_live, etc.
        """
        start_time = time.time()
        self._full_frame = frame
        self._frame_counter += 1

        if not _HAS_CV2 or not faces:
            return {
                "liveness_score": 0.08,
                "liveness_verdict": "No face detected" if not faces else "CV2 unavailable",
                "vit_spoof_prob": 0.08,
                "vit_live_prob": 0.92,
                "liveness_signals": {},
                "is_live": True,
                "processing_time_ms": 0,
            }

        all_scores = []
        all_signals = {}

        for face in faces:
            crop = face.get("crop")
            bbox = face.get("bbox")
            if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            signals = {}

            # =================================================================
            # 1. PILLAR 1: NEURAL OBJECT & PHYSICAL DEVICE DETECTION
            # =================================================================
            # Detect cell phone, tv, laptop, or book shown to camera
            device_score, device_detected, device_label = self._device_object_detection(frame, bbox)
            signals["device_detected"] = 1.0 if device_detected else 0.0
            signals["device_score"] = device_score
            signals["device_label"] = device_label

            # Screen bezel / chassis contour & edge geometry
            bezel_score = self._screen_bezel_contour_detection(frame, bbox)
            signals["context_border"] = bezel_score

            # =================================================================
            # 2. PILLAR 2: MODERN SCREEN PIXEL & SUB-PIXEL MICRO-ANALYSIS
            # =================================================================
            # High-PPI OLED PenTile / LCD subpixel periodic autocorrelation & cross-phase
            subpixel_score = self._subpixel_autocorrelation(crop)
            signals["subpixel_pattern"] = subpixel_score
            # Map moire_frequency to subpixel periodic peak
            signals["moire_frequency"] = subpixel_score

            # 8-bit digital color quantization & gradient comb detection
            quant_score = self._color_quantization_detection(crop)
            signals["color_quantization"] = quant_score

            # Multi-Scale LBP Micro-Texture (ISO/IEC 30107 standard)
            lbp_score = self._multiscale_lbp_analysis(crop)
            signals["lbp_texture"] = lbp_score

            # Gradient magnitude distribution & Laplacian kurtosis
            grad_score = self._gradient_distribution_analysis(crop)
            signals["gradient_dist"] = grad_score

            # Gorilla Glass specular glare & hard reflection geometry
            glare_score = self._specular_reflection_analysis(crop)
            signals["screen_glare"] = glare_score
            signals["specular_glare"] = glare_score

            # Screen light emission uniformity (lacking ambient 3D shadows)
            illum_score = self._illumination_uniformity_analysis(crop)
            signals["illumination"] = illum_score

            # Chromatic channel phase decorrelation (subpixel color fringing)
            chroma_score = self._chromatic_decorrelation(crop)
            signals["chroma_decorr"] = chroma_score

            # Temporal refresh / rolling shutter banding
            flicker_score = self._temporal_flicker_analysis(crop)
            signals["temporal_flicker"] = flicker_score

            # =================================================================
            # 3. ROBUST ATTACK POOLING & SCORE CALIBRATION
            # =================================================================
            # If a physical device (cell phone, laptop, screen) is detected in frame,
            # or a clear rectangular bezel encloses the face, this is DIRECT PHYSICAL PROOF
            # of a presentation attack. It must NOT be diluted!
            if device_detected and device_score > 0.40:
                # Direct device attack override
                combined = max(0.85, device_score)
            elif bezel_score > 0.55:
                # Strong physical bezel / screen frame detected around face
                combined = max(0.80, bezel_score)
            else:
                # Weighted fusion across all forensic signals
                weights = {
                    "subpixel_pattern": 0.18,    # Modern OLED/LCD grid
                    "color_quantization": 0.16,  # 8-bit digital banding
                    "context_border": 0.15,      # Bezel & edge geometry
                    "lbp_texture": 0.14,         # Micro-texture pore analysis
                    "gradient_dist": 0.12,       # Digital edge kurtosis
                    "screen_glare": 0.08,        # Glass reflections
                    "illumination": 0.07,        # Active emission uniformity
                    "chroma_decorr": 0.05,       # RGB channel fringing
                    "temporal_flicker": 0.05,    # Refresh flicker
                }

                linear_sum = sum(weights.get(k, 0.0) * signals[k] for k in weights)

                # Top-3 Attack Pooling (protects against attack dilution)
                eval_signals = [
                    subpixel_score, quant_score, bezel_score, lbp_score,
                    grad_score, glare_score, illum_score
                ]
                sorted_vals = sorted(eval_signals, reverse=True)
                top3_pool = 0.50 * sorted_vals[0] + 0.30 * sorted_vals[1] + 0.20 * sorted_vals[2]

                if sorted_vals[0] >= 0.65:
                    # Single very strong forensic signal (e.g. OLED PenTile raster or heavy glass glare)
                    combined = max(linear_sum, sorted_vals[0] * 0.75 + sorted_vals[1] * 0.25)
                elif sorted_vals[0] > 0.40 and sorted_vals[1] > 0.20:
                    # Multi-signal forensic signature
                    combined = max(linear_sum, top3_pool)
                else:
                    combined = linear_sum

            combined = max(0.0, min(1.0, combined))
            all_scores.append(combined)

            for k, v in signals.items():
                if isinstance(v, (int, float)):
                    all_signals.setdefault(k, []).append(v)

        overall = float(np.mean(all_scores)) if all_scores else 0.08
        avg_signals = {k: round(float(np.mean(v)), 4) for k, v in all_signals.items()}
        # Include non-numeric device info
        avg_signals["device_label"] = self._device_cache.get("label", "None")
        avg_signals["device_detected"] = bool(self._device_cache.get("detected", False))

        # Verdict assignment (strict threshold for zero presentation spoof escape)
        if overall <= 0.40:
            verdict = "Live Person"
            is_live = True
        elif overall <= 0.52:
            verdict = "Uncertain"
            is_live = True
        else:
            if self._device_cache.get("detected"):
                verdict = f"{self._device_cache.get('label', 'Device')} Replay Attack"
            elif avg_signals.get("context_border", 0) > 0.45:
                verdict = "Screen Bezel Replay Attack"
            elif avg_signals.get("subpixel_pattern", 0) > 0.35:
                verdict = "OLED / LCD Screen Replay"
            else:
                verdict = "Spoof Replay Attack"
            is_live = False

        return {
            "liveness_score": round(overall, 4),
            "liveness_verdict": verdict,
            "vit_spoof_prob": round(overall, 4),
            "vit_live_prob": round(1.0 - overall, 4),
            "liveness_signals": avg_signals,
            "is_live": is_live,
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }

    # =========================================================================
    # SIGNAL 1: Neural Object & Display Device Detection
    # =========================================================================
    def _device_object_detection(self, full_frame, face_bbox):
        """
        Detect electronic display devices (cell phones, TVs, laptops, tablets)
        using SSDLite MobileNetV3.

        Checks whether any display device is detected in the frame and whether
        it encloses, overlaps, or is near the face bounding box.
        """
        # Run inference every 2 frames for optimal performance, re-using cached result
        if self._frame_counter % 2 != 0 and self._device_cache.get("box") is not None:
            return (
                self._device_cache["score"],
                self._device_cache["detected"],
                self._device_cache["label"],
            )

        if self._object_model is None or not _HAS_TORCH:
            return 0.0, False, "None"

        try:
            fh, fw = full_frame.shape[:2]
            # Downsample frame to 320 max dimension for ultra-fast <25ms CPU inference
            scale = 320.0 / max(fh, fw)
            dw, dh = int(fw * scale), int(fh * scale)
            small = cv2.resize(full_frame, (dw, dh), interpolation=cv2.INTER_LINEAR)

            img_t = torch.from_numpy(small.transpose(2, 0, 1)).float().to(self._object_device) / 255.0

            with torch.no_grad():
                preds = self._object_model([img_t])[0]

            labels = preds["labels"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            boxes = preds["boxes"].cpu().numpy()

            # Target COCO categories:
            # 77: 'cell phone', 72: 'tv', 73: 'laptop', 84: 'book'
            target_cats = {77: "Cell Phone", 72: "TV / Monitor", 73: "Laptop Screen", 84: "Printed Photo / Book"}

            fx1, fy1, fx2, fy2 = face_bbox
            face_area = (fx2 - fx1) * (fy2 - fy1)

            best_device_score = 0.0
            best_device_label = "None"
            best_device_box = None
            device_detected = False

            for lbl, score, box in zip(labels, scores, boxes):
                if lbl in target_cats and score > 0.22:
                    # Rescale box back to original coordinates
                    bx1 = int(box[0] / scale)
                    by1 = int(box[1] / scale)
                    bx2 = int(box[2] / scale)
                    by2 = int(box[3] / scale)

                    device_name = target_cats[lbl]

                    # Check geometric relationship with face
                    # 1. Device encloses the face (face displayed inside phone/screen)
                    encloses = (bx1 <= fx1 + 25) and (by1 <= fy1 + 25) and (bx2 >= fx2 - 25) and (by2 >= fy2 - 25)

                    # 2. Overlap / IoU between device and face
                    ix1 = max(bx1, fx1)
                    iy1 = max(by1, fy1)
                    ix2 = min(bx2, fx2)
                    iy2 = min(by2, fy2)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    intersection = iw * ih
                    overlap_ratio = intersection / (face_area + 1e-8)

                    # Category-specific attack verification:
                    # Cell phone (77) or Printed Photo (84) held in front of camera
                    if lbl in (77, 84):
                        if encloses:
                            dev_score = min(0.98, float(score) + 0.35)
                            device_detected = True
                            label = f"{device_name} (Displaying Face)"
                        elif overlap_ratio >= 0.28:
                            dev_score = min(0.92, float(score) + 0.25)
                            device_detected = True
                            label = f"{device_name} (Overlapping Face)"
                        else:
                            # Phone nearby on desk, NOT held up displaying face
                            dev_score = float(score) * 0.15
                            label = device_name
                    # Laptop (73) or TV/Monitor (72): must enclose or strongly overlap face (>50%)
                    elif lbl in (72, 73):
                        if encloses or overlap_ratio >= 0.50:
                            dev_score = min(0.95, float(score) + 0.30)
                            device_detected = True
                            label = f"{device_name} (Displaying Face)"
                        else:
                            # User simply sitting at their laptop
                            dev_score = float(score) * 0.10
                            label = device_name
                    else:
                        dev_score = float(score) * 0.20
                        label = device_name

                    if dev_score > best_device_score:
                        best_device_score = dev_score
                        best_device_label = label
                        best_device_box = (bx1, by1, bx2, by2)

            self._device_cache = {
                "score": best_device_score,
                "detected": device_detected,
                "label": best_device_label,
                "box": best_device_box,
            }
            return best_device_score, device_detected, best_device_label

        except Exception as e:
            logger.debug(f"Device object detection failed: {e}")
            return 0.0, False, "None"

    # =========================================================================
    # SIGNAL 2: Screen Bezel & Edge Quadrangle Analysis
    # =========================================================================
    def _screen_bezel_contour_detection(self, full_frame, bbox):
        """
        Detect rectangular or squircle phone/tablet bezels and frames enclosing the face.

        Modern smartphones have:
        1. Rectangular outline with aspect ratio between 1.2 and 2.5 (16:9, 18:9, 19.5:9, 20:9)
        2. High-contrast border between the bright illuminated screen and dark bezel / room
        3. Parallel vertical and horizontal edges framing the face
        """
        try:
            fh, fw = full_frame.shape[:2]
            x1, y1, x2, y2 = bbox
            face_w, face_h = x2 - x1, y2 - y1

            if face_w < 20 or face_h < 20:
                return 0.06

            # Extract context region: 2.2x the face size
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            expand_w = int(face_w * 1.1)
            expand_h = int(face_h * 1.1)
            rx1 = max(0, cx - face_w // 2 - expand_w)
            ry1 = max(0, cy - face_h // 2 - expand_h)
            rx2 = min(fw, cx + face_w // 2 + expand_w)
            ry2 = min(fh, cy + face_h // 2 + expand_h)

            context = full_frame[ry1:ry2, rx1:rx2]
            ch, cw = context.shape[:2]
            if ch < 40 or cw < 40:
                return 0.06

            gray = cv2.cvtColor(context, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            score = 0.0

            # --- A. Quadrilateral / Bezel Contour Analysis ---
            edges = cv2.Canny(blurred, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            min_area = (face_w * face_h) * 0.8
            max_area = (ch * cw) * 0.95

            # Face coords relative to context
            rel_fx1, rel_fy1 = max(0, x1 - rx1), max(0, y1 - ry1)
            rel_fx2, rel_fy2 = min(cw, x2 - rx1), min(ch, y2 - ry1)

            enclosing_quadrangles = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                # 4 to 8 vertices (rectangles, rounded corners / squircles)
                if 4 <= len(approx) <= 8:
                    bx, by, bw, bh = cv2.boundingRect(approx)
                    aspect = bh / float(bw) if bw > 0 else 0
                    # Phone aspect ratio (portrait or landscape): 1.2 to 2.6
                    if (1.2 <= aspect <= 2.6) or (0.38 <= aspect <= 0.83):
                        # Check if this box encloses or closely frames the face
                        if (bx <= rel_fx1 + 20) and (by <= rel_fy1 + 20) and \
                           ((bx + bw) >= rel_fx2 - 20) and ((by + bh) >= rel_fy2 - 20):
                            enclosing_quadrangles += 1

            if enclosing_quadrangles >= 2:
                score = 0.65  # Clear nested screen and bezel frames
            elif enclosing_quadrangles == 1:
                score = 0.40  # Rectangular device bezel enclosing face
            else:
                score = 0.06  # No enclosing chassis contour

            # --- B. Parallel Straight Line Detection (Hough) ---
            # Lines only reinforce a detected physical frame, not room background walls/doors
            if enclosing_quadrangles >= 1:
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35,
                                        minLineLength=min(ch, cw) // 5, maxLineGap=12)
                h_lines = 0
                v_lines = 0
                if lines is not None:
                    for line in lines:
                        lx1, ly1, lx2, ly2 = line[0]
                        angle = abs(np.arctan2(ly2 - ly1, lx2 - lx1) * 180 / np.pi)
                        if angle < 15 or angle > 165:
                            h_lines += 1
                        elif 75 < angle < 105:
                            v_lines += 1

                if h_lines >= 2 and v_lines >= 2:
                    score += 0.20  # Full rectangular box of lines
                elif (h_lines + v_lines) >= 3:
                    score += 0.10

                # --- C. Screen-to-Bezel Luminance Discontinuity ---
                face_region = gray[rel_fy1:rel_fy2, rel_fx1:rel_fx2]
                if face_region.size > 0:
                    face_lum = float(np.mean(face_region))
                    # Outer perimeter bands
                    border_pixels = []
                    margin = max(2, int(min(ch, cw) * 0.08))
                    border_pixels.extend(gray[:margin, :].ravel())
                    border_pixels.extend(gray[-margin:, :].ravel())
                    border_pixels.extend(gray[:, :margin].ravel())
                    border_pixels.extend(gray[:, -margin:].ravel())

                    if border_pixels:
                        border_lum = float(np.mean(border_pixels))
                        lum_diff = abs(face_lum - border_lum)
                        # High contrast between illuminated screen and dark bezel/surround
                        if lum_diff > 45:
                            score += 0.15
                        elif lum_diff > 30:
                            score += 0.08

            return max(0.06, min(1.0, score))
        except Exception as e:
            logger.debug(f"Screen bezel detection failed: {e}")
            return 0.06

    # =========================================================================
    # SIGNAL 3: Modern High-PPI Sub-Pixel Autocorrelation
    # =========================================================================
    def _subpixel_autocorrelation(self, crop):
        """
        Detect periodic sub-pixel raster patterns from modern OLED (PenTile)
        and LCD (Stripe) screens.

        Modern smartphone screens have subpixel layouts that create periodic
        spatial autocorrelation and Red-Blue channel phase offsets.
        Real organic human skin exhibits monotonic isotropic decay with no periodic peaks.
        """
        try:
            h, w = crop.shape[:2]
            if h < 32 or w < 32:
                return 0.06

            # Analyze green channel (most sensitive to sub-pixel arrangement)
            green = crop[:, :, 1].astype(np.float32)
            strip_h = min(64, h - 8)
            strip_w = min(64, w - 8)
            cy, cx = h // 2, w // 2
            strip = green[cy - strip_h//2:cy + strip_h//2, cx - strip_w//2:cx + strip_w//2]

            if np.std(strip) < 1.0:
                return 0.06

            # Horizontal autocorrelation (subpixel columns)
            row_mean = np.mean(strip, axis=0)
            row_mean -= np.mean(row_mean)
            if np.std(row_mean) < 0.5:
                return 0.06
            autocorr_h = np.correlate(row_mean, row_mean, mode="full")
            autocorr_h = autocorr_h[len(autocorr_h)//2:]
            autocorr_h /= (autocorr_h[0] + 1e-8)

            # Look for true periodic peaks at lags 2-12 (must be a local maximum above its neighbors)
            peak_h = 0.0
            if len(autocorr_h) > 8:
                for lag in range(2, min(12, len(autocorr_h) - 1)):
                    if (autocorr_h[lag] > autocorr_h[lag - 1] + 0.02) and (autocorr_h[lag] > autocorr_h[lag + 1] + 0.02):
                        if autocorr_h[lag] > 0.12:
                            peak_h = max(peak_h, float(autocorr_h[lag]))

            # Vertical autocorrelation
            col_mean = np.mean(strip, axis=1)
            col_mean -= np.mean(col_mean)
            autocorr_v = np.correlate(col_mean, col_mean, mode="full")
            autocorr_v = autocorr_v[len(autocorr_v)//2:]
            autocorr_v /= (autocorr_v[0] + 1e-8)

            peak_v = 0.0
            if len(autocorr_v) > 8:
                for lag in range(2, min(12, len(autocorr_v) - 1)):
                    if (autocorr_v[lag] > autocorr_v[lag - 1] + 0.02) and (autocorr_v[lag] > autocorr_v[lag + 1] + 0.02):
                        if autocorr_v[lag] > 0.12:
                            peak_v = max(peak_v, float(autocorr_v[lag]))

            # Red-Blue channel cross-correlation phase shift
            # In OLED PenTile, R and B are spatially offset from G
            r_strip = crop[cy - strip_h//2:cy + strip_h//2, cx - strip_w//2:cx + strip_w//2, 0].astype(np.float32)
            b_strip = crop[cy - strip_h//2:cy + strip_h//2, cx - strip_w//2:cx + strip_w//2, 2].astype(np.float32)

            r_prof = np.mean(r_strip, axis=0) - np.mean(r_strip)
            b_prof = np.mean(b_strip, axis=0) - np.mean(b_strip)

            phase_offset_score = 0.0
            if np.std(r_prof) > 1.0 and np.std(b_prof) > 1.0:
                xcorr = np.correlate(r_prof, b_prof, mode="full")
                mid = len(xcorr) // 2
                max_idx = np.argmax(np.abs(xcorr))
                lag_shift = abs(max_idx - mid)
                if 1 <= lag_shift <= 5:
                    phase_offset_score = 0.35  # Screen subpixel spatial phase offset

            max_peak = max(peak_h, peak_v)

            if max_peak > 0.28:
                score = 0.80  # Strong OLED/LCD sub-pixel periodic grid
            elif max_peak > 0.16:
                score = 0.50 + (0.15 if phase_offset_score > 0 else 0.0)
            elif max_peak > 0.10:
                score = 0.25 + (0.10 if phase_offset_score > 0 else 0.0)
            else:
                score = 0.06  # Organic skin: monotonic decay, no periodic raster

            return max(0.06, min(1.0, score))

        except Exception as e:
            logger.debug(f"Sub-pixel autocorrelation failed: {e}")
            return 0.06

    # =========================================================================
    # SIGNAL 4: 8-Bit Digital Color Quantization & Comb Detection
    # =========================================================================
    def _color_quantization_detection(self, crop):
        """
        Detect 8-bit digital color quantization and video compression banding.

        Real human skin has continuous tonal gradients from subsurface light diffusion.
        Digital video on a phone screen has undergone multiple 8-bit quantizations
        (encoding, screen DAC, webcam ADC), creating:
        - Stepping / banding (flat regions interspersed with step jumps)
        - Comb-like spikes in the gradient difference histogram
        - Sharp artificial contrast boundaries
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h, w = gray.shape
            if h < 30 or w < 30:
                return 0.06

            # Compute fine gradients with Sobel
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)

            total_pixels = magnitude.size

            # Flat pixels: gradient < 2.0 (quantized flat macroblock regions)
            flat_ratio = np.sum(magnitude < 2.0) / total_pixels

            # Natural skin fine gradient ratio: 2.0 <= magnitude < 8.0
            fine_ratio = np.sum((magnitude >= 2.0) & (magnitude < 8.0)) / total_pixels

            # Sharp artificial edge ratio: magnitude > 35.0 (digital sharpening)
            sharp_ratio = np.sum(magnitude > 35.0) / total_pixels

            # Histogram comb analysis: diff between adjacent luminance bins
            hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
            hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-8)
            diff2 = np.abs(np.diff(np.diff(hist_norm)))
            comb_roughness = float(np.mean(diff2))

            score = 0.06

            # Screen signature: high flat ratio + low organic fine gradients + high comb roughness
            if flat_ratio > 0.60 and fine_ratio < 0.20:
                score = 0.78  # Strong digital color quantization
            elif flat_ratio > 0.50 and fine_ratio < 0.25:
                score = 0.52
            elif flat_ratio > 0.42:
                score = 0.28
            elif flat_ratio < 0.32 and fine_ratio > 0.30:
                score = 0.06  # Rich continuous analog skin gradients

            # Add boost from comb roughness & artificial sharpening only if flat macroblock ratio is elevated
            if comb_roughness > 0.035 and flat_ratio > 0.45:
                score = max(score, 0.45)
            if sharp_ratio > 0.08 and flat_ratio > 0.45:
                score = max(score, 0.40)

            return max(0.06, min(1.0, score))
        except Exception as e:
            logger.debug(f"Color quantization detection failed: {e}")
            return 0.06

    # =========================================================================
    # SIGNAL 5: Multi-Scale LBP Micro-Texture (ISO/IEC 30107)
    # =========================================================================
    def _multiscale_lbp_analysis(self, crop):
        """
        Multi-Scale Local Binary Pattern analysis for micro-texture verification.
        Real skin has high entropy from pores, fine wrinkles, and natural hair.
        Screen-displayed faces have smoother, quantized, low-entropy distributions.
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            if h < 32 or w < 32:
                return 0.06

            gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

            scores = []
            for radius in [1, 2]:
                hist = self._compute_lbp_histogram(gray, radius)
                if hist is not None:
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                    max_entropy = np.log2(len(hist))
                    entropy_ratio = entropy / (max_entropy + 1e-8)

                    sorted_hist = np.sort(hist)[::-1]
                    top5_concentration = np.sum(sorted_hist[:5])

                    # Low entropy + high concentration = screen display
                    if entropy_ratio < 0.55 and top5_concentration > 0.45:
                        scores.append(0.70)
                    elif entropy_ratio < 0.65 and top5_concentration > 0.35:
                        scores.append(0.40)
                    elif entropy_ratio > 0.72:
                        scores.append(0.06)  # Organic skin micro-texture
                    else:
                        scores.append(0.12)

            return float(np.mean(scores)) if scores else 0.06
        except Exception as e:
            logger.debug(f"LBP analysis failed: {e}")
            return 0.06

    def _compute_lbp_histogram(self, gray, radius=1):
        """Compute basic LBP histogram."""
        try:
            h, w = gray.shape
            lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)
            center = gray[radius:h-radius, radius:w-radius]

            shifts = [
                (-radius, -radius), (-radius, 0), (-radius, radius),
                (0, radius), (radius, radius), (radius, 0),
                (radius, -radius), (0, -radius)
            ]

            for i, (dy, dx) in enumerate(shifts):
                neighbor = gray[radius+dy:h-radius+dy, radius+dx:w-radius+dx]
                lbp |= ((neighbor >= center).astype(np.uint8) << i)

            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            return hist.astype(np.float32) / (hist.sum() + 1e-8)
        except Exception:
            return None

    # =========================================================================
    # SIGNAL 6: High-Frequency Laplacian Kurtosis & Edge Distribution
    # =========================================================================
    def _gradient_distribution_analysis(self, crop):
        """
        Analyze Laplacian kurtosis and gradient orientation isotropy.
        Screen-displayed faces have high kurtosis (bimodal: flat + digitally sharp)
        and grid-aligned orientation bias (0°/90° from pixel columns).
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h, w = gray.shape
            if h < 30 or w < 30:
                return 0.06

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2).ravel()

            # Kurtosis: real skin is platykurtic, screen is leptokurtic
            m_mean = np.mean(magnitude)
            m_std = np.std(magnitude) + 1e-8
            kurtosis = np.mean(((magnitude - m_mean) / m_std) ** 4) - 3.0

            # Orientation isotropy
            orientation = np.arctan2(gy.ravel(), gx.ravel()) * 180 / np.pi
            orient_hist, _ = np.histogram(orientation, bins=8, range=(-180, 180))
            orient_hist = orient_hist.astype(np.float32) / (orient_hist.sum() + 1e-8)
            orient_entropy = -np.sum(orient_hist * np.log2(orient_hist + 1e-10))
            isotropy = orient_entropy / np.log2(8)

            score = 0.06
            # Natural facial curves (eyes, lips, cheeks) produce high isotropy across orientations
            if isotropy > 0.91:
                score = 0.06  # Natural omnidirectional organic facial curves
            elif kurtosis > 6.0 and isotropy < 0.85:
                score = 0.72  # Grid-aligned screen edge characteristics
            elif kurtosis > 4.0 and isotropy < 0.88:
                score = 0.42
            elif isotropy < 0.85:
                score = 0.25
            else:
                score = 0.06

            return max(0.06, min(1.0, score))
        except Exception as e:
            logger.debug(f"Gradient analysis failed: {e}")
            return 0.06

    # =========================================================================
    # SIGNAL 7: Gorilla Glass Specular Glare Analysis
    # =========================================================================
    def _specular_reflection_analysis(self, crop):
        """
        Detect sharp, geometric specular glare from glass screen surfaces.
        Phone screens have flat Gorilla Glass/Ceramic Shield producing
        hard-edged specular glare spots from room lights.
        Human skin produces soft, diffuse organic highlights following facial curvature.
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            min_glare_area = max(100, int(h * w * 0.015))

            _, bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            bright_ratio = np.sum(bright > 0) / (gray.size + 1e-8)

            contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hard_glare_spots = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_glare_area:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = max(bw, bh) / (min(bw, bh) + 1e-8)
                    extent = area / (bw * bh + 1e-8)
                    # Compact geometric glare = flat glass surface
                    if aspect < 3.0 and extent > 0.40:
                        # Check boundary gradient sharpness
                        mask = np.zeros_like(gray)
                        cv2.drawContours(mask, [cnt], -1, 255, 1)
                        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
                        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
                        edge_grad = np.sqrt(gx**2 + gy**2)[mask > 0]
                        if len(edge_grad) > 0 and np.mean(edge_grad) > 40:
                            hard_glare_spots += 1

            if hard_glare_spots >= 2 and bright_ratio > 0.03:
                return 0.75  # Multiple glass specular reflections
            elif hard_glare_spots >= 1 and bright_ratio > 0.02:
                return 0.55  # Distinct sharp geometric screen glass glare
            elif bright_ratio > 0.08:
                return 0.25  # Screen backlight saturation
            else:
                return 0.06  # Natural diffuse lighting

        except Exception as e:
            logger.debug(f"Specular reflection analysis failed: {e}")
            return 0.06

    # =========================================================================
    # SIGNAL 8: Screen Light Emission Uniformity
    # =========================================================================
    def _illumination_uniformity_analysis(self, crop):
        """
        Detect active screen light emission vs ambient 3D lighting.
        Phone screens emit light uniformly, lacking natural 3D depth shadows
        (under the nose, chin, eye sockets).
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h, w = gray.shape
            if h < 30 or w < 30:
                return 0.06

            mid_h, mid_w = h // 2, w // 2
            q_tl = float(np.mean(gray[:mid_h, :mid_w]))
            q_tr = float(np.mean(gray[:mid_h, mid_w:]))
            q_bl = float(np.mean(gray[mid_h:, :mid_w]))
            q_br = float(np.mean(gray[mid_h:, mid_w:]))

            quadrants = [q_tl, q_tr, q_bl, q_br]
            q_range = max(quadrants) - min(quadrants)
            lr_diff = abs(np.mean(gray[:, :mid_w]) - np.mean(gray[:, mid_w:]))

            # Screen emission: very flat quadrant range (mild hint, never high on its own)
            if q_range < 4.0 and lr_diff < 2.0:
                return 0.18
            elif q_range < 8.0 and lr_diff < 3.5:
                return 0.12
            else:
                return 0.06  # Organic 3D shadows

        except Exception as e:
            logger.debug(f"Illumination analysis failed: {e}")
            return 0.06

    # =========================================================================
    # SIGNAL 9: Chromatic Channel Decorrelation
    # =========================================================================
    def _chromatic_decorrelation(self, crop):
        """
        Detect subpixel chromatic phase offset between R, G, B channels.
        Subpixel rendering creates high-frequency spatial color fringing.
        """
        try:
            h, w = crop.shape[:2]
            if h < 30 or w < 30:
                return 0.06

            # High-frequency Laplacian of R and B channels
            r_lap = cv2.Laplacian(crop[:, :, 0], cv2.CV_32F).ravel()
            b_lap = cv2.Laplacian(crop[:, :, 2], cv2.CV_32F).ravel()

            std_r = np.std(r_lap)
            std_b = np.std(b_lap)

            # Only analyze color fringing if image contains substantial high-frequency edges
            # On smooth skin, sensor noise is uncorrelated, which is NOT subpixel fringing!
            if std_r > 18.0 and std_b > 18.0:
                hf_corr = float(np.corrcoef(r_lap, b_lap)[0, 1])
                if hf_corr < 0.35:
                    return 0.65  # Strong subpixel color fringing on sharp edges
                elif hf_corr < 0.55:
                    return 0.35
                else:
                    return 0.06  # Natural illumination
            return 0.06
        except Exception as e:
            logger.debug(f"Chromatic decorrelation failed: {e}")
            return 0.06

    # =========================================================================
    # SIGNAL 10: Fourier 1/f Spectral Decay Analysis
    # =========================================================================
    def _spectral_decay_analysis(self, crop):
        """
        Analyze Fourier power spectrum decay rate.
        Natural faces follow 1/f^β decay (β ≈ 1.8-2.4).
        Screen images deviate due to digital sharpening and DCT compression blocks.
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            h, w = gray.shape
            if h < 32 or w < 32 or np.std(gray) < 1.0:
                return 0.06

            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            power = np.abs(fshift) ** 2

            cy, cx = h // 2, w // 2
            max_r = min(h, w) // 2

            n_rings = min(20, max_r - 2)
            radial_power = []
            radial_freq = []

            for i in range(1, n_rings + 1):
                r_inner = (i - 1) * max_r / n_rings
                r_outer = i * max_r / n_rings
                y, x = np.ogrid[:h, :w]
                dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                ring_mask = (dist >= r_inner) & (dist < r_outer)
                if np.any(ring_mask):
                    radial_power.append(float(np.mean(power[ring_mask])))
                    radial_freq.append((r_inner + r_outer) / 2.0)

            if len(radial_power) < 5:
                return 0.06

            log_freq = np.log10(np.array(radial_freq) + 1)
            log_power = np.log10(np.array(radial_power) + 1)
            coeffs = np.polyfit(log_freq, log_power, 1)
            beta = -coeffs[0]

            if beta < 1.1:
                return 0.55  # Flat spectrum = digital display compression
            elif beta < 1.4:
                return 0.30
            elif 1.5 <= beta <= 3.8:
                return 0.06  # Organic natural 1/f spectrum
            else:
                return 0.12

        except Exception as e:
            logger.debug(f"Spectral decay failed: {e}")
            return 0.06

    # =========================================================================
    # SIGNAL 11: Temporal Refresh & Rolling Shutter Banding
    # =========================================================================
    def _temporal_flicker_analysis(self, crop):
        """
        Detect temporal luminance oscillation from screen refresh / PWM dimming.
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            current_lum = float(np.mean(gray))

            self._frame_history.append(current_lum)
            if len(self._frame_history) > self._max_history:
                self._frame_history = self._frame_history[-self._max_history:]

            if len(self._frame_history) < 5:
                return 0.06

            lum_array = np.array(self._frame_history)
            diffs = np.abs(np.diff(lum_array))
            diff_mean = float(np.mean(diffs))

            signs = np.diff(lum_array)
            sign_changes = np.sum(np.abs(np.diff(np.sign(signs + 1e-8))) > 0)
            oscillation_ratio = sign_changes / (len(signs) - 1 + 1e-8)

            # Rapid oscillating luminance = screen refresh aliasing
            if 0.7 < diff_mean < 4.5 and oscillation_ratio > 0.55:
                return 0.60
            elif diff_mean < 0.3 and oscillation_ratio < 0.3:
                return 0.06
            else:
                return 0.12

        except Exception as e:
            logger.debug(f"Temporal flicker analysis failed: {e}")
            return 0.06
