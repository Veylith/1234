"""
Webcam Pipeline — Two-Phase Identity Verification & Live Monitoring.

Phase 1: Active Challenge Gate — Prove human liveness before a call.
Phase 2: Live Monitoring — Continuous deepfake/spoof detection during a call.

Fuses core detection methodologies with temporal smoothing & hysteresis:
  1. Deep Neural Face Classification (Vision Transformer / EfficientNet)
  2. Anti-Spoofing & Liveness (Moiré FFT, Skin Chromaticity, Glare Geometry)
  3. Eye Blink Dynamics (EAR)
  4. 3D Head Pose Perspective (PnP)
  + Active Challenge-Response Verification Engine
  + Temporal EMA Smoothing & Hysteresis State Machine (Zero-Flicker)
"""

import logging
import time
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from .webcam_behavioral import WebcamBehavioralAnalyzer

logger = logging.getLogger("deepscan.webcam_pipeline")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class WebcamPipeline:
    """Two-phase webcam analysis: challenge gate + live monitoring."""

    def __init__(self, face_detector, liveness_detector, deepfake_detector):
        self.face_detector = face_detector
        self.liveness_detector = liveness_detector
        self.deepfake_detector = deepfake_detector
        self.behavioral = WebcamBehavioralAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=2)

        # --- Anti-Drift Dual-Track Temporal Smoothing ---
        self.ema_alpha = 0.18
        self.smooth_df_score = 0.06
        self.smooth_spoof_score = 0.08
        self.smooth_combined_score = 0.06

        # Rolling median anchor
        self.recent_raw_combined = deque(maxlen=20)
        self.recent_raw_df = deque(maxlen=20)
        self.recent_raw_spoof = deque(maxlen=20)

        # Pose variance tracking
        self.recent_yaws = deque(maxlen=20)
        self.recent_pitches = deque(maxlen=20)

        # Hysteresis state
        self.current_tier = 1
        self.tier_history = deque(maxlen=8)

        # Confirmed-Live Lock (blink + pose + anti-spoofing)
        self.confirmed_live_lock = False
        self.live_frame_streak = 0
        self.LOCK_ENGAGE_FRAMES = 8

        # Temporal behavioral gate — blink tracking for spoof detection
        self.session_start_time = time.time()
        self.total_blinks_in_session = 0
        self.frames_since_last_blink = 0
        self.NO_BLINK_SPOOF_FRAMES = 90  # ~3 seconds at 30fps — if no blink, escalate

        # Phase tracking
        self.phase = "monitoring"

        # Interactive states
        self.last_face_crop = None
        self._has_deep_model = None
        self._processing = False
        self._last_result = None

        logger.info("WebcamPipeline initialized — Direct Live Monitoring Mode")

    def reset_session(self):
        """Completely reset all smoothed metrics, histories, locks, and counters for a fresh session."""
        # Shutdown and recreate thread pool to kill any stale hanging futures
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.smooth_df_score = 0.05
        self.smooth_spoof_score = 0.15
        self.smooth_combined_score = 0.05
        self.recent_raw_df.clear()
        self.recent_raw_spoof.clear()
        self.recent_raw_combined.clear()
        self.recent_yaws.clear()
        self.recent_pitches.clear()
        self.tier_history.clear()
        self.current_tier = 1
        self.confirmed_live_lock = False
        self.live_frame_streak = 0
        self.last_face_crop = None
        self.session_start_time = time.time()
        self.total_blinks_in_session = 0
        self.frames_since_last_blink = 0
        self.phase = "monitoring"
        self._processing = False  # Frame throttle flag
        if hasattr(self, 'behavioral') and self.behavioral is not None:
            self.behavioral.reset()
        logger.info("WebcamPipeline session state completely reset")

    def handle_command(self, cmd_type, payload=None):
        """Handle interactive WebSocket commands."""
        if cmd_type in ("reset", "reset_session", "start", "stop"):
            self.reset_session()
            return {"reset": True, "message": "Session reset successfully"}
        elif cmd_type == "start_challenge":
            self.phase = "challenge"
            return self.behavioral.trigger_challenge()
        elif cmd_type == "cancel_challenge":
            return self.behavioral.cancel_challenge()
        elif cmd_type == "start_monitoring":
            self.phase = "monitoring"
            # Auto-enroll the current face as reference identity
            if self.last_face_crop is not None:
                self.behavioral.enroll_identity(self.last_face_crop)
            return {"phase": "monitoring", "message": "Live monitoring started"}
        elif cmd_type == "enroll_identity":
            if self.last_face_crop is not None:
                return self.behavioral.enroll_identity(self.last_face_crop)
            return {"enrolled": False, "message": "No face detected"}
        elif cmd_type == "clear_identity":
            return self.behavioral.clear_identity()
        return {}

    def _detect_deep_model(self):
        """Check if a deep learning model is loaded."""
        if self._has_deep_model is not None:
            return self._has_deep_model
        pipe = self.deepfake_detector._get_active_pipe()
        model = getattr(self.deepfake_detector, 'model', None)
        self._has_deep_model = (pipe is not None) or (model is not None)
        return self._has_deep_model

    def analyze_frame(self, frame_bytes):
        """
        Analyze a single webcam frame with anti-drift temporal smoothing.
        Returns telemetry for the current phase (challenge or monitoring).
        Includes frame throttling to prevent backlog when frames arrive faster than processing.
        """
        # Frame throttle: skip if already processing a frame (prevents 5-min restart backlog)
        if getattr(self, '_processing', False):
            return self._last_result if hasattr(self, '_last_result') else self._empty_result("Processing previous frame")

        self._processing = True
        start_time = time.time()

        if not _HAS_CV2:
            self._processing = False
            return self._empty_result("OpenCV not available")

        try:
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                return self._empty_result("Could not decode frame")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Detect faces & landmarks
            faces = self.face_detector.detect(frame_rgb)

            if not faces:
                self.last_face_crop = None
                result = self._empty_result("No face detected")
                self._last_result = result
                self._processing = False
                return result

            self.last_face_crop = faces[0].get("crop")

            # Run behavioral analysis (blink, pose, challenge)
            behavioral_result = self.behavioral.process(frame_rgb, faces)

            # Run deepfake and liveness analysis in parallel
            liveness_future = self.executor.submit(
                self.liveness_detector.analyze, frame_rgb, faces
            )
            authenticity_future = self.executor.submit(
                self.deepfake_detector.analyze, frame_rgb, faces
            )

            liveness_result = liveness_future.result(timeout=6.0)
            auth_result = authenticity_future.result(timeout=6.0)

            # Raw frame scores
            raw_spoof = float(liveness_result.get("liveness_score", 0.15))
            raw_df = float(auth_result.get("authenticity_score", 0.05))

            liveness_signals = liveness_result.get("liveness_signals", {})
            device_detected = bool(liveness_signals.get("device_detected", False))
            device_label = liveness_signals.get("device_label", "None")
            context_border = float(liveness_signals.get("context_border", 0.08))
            subpixel_pattern = float(liveness_signals.get("subpixel_pattern", 0.08))
            color_quantization = float(liveness_signals.get("color_quantization", 0.08))
            moire_frequency = float(liveness_signals.get("moire_frequency", 0.08))
            screen_glare = float(liveness_signals.get("screen_glare", 0.08))

            device_score = float(liveness_signals.get("device_score", 0.0))
            is_device_attack = device_detected and (device_score >= 0.45)
            is_bezel_attack = (context_border >= 0.60)

            if is_device_attack:
                raw_spoof = max(raw_spoof, device_score)
            elif is_bezel_attack:
                raw_spoof = max(raw_spoof, context_border)

            has_model = self._detect_deep_model()

            # Heuristic-only clamping for deepfake generation artifacts
            if not has_model:
                raw_df = min(raw_df, 0.25)

            # --- ASYMMETRIC TEMPORAL SMOOTHING ---
            # Fast reaction to RISING scores (spoof appearing): alpha = 0.40
            # Smooth reaction to FALLING scores: alpha = 0.15
            alpha_up = 0.40
            alpha_down = 0.15

            df_alpha = alpha_up if raw_df > self.smooth_df_score else alpha_down
            spoof_alpha = alpha_up if raw_spoof > self.smooth_spoof_score else alpha_down

            self.smooth_df_score = (1 - df_alpha) * self.smooth_df_score + df_alpha * raw_df
            self.smooth_spoof_score = (1 - spoof_alpha) * self.smooth_spoof_score + spoof_alpha * raw_spoof

            self.recent_raw_df.append(raw_df)
            self.recent_raw_spoof.append(raw_spoof)

            median_df = float(np.median(list(self.recent_raw_df)))
            median_spoof = float(np.median(list(self.recent_raw_spoof)))

            df_score = max(self.smooth_df_score, median_df)
            spoof_score = max(self.smooth_spoof_score, median_spoof)

            is_forensic_attack = (spoof_score >= 0.52) or (raw_spoof >= 0.58)
            is_presentation_attack = is_device_attack or is_bezel_attack or is_forensic_attack

            # Liveness: True if NOT a presentation attack and spoof_score is low/normal
            is_live = (not is_presentation_attack) and (spoof_score <= 0.45)

            # Debug logging
            logger.debug(
                f"[PAD] raw_spoof={raw_spoof:.3f} smooth_spoof={self.smooth_spoof_score:.3f} "
                f"final_spoof={spoof_score:.3f} is_presentation_attack={is_presentation_attack} "
                f"device={device_detected} ({device_label}) | is_live={is_live}"
            )
            if len(self.recent_raw_spoof) % 15 == 0:
                logger.info(
                    f"[PAD MONITOR] spoof={spoof_score:.3f} is_live={is_live} "
                    f"device={device_detected} ({device_label}) bezel={context_border:.2f} "
                    f"subpixel={subpixel_pattern:.2f} quant={color_quantization:.2f}"
                )

            # Extract behavioral signals
            blink_data = behavioral_result.get("blink", {})
            pose_data = behavioral_result.get("pose", {})
            self.recent_yaws.append(pose_data.get("yaw", 0.0))
            self.recent_pitches.append(pose_data.get("pitch", 0.0))

            # --- TEMPORAL BEHAVIORAL GATE ---
            current_blinks = blink_data.get("blink_count", 0)
            if current_blinks > self.total_blinks_in_session:
                self.total_blinks_in_session = current_blinks
                self.frames_since_last_blink = 0
            else:
                self.frames_since_last_blink += 1

            # Behavioral boosts
            session_duration = time.time() - self.session_start_time
            no_blink_spoof_boost = 0.0
            if session_duration > 4.0 and self.total_blinks_in_session == 0:
                no_blink_spoof_boost = 0.15
            elif self.frames_since_last_blink > self.NO_BLINK_SPOOF_FRAMES:
                no_blink_spoof_boost = 0.10

            pose_variance_boost = 0.0
            if len(self.recent_yaws) >= 10:
                yaw_std = float(np.std(list(self.recent_yaws)))
                pitch_std = float(np.std(list(self.recent_pitches)))
                combined_std = (yaw_std + pitch_std) / 2.0
                if combined_std < 0.20 and session_duration > 3.0:
                    pose_variance_boost = 0.08

            behavioral_boost = no_blink_spoof_boost + pose_variance_boost
            if behavioral_boost > 0 and (not is_live):
                raw_spoof = min(1.0, raw_spoof + behavioral_boost)
                spoof_score = max(spoof_score, raw_spoof)

            # --- CONFIRMED-LIVE LOCK (STRICT SAFETY GATE) ---
            if is_presentation_attack or (not is_live) or spoof_score > 0.42:
                self.confirmed_live_lock = False
                self.live_frame_streak = 0
            else:
                has_any_blinks = self.total_blinks_in_session > 0
                if is_live and has_any_blinks:
                    self.live_frame_streak += 1
                else:
                    self.live_frame_streak = max(0, self.live_frame_streak - 2)

                if self.live_frame_streak >= self.LOCK_ENGAGE_FRAMES:
                    self.confirmed_live_lock = True

            # Identity check
            identity_data = behavioral_result.get("identity", {})
            identity_mismatch = False
            if identity_data.get("enrolled") and identity_data.get("is_identity_match") is False:
                identity_mismatch = True

            # --- LIVENESS-ANCHORED FUSION ---
            if identity_mismatch:
                raw_combined = 0.90
            elif is_presentation_attack:
                raw_combined = max(spoof_score, 0.88 if is_device_attack else 0.78)
            elif self.confirmed_live_lock and not is_presentation_attack:
                effective_df = min(df_score, 0.15)
                raw_combined = effective_df * 0.50 + spoof_score * 0.20 + 0.02
            elif is_live and spoof_score <= 0.35:
                effective_df = min(df_score, 0.20)
                raw_combined = effective_df * 0.65 + spoof_score * 0.35
            else:
                raw_combined = df_score * 0.50 + spoof_score * 0.50

            # Only clamp if live person and no deep learning weights loaded
            if not has_model and is_live and (not is_presentation_attack):
                raw_combined = min(raw_combined, 0.20)

            self.recent_raw_combined.append(raw_combined)

            combined_alpha = alpha_up if raw_combined > self.smooth_combined_score else alpha_down
            self.smooth_combined_score = (1 - combined_alpha) * self.smooth_combined_score + combined_alpha * raw_combined

            median_combined = float(np.median(list(self.recent_raw_combined)))
            combined_fake_score = max(self.smooth_combined_score, median_combined)

            # --- HYSTERESIS TIER STATE MACHINE ---
            if identity_mismatch:
                candidate_tier = 3
            elif is_presentation_attack or spoof_score >= 0.52:
                candidate_tier = 3
            elif is_live and combined_fake_score < 0.40:
                candidate_tier = 1
            elif self.current_tier == 1:
                if combined_fake_score > 0.48 or spoof_score > 0.48:
                    candidate_tier = 2
                else:
                    candidate_tier = 1
            elif self.current_tier == 2:
                if combined_fake_score > 0.65 or spoof_score > 0.55:
                    candidate_tier = 3
                elif combined_fake_score < 0.35 and spoof_score < 0.40:
                    candidate_tier = 1
                else:
                    candidate_tier = 2
            else:
                if combined_fake_score < 0.38 and spoof_score < 0.40 and (not is_presentation_attack):
                    candidate_tier = 2
                else:
                    candidate_tier = 3

            self.tier_history.append(candidate_tier)
            if self.tier_history.count(candidate_tier) >= 3:
                self.current_tier = candidate_tier

            # Classify presentation attack object
            if is_live and (not is_presentation_attack):
                liveness_object = "Live Person"
            else:
                if is_device_attack:
                    liveness_object = f"{device_label} (Replay Attack)"
                elif is_bezel_attack:
                    liveness_object = "Phone/Screen Bezel Replay Attack"
                elif subpixel_pattern > 0.35 or moire_frequency > 0.35:
                    liveness_object = "OLED/LCD Screen Replay"
                elif color_quantization > 0.45:
                    liveness_object = "Digital Video Screen Replay"
                elif screen_glare > 0.35:
                    liveness_object = "Glass Screen Glare Replay"
                else:
                    liveness_object = "Screen Video Replay (Spoof)"

            # Overall unequivocal verdict
            if identity_mismatch:
                tier_label = "Identity Mismatch"
                overall_verdict = "Face-Swap Attack Detected"
                liveness_object = "Face Swap / Identity Anomaly"
                face_verdict = "Fake Face"
                self.current_tier = 3
            elif is_presentation_attack or not is_live:
                tier_label = "Spoof Detected"
                face_verdict = "Spoof / Replay"
                overall_verdict = f"{liveness_object}"
                self.current_tier = 3
            elif combined_fake_score < 0.35:
                tier_label = "Authentic"
                liveness_object = "Live Person"
                face_verdict = "Real Face"
                overall_verdict = "Live Person (Authentic Real)"
                self.current_tier = 1
            elif combined_fake_score < 0.60:
                tier_label = "Uncertain"
                face_verdict = "Uncertain"
                overall_verdict = "Live Person (Analyzing Feed)"
                self.current_tier = 2
            else:
                tier_label = "Deepfake Detected"
                face_verdict = "Fake Face"
                overall_verdict = "Live Deepfake Synthetic Face"
                self.current_tier = 3

            # Build face list
            face_list = auth_result.get("faces", [])
            if not face_list:
                for face in faces:
                    face_list.append({
                        "bbox": face["bbox"],
                        "confidence": round(face.get("confidence", 0.0), 4),
                        "fake_score": round(combined_fake_score, 4),
                    })

            processing_time = int((time.time() - start_time) * 1000)

            result = {
                "phase": self.phase,
                "tier": self.current_tier,
                "tier_label": tier_label,
                "overall_verdict": overall_verdict,
                "combined_fake_score": round(combined_fake_score, 4),

                "liveness_score": round(spoof_score, 4),
                "liveness_verdict": liveness_object,
                "liveness_signals": liveness_result.get("liveness_signals", {}),
                "is_live": is_live,
                "liveness_object": liveness_object,
                "face_verdict": face_verdict,

                "authenticity_score": round(df_score, 4),
                "authenticity_verdict": "Authentic" if df_score < 0.35 else ("Uncertain" if df_score < 0.65 else "Deepfake"),
                "authenticity_signals": auth_result.get("authenticity_signals", {}),

                "blink": blink_data,
                "pose": pose_data,
                "hand": behavioral_result.get("hand", {}),
                "challenge": behavioral_result.get("challenge", {}),
                "identity": identity_data,

                "confirmed_live_lock": self.confirmed_live_lock,
                "faces": face_list,
                "faces_detected": len(face_list),
                "processing_time_ms": processing_time,
            }

            self._last_result = result
            self._processing = False
            return result

        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            self._processing = False
            return self._empty_result(str(e))

    def _empty_result(self, error=""):
        """Return empty result structure."""
        return {
            "phase": self.phase,
            "tier": 1,
            "tier_label": "No face detected",
            "overall_verdict": "No face detected" if error in ("", "No face detected") else "Error: " + error,
            "combined_fake_score": 0.05,
            "liveness_score": 0.15,
            "liveness_verdict": "No face",
            "liveness_signals": {},
            "is_live": True,
            "liveness_object": "No face",
            "face_verdict": "Unknown",
            "authenticity_score": 0.05,
            "authenticity_verdict": "No face",
            "authenticity_signals": {},
            "blink": {"ear": 0.32, "blink_count": 0, "blinks_per_min": 0, "eye_state": "open"},
            "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "landmark_ratio": 0.0, "orientation": "Center"},
            "hand": {"hand_detected": False, "finger_count": 0},
            "challenge": {"state": "idle", "challenge": None, "time_left": 0.0, "progress": 0.0},
            "identity": {"enrolled": False, "similarity": None, "similarity_pct": None, "is_identity_match": True},
            "confirmed_live_lock": self.confirmed_live_lock,
            "faces": [],
            "faces_detected": 0,
            "processing_time_ms": 0,
            "error": error,
        }

    def shutdown(self):
        self.executor.shutdown(wait=False)
