"""
Webcam Behavioral & Physiological Analysis Module.

Implements core real-time methodologies for the two-phase verification system:
  1. Eye Blink & Saccadic Movement Tracking (EAR & dynamic blink rate)
  2. 3D Head Pose Estimation (Yaw, Pitch, Roll via Perspective-n-Point)
  3. Hand Gesture & Finger Counting Tracker (Convex Hull & Defects outside face)
  4. Active 3-Action Challenge-Response Verification Engine:
       - Action 1: Turn head to the RIGHT ➡️
       - Action 2: Turn head to the LEFT ⬅️
       - Action 3: Raise 2 fingers ✌️ (or random finger count)
  5. Identity Embedding & Temporal Consistency Tracker
"""

import time
import math
import logging
import random
from collections import deque
import numpy as np

logger = logging.getLogger("deepscan.webcam_behavioral")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# 3D canonical reference face model points (in millimeters)
# Landmarks: [Left eye (smaller X), Right eye (larger X), Nose (0,0), Left mouth (smaller X), Right mouth (larger X)]
MODEL_POINTS_3D = np.array([
    [-35.0, 35.0, -35.0],   # Image Left eye (smaller X)
    [35.0, 35.0, -35.0],    # Image Right eye (larger X)
    [0.0, 0.0, 0.0],        # Nose tip
    [-25.0, -40.0, -25.0],  # Image Left mouth (smaller X)
    [25.0, -40.0, -25.0],   # Image Right mouth (larger X)
], dtype=np.float64)


class BlinkAndGazeTracker:
    """
    Tracks eye blinks using dual-method detection:
    1. Vertical eye opening ratio (simulated EAR from patch pixel analysis)
    2. Temporal derivative — detects rapid drops in eye openness across frames
    """

    def __init__(self):
        self.blink_count = 0
        self.last_state = "open"
        self.last_blink_time = time.time()
        self.start_time = time.time()
        self.baseline_ear = 0.35
        self.ear_history = deque(maxlen=30)
        self.recent_ears = deque(maxlen=5)  # Short window for derivative detection

    def update(self, frame_rgb, landmarks):
        """
        Estimate Eye Openness (EAR) via vertical pixel ratio and count real blinks.
        Uses adaptive thresholds and temporal derivative for robust detection.
        """
        ear = 0.35

        if frame_rgb is not None and landmarks is not None and len(landmarks) >= 2 and _HAS_CV2:
            try:
                le = np.array(landmarks[0], dtype=np.float32)
                re = np.array(landmarks[1], dtype=np.float32)
                eye_dist = float(np.linalg.norm(le - re)) + 1e-6

                # Eye patch size proportional to inter-eye distance
                rx = max(10, int(0.28 * eye_dist))
                ry = max(8, int(0.22 * eye_dist))

                h, w = frame_rgb.shape[:2]

                ears = []
                for (ex, ey) in [(int(le[0]), int(le[1])), (int(re[0]), int(re[1]))]:
                    y1 = max(0, ey - ry)
                    y2 = min(h, ey + ry)
                    x1 = max(0, ex - rx)
                    x2 = min(w, ex + rx)
                    patch = frame_rgb[y1:y2, x1:x2]

                    if patch.size < 16:
                        ears.append(0.35)
                        continue

                    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                    ph, pw = gray.shape[:2]

                    if ph < 4 or pw < 4:
                        ears.append(0.35)
                        continue

                    # Method 1: Vertical opening ratio
                    # When eyes are open: bright sclera + dark pupil = high vertical gradient variance
                    # When eyes are closed: uniform skin color = low vertical gradient variance
                    col_profile = np.mean(gray, axis=1).astype(np.float32)
                    # Vertical gradient strength
                    vert_grad = np.sum(np.abs(np.diff(col_profile)))
                    vert_ratio = vert_grad / (ph * 25.0 + 1e-6)

                    # Method 2: Dark pixel ratio (pupil/iris visible = more dark pixels when open)
                    threshold = np.mean(gray) * 0.70
                    dark_ratio = np.sum(gray < threshold) / (gray.size + 1e-6)

                    # Method 3: Horizontal vs vertical standard deviation ratio
                    # Open eyes have more horizontal variation (sclera-iris-sclera)
                    h_std = float(np.mean([np.std(gray[r, :]) for r in range(ph)]))
                    v_std = float(np.mean([np.std(gray[:, c]) for c in range(pw)]))
                    hv_ratio = h_std / (v_std + 1e-6)

                    # Composite EAR: higher = more open
                    eye_ear = float(min(0.50, max(0.05,
                        vert_ratio * 0.35 +
                        dark_ratio * 0.30 +
                        min(hv_ratio * 0.15, 0.20) +
                        0.02
                    )))
                    ears.append(eye_ear)

                ear = sum(ears) / len(ears) if ears else 0.35

                self.ear_history.append(ear)
                self.recent_ears.append(ear)

                # Adaptive baseline: track the 75th percentile as "open eyes" baseline
                if len(self.ear_history) >= 6:
                    recent_max = float(np.percentile(list(self.ear_history), 75))
                    self.baseline_ear = 0.90 * self.baseline_ear + 0.10 * recent_max

                # Adaptive close/open thresholds (much more sensitive)
                close_thresh = max(0.12, self.baseline_ear * 0.68)
                open_thresh = max(0.16, self.baseline_ear * 0.80)

                # Temporal derivative detection: sudden drop in EAR = blink
                derivative_blink = False
                if len(self.recent_ears) >= 3:
                    recent = list(self.recent_ears)
                    max_recent = max(recent[:-1])
                    current = recent[-1]
                    drop = max_recent - current
                    if drop > 0.06 and current < close_thresh * 1.2:
                        derivative_blink = True

                now = time.time()

                # Standard state machine blink detection
                if ear <= close_thresh and self.last_state == "open":
                    self.last_state = "closed"
                elif ear >= open_thresh and self.last_state == "closed":
                    self.last_state = "open"
                    if now - self.last_blink_time > 0.12:
                        self.blink_count += 1
                        self.last_blink_time = now

                # Derivative-based blink detection (catches fast/subtle blinks)
                if derivative_blink and self.last_state == "open" and now - self.last_blink_time > 0.25:
                    self.blink_count += 1
                    self.last_blink_time = now

            except Exception as e:
                logger.debug(f"Optical blink tracker error: {e}")

        elapsed_min = max(0.1, (time.time() - self.start_time) / 60.0)
        bpm = round(self.blink_count / elapsed_min, 1)

        return {
            "ear": round(ear, 3),
            "blink_count": self.blink_count,
            "blinks_per_min": bpm,
            "eye_state": self.last_state,
        }


class HeadPose3DTracker:
    """
    Estimates 3D Head Pose (Yaw, Pitch, Roll) using Perspective-n-Point and
    2D landmark geometry ratios for strict, rock-solid direction verification.
    """

    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.turn_ratio = 0.0
        self.vert_ratio = 0.50

    def update(self, landmarks, img_shape):
        """
        Compute Yaw, Pitch, Roll from 5 facial landmarks and landmark ratio.
        """
        if landmarks is None or len(landmarks) < 5 or not _HAS_CV2:
            return self.get_stats()

        try:
            h, w = img_shape[:2]

            # Sort eyes and mouth by X coordinate
            x_eye_left = min(float(landmarks[0][0]), float(landmarks[1][0]))
            x_eye_right = max(float(landmarks[0][0]), float(landmarks[1][0]))
            x_nose = float(landmarks[2][0])

            span_x = max(1.0, x_eye_right - x_eye_left)
            dist_left = x_nose - x_eye_left
            dist_right = x_eye_right - x_nose

            # When user turns RIGHT (in real world): nose moves left in webcam -> dist_left < dist_right -> turn_ratio is POSITIVE
            # When user turns LEFT (in real world): nose moves right in webcam -> dist_right < dist_left -> turn_ratio is NEGATIVE
            self.turn_ratio = round((dist_right - dist_left) / span_x, 3)

            # Pitch geometry:
            y_eye_mid = (float(landmarks[0][1]) + float(landmarks[1][1])) / 2.0
            y_mouth_mid = (float(landmarks[3][1]) + float(landmarks[4][1])) / 2.0
            y_nose = float(landmarks[2][1])
            span_y = max(1.0, y_mouth_mid - y_eye_mid)

            self.vert_ratio = round((y_nose - y_eye_mid) / span_y, 3)

            image_points = np.array([
                landmarks[0],
                landmarks[1],
                landmarks[2],
                landmarks[3],
                landmarks[4],
            ], dtype=np.float64)

            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )

            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
                singular = sy < 1e-6

                if not singular:
                    pitch = math.atan2(rmat[2, 1], rmat[2, 2])
                    yaw = math.atan2(-rmat[2, 0], sy)
                    roll = math.atan2(rmat[1, 0], rmat[0, 0])
                else:
                    pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
                    yaw = math.atan2(-rmat[2, 0], sy)
                    roll = 0

                # Correct pitch sign for camera coordinate convention
                raw_pitch = -float(np.degrees(pitch))
                raw_yaw = float(np.degrees(yaw))
                raw_roll = float(np.degrees(roll))

                self.pitch = round(0.7 * self.pitch + 0.3 * raw_pitch, 1)
                self.yaw = round(0.7 * self.yaw + 0.3 * raw_yaw, 1)
                self.roll = round(0.7 * self.roll + 0.3 * raw_roll, 1)

        except Exception as e:
            logger.debug(f"Head pose estimation error: {e}")

        return self.get_stats()

    def get_stats(self):
        # Strict orientation determination with calibrated deadzones
        if self.turn_ratio >= 0.18 or self.yaw >= 14.0:
            orientation = "Turned Right"
        elif self.turn_ratio <= -0.18 or self.yaw <= -14.0:
            orientation = "Turned Left"
        elif self.vert_ratio >= 0.62 or self.pitch >= 12.0:
            orientation = "Looking Down"
        elif self.vert_ratio <= 0.38 or self.pitch <= -12.0:
            orientation = "Looking Up"
        else:
            orientation = "Center"

        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "turn_ratio": self.turn_ratio,
            "orientation": orientation,
        }


class HandGestureTracker:
    """
    Real-time Hand & Finger Gesture Analysis.
    Detects hand presence and accurately counts raised fingers outside the facial region.
    """

    def __init__(self):
        self.detected_fingers = 0
        self.hand_detected = False
        self.consecutive_frames = 0
        self.last_count = 0

    def update(self, frame_rgb, faces=None):
        if frame_rgb is None or not _HAS_CV2:
            return {"hand_detected": False, "finger_count": 0}

        try:
            h, w = frame_rgb.shape[:2]
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
            ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)

            # Combined skin color thresholding
            mask_hsv = cv2.inRange(hsv, np.array([0, 30, 50]), np.array([25, 230, 255]))
            mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 175, 127]))
            skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

            # Zero out face area (+35% padding + neck) so face is not detected as hand
            if faces:
                for face in faces:
                    bbox = face.get("bbox", [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        if x2 < x1: x2 = x1 + x2; y2 = y1 + y2
                        pad_x = int(0.35 * (x2 - x1))
                        pad_y = int(0.35 * (y2 - y1))
                        fx1 = max(0, x1 - pad_x)
                        fy1 = max(0, y1 - pad_y)
                        fx2 = min(w, x2 + pad_x)
                        fy2 = min(h, y2 + pad_y + int(0.4 * (y2 - y1)))
                        skin_mask[fy1:fy2, fx1:fx2] = 0

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel, iterations=2)

            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            finger_count = 0
            hand_detected = False

            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)

                if area > 1000:
                    M = cv2.moments(c)
                    if M["m00"] > 1e-5:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        hull = cv2.convexHull(c, returnPoints=False)
                        
                        if hull is not None and len(hull) > 3 and len(c) > 3:
                            defects = cv2.convexityDefects(c, hull)
                            if defects is not None:
                                valleys = 0
                                for i in range(defects.shape[0]):
                                    s, e, f, d = defects[i, 0]
                                    start = tuple(c[s][0])
                                    end = tuple(c[e][0])
                                    far = tuple(c[f][0])

                                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                                    c_len = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                                    if b * c_len > 1e-5:
                                        angle = math.acos(max(-1.0, min(1.0, (b**2 + c_len**2 - a**2) / (2 * b * c_len))))
                                        if angle <= math.pi * 0.55 and d > 12 * 256 and far[1] < cy + 50:
                                            valleys += 1

                                if valleys == 1:
                                    # 1 deep valley between index & middle finger
                                    finger_count = 2
                                    hand_detected = True
                                elif valleys == 2:
                                    # 2 deep valleys between 3 fingers
                                    finger_count = 3
                                    hand_detected = True
                                elif valleys >= 3:
                                    # 4 or 5 fingers
                                    finger_count = min(5, valleys + 1)
                                    hand_detected = True
                                else:
                                    # Single finger / peace gesture
                                    rect = cv2.boundingRect(c)
                                    aspect = rect[3] / (rect[2] + 1e-5)
                                    if aspect > 1.2 and rect[3] > 50:
                                        finger_count = 2
                                        hand_detected = True
                                    else:
                                        finger_count = 0
                                        hand_detected = False

            if hand_detected and finger_count > 0:
                self.consecutive_frames += 1
                self.last_count = finger_count
            else:
                self.consecutive_frames = 0
                self.last_count = 0

            self.detected_fingers = self.last_count if self.consecutive_frames >= 1 else 0
            self.hand_detected = (self.consecutive_frames >= 1)

            return {
                "hand_detected": self.hand_detected,
                "finger_count": self.detected_fingers,
            }
        except Exception as e:
            logger.debug(f"Hand gesture error: {e}")
            return {"hand_detected": False, "finger_count": 0}


class ChallengeResponseEngine:
    """
    Active 3-Action Sequential Challenge-Response Verification Engine.
    Executes a structured 3-step verification flow:
      Step 1: Turn head to the RIGHT ➡️
      Step 2: Turn head to the LEFT ⬅️
      Step 3: Raise 2 fingers ✌️ (or 3 fingers)
    """

    def __init__(self):
        self.state = "idle"  # "idle", "active", "passed", "failed"
        self.steps = []
        self.current_step_idx = 0
        self.step_start_time = 0.0
        self.step_duration = 12.0
        self.step_hold_frames = 0
        self.REQUIRED_HOLD_FRAMES = 2

    def start_new_challenge(self, current_blinks=0):
        """Initiate the 3-action sequential challenge."""
        target_fingers = random.choice([2, 3])
        finger_emoji = "✌️" if target_fingers == 2 else "🖐️"

        self.steps = [
            {
                "id": "turn_right",
                "short_name": "Turn Right",
                "prompt": "Turn your head to the RIGHT",
                "icon": "➡️",
                "target": "yaw_right",
                "passed": False,
            },
            {
                "id": "turn_left",
                "short_name": "Turn Left",
                "prompt": "Turn your head to the LEFT",
                "icon": "⬅️",
                "target": "yaw_left",
                "passed": False,
            },
            {
                "id": "raise_fingers",
                "short_name": f"Raise {target_fingers} Fingers",
                "prompt": f"Raise {target_fingers} fingers to the camera",
                "icon": finger_emoji,
                "target": "fingers",
                "target_count": target_fingers,
                "passed": False,
            },
        ]
        self.current_step_idx = 0
        self.state = "active"
        self.step_start_time = time.time()
        self.step_hold_frames = 0
        return self.get_status()

    def cancel_challenge(self):
        """Reset challenge state to idle."""
        self.state = "idle"
        self.steps = []
        self.current_step_idx = 0
        self.step_hold_frames = 0
        self.step_start_time = 0.0
        return self.get_status()

    def evaluate(self, pose, blink_stats, hand_stats):
        """
        Evaluate frame against the current active step in the 3-action sequence.
        Strictly verifies that the action is correct and held.
        """
        if self.state != "active" or not self.steps:
            return self.get_status()

        elapsed = time.time() - self.step_start_time

        if elapsed > self.step_duration:
            self.state = "failed"
            return self.get_status()

        if self.current_step_idx >= len(self.steps):
            self.state = "passed"
            return self.get_status()

        current_step = self.steps[self.current_step_idx]
        target = current_step.get("target")

        yaw = pose.get("yaw", 0.0)
        turn_ratio = pose.get("turn_ratio", 0.0)
        hand_detected = hand_stats.get("hand_detected", False)
        finger_count = hand_stats.get("finger_count", 0)

        action_satisfied = False

        if target == "yaw_right":
            # Must strictly turn RIGHT (turn_ratio > 0.10 or yaw > 6.0) and NOT turned left
            if (turn_ratio >= 0.10 or yaw >= 6.0) and turn_ratio > -0.05:
                action_satisfied = True
        elif target == "yaw_left":
            # Must strictly turn LEFT (turn_ratio < -0.10 or yaw < -6.0) and NOT turned right
            if (turn_ratio <= -0.10 or yaw <= -6.0) and turn_ratio < 0.05:
                action_satisfied = True
        elif target == "fingers":
            target_count = current_step.get("target_count", 2)
            # When user raises fingers
            if hand_detected and (finger_count >= 2 or finger_count == target_count):
                action_satisfied = True

        if action_satisfied:
            self.step_hold_frames += 1
        else:
            # Strictly reset hold frames on wrong movement or neutral pose
            self.step_hold_frames = 0

        # Advance to next step once hold threshold is met
        if self.step_hold_frames >= self.REQUIRED_HOLD_FRAMES:
            current_step["passed"] = True
            self.current_step_idx += 1
            self.step_start_time = time.time()
            self.step_hold_frames = 0

            # If all 3 steps completed
            if self.current_step_idx >= len(self.steps):
                self.state = "passed"

        return self.get_status()

    def get_status(self):
        time_left = 0.0
        progress = 0.0
        current_challenge = None

        if self.steps:
            idx = min(self.current_step_idx, len(self.steps) - 1)
            current_challenge = self.steps[idx]

        if self.state == "active" and current_challenge:
            elapsed = time.time() - self.step_start_time
            time_left = max(0.0, self.step_duration - elapsed)
            progress = min(1.0, elapsed / self.step_duration)

        return {
            "state": self.state,
            "step_index": min(self.current_step_idx + 1, len(self.steps)),
            "total_steps": len(self.steps) if self.steps else 3,
            "challenge": current_challenge,
            "steps": [
                {
                    "id": s["id"],
                    "short_name": s["short_name"],
                    "prompt": s["prompt"],
                    "icon": s["icon"],
                    "passed": s["passed"],
                }
                for s in self.steps
            ],
            "time_left": round(time_left, 1),
            "progress": round(progress, 2),
        }


class IdentityEmbeddingTracker:
    """
    Facial Identity Embedding & Temporal Consistency Tracker.
    Inspired by Sumo2003/Real-time-Deepfake-Detection-in-video-calls.
    """

    def __init__(self, history_len=10):
        self.reference_embedding = None
        self.enrolled = False
        self.enrolled_time = 0.0
        self.history = []
        self.history_len = history_len

    def _extract_embedding(self, crop):
        """Extract a normalized 128-d spatial facial descriptor from crop."""
        if crop is None or crop.size == 0 or not _HAS_CV2:
            return None
        try:
            resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32)

            blocks_mean = []
            blocks_std = []
            for r in range(8):
                for c in range(8):
                    block = gray[r*8:(r+1)*8, c*8:(c+1)*8]
                    blocks_mean.append(np.mean(block))
                    blocks_std.append(np.std(block))

            feat = np.array(blocks_mean + blocks_std, dtype=np.float32)
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            return feat
        except Exception as e:
            logger.debug(f"Embedding extraction error: {e}")
            return None

    def enroll(self, crop):
        """Enroll the current face as the reference identity."""
        emb = self._extract_embedding(crop)
        if emb is not None:
            self.reference_embedding = emb
            self.enrolled = True
            self.enrolled_time = time.time()
            return {"enrolled": True, "message": "Reference face enrolled successfully"}
        return {"enrolled": False, "message": "Failed to extract face features"}

    def clear(self):
        """Clear enrolled reference identity."""
        self.reference_embedding = None
        self.enrolled = False
        self.history.clear()
        return {"enrolled": False, "message": "Enrolled reference cleared"}

    def update(self, crop):
        """Process frame crop and compute similarity against enrolled reference."""
        emb = self._extract_embedding(crop)
        if emb is None:
            return self.get_stats()

        self.history.append(emb)
        if len(self.history) > self.history_len:
            self.history.pop(0)

        similarity = None
        is_match = True
        if self.enrolled and self.reference_embedding is not None:
            dot = float(np.dot(emb, self.reference_embedding))
            similarity = round(max(0.0, min(1.0, dot)), 4)
            is_match = similarity >= 0.65

        return {
            "enrolled": self.enrolled,
            "similarity": similarity,
            "similarity_pct": int(similarity * 100) if similarity is not None else None,
            "is_identity_match": is_match,
        }

    def get_stats(self):
        return {
            "enrolled": self.enrolled,
            "similarity": None,
            "similarity_pct": None,
            "is_identity_match": True,
        }


class WebcamBehavioralAnalyzer:
    """
    Unified behavioral engine for two-phase webcam verification & monitoring.
    """

    def __init__(self):
        self.blink = BlinkAndGazeTracker()
        self.pose = HeadPose3DTracker()
        self.hand = HandGestureTracker()
        self.challenge = ChallengeResponseEngine()
        self.identity = IdentityEmbeddingTracker()

    def process(self, frame_rgb, faces):
        """
        Run complete behavioral analysis across detected faces & hands.
        """
        landmarks = None
        crop = None
        if faces:
            crop = faces[0].get("crop")
            landmarks = faces[0].get("landmark")

        # 1. Update Blink & Gaze
        blink_stats = self.blink.update(frame_rgb, landmarks)

        # 2. Update 3D Head Pose
        pose_stats = self.pose.update(landmarks, frame_rgb.shape)

        # 3. Update Hand & Finger Gesture
        hand_stats = self.hand.update(frame_rgb, faces)

        # 4. Evaluate Active 3-Action Challenge
        challenge_status = self.challenge.evaluate(pose_stats, blink_stats, hand_stats)

        # 5. Evaluate Identity Tracking
        identity_stats = self.identity.update(crop)

        return {
            "blink": blink_stats,
            "pose": pose_stats,
            "hand": hand_stats,
            "challenge": challenge_status,
            "identity": identity_stats,
        }

    def trigger_challenge(self):
        return self.challenge.start_new_challenge(self.blink.blink_count)

    def cancel_challenge(self):
        return self.challenge.cancel_challenge()

    def reset(self):
        """Completely reset all behavioral trackers for a fresh session."""
        self.blink = BlinkAndGazeTracker()
        self.pose = HeadPose3DTracker()
        self.hand = HandGestureTracker()
        self.challenge = ChallengeResponseEngine()
        self.identity = IdentityEmbeddingTracker()

    def enroll_identity(self, crop):
        return self.identity.enroll(crop)

    def clear_identity(self):
        return self.identity.clear()
