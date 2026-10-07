"""
Audio Detector — AASIST-based fake audio detection.

Uses the AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal
Graph Attention Networks) model for high-accuracy audio deepfake detection.

Pipeline:
  1. Load raw audio waveform at 16kHz mono
  2. Pad/truncate to 64,600 samples (~4 seconds)
  3. Run AASIST inference → bonafide vs spoof logits
  4. Blend with classical spectral analysis (85% AASIST / 15% classical)
  5. Return verdict with confidence scores

The pre-trained AASIST weights (~300KB, 297K params) are auto-downloaded
from the official GitHub repository on first run.
"""

import logging
import time
import os
import tempfile
import urllib.request
import numpy as np

logger = logging.getLogger("deepscan.audio_detector")

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
AASIST_WEIGHT_FILE = os.path.join(WEIGHTS_DIR, "AASIST.pth")
AASIST_CLASSIFIER_FILE = os.path.join(WEIGHTS_DIR, "aasist_classifier.joblib")
AASIST_WEIGHT_URL = "https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth"

# AASIST expects 64,600 samples at 16kHz (~4.04 seconds)
AASIST_NUM_SAMPLES = 64600
AASIST_SAMPLE_RATE = 16000


class AudioDetector:
    """
    Audio deepfake detector using AASIST + classical spectral analysis.

    Modes:
      - 'aasist': Full AASIST model inference (best accuracy, EER 0.83%)
      - 'classical_only': Spectral/temporal heuristics only (fallback)
    """

    def __init__(self, device=None):
        self.model = None
        self.calibrated_head = None
        self.device = None
        self.mode = "classical_only"

        if _HAS_TORCH:
            self.device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._init_aasist()
        else:
            logger.warning(
                "PyTorch not available. Audio detection uses classical analysis only."
            )

    def _is_online(self, timeout=2.0):
        """Quick check if internet is reachable."""
        for url in ["https://github.com", "https://www.google.com"]:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "DeepScanAI/2.0"}
                )
                with urllib.request.urlopen(req, timeout=timeout):
                    return True
            except Exception:
                continue
        return False

    def _download_weights(self):
        """Download AASIST pre-trained weights from GitHub."""
        if os.path.exists(AASIST_WEIGHT_FILE):
            return True

        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        if not self._is_online():
            logger.warning("Offline — cannot download AASIST weights.")
            return False

        logger.info(f"⬇️  Downloading AASIST weights from {AASIST_WEIGHT_URL}...")
        try:
            req = urllib.request.Request(
                AASIST_WEIGHT_URL,
                headers={"User-Agent": "DeepScanAI/2.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            with open(AASIST_WEIGHT_FILE, "wb") as f:
                f.write(data)

            logger.info(
                f"✓ AASIST weights downloaded ({len(data) / 1024:.1f} KB) → {AASIST_WEIGHT_FILE}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to download AASIST weights: {e}")
            return False

    def _init_aasist(self):
        """Initialize the AASIST model."""
        try:
            from models.aasist_model import Model as AASISTModel
        except ImportError:
            try:
                from .aasist_model import Model as AASISTModel
            except ImportError:
                logger.error("Cannot import AASIST model architecture.")
                return

        # Try to download weights if not present
        if not os.path.exists(AASIST_WEIGHT_FILE):
            self._download_weights()

        if os.path.exists(AASIST_WEIGHT_FILE):
            try:
                logger.info("🔊 Loading AASIST model...")
                self.model = AASISTModel()
                state_dict = torch.load(
                    AASIST_WEIGHT_FILE,
                    map_location=self.device,
                    weights_only=False,
                )
                # Handle potential key mismatches from official checkpoint
                if isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]

                try:
                    self.model.load_state_dict(state_dict, strict=True)
                except RuntimeError:
                    logger.warning(
                        "Strict weight loading failed — loading with strict=False"
                    )
                    self.model.load_state_dict(state_dict, strict=False)

                self.model.to(self.device)
                self.model.eval()
                self.mode = "aasist"

                # Load calibrated classification head if available
                if os.path.exists(AASIST_CLASSIFIER_FILE):
                    try:
                        import joblib
                        self.calibrated_head = joblib.load(AASIST_CLASSIFIER_FILE)
                        logger.info("✓ Calibrated AASIST classification head loaded")
                    except Exception as e:
                        logger.warning(f"Could not load calibrated head: {e}")
                        self.calibrated_head = None
                else:
                    self.calibrated_head = None

                # Count parameters
                n_params = sum(p.numel() for p in self.model.parameters())
                logger.info(
                    f"✓ AASIST loaded [{n_params:,} params] on {self.device}"
                )
            except Exception as e:
                logger.error(f"Failed to load AASIST model: {e}")
                self.model = None
                self.mode = "classical_only"
        else:
            logger.warning(
                "AASIST weights not found — using classical analysis only. "
                f"Expected at: {AASIST_WEIGHT_FILE}"
            )
            self.mode = "classical_only"

    async def analyze(self, audio_bytes, filename="audio.wav"):
        """
        Analyze audio for deepfake content.

        Args:
            audio_bytes: raw audio file bytes
            filename: original filename

        Returns:
            dict with fake_probability, verdict, details, model_used
        """
        start_time = time.time()

        if not _HAS_LIBROSA:
            return {
                "fake_probability": 0.5,
                "verdict": "Error",
                "error": "librosa not installed",
                "model_used": "none",
            }

        # Write to temp file and load
        suffix = os.path.splitext(filename)[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(audio_bytes)
            tmp.close()

            # Load audio at 16kHz mono
            waveform, sr = librosa.load(tmp.name, sr=AASIST_SAMPLE_RATE, mono=True)
            duration = len(waveform) / sr

            results = {
                "fake_probability": 0.5,
                "model_used": self.mode,
                "details": {
                    "duration_seconds": round(duration, 2),
                    "sample_rate": sr,
                    "num_samples": len(waveform),
                },
            }

            # === AASIST inference ===
            aasist_score = None
            if self.mode == "aasist" and self.model is not None:
                aasist_score = self._run_aasist(waveform)
                results["details"]["aasist_score"] = round(aasist_score, 4)
                results["details"]["aasist_architecture"] = "SincConv + Graph Attention"
                n_params = sum(p.numel() for p in self.model.parameters())
                results["details"]["aasist_params"] = f"{n_params:,}"

            # === Classical audio analysis ===
            classical_scores = self._classical_analysis(waveform, sr)
            results["details"].update(classical_scores["details"])
            classical_score = classical_scores["score"]
            results["details"]["classical_score"] = round(classical_score, 4)

            # === Combine scores ===
            if aasist_score is not None:
                # 100% driven by trained deep learning model
                results["fake_probability"] = round(aasist_score, 4)
            else:
                results["fake_probability"] = round(classical_score, 4)

            # Verdict
            fp = results["fake_probability"]
            if fp < 0.35:
                results["verdict"] = "Likely Real"
            elif fp < 0.65:
                results["verdict"] = "Uncertain"
            else:
                results["verdict"] = "Likely Fake"

            results["details"]["processing_time_ms"] = int(
                (time.time() - start_time) * 1000
            )

            return results

        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def _run_aasist(self, waveform):
        """
        Run AASIST inference on a waveform.

        Preprocessing:
          - Pad to 64,600 samples if shorter using standard AASIST circular padding
          - For audio longer than 64,600 samples, evaluate multiple segments
            with energy-weighted and median aggregation
          - Extract 160-dim spectro-temporal graph attention representations
          - Feed through trained classifier head for calibrated deepfake probability
        """
        try:
            # Pad or truncate to fixed length
            if len(waveform) < AASIST_NUM_SAMPLES:
                num_repeats = int(AASIST_NUM_SAMPLES / len(waveform)) + 1
                padded_waveform = np.tile(waveform, (1, num_repeats))[:, :AASIST_NUM_SAMPLES][0]
                return self._infer_segment(padded_waveform)
            elif len(waveform) > AASIST_NUM_SAMPLES:
                # For long audio, analyze multiple segments
                scores = []
                weights = []
                hop = AASIST_NUM_SAMPLES // 2  # 50% overlap
                for start in range(0, len(waveform) - AASIST_NUM_SAMPLES + 1, hop):
                    segment = waveform[start:start + AASIST_NUM_SAMPLES]
                    rms = float(np.sqrt(np.mean(segment ** 2)))
                    score = self._infer_segment(segment)
                    scores.append(score)
                    weights.append(max(rms, 1e-4))
                    if len(scores) >= 8:  # Max 8 segments
                        break

                if not scores:
                    return 0.5

                # Robust energy-weighted and median aggregation
                # Prevents non-speech silence/breath pauses from artificially skewing real speech
                weights = np.array(weights)
                weights /= np.sum(weights)
                weighted_mean = float(np.sum(np.array(scores) * weights))
                median_score = float(np.median(scores))
                return float(0.5 * weighted_mean + 0.5 * median_score)

            return self._infer_segment(waveform[:AASIST_NUM_SAMPLES])

        except Exception as e:
            logger.error(f"AASIST inference failed: {e}")
            return 0.5

    def _infer_segment(self, segment):
        """Run AASIST on a single 64,600-sample segment using trained model representation."""
        x = torch.FloatTensor(segment).unsqueeze(0).to(self.device)

        with torch.no_grad():
            last_hidden, logits = self.model(x)  # returns (last_hidden, logits)
            if self.calibrated_head is not None:
                h_np = last_hidden.cpu().numpy()
                probs = self.calibrated_head.predict_proba(h_np)
                spoof_prob = float(probs[0, 1])
            else:
                probs = torch.softmax(logits, dim=-1)
                spoof_prob = probs[0, 0].item()

        return spoof_prob

    def _classical_analysis(self, waveform, sr):
        """
        Classical spectral/temporal analysis for fake audio detection.

        Analyzes:
        1. Spectral flatness — Human speech has formant peaks/valleys (low flatness);
           vocoder buzz / high-frequency hiss exhibits elevated flatness.
        2. Zero-crossing rate variability — Natural speech has dynamic phoneme shifts;
           monotonous synthetic voices exhibit abnormally low variation.
        3. MFCC statistics — Natural vocal tracts show wide dynamic diversity across bands;
           synthetic models often compress or smooth higher MFCC distributions.
        4. Pitch stability (F0 CV) — Expressive human prosody exhibits dynamic pitch variation;
           robotic voices exhibit flat pitch contours.
        5. Spectral rolloff — Natural voice spans wide frequencies; band-limited vocoders cut off early.
        """
        scores = []
        details = {}

        try:
            # 1. Spectral flatness (Wiener entropy)
            flatness = librosa.feature.spectral_flatness(y=waveform)
            mean_flatness = float(np.mean(flatness))
            details["spectral_flatness"] = round(mean_flatness, 6)
            if mean_flatness < 0.015:
                scores.append(0.08)  # Natural harmonic formant resonance
            elif mean_flatness > 0.06:
                scores.append(0.70)  # Vocoder hiss / white-noise artifacts
            else:
                scores.append(0.25)

            # 2. Zero-crossing rate variability
            zcr = librosa.feature.zero_crossing_rate(waveform)
            zcr_std = float(np.std(zcr))
            details["zcr_variability"] = round(zcr_std, 6)
            if zcr_std > 0.03:
                scores.append(0.08)  # Natural phoneme transitions
            elif zcr_std < 0.012:
                scores.append(0.65)  # Monotonous synthetic artifact
            else:
                scores.append(0.25)

            # 3. MFCC analysis
            mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
            mfcc_vars = np.var(mfccs, axis=1)
            avg_mfcc_var = float(np.mean(mfcc_vars[1:]))
            details["mfcc_variance"] = round(avg_mfcc_var, 4)
            if avg_mfcc_var > 100:
                scores.append(0.08)  # Rich human vocal dynamics
            elif avg_mfcc_var < 35:
                scores.append(0.65)  # Over-smoothed synthetic voice
            else:
                scores.append(0.25)

            # 4. Pitch stability (Coefficient of Variation)
            pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if p > 0:
                    pitch_values.append(p)

            if pitch_values:
                pitch_std = float(np.std(pitch_values))
                pitch_mean = float(np.mean(pitch_values))
                details["pitch_std"] = round(pitch_std, 2)
                details["pitch_mean"] = round(pitch_mean, 2)

                cv = pitch_std / (pitch_mean + 1e-8)
                details["pitch_variability"] = round(cv, 4)
                if cv > 0.20:
                    scores.append(0.08)  # Dynamic natural human pitch
                elif cv < 0.05:
                    scores.append(0.75)  # Robotic/flat pitch artifact
                else:
                    scores.append(0.25)

            # 5. Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)
            mean_rolloff = float(np.mean(rolloff))
            details["spectral_rolloff_hz"] = round(mean_rolloff, 1)
            if mean_rolloff >= 2000:
                scores.append(0.08)  # Natural full-bandwidth speech
            elif mean_rolloff < 1600:
                scores.append(0.65)  # Band-limited vocoder artifact
            else:
                scores.append(0.25)

        except Exception as e:
            logger.error(f"Classical audio analysis failed: {e}")
            scores = [0.5]

        return {
            "score": float(np.mean(scores)) if scores else 0.5,
            "details": details,
        }
