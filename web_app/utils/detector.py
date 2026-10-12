# utils/detector.py
# Shared emotion detection pipeline used by app.py, gui_advanced.py, and test.py.
# Eliminates code duplication and ensures consistent behaviour across all entry points.

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    CASCADE_PATH, ENGAGEMENT_WEIGHTS, IMG_SIZE, LABELS,
    META_PATH, MIN_FACE_SIZE, MIN_NEIGHBORS, MODEL_PATH,
    SCALE_FACTOR, SMOOTHING_ALPHA, TFLITE_PATH, USE_TFLITE,
)

log = logging.getLogger(__name__)


# ── Model loading helpers ───────────────────────────────────────────────────────

def _should_use_tflite() -> bool:
    if USE_TFLITE == "true":
        return True
    if USE_TFLITE == "false":
        return False
    # "auto": use TFLite if the file exists
    return TFLITE_PATH.exists()


def _load_model():
    """Load the best available model. Prefers TFLite for speed, falls back to Keras."""
    if _should_use_tflite() and TFLITE_PATH.exists():
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore
            interpreter = tflite.Interpreter(model_path=str(TFLITE_PATH))
        except ImportError:
            import tensorflow.lite as tflite  # type: ignore
            interpreter = tflite.Interpreter(model_path=str(TFLITE_PATH))
        interpreter.allocate_tensors()
        log.info("Loaded TFLite model from %s", TFLITE_PATH)
        return ("tflite", interpreter)

    # Fall back to Keras / TF SavedModel
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run train.py first or set EMOTEVISION_MODEL env var to the correct path."
        )
    from tensorflow.keras.models import load_model as keras_load  # type: ignore
    keras_model = keras_load(str(MODEL_PATH))
    log.info("Loaded Keras model from %s", MODEL_PATH)
    return ("keras", keras_model)


# ── Metadata ───────────────────────────────────────────────────────────────────

def load_model_meta() -> dict:
    """Return metadata saved during training (accuracy, epochs, timestamp, etc.)."""
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {"val_accuracy": None, "note": "model_meta.json not found"}


# ── Face cascade ───────────────────────────────────────────────────────────────

def _build_cascade() -> cv2.CascadeClassifier:
    path = CASCADE_PATH or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade from '{path}'")
    return cascade


# ── EmotionDetector class ───────────────────────────────────────────────────────

class EmotionDetector:
    """
    Thread-safe emotion detector with optional MediaPipe face detection
    and exponential-weighted-average temporal smoothing.

    Usage::

        detector = EmotionDetector()
        annotated_frame, result = detector.detect(frame)
        # result = {
        #   'faces_detected': int,
        #   'emotions': [{'emotion', 'confidence', 'probabilities', 'bbox', 'latency'}, ...],
        #   'engagement': float,     # 0-100
        #   'smoothed_probs': dict,  # EWA-smoothed probabilities for primary face
        # }
    """

    def __init__(self, draw: bool = True, use_mediapipe: bool = False):
        self.draw = draw
        self._use_mediapipe = use_mediapipe
        self._model_type, self._model = _load_model()
        self._cascade = _build_cascade()
        self._mp_detector = self._init_mediapipe() if use_mediapipe else None

        # CLAHE for adaptive contrast normalization (improves accuracy in varied lighting)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # EWA smoothing state (per-class probability vector)
        self._ewa_probs: Optional[np.ndarray] = None
        self.meta = load_model_meta()

    # ── MediaPipe ──────────────────────────────────────────────────────────────

    @staticmethod
    def _init_mediapipe():
        try:
            import mediapipe as mp  # type: ignore
            detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            log.info("MediaPipe face detection enabled")
            return detector
        except ImportError:
            log.warning("mediapipe not installed — falling back to Haar Cascade")
            return None

    def _detect_faces(self, frame: np.ndarray, gray: np.ndarray):
        """Detect faces using MediaPipe (preferred) or Haar Cascade (fallback).

        Returns list of (x, y, w, h) tuples in pixel coordinates.
        """
        if self._mp_detector is not None:
            try:
                import mediapipe as mp  # type: ignore
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_detector.process(rgb)
                boxes = []
                if results.detections:
                    h, w = frame.shape[:2]
                    for det in results.detections:
                        bb = det.location_data.relative_bounding_box
                        x = max(0, int(bb.xmin * w))
                        y = max(0, int(bb.ymin * h))
                        bw = int(bb.width * w)
                        bh = int(bb.height * h)
                        boxes.append((x, y, bw, bh))
                return boxes
            except Exception as exc:
                log.debug("MediaPipe detection failed (%s); using Haar", exc)

        # Haar Cascade fallback
        boxes = self._cascade.detectMultiScale(
            gray, SCALE_FACTOR, MIN_NEIGHBORS, minSize=MIN_FACE_SIZE
        )
        if len(boxes) == 0:
            return []
        return [tuple(b) for b in boxes]

    # ── Inference ──────────────────────────────────────────────────────────────

    def _predict(self, roi_gray: np.ndarray) -> np.ndarray:
        """Run model inference on a grayscale ROI. Returns probability vector."""
        resized = cv2.resize(roi_gray, IMG_SIZE)
        # Apply CLAHE to normalize contrast before inference
        resized = self._clahe.apply(resized)
        inp = resized.reshape(1, *IMG_SIZE, 1).astype(np.float32) / 255.0

        if self._model_type == "tflite":
            interpreter = self._model
            in_idx  = interpreter.get_input_details()[0]["index"]
            out_idx = interpreter.get_output_details()[0]["index"]
            interpreter.set_tensor(in_idx, inp)
            interpreter.invoke()
            return interpreter.get_tensor(out_idx)[0]
        else:
            return self._model.predict(inp, verbose=0)[0]

    # ── Smoothing ──────────────────────────────────────────────────────────────

    def _smooth(self, probs: np.ndarray) -> np.ndarray:
        """Apply EWA temporal smoothing to reduce emotion flicker."""
        if self._ewa_probs is None:
            self._ewa_probs = probs.copy()
        else:
            self._ewa_probs = (
                SMOOTHING_ALPHA * probs + (1 - SMOOTHING_ALPHA) * self._ewa_probs
            )
        return self._ewa_probs

    def reset_smoothing(self):
        self._ewa_probs = None

    # ── Engagement ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_engagement(probs) -> float:
        """
        Weighted engagement score in [0, 100].
        Accepts a numpy probability array OR a label→float dict.
        Positive emotions (Happy, Surprise) increase it;
        negative emotions (Sad, Angry, Disgust, Fear) decrease it.
        """
        if isinstance(probs, dict):
            score = sum(
                ENGAGEMENT_WEIGHTS.get(label, 0) * float(val)
                for label, val in probs.items()
            ) * 100
        else:
            score = sum(
                ENGAGEMENT_WEIGHTS.get(LABELS[i], 0) * float(probs[i])
                for i in range(len(LABELS))
            ) * 100
        return float(np.clip(score + 50, 0, 100))  # centre at 50

    # ── Main entry point ───────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Detect emotions in *frame* (BGR).

        Returns
        -------
        annotated_frame : np.ndarray
            Frame with bounding boxes / labels drawn (if ``draw=True``).
        result : dict
            Structured detection output.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE to improve face detection in uneven lighting
        gray = self._clahe.apply(gray)
        faces = self._detect_faces(frame, gray)

        result: dict = {
            "faces_detected": len(faces),
            "emotions": [],
            "engagement": 50.0,
            "smoothed_probs": {},
        }

        for i, (x, y, w, h) in enumerate(faces):
            roi = gray[y : y + h, x : x + w]
            t0 = time.time()
            probs = self._predict(roi)
            latency_ms = (time.time() - t0) * 1000

            # EWA smoothing only applied to the primary (first) face
            display_probs = self._smooth(probs) if i == 0 else probs

            idx = int(np.argmax(display_probs))
            emotion = LABELS[idx]
            confidence = float(display_probs[idx] * 100)
            prob_dict = {LABELS[j]: float(display_probs[j] * 100) for j in range(len(LABELS))}

            if i == 0:
                result["smoothed_probs"] = prob_dict
                result["engagement"] = self.compute_engagement(display_probs)

            result["emotions"].append({
                "emotion":       emotion,
                "confidence":    confidence,
                "probabilities": prob_dict,
                "bbox":          {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "latency":       round(latency_ms, 2),
            })

            if self.draw:
                self._draw(frame, x, y, w, h, emotion, confidence)

        return frame, result

    # ── Drawing ────────────────────────────────────────────────────────────────

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    def _draw(self, frame, x, y, w, h, emotion, confidence):
        from config import EMOTION_COLORS
        color_hex = EMOTION_COLORS.get(emotion, "#00FF88")
        color_bgr = self._hex_to_bgr(color_hex)

        # Outer glow rectangle
        cv2.rectangle(frame, (x - 4, y - 4), (x + w + 4, y + h + 4), color_bgr, 2)
        # Main bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

        # Corner L-brackets
        corner = 18
        for (cx, cy) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            dx = corner if cx == x else -corner
            dy = corner if cy == y else -corner
            cv2.line(frame, (cx, cy), (cx + dx, cy), color_bgr, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy), color_bgr, 3)

        # Label background + text
        label = f"{emotion}: {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - th - 16), (x + tw + 10, y), color_bgr, -1)
        cv2.putText(frame, label, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
