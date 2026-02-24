"""
tests/test_detector.py — pytest unit tests for EmotionDetector and FrameBuffer.

Run from the project root:
    pytest src/tests/ -v --tb=short

Or with coverage:
    pytest src/tests/ -v --cov=src --cov-report=term-missing
"""

import os
import sys
import types
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the src directory is on sys.path so that `config` and `utils`
# can be imported without installing the package.
# ---------------------------------------------------------------------------
SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    """Return a shared EmotionDetector instance (draws annotations)."""
    from utils.detector import EmotionDetector  # noqa: PLC0415
    return EmotionDetector(draw=True, use_mediapipe=False)


@pytest.fixture()
def black_frame():
    """480×640 all-black BGR frame — no faces, safe to process."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def face_frame():
    """Synthetic frame with a white 96×96 square that mimics a skin region."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # A bright patch roughly in face-detection territory.
    frame[192:288, 272:368] = 200
    return frame


# ---------------------------------------------------------------------------
# EmotionDetector — basic contract tests
# ---------------------------------------------------------------------------

class TestEmotionDetectorContract:
    """Verify the public API shape & invariants of EmotionDetector."""

    def test_detect_returns_two_items(self, detector, black_frame):
        result = detector.detect(black_frame)
        assert isinstance(result, tuple), "detect() must return a tuple"
        assert len(result) == 2, "detect() must return (annotated_frame, result_dict)"

    def test_annotated_frame_same_shape(self, detector, black_frame):
        annotated, _ = detector.detect(black_frame)
        assert annotated.shape == black_frame.shape, \
            "Annotated frame must preserve original HxWxC shape"

    def test_result_dict_has_required_keys(self, detector, black_frame):
        _, info = detector.detect(black_frame)
        required = {"faces_detected", "emotions", "engagement", "smoothed_probs"}
        assert required.issubset(info.keys()), \
            f"result_dict missing keys: {required - info.keys()}"

    def test_no_exception_on_black_frame(self, detector, black_frame):
        """detect() must never raise on a valid all-black frame."""
        try:
            detector.detect(black_frame)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"detect() raised an unexpected exception: {exc}")

    def test_faces_detected_is_non_negative_int(self, detector, black_frame):
        _, info = detector.detect(black_frame)
        assert isinstance(info["faces_detected"], int)
        assert info["faces_detected"] >= 0

    def test_emotions_is_list(self, detector, black_frame):
        _, info = detector.detect(black_frame)
        assert isinstance(info["emotions"], list)

    def test_engagement_range(self, detector, black_frame):
        """Engagement score must always be in [0, 100]."""
        _, info = detector.detect(black_frame)
        assert 0.0 <= info["engagement"] <= 100.0, \
            f"Engagement {info['engagement']} out of [0, 100]"


# ---------------------------------------------------------------------------
# EmotionDetector — single-face prediction quality tests
# ---------------------------------------------------------------------------

class TestEmotionDetectorPrediction:
    """Verify prediction structure when a face is detected."""

    def test_probabilities_sum_to_one(self, detector, face_frame):
        """If any faces are detected, probabilities must sum to ~1."""
        _, info = detector.detect(face_frame)
        if not info["emotions"]:
            pytest.skip("No face detected in synthetic frame — skipping probability test")
        probs = info["emotions"][0]["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-3, f"Probabilities sum to {total:.4f}, expected ~1.0"

    def test_emotion_label_is_valid(self, detector, face_frame):
        """Emotion label must be one of the 7 canonical class names."""
        from config import LABELS  # noqa: PLC0415
        _, info = detector.detect(face_frame)
        if not info["emotions"]:
            pytest.skip("No face detected in synthetic frame — skipping label test")
        emotion = info["emotions"][0]["emotion"]
        assert emotion in LABELS, f"Unexpected emotion label: {emotion!r}"

    def test_confidence_in_range(self, detector, face_frame):
        _, info = detector.detect(face_frame)
        if not info["emotions"]:
            pytest.skip("No face detected in synthetic frame — skipping confidence test")
        conf = info["emotions"][0]["confidence"]
        assert 0.0 <= conf <= 100.0, f"Confidence {conf} out of [0, 100]"

    def test_latency_is_positive(self, detector, face_frame):
        _, info = detector.detect(face_frame)
        if not info["emotions"]:
            pytest.skip("No face detected in synthetic frame — skipping latency test")
        latency = info["emotions"][0].get("latency", -1)
        assert latency > 0, f"Latency should be positive, got {latency}"


# ---------------------------------------------------------------------------
# EmotionDetector — smoothing and reset
# ---------------------------------------------------------------------------

class TestEmotionDetectorSmoothing:
    """Verify EWA temporal smoothing behaviour."""

    def test_smoothed_probs_same_keys_as_labels(self, detector, black_frame):
        from config import LABELS  # noqa: PLC0415
        detector.reset_smoothing()
        _, info = detector.detect(black_frame)
        if info["smoothed_probs"]:
            diff = set(info["smoothed_probs"].keys()) ^ set(LABELS)
            assert not diff, f"Unexpected keys in smoothed_probs: {diff}"

    def test_reset_smoothing_clears_state(self, detector, black_frame):
        detector.detect(black_frame)  # Build up some state
        detector.reset_smoothing()
        assert detector._ewa_probs is None, \
            "reset_smoothing() must clear internal EWA state (_ewa_probs should be None)"


# ---------------------------------------------------------------------------
# EmotionDetector — engagement score computation
# ---------------------------------------------------------------------------

class TestEngagementScore:
    """Unit-test the engagement calculation using synthetic probability dicts."""

    def test_happy_gives_high_engagement(self, detector):
        probs = {label: 0.0 for label in ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]}
        probs["Happy"] = 1.0
        score = detector.compute_engagement(probs)
        assert score > 60, f"Happy should give high engagement, got {score}"

    def test_neutral_gives_mid_engagement(self, detector):
        probs = {label: 0.0 for label in ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]}
        probs["Neutral"] = 1.0
        score = detector.compute_engagement(probs)
        assert 30 <= score <= 70, f"Neutral should give mid engagement, got {score}"

    def test_engagement_clamped(self, detector):
        """Extreme inputs must not produce values outside [0, 100]."""
        probs = {label: 0.0 for label in ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]}
        # All weight on highest-weight emotion
        probs["Surprise"] = 1.0
        score = detector.compute_engagement(probs)
        assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# Config integrity tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Sanity-check config.py for correctness."""

    def test_labels_count(self):
        from config import LABELS  # noqa: PLC0415
        assert len(LABELS) == 7, f"Expected 7 LABELS, found {len(LABELS)}"

    def test_labels_title_case(self):
        from config import LABELS  # noqa: PLC0415
        for label in LABELS:
            assert label == label.title(), \
                f"Label {label!r} is not Title Case — inconsistency detected"

    def test_img_size_48(self):
        from config import IMG_SIZE  # noqa: PLC0415
        assert IMG_SIZE == (48, 48), f"Expected (48, 48), got {IMG_SIZE}"

    def test_emotion_colors_all_labels(self):
        from config import LABELS, EMOTION_COLORS  # noqa: PLC0415
        missing = [lbl for lbl in LABELS if lbl not in EMOTION_COLORS]
        assert not missing, f"EMOTION_COLORS missing entries for: {missing}"

    def test_engagement_weights_all_labels(self):
        from config import LABELS, ENGAGEMENT_WEIGHTS  # noqa: PLC0415
        missing = [lbl for lbl in LABELS if lbl not in ENGAGEMENT_WEIGHTS]
        assert not missing, f"ENGAGEMENT_WEIGHTS missing entries for: {missing}"
