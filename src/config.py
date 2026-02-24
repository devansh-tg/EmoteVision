# config.py - Single source of truth for all project constants
# All other modules import from here; never define these values twice.

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent.resolve()
MODEL_PATH = SRC_DIR / os.getenv("EMOTEVISION_MODEL", "model.h5")
TFLITE_PATH = SRC_DIR / os.getenv("EMOTEVISION_TFLITE", "model.tflite")
META_PATH  = SRC_DIR / "model_meta.json"
EMOJI_DIR  = SRC_DIR / "emojis"
CASCADE_PATH = None  # None → use cv2.data.haarcascades default

# ── Model ──────────────────────────────────────────────────────────────────────
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = (48, 48)

# Use TFLite runtime for faster CPU inference when the .tflite file exists
USE_TFLITE = os.getenv("USE_TFLITE", "auto")  # "auto" | "true" | "false"

# ── Camera ─────────────────────────────────────────────────────────────────────
CAMERA_INDEX     = int(os.getenv("CAMERA_INDEX", 0))
CAMERA_WIDTH     = 640
CAMERA_HEIGHT    = 480
GUI_CAMERA_WIDTH = 1280
GUI_CAMERA_HEIGHT= 720

# ── Inference ──────────────────────────────────────────────────────────────────
# Temporal smoothing: EWA alpha (0=fully smooth, 1=no smoothing)
SMOOTHING_ALPHA = 0.4
# Haar Cascade detection params
SCALE_FACTOR    = 1.3
MIN_NEIGHBORS   = 5
MIN_FACE_SIZE   = (48, 48)

# ── Server ─────────────────────────────────────────────────────────────────────
HOST  = os.getenv("EMOTEVISION_HOST", "0.0.0.0")
PORT  = int(os.getenv("EMOTEVISION_PORT", 5000))
DEBUG = os.getenv("EMOTEVISION_DEBUG", "false").lower() == "true"

# ── UI Colors (shared between Python GUI and served to JS via /api/config) ─────
EMOTION_COLORS = {
    'Angry':    '#FF1744',
    'Disgust':  '#00E676',
    'Fear':     '#D500F9',
    'Happy':    '#FFEA00',
    'Neutral':  '#B0BEC5',
    'Sad':      '#00B0FF',
    'Surprise': '#FF6D00',
}

GRADIENT_COLORS = {
    'Angry':    ['#FF1744', '#F50057', '#D50000'],
    'Disgust':  ['#00E676', '#00C853', '#1DE9B6'],
    'Fear':     ['#D500F9', '#AA00FF', '#9C27B0'],
    'Happy':    ['#FFEA00', '#FFD600', '#FFC400'],
    'Neutral':  ['#B0BEC5', '#90A4AE', '#78909C'],
    'Sad':      ['#00B0FF', '#0091EA', '#2196F3'],
    'Surprise': ['#FF6D00', '#FF9100', '#FF9800'],
}

# Engagement score weights per emotion (clamped 0-100 by detector)
ENGAGEMENT_WEIGHTS = {
    'Happy':    1.0,
    'Surprise': 0.8,
    'Neutral':  0.2,
    'Fear':    -0.3,
    'Angry':   -0.3,
    'Disgust': -0.5,
    'Sad':     -0.5,
}
