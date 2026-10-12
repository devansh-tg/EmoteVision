"""Central configuration for EmoteVision."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency may not be installed yet
    load_dotenv = None


WEB_APP_DIR = Path(__file__).parent.resolve()

if load_dotenv is not None:
    load_dotenv(WEB_APP_DIR / ".env")


def _env(name: str, default=None, *aliases: str):
    for key in (name, *aliases):
        value = os.getenv(key)
        if value not in (None, ""):
            return value
    return default


def _env_bool(name: str, default: bool = False, *aliases: str) -> bool:
    return str(_env(name, str(default).lower(), *aliases)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str, default: int, *aliases: str) -> int:
    return int(_env(name, default, *aliases))


def _env_float(name: str, default: float, *aliases: str) -> float:
    return float(_env(name, default, *aliases))


# Paths
MODEL_PATH = WEB_APP_DIR / _env("EMOTEVISION_MODEL", "../model/custom_cnn/model.h5")
TFLITE_PATH = WEB_APP_DIR / _env("EMOTEVISION_TFLITE", "../model/custom_cnn/model.tflite")
META_PATH = WEB_APP_DIR / _env("EMOTEVISION_META", "../model/custom_cnn/model_meta.json")
EMOJI_DIR = WEB_APP_DIR / "emojis"
CASCADE_PATH = _env("EMOTEVISION_CASCADE", None)

# Model
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = (48, 48)
USE_TFLITE = _env("EMOTEVISION_USE_TFLITE", "auto", "USE_TFLITE").lower()

# Camera
CAMERA_INDEX = _env_int("EMOTEVISION_CAMERA", 0, "CAMERA_INDEX")
CAMERA_WIDTH = _env_int("EMOTEVISION_CAMERA_WIDTH", 640, "CAMERA_WIDTH")
CAMERA_HEIGHT = _env_int("EMOTEVISION_CAMERA_HEIGHT", 480, "CAMERA_HEIGHT")
GUI_CAMERA_WIDTH = _env_int("EMOTEVISION_GUI_CAMERA_WIDTH", 1280, "GUI_CAMERA_WIDTH")
GUI_CAMERA_HEIGHT = _env_int("EMOTEVISION_GUI_CAMERA_HEIGHT", 720, "GUI_CAMERA_HEIGHT")

# Inference
SMOOTHING_ALPHA = _env_float("EMOTEVISION_SMOOTHING", 0.4, "SMOOTHING_ALPHA")
SCALE_FACTOR = _env_float("EMOTEVISION_SCALE_FACTOR", 1.3)
MIN_NEIGHBORS = _env_int("EMOTEVISION_MIN_NEIGHBORS", 5)
MIN_FACE_SIZE = (48, 48)

# Server
HOST = _env("EMOTEVISION_HOST", "0.0.0.0")
PORT = _env_int("EMOTEVISION_PORT", 5000)
DEBUG = _env_bool("EMOTEVISION_DEBUG", False)
SECRET_KEY = _env("EMOTEVISION_SECRET_KEY", "change-me-for-production")
CORS_ORIGINS = _env("EMOTEVISION_CORS_ORIGINS", "*")

# UI colors shared by backend and frontend
EMOTION_COLORS = {
    "Angry": "#E5484D",
    "Disgust": "#30A46C",
    "Fear": "#8E4EC6",
    "Happy": "#F5A524",
    "Neutral": "#8B949E",
    "Sad": "#3E8EDE",
    "Surprise": "#F76808",
}

GRADIENT_COLORS = {
    "Angry": ["#E5484D", "#C62A2F", "#9F1D24"],
    "Disgust": ["#30A46C", "#218358", "#196C49"],
    "Fear": ["#8E4EC6", "#793AAF", "#5B2A86"],
    "Happy": ["#F5A524", "#D88912", "#AD6B08"],
    "Neutral": ["#8B949E", "#6E7781", "#57606A"],
    "Sad": ["#3E8EDE", "#2870BD", "#1D5D9E"],
    "Surprise": ["#F76808", "#D95700", "#A84300"],
}

ENGAGEMENT_WEIGHTS = {
    "Happy": 1.0,
    "Surprise": 0.8,
    "Neutral": 0.2,
    "Fear": -0.3,
    "Angry": -0.3,
    "Disgust": -0.5,
    "Sad": -0.5,
}
