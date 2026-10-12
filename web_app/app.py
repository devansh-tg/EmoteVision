# app.py - EmoteVision Flask Backend (optimized)
#
# Key improvements over original:
#   - Flask-SocketIO replaces 500ms REST polling (true real-time push)
#   - Single shared FrameBuffer thread eliminates camera read races
#   - EmotionDetector (utils/detector.py) replaces duplicated pipeline
#   - Session logging with CSV export endpoint
#   - Real model accuracy read from model_meta.json
#   - Proper error handling throughout
#   - atexit camera cleanup
#   - Config-driven host / port / debug

from __future__ import annotations

import atexit
import csv
import io
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, Response, jsonify, make_response, render_template
from flask_socketio import SocketIO

import config
from utils.detector import EmotionDetector, load_model_meta
from utils.frame_buffer import FrameBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
_cors_origins = (
    config.CORS_ORIGINS
    if config.CORS_ORIGINS == "*"
    else [origin.strip() for origin in config.CORS_ORIGINS.split(",") if origin.strip()]
)
socketio = SocketIO(app, cors_allowed_origins=_cors_origins, async_mode="threading")

# ── Global state ───────────────────────────────────────────────────────────────
_detector: EmotionDetector | None = None
_buffer:   FrameBuffer | None     = None
_session_log: list[dict]          = []
_prediction_times: deque          = deque(maxlen=60)
_total_predictions: int           = 0
_session_start: float             = time.time()
_model_meta: dict                 = {}
_cached_annotated: bytes | None   = None   # shared MJPEG frame from emit loop
_cached_frame_lock                = threading.Lock()
_detector_lock                    = threading.Lock()
_session_lock                     = threading.Lock()
_background_lock                  = threading.Lock()
_background_started: bool         = False


def get_detector() -> EmotionDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            log.info("Loading AI model…")
            _detector = EmotionDetector(draw=True)
            log.info("Model loaded.")
    return _detector


def get_buffer() -> FrameBuffer:
    global _buffer
    if _buffer is None:
        _buffer = FrameBuffer.get_instance()
    return _buffer


def ensure_background_task() -> None:
    """Start the detection broadcaster once, when a route or client needs it."""
    global _background_started
    with _background_lock:
        if not _background_started:
            socketio.start_background_task(_emit_loop)
            _background_started = True


# ── Cleanup ────────────────────────────────────────────────────────────────────
@atexit.register
def _cleanup():
    global _buffer
    if _buffer is not None:
        _buffer.release()
        log.info("Camera released on exit.")


# ── SocketIO background broadcaster ───────────────────────────────────────────
def _emit_loop():
    """Background thread: grab frames, run detection, push to all WS clients.
    Also caches the annotated JPEG bytes so _generate_mjpeg() never re-infers.
    """
    global _total_predictions, _cached_annotated
    import cv2 as _cv2

    while True:
        try:
            detector = get_detector()
            buf = get_buffer()
        except Exception as exc:
            log.exception("Detection worker is waiting for startup: %s", exc)
            socketio.sleep(2.0)
            continue

        frame = buf.get_frame()
        if frame is None:
            socketio.sleep(0.033)
            continue

        try:
            annotated, result = detector.detect(frame)
        except Exception as exc:
            log.exception("Detection failed for current frame: %s", exc)
            socketio.sleep(0.25)
            continue

        # Cache the annotated frame for the MJPEG stream (avoids double inference)
        ok, enc = _cv2.imencode(".jpg", annotated, [_cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _cached_frame_lock:
                _cached_annotated = enc.tobytes()

        if result["emotions"]:
            _total_predictions += 1
            em = result["emotions"][0]
            _prediction_times.append(em["latency"])

            with _session_lock:
                _session_log.append({
                    "timestamp":  round(time.time(), 3),
                    "emotion":    em["emotion"],
                    "confidence": round(em["confidence"], 2),
                    "latency_ms": round(em["latency"], 2),
                })

            socketio.emit("emotion_update", {
                "emotion":           em["emotion"],
                "confidence":        em["confidence"],
                "probabilities":     em["probabilities"],
                "smoothed_probs":    result["smoothed_probs"],
                "engagement":        result["engagement"],
                "latency":           em["latency"],
                "faces_detected":    result["faces_detected"],
                "total_predictions": _total_predictions,
                "uptime":            int(time.time() - _session_start),
            })
        else:
            socketio.emit("emotion_update", {
                "emotion":           None,
                "faces_detected":    0,
                "uptime":            int(time.time() - _session_start),
                "total_predictions": _total_predictions,
            })

        socketio.sleep(0.1)   # ~10 updates/sec for lower CPU and smoother UI




# ── MJPEG stream — serves cached annotated frames from _emit_loop ─────────────
def _generate_mjpeg():
    """Yield MJPEG frames. Uses frames annotated by _emit_loop (zero extra inference)."""
    import cv2 as _cv2
    import numpy as _np

    placeholder = _np.zeros((480, 640, 3), dtype=_np.uint8)
    _cv2.putText(placeholder, "Camera unavailable", (120, 240),
                 _cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 212, 255), 2)
    _, _ph = _cv2.imencode(".jpg", placeholder)
    placeholder_bytes = _ph.tobytes()

    while True:
        with _cached_frame_lock:
            jpg = _cached_annotated
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + (jpg if jpg is not None else placeholder_bytes)
               + b"\r\n")
        time.sleep(0.04)  # cap stream at 25 fps max


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/video_feed")
def video_feed():
    ensure_background_task()
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/predict")
def predict():
    """Snapshot prediction for external REST consumers."""
    ensure_background_task()
    frame = get_buffer().get_frame()
    if frame is None:
        return jsonify({"error": "Camera not available"}), 503

    _, result = get_detector().detect(frame.copy())
    result["stats"] = {
        "total_predictions": _total_predictions,
        "avg_latency": round(sum(_prediction_times) / len(_prediction_times), 2)
        if _prediction_times else 0,
        "uptime": int(time.time() - _session_start),
    }
    return jsonify(result)


@app.route("/api/stats")
def get_stats():
    meta = _model_meta or load_model_meta()
    return jsonify({
        "total_predictions": _total_predictions,
        "avg_latency": round(sum(_prediction_times) / len(_prediction_times), 2)
        if _prediction_times else 0,
        "uptime": int(time.time() - _session_start),
        "model_accuracy": meta.get("val_accuracy"),
        "model_trained_at": meta.get("timestamp"),
        "emotions_count": len(config.LABELS),
    })


@app.route("/api/config")
def get_config():
    """Serve shared colour/label config to the frontend."""
    return jsonify({
        "labels":         config.LABELS,
        "emotion_colors": config.EMOTION_COLORS,
    })


@app.route("/api/session/export")
def session_export():
    """Download current session as CSV."""
    with _session_lock:
        rows = list(_session_log)

    if not rows:
        return jsonify({"error": "No session data yet"}), 404

    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=["timestamp", "emotion", "confidence", "latency_ms"]
    )
    writer.writeheader()
    writer.writerows(rows)

    resp = make_response(output.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=emotevision_session.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route("/api/session/reset", methods=["POST"])
def session_reset():
    global _session_log, _total_predictions, _session_start
    with _session_lock:
        _session_log = []
    _total_predictions = 0
    _session_start = time.time()
    if _detector is not None:
        get_detector().reset_smoothing()
    return jsonify({"status": "ok"})


# ── SocketIO events ────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    ensure_background_task()
    log.info("WS client connected")


@socketio.on("disconnect")
def on_disconnect():
    log.info("WS client disconnected")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting EmoteVision on http://%s:%d", config.HOST, config.PORT)
    try:
        _model_meta = load_model_meta()
    except Exception as exc:
        log.warning("Model metadata could not be loaded: %s", exc)

    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        allow_unsafe_werkzeug=True,
    )
