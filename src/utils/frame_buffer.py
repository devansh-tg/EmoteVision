# utils/frame_buffer.py
# Thread-safe camera frame buffer.
#
# A background thread continuously grabs frames from the camera so that:
#   1. The MJPEG stream (/video_feed) and the /api/predict endpoint never
#      race against each other on cam.read().
#   2. The most recent frame is always available without blocking I/O.

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from config import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH

log = logging.getLogger(__name__)


class FrameBuffer:
    """
    Singleton-style thread-safe frame buffer.

    Usage::

        buf = FrameBuffer.get_instance()
        frame = buf.get_frame()   # returns latest BGR ndarray or None
        buf.release()             # call on shutdown
    """

    _instance: Optional["FrameBuffer"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._open()

    # ── Singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "FrameBuffer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Camera ───────────────────────────────────────────────────────────────

    def _open(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {CAMERA_INDEX}. "
                "Check that no other application is using it and that CAMERA_INDEX is correct."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True, name="FrameBuffer")
        self._thread.start()
        log.info("FrameBuffer started (camera index %d, %dx%d)", CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)

    # ── Background reader ────────────────────────────────────────────────────

    def _reader(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._frame_lock:
                    self._frame = frame
            else:
                log.warning("Camera read failed; retrying in 100 ms")
                time.sleep(0.1)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame, or None if none has been captured yet."""
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    def is_open(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    def release(self):
        """Stop the reader thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        log.info("FrameBuffer released")
        with FrameBuffer._lock:
            FrameBuffer._instance = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
