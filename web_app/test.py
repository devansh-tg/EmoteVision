# test.py — EmoteVision quick webcam test
# Uses the shared EmotionDetector and config — no more duplicated pipeline or
# inconsistent label casing vs app.py / gui_advanced.py.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from config import CAMERA_INDEX, EMOJI_DIR, LABELS
from utils.detector import EmotionDetector

# ── Load shared detector ────────────────────────────────────────────────────────
try:
    detector = EmotionDetector(draw=True)
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ── Preload emoji images ────────────────────────────────────────────────────────
emoji_imgs: dict[str, np.ndarray | None] = {}
for label in LABELS:
    path = EMOJI_DIR / f"{label.lower()}.png"
    emoji_imgs[label] = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None


def overlay_image(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-composite an RGBA/RGB image onto bg at (x, y)."""
    h, w = fg.shape[:2]
    x = max(0, x); y = max(0, y)
    # Clip to frame boundaries
    bh, bw = bg.shape[:2]
    if y >= bh or x >= bw:
        return bg
    h = min(h, bh - y)
    w = min(w, bw - x)
    fg = fg[:h, :w]

    if fg.shape[2] == 4:
        alpha = fg[:, :, 3:4] / 255.0
        bg[y:y+h, x:x+w] = (alpha * fg[:, :, :3] + (1 - alpha) * bg[y:y+h, x:x+w]).astype(np.uint8)
    else:
        bg[y:y+h, x:x+w] = fg[:, :, :3]
    return bg


# ── Webcam loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera at index {CAMERA_INDEX}.")
    sys.exit(1)

print("EmoteVision test running — press Q to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed; exiting.")
        break

    annotated, result = detector.detect(frame)

    for em in result["emotions"]:
        label   = em["emotion"]
        bbox    = em["bbox"]
        x, y, w = bbox["x"], bbox["y"], bbox["w"]
        emoji   = emoji_imgs.get(label)
        if emoji is not None:
            resized = cv2.resize(emoji, (100, 100))
            annotated = overlay_image(annotated, resized, x + w // 2 - 50, max(0, y - 120))

    # Engagement score overlay
    eng = result.get("engagement", 50)
    cv2.putText(annotated, f"Engagement: {eng:.0f}/100",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)

    cv2.imshow("EmoteVision — press Q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

