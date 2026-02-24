<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Socket.IO-4.x-010101?logo=socketdotio&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<h1 align="center">🎭 EmoteVision</h1>
<p align="center">
  <strong>Real-time facial emotion detection powered by deep learning</strong><br>
  Live webcam analysis · WebSocket streaming · 3D glassmorphism UI
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Real-Time Detection** | Detects 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from webcam feed at ~15-30 FPS |
| **Deep Learning CNN** | Custom 4-block convolutional neural network (64→128→256→512 filters) trained on 35,000+ images |
| **WebSocket Streaming** | Flask-SocketIO pushes emotion data instantly — no polling, sub-100ms latency |
| **3D Interactive UI** | Glassmorphic cards with mouse-tilt perspective, particle neural network background, neon glow system |
| **Live Analytics** | Confidence bars, emotion trend chart (Chart.js), engagement gauge, inference timer |
| **Session Management** | Export session data as CSV, reset stats, persistent theme preference |
| **CLAHE Preprocessing** | Adaptive histogram equalization at inference for robust detection in varied lighting |
| **EWA Smoothing** | Exponential weighted average reduces emotion flicker for stable real-time output |
| **Dark / Light Theme** | Full theme toggle with localStorage persistence |
| **Keyboard Shortcuts** | `R` = manual predict, `H` = home |

---

## 🏗️ Architecture

```
┌──────────────┐    WebSocket     ┌──────────────────┐
│   Browser    │◄────────────────►│  Flask-SocketIO   │
│  (JS + CSS)  │   emotion_update │    (app.py)       │
│              │◄─── MJPEG ──────│                    │
└──────────────┘                  └────────┬───────────┘
                                           │
                                  ┌────────▼───────────┐
                                  │  EmotionDetector    │
                                  │  (utils/detector.py)│
                                  │                     │
                                  │  ┌───────────────┐  │
                                  │  │ Keras / TFLite│  │
                                  │  │   CNN Model   │  │
                                  │  └───────────────┘  │
                                  │  ┌───────────────┐  │
                                  │  │  Haar / Media- │  │
                                  │  │  Pipe Face Det│  │
                                  │  └───────────────┘  │
                                  │  ┌───────────────┐  │
                                  │  │ CLAHE + EWA   │  │
                                  │  │ Preprocessing │  │
                                  │  └───────────────┘  │
                                  └─────────────────────┘
```

---

## 📁 Project Structure

```
src/
├── app.py                 # Flask-SocketIO server (routes, WebSocket, MJPEG stream)
├── config.py              # Centralized configuration (env-var overridable)
├── train.py               # Model training script (argparse, callbacks, confusion matrix)
├── test.py                # Quick webcam test with emoji overlay
├── export_tflite.py       # Convert .h5 → .tflite with optional quantization
├── model.h5               # Trained Keras model (~68-70% val accuracy on FER-2013)
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable reference
│
├── utils/
│   ├── __init__.py
│   ├── detector.py        # EmotionDetector class (inference, smoothing, drawing)
│   └── frame_buffer.py    # Thread-safe singleton camera buffer
│
├── static/
│   ├── style.css          # 3D glassmorphism UI (dark/light themes)
│   └── script.js          # Socket.IO client, Chart.js, particles, 3D tilt
│
├── templates/
│   ├── index.html         # Main detection page
│   └── about.html         # About / tech stack page
│
├── emojis/                # 7 emotion emoji PNGs
│   ├── angry.png ... surprise.png
│
├── tests/
│   ├── __init__.py
│   └── test_detector.py   # Pytest unit tests for detector & config
│
└── data/                  # FER-2013 dataset (not included in repo)
    ├── train/
    │   ├── angry/ ... surprise/
    └── test/
        ├── angry/ ... surprise/
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested on 3.11)
- **Webcam** (built-in or USB)
- ~2 GB disk space (TensorFlow)

### 1. Clone the repository

```bash
git clone https://github.com/devansh-tg/emojify--copy-.git
cd emojify--copy-/src
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. The model takes ~60-80s to load on first run (TensorFlow initialization).

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Architecture** | Sequential CNN — 4 conv blocks + GlobalAveragePooling + 2 Dense layers |
| **Input** | 48×48 grayscale face crop |
| **Output** | 7-class softmax (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) |
| **Parameters** | ~2.5M |
| **Training Data** | FER-2013 (35,887 images) |
| **Validation Accuracy** | ~68-70% (human agreement on FER-2013 is ~65-72%) |
| **Optimizer** | Adam (lr=2e-4) |
| **Regularization** | L2 (5e-4), Dropout (0.20–0.40), BatchNorm |
| **Augmentation** | Rotation, shift, flip, zoom, brightness, channel shift |

### Conv Block Structure
```
Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout
Filters: 64 → 128 → 256 → 512
```

### Training

To retrain from scratch (requires dataset in `data/` folder):

```bash
python train.py --train-dir data/train --val-dir data/test --epochs 150 --batch-size 64
```

**Output files:**
- `model.h5` / `model_best.h5` — trained model
- `model_meta.json` — accuracy metrics + metadata
- `confusion_matrix.png` — per-class detection heatmap
- `classification_report.txt` — precision / recall / F1
- `training_history.png` — accuracy, loss, gap curves

### Export to TFLite (optional, for faster inference)

```bash
python export_tflite.py --input model.h5 --output model.tflite --quantize dynamic
```

---

## ⚙️ Configuration

All settings are configurable via **environment variables** or the `.env` file. See [.env.example](src/.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMOTEVISION_MODEL` | `model.h5` | Path to Keras model |
| `EMOTEVISION_TFLITE` | `model.tflite` | Path to TFLite model |
| `EMOTEVISION_USE_TFLITE` | `auto` | `true`, `false`, or `auto` |
| `EMOTEVISION_CAMERA` | `0` | Camera index |
| `EMOTEVISION_HOST` | `0.0.0.0` | Server bind address |
| `EMOTEVISION_PORT` | `5000` | Server port |
| `EMOTEVISION_SMOOTHING` | `0.4` | EWA smoothing alpha (0=max smooth, 1=no smooth) |

---

## 🧪 Testing

```bash
pytest tests/ -v --tb=short
```

Tests cover:
- Detector initialization and model loading
- Emotion prediction output format
- Engagement score calculation
- Configuration validation

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, Flask 3.x, Flask-SocketIO 5.x |
| **Deep Learning** | TensorFlow / Keras 2.x, custom CNN |
| **Computer Vision** | OpenCV 4.x, Haar Cascade (default) or MediaPipe |
| **Frontend** | Vanilla JS, CSS3 (custom properties, glassmorphism, keyframe animations) |
| **Charts** | Chart.js 4.x (60-point sliding trend) |
| **Real-Time** | Socket.IO 4.x (WebSocket transport) |
| **Preprocessing** | CLAHE (adaptive contrast), EWA temporal smoothing |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Inference time** | ~15-40ms per frame (CPU) |
| **FPS** | ~15-30 FPS |
| **Model load** | ~60-80s (first run, TensorFlow CPU init) |
| **Memory** | ~400-600 MB (TensorFlow + OpenCV) |
| **WebSocket latency** | <50ms |

---

## 📄 License

This project is for educational purposes. The FER-2013 dataset is provided under its own license via Kaggle.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/devansh-tg">Devansh</a>
</p>
