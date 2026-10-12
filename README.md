# EmoteVision

FER2013 facial-emotion recognition with a Flask/Socket.IO web application and
four model research tracks: Custom CNN, EfficientNet-B0, ResNet50, and
ConvNeXt-Small.

## Repository structure

```text
web_app/                         # Production web application and inference
|-- app.py
|-- config.py
|-- requirements.txt
|-- .env.example
|-- utils/                       # Detector and frame-buffer code
|-- templates/                   # Flask HTML templates
|-- static/                      # Browser JavaScript and CSS
|-- emojis/                      # Optional local assets
|-- test.py                      # Local webcam utility
|-- export_tflite.py             # TFLite conversion utility
`-- tests/                       # Web/inference tests

training/
`-- custom_cnn/
    `-- train.py                 # Legacy split training entry point

models/
|-- custom_cnn/
|   |-- model.h5                 # Production checkpoint
|   `-- model_meta.json          # Production model metadata
`-- efficientnet_b0/
    |-- efficientnet_b0.h5
    |-- efficientnet_b0_best.h5
    |-- efficientnet_b0_checkpoint.h5
    |-- efficientnet_b0_model_meta.json
    |-- efficientnet_b0_classification_report.txt
    |-- efficientnet_b0_confusion_matrix.png
    `-- efficientnet_b0_training_history.png

data/
`-- fer2013/
    |-- train/<class>/*.jpg
    `-- test/<class>/*.jpg
```

The old `src/` folder has been replaced by feature-specific folders. The
dataset and model artifacts are intentionally outside the web application.

## Current experiment protocol

The existing training workflow uses the original FER2013 directory split:

```text
Training:   data/fer2013/train
Validation: data/fer2013/test
```

This is a legacy validation protocol. It is useful for reproducing the
existing project results, but it is not an untouched final-test evaluation
because the original test directory is used during model selection. Use the
same protocol for all four models if you want a directly comparable table.

## Run the web application

```powershell
cd web_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

The application loads the production Custom CNN from
`models/custom_cnn/model.h5` by default. Copy `web_app/.env.example` to
`web_app/.env` to override model, camera, server, or TFLite settings.

## Train the legacy Custom CNN

From the repository root:

```powershell
python training/custom_cnn/train.py `
  --train-dir /kaggle/input/emotion-detection-dataset/train `
  --val-dir /kaggle/input/emotion-detection-dataset/test `
  --output-dir models/custom_cnn/runs/legacy_split `
  --epochs 150 `
  --batch-size 64
```

The script saves the final model, best checkpoint, metadata, confusion matrix,
classification report, and training-history plot into the selected output
folder.

## Model research folders

The current repository contains preserved Custom CNN and EfficientNet-B0
artifacts. ResNet50 and ConvNeXt-Small do not yet have trained checkpoints in
this working tree. Their Kaggle training scripts/notebooks should be kept
separate from the production application and should use the same legacy split
if they are being compared with the existing results.

## Testing

```powershell
pytest web_app/tests/ -v --tb=short
```

## Academic reporting notes

- Report the split and preprocessing beside every model metric.
- Label results from `data/fer2013/test` as legacy validation results.
- Do not describe estimated real-world accuracy as measured accuracy.
- Preserve the production Custom CNN checkpoint while writing new runs under
  `models/custom_cnn/runs/`.
- Keep datasets, checkpoints, reports, and application code in their separate
  feature folders.
