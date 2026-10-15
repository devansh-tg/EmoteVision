# EmoteVision

Research and deployment workspace for seven-class FER2013 facial-emotion
recognition. The repository contains a production Flask/Socket.IO camera
application, a preserved Custom CNN checkpoint, reproducible training code,
and evaluation artifacts for four controlled model families:

- Custom CNN
- EfficientNet-B0
- ResNet50
- ConvNeXt-Small

The application and research pipelines are intentionally separated. Training
artifacts are never used as the production model unless explicitly configured.

## Repository structure

```text
web_app/                         # Production application and inference
|-- app.py                       # Flask/Socket.IO entry point
|-- config.py                    # Environment-driven configuration
|-- utils/                       # Camera and inference pipeline
|-- templates/                   # Dashboard HTML
|-- static/                      # Browser JavaScript and CSS
|-- emojis/                      # Local UI assets
`-- tests/                       # Application tests

model/                           # Preserved checkpoints and research outputs
|-- custom_cnn/                  # Production model used by the app
|-- efficientnet_b0/             # Frozen-backbone experiment
|-- resnet/                      # Frozen-backbone experiment
`-- convnext/                    # Frozen-backbone experiment

training/                        # Reproducible training entry points
|-- common.py                    # Shared repository training helpers
|-- run_all_feature_extraction.py # Standalone Kaggle runner
|-- custom_cnn/train.py          # Custom CNN baseline
|-- efficientnet_b0/train.py
|-- resnet/train.py
`-- convnext/train.py

data/fer2013/                    # Local legacy directory split
|-- train/<class>/*.jpg
`-- test/<class>/*.jpg

results/                         # Aggregate comparison metrics
```

The dataset is intentionally ignored for future commits because it is large
and should be obtained from the approved FER2013 source. Existing checkpoints
and evaluation artifacts are retained because they document the experiments
and include the production model required by the web application.

## Experiment protocol

The recovered comparison outputs use the original FER2013 directory split:

```text
Training:   data/fer2013/train
Validation: data/fer2013/test
```

This is a legacy validation protocol. The original `test/` directory is used
during model selection, so these are not untouched final-test results. Report
all metrics as legacy validation results rather than as untouched final-test
performance.

The EfficientNet-B0, ResNet50, and ConvNeXt-Small comparison runs use frozen
ImageNet backbones and train only a seven-class classification head. The
Custom CNN is trained from scratch.

## Recovered comparison results

| Model | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| EfficientNet-B0 | 54.54% | 55.95% | 55.18% |
| ResNet50 | 56.24% | 56.20% | 56.55% |
| ConvNeXt-Small | 49.11% | 50.46% | 48.32% |

The source metrics, reports, metadata, and plots are stored beside each model
under `model/<name>/`. These values are measured on the legacy validation
protocol and must not be presented as a clean held-out test benchmark.

## Run the web application

The tested Windows runtime uses Python 3.11 with TensorFlow 2.20:

```powershell
cd web_app
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The application loads `model/custom_cnn/model.h5` by default. Copy
`web_app/.env.example` to `web_app/.env` to override model, camera, server, or
TFLite settings. Open `http://127.0.0.1:5000` after the server starts.

## Train the legacy Custom CNN

From the repository root:

```powershell
python training/custom_cnn/train.py `
  --train-dir /kaggle/input/emotion-detection-dataset/train `
  --val-dir /kaggle/input/emotion-detection-dataset/test `
  --output-dir model/custom_cnn/runs/legacy_split `
  --epochs 150 `
  --batch-size 64
```

## Train the feature-extraction baselines locally

EfficientNet-B0, ResNet, and ConvNeXt use frozen ImageNet backbones with a
trainable seven-class emotion head:

```powershell
python training/efficientnet_b0/train.py `
  --train-dir data/fer2013/train `
  --val-dir data/fer2013/test `
  --output-dir model/efficientnet_b0/runs/legacy_split

python training/resnet/train.py `
  --train-dir data/fer2013/train `
  --val-dir data/fer2013/test `
  --output-dir model/resnet/runs/legacy_split

python training/convnext/train.py `
  --train-dir data/fer2013/train `
  --val-dir data/fer2013/test `
  --output-dir model/convnext/runs/legacy_split
```

Generated checkpoints, metadata, reports, confusion matrices, and histories
remain under the corresponding `model/<name>/` directory.

## Run all comparison models in Kaggle

`training/run_all_feature_extraction.py` is standalone and is intended to be
uploaded as a Kaggle code dataset. In a fresh Kaggle session, copy it from
`/kaggle/input` into writable storage before execution:

```python
!cp "$(find /kaggle/input -name 'run_all_feature_extraction.py' | head -n 1)" \
    /kaggle/working/run_all_feature_extraction.py
```

Then run the models sequentially:

```python
!python /kaggle/working/run_all_feature_extraction.py \
    --dataset-root /kaggle/input/datasets/devanshtyagi824/emotion-detection-dataset/data \
    --output-root /kaggle/working/emotevision_outputs \
    --models efficientnet_b0 resnet convnext \
    --weights imagenet \
    --epochs 100 \
    --batch-size 64 \
    --seed 42
```

Package and download the output before ending the Kaggle session:

```python
!cd /kaggle/working && zip -r emotevision_outputs.zip emotevision_outputs
from IPython.display import FileLink, display
display(FileLink("/kaggle/working/emotevision_outputs.zip"))
```

## Testing

```powershell
pytest web_app/tests/ -v --tb=short
```

## Academic reporting

Report the split, input size, color mode, preprocessing, backbone status, and
checkpoint format beside every metric. Never describe estimated real-world
accuracy as measured accuracy. Keep `model/custom_cnn/model.h5` intact unless
the production application is deliberately migrated and tested against a
replacement.
