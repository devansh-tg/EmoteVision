"""Train ResNet50 as a frozen ImageNet feature extractor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ArchitectureConfig, run_training


CONFIG = ArchitectureConfig(
    key="resnet50",
    display_name="ResNet50",
    folder_name="resnet50",
    keras_application="tensorflow.keras.applications.ResNet50",
    preprocess_function="tensorflow.keras.applications.resnet50.preprocess_input",
    default_input_size=96,
)


if __name__ == "__main__":
    run_training(CONFIG)
