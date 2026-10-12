"""Train EfficientNet-B0 as a frozen ImageNet feature extractor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ArchitectureConfig, run_training


CONFIG = ArchitectureConfig(
    key="efficientnet_b0",
    display_name="EfficientNet-B0",
    folder_name="efficientnet_b0",
    keras_application="tensorflow.keras.applications.EfficientNetB0",
    preprocess_function="tensorflow.keras.applications.efficientnet.preprocess_input",
    default_input_size=96,
)


if __name__ == "__main__":
    run_training(CONFIG)
