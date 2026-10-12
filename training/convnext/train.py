"""Train ConvNeXt-Small as a frozen ImageNet feature extractor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ArchitectureConfig, run_training


CONFIG = ArchitectureConfig(
    key="convnext_small",
    display_name="ConvNeXt-Small",
    folder_name="convnext_small",
    keras_application="tensorflow.keras.applications.ConvNeXtSmall",
    preprocess_function="tensorflow.keras.applications.convnext.preprocess_input",
    default_input_size=96,
)


if __name__ == "__main__":
    run_training(CONFIG)
