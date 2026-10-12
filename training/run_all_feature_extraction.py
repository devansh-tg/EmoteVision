"""Run the three frozen-backbone FER2013 baselines in one Kaggle cell.

This file is standalone: it does not import any other project file.
It automatically finds an attached Kaggle dataset containing train/ and test/
directories, trains EfficientNet-B0, ResNet50, and ConvNeXt-Small in sequence,
and writes complete models plus weights-only files under --output-root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.applications import ConvNeXtSmall, EfficientNetB0, ResNet50
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODELS = {
    "efficientnet_b0": (EfficientNetB0, efficientnet_preprocess),
    "resnet": (ResNet50, resnet_preprocess),
    "convnext": (ConvNeXtSmall, convnext_preprocess),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/emotevision_outputs"))
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_dataset_root() -> Path:
    roots = sorted({
        path.parent
        for path in Path("/kaggle/input").rglob("train")
        if path.is_dir() and (path.parent / "test").is_dir()
    })
    if len(roots) != 1:
        print("Dataset roots found:")
        for root in roots:
            print(" ", root)
        raise RuntimeError(
            "Expected exactly one attached dataset containing train/ and test/. "
            "Pass --dataset-root explicitly if necessary."
        )
    return roots[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_model(name: str, input_size: int, class_count: int, weights: str) -> Model:
    application, _ = MODELS[name]
    backbone = application(
        include_top=False,
        weights=None if weights == "none" else weights,
        input_shape=(input_size, input_size, 3),
    )
    backbone.trainable = False
    x = GlobalAveragePooling2D(name="global_average_pooling")(backbone.output)
    x = BatchNormalization(name="head_batch_normalization")(x)
    x = Dropout(0.35, name="head_dropout")(x)
    x = Dense(256, activation="relu", name="head_dense")(x)
    x = Dropout(0.35, name="head_dense_dropout")(x)
    output = Dense(class_count, activation="softmax", name="emotion")(x)
    model = Model(backbone.input, output, name=f"{name}_feature_extractor")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(loader) -> dict[int, float]:
    labels = np.unique(loader.classes)
    weights = compute_class_weight("balanced", classes=labels, y=loader.classes)
    return {int(label): float(weight) for label, weight in zip(labels, weights)}


def json_shape(value):
    if value is None:
        return None
    try:
        return [None if item is None else int(item) for item in value]
    except TypeError:
        return str(value)


def describe_model(model: Model) -> list[dict]:
    layers = []
    for layer in model.layers:
        layers.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "trainable": bool(layer.trainable),
                "parameter_count": int(layer.count_params()),
                "output_shape": json_shape(getattr(layer, "output_shape", None)),
            }
        )
    return layers


def file_manifest(output_dir: Path) -> dict[str, dict]:
    manifest = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[path.name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    return manifest


def save_plots_and_metrics(
    name: str,
    output_dir: Path,
    history,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    matrix = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
    }
    (output_dir / f"{name}_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / f"{name}_classification_report.txt").write_text(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    np.savetxt(output_dir / f"{name}_confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    with (output_dir / f"{name}_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"{name} confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=180)
    plt.close()

    epochs = range(1, len(history.history["accuracy"]) + 1)
    history_keys = list(history.history)
    with (output_dir / f"{name}_training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *history_keys])
        for index in range(len(epochs)):
            writer.writerow([index + 1, *(history.history[key][index] for key in history_keys)])
    (output_dir / f"{name}_training_history.json").write_text(
        json.dumps(
            {key: [float(value) for value in values] for key, values in history.history.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history.history["accuracy"], label="train")
    axes[0].plot(epochs, history.history["val_accuracy"], label="validation")
    axes[0].set_title(f"{name} accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].plot(epochs, history.history["loss"], label="train")
    axes[1].plot(epochs, history.history["val_loss"], label="validation")
    axes[1].set_title(f"{name} loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}_training_history.png", dpi=180)
    plt.close(figure)

    return metrics


def save_pytorch_weights(model: Model, output_path: Path) -> None:
    """Serialize Keras best weights as a PyTorch .pth state dictionary.

    The .pth file contains tensors only; it does not contain a model
    architecture and cannot be loaded directly into a Keras model.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to create .pth exports. "
            "Install torch in the Kaggle environment or use the native .weights.h5 file."
        ) from exc

    state = {}
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            state[layer.name] = [torch.from_numpy(np.asarray(value)) for value in weights]
    torch.save(state, output_path)


def train_one(
    name: str,
    train_dir: Path,
    val_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, float] | None:
    set_seed(args.seed)
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    application, preprocess = MODELS[name]

    train_generator = ImageDataGenerator(preprocessing_function=preprocess)
    validation_generator = ImageDataGenerator(preprocessing_function=preprocess)
    train_loader = train_generator.flow_from_directory(
        str(train_dir),
        target_size=(args.input_size, args.input_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = validation_generator.flow_from_directory(
        str(val_dir),
        target_size=(args.input_size, args.input_size),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = make_model(name, args.input_size, len(train_loader.class_indices), args.weights)
    params = {
        "total": int(model.count_params()),
        "trainable": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "non_trainable": int(sum(np.prod(w.shape) for w in model.non_trainable_weights)),
    }
    weights = class_weights(train_loader)
    print(f"\n{name}: backbone={application.__name__}, frozen=True, parameters={params}")
    if args.dry_run:
        batch_x, batch_y = next(train_loader)
        prediction = model(batch_x, training=False)
        if prediction.shape[-1] != len(train_loader.class_indices):
            raise RuntimeError("Model output class count does not match dataset classes.")
        print(f"{name}: dry run passed; batch={batch_x.shape}, output={prediction.shape}")
        return None

    best_model = output_dir / f"{name}_best.h5"
    checkpoint_model = output_dir / f"{name}_checkpoint.h5"
    final_model = output_dir / f"{name}.h5"
    best_weights = output_dir / f"{name}_best.weights.h5"
    final_weights = output_dir / f"{name}.weights.h5"
    best_pytorch_weights = output_dir / f"{name}_best.pth"
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-7, mode="min"),
        ModelCheckpoint(str(best_model), monitor="val_accuracy", save_best_only=True, mode="max"),
        ModelCheckpoint(str(checkpoint_model), monitor="val_loss", save_best_only=True, mode="min"),
        ModelCheckpoint(
            str(best_weights),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        ),
    ]
    started = time.time()
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs,
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(final_model)
    model.save_weights(final_weights)
    save_pytorch_weights(model, best_pytorch_weights)
    val_loader.reset()
    probabilities = model.predict(val_loader, verbose=1)
    predicted = np.argmax(probabilities, axis=1)
    labels = list(train_loader.class_indices.keys())
    metrics = save_plots_and_metrics(
        name=name,
        output_dir=output_dir,
        history=history,
        true_labels=val_loader.classes,
        predicted_labels=predicted,
        class_names=labels,
    )
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)
    best_val_accuracy = float(max(history.history["val_accuracy"]))
    history_final = {
        key: float(values[-1]) for key, values in history.history.items()
    }
    history_best = {
        "epoch": best_epoch,
        "val_accuracy": best_val_accuracy,
        "val_loss": float(history.history["val_loss"][best_epoch - 1]),
        "accuracy": float(history.history["accuracy"][best_epoch - 1]),
        "loss": float(history.history["loss"][best_epoch - 1]),
    }
    training_seconds = round(time.time() - started, 3)
    dataset_counts = {
        "train_samples": int(train_loader.samples),
        "validation_samples": int(val_loader.samples),
        "train_batches_per_epoch": int(len(train_loader)),
        "validation_batches": int(len(val_loader)),
        "class_counts_train": {
            label: int(np.sum(train_loader.classes == index))
            for label, index in train_loader.class_indices.items()
        },
        "class_counts_validation": {
            label: int(np.sum(val_loader.classes == index))
            for label, index in val_loader.class_indices.items()
        },
    }
    trainable_layers = [layer.name for layer in model.layers if layer.trainable]
    non_trainable_layers = [layer.name for layer in model.layers if not layer.trainable]
    metadata = {
        "schema_version": "1.0",
        "experiment_name": "FER2013 legacy directory split feature extraction",
        "model_family": name,
        "display_name": application.__name__,
        "architecture": {
            "keras_application": application.__module__ + "." + application.__name__,
            "model_name": model.name,
            "input_shape": json_shape(model.input_shape),
            "output_shape": json_shape(model.output_shape),
            "input_channels": 3,
            "input_color_mode": "rgb",
            "input_size": [args.input_size, args.input_size],
            "class_count": len(labels),
            "layers": describe_model(model),
        },
        "training_method": "feature_extraction",
        "backbone_weights": args.weights,
        "backbone_trainable": False,
        "trainable_layers": trainable_layers,
        "non_trainable_layers": non_trainable_layers,
        "class_mapping": train_loader.class_indices,
        "dataset": {
            **dataset_counts,
            "dataset_root": str(train_dir.parent),
            "train_dir": str(train_dir),
            "validation_dir": str(val_dir),
            "test_dir": str(val_dir),
            "protocol": "legacy_train_directory_and_test_directory_as_validation",
            "test_set_used_for_model_selection": True,
        },
        "preprocessing": {
            "preprocessing_function": application.__module__ + ".preprocess_input",
            "image_resizing": f"{args.input_size}x{args.input_size}",
            "color_mode": "rgb",
            "augmentation": "none",
            "normalization": "architecture-specific preprocess_input",
        },
        "training_configuration": {
            "requested_epochs": args.epochs,
            "epochs_trained": len(history.history["accuracy"]),
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "loss": "categorical_crossentropy",
            "class_weights": weights,
            "early_stopping": {
                "monitor": "val_accuracy",
                "patience": 12,
                "restore_best_weights": True,
                "mode": "max",
            },
            "reduce_learning_rate": {
                "monitor": "val_loss",
                "patience": 5,
                "factor": 0.5,
                "min_learning_rate": 1e-7,
            },
            "checkpoint_selection": "best validation accuracy",
        },
        "seed": args.seed,
        "random_seed": {
            "python": args.seed,
            "numpy": args.seed,
            "tensorflow": args.seed,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
            "cpu_devices": [device.name for device in tf.config.list_physical_devices("CPU")],
        },
        "parameters": params,
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "best_validation_metrics": history_best,
        "last_epoch_metrics": history_final,
        "metrics": metrics,
        "artifacts": {
            "best_model": str(best_model),
            "best_weights": str(best_weights),
            "best_pytorch_weights": str(best_pytorch_weights),
            "final_model": str(final_model),
            "final_weights": str(final_weights),
            "checkpoint_model": str(checkpoint_model),
            "classification_report_json": str(output_dir / f"{name}_classification_report.json"),
            "classification_report_text": str(output_dir / f"{name}_classification_report.txt"),
            "metrics_json": str(output_dir / f"{name}_metrics.json"),
            "metrics_csv": str(output_dir / f"{name}_metrics.csv"),
            "confusion_matrix_csv": str(output_dir / f"{name}_confusion_matrix.csv"),
            "confusion_matrix_png": str(output_dir / f"{name}_confusion_matrix.png"),
            "training_history_json": str(output_dir / f"{name}_training_history.json"),
            "training_history_csv": str(output_dir / f"{name}_training_history.csv"),
            "training_history_png": str(output_dir / f"{name}_training_history.png"),
        },
    }
    (output_dir / f"{name}_model_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / f"{name}_artifact_manifest.json").write_text(
        json.dumps(file_manifest(output_dir), indent=2), encoding="utf-8"
    )
    print(f"{name}: complete. Outputs saved to {output_dir}")
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    dataset_root = args.dataset_root or find_dataset_root()
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "test"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected train/ and test/ under {dataset_root}")
    print("Dataset root:", dataset_root)
    print("Training:", train_dir)
    print("Validation:", val_dir)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    results = {}
    for name in args.models:
        metrics = train_one(name, train_dir, val_dir, args.output_root, args)
        if metrics is not None:
            results[name] = metrics
    if results:
        (args.output_root / "model_comparison_metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        with (args.output_root / "model_comparison_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            fieldnames = ["model", *next(iter(results.values())).keys()]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for model_name, metrics in results.items():
                writer.writerow({"model": model_name, **metrics})
        print("Comparison metrics saved to:", args.output_root)


if __name__ == "__main__":
    main()
