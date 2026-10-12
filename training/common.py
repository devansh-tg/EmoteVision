"""Shared feature-extraction training utilities for FER2013 models."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


TRAINING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_ROOT.parent
MODEL_ROOT = REPO_ROOT / "model"
MODEL_DIRECTORY_BY_KEY = {
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet",
    "convnext_small": "convnext",
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class ArchitectureConfig:
    key: str
    display_name: str
    folder_name: str
    keras_application: str
    preprocess_function: str
    default_input_size: int = 96


def build_parser(config: ArchitectureConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Train a frozen-backbone {config.display_name} feature extractor "
            "on the FER2013 directory split."
        )
    )
    parser.add_argument("--train-dir", default=str(REPO_ROOT / "data" / "fer2013" / "train"))
    parser.add_argument("--val-dir", default=str(REPO_ROOT / "data" / "fer2013" / "test"))
    parser.add_argument(
        "--output-dir",
        default=str(
            MODEL_ROOT
            / MODEL_DIRECTORY_BY_KEY.get(config.key, config.folder_name)
            / "runs"
            / "legacy_split"
        ),
    )
    parser.add_argument("--input-size", type=int, default=config.default_input_size)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument(
        "--inference-samples",
        type=int,
        default=256,
        help="Validation samples used for timing after training. Use 0 to skip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data loading and model construction without training.",
    )
    parser.add_argument(
        "--show-model-summary",
        action="store_true",
        help="Print the full Keras model summary.",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover - depends on TensorFlow build
            logging.warning("Could not enable TensorFlow op determinism: %s", exc)


def configure_device(device: str) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
            logging.info("Using CPU only.")
        except RuntimeError as exc:
            logging.warning("Could not hide GPU devices after TensorFlow initialization: %s", exc)
        return

    if device == "cuda" and not gpus:
        raise RuntimeError("--device cuda was requested, but TensorFlow sees no GPU.")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            logging.warning("Could not set GPU memory growth for %s: %s", gpu, exc)

    if gpus:
        logging.info("Using TensorFlow GPU device(s): %s", ", ".join(g.name for g in gpus))
    else:
        logging.info("No TensorFlow GPU detected; using CPU.")


def load_object(dotted_path: str):
    module_name, object_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def validate_dataset_dir(path: Path, label: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")

    class_dirs = sorted(item for item in path.iterdir() if item.is_dir())
    if not class_dirs:
        raise ValueError(f"{label} directory has no class subdirectories: {path}")

    empty_classes = []
    for class_dir in class_dirs:
        image_count = sum(
            1
            for item in class_dir.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count == 0:
            empty_classes.append(class_dir.name)
    if empty_classes:
        raise ValueError(f"{label} has empty class folder(s): {', '.join(empty_classes)}")

    return [item.name for item in class_dirs]


def validate_dataset(train_dir: Path, val_dir: Path) -> None:
    train_classes = validate_dataset_dir(train_dir, "Training")
    val_classes = validate_dataset_dir(val_dir, "Validation")
    if train_classes != val_classes:
        raise ValueError(
            "Training and validation class folders differ: "
            f"train={train_classes}, validation={val_classes}"
        )
    logging.info("Detected %d classes: %s", len(train_classes), ", ".join(train_classes))


def make_generators(
    train_dir: Path,
    val_dir: Path,
    input_size: int,
    batch_size: int,
    seed: int,
    preprocess_input: Callable,
):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.15,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    target_size = (input_size, input_size)

    train_loader = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=target_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=target_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )
    if train_loader.class_indices != val_loader.class_indices:
        raise ValueError(
            "Keras class indices differ between splits: "
            f"{train_loader.class_indices} vs {val_loader.class_indices}"
        )
    return train_loader, val_loader


def build_model(
    config: ArchitectureConfig,
    input_size: int,
    class_count: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
    weights: str,
) -> Model:
    application_model = load_object(config.keras_application)
    resolved_weights = None if weights == "none" else weights
    try:
        backbone = application_model(
            include_top=False,
            weights=resolved_weights,
            input_shape=(input_size, input_size, 3),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize {config.display_name} with weights={weights!r}. "
            "If ImageNet weights are not cached and this machine is offline, rerun "
            "after the weights are available or use --weights none for a smoke test."
        ) from exc

    backbone.trainable = False
    x = GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = BatchNormalization(name="head_bn")(x)
    x = Dropout(dropout, name="head_dropout")(x)
    x = Dense(dense_units, activation="relu", name="head_dense")(x)
    x = BatchNormalization(name="head_dense_bn")(x)
    x = Dropout(dropout, name="head_dense_dropout")(x)
    output = Dense(class_count, activation="softmax", name="emotion")(x)

    model = Model(backbone.input, output, name=f"{config.key}_feature_extractor")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model: Model) -> dict[str, int]:
    trainable = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable = int(
        sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    )
    return {
        "total": trainable + non_trainable,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def compute_weights(loader) -> dict[int, float]:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(loader.classes),
        y=loader.classes,
    )
    return {index: float(weight) for index, weight in enumerate(weights)}


def save_history(history, out_dir: Path, key: str) -> tuple[str, str | None]:
    history_path = out_dir / f"{key}_training_history.json"
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
        axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
        axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history.history["loss"], label="Train", linewidth=2)
        axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
        axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out_dir / f"{key}_training_history.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(history_path), str(plot_path)
    except Exception as exc:  # pragma: no cover - reporting should not fail training
        logging.warning("Could not save training-history plot: %s", exc)
        return str(history_path), None


def save_evaluation_artifacts(model: Model, val_loader, out_dir: Path, key: str) -> dict:
    import matplotlib.pyplot as plt
    import seaborn as sns

    val_loader.reset()
    probabilities = model.predict(val_loader, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = val_loader.classes
    class_names = list(val_loader.class_indices.keys())

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_path = out_dir / f"{key}_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{key} Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    matrix_path = out_dir / f"{key}_confusion_matrix.png"
    fig.savefig(matrix_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "classification_report": report_dict,
        "classification_report_path": str(report_path),
        "confusion_matrix_path": str(matrix_path),
    }


def measure_inference_speed(model: Model, val_loader, sample_count: int) -> dict:
    if sample_count <= 0:
        return {"samples": 0, "mean_ms": None, "p95_ms": None, "fps": None}

    val_loader.reset()
    seen = 0
    timings = []
    for batch_x, _ in val_loader:
        start = time.perf_counter()
        model.predict(batch_x, verbose=0)
        elapsed = time.perf_counter() - start
        per_sample = elapsed / len(batch_x)
        timings.extend([per_sample] * len(batch_x))
        seen += len(batch_x)
        if seen >= sample_count:
            break

    timings_ms = np.array(timings[:sample_count]) * 1000.0
    mean_ms = float(np.mean(timings_ms))
    return {
        "samples": int(len(timings_ms)),
        "mean_ms": mean_ms,
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "fps": float(1000.0 / mean_ms) if mean_ms > 0 else None,
    }


def run_dry_check(model: Model, train_loader, class_count: int) -> None:
    batch_x, batch_y = next(train_loader)
    predictions = model(batch_x[: min(2, len(batch_x))], training=False)
    if predictions.shape[-1] != class_count:
        raise RuntimeError(
            f"Classifier output has {predictions.shape[-1]} units; expected {class_count}."
        )
    logging.info("Dry-run batch input shape: %s", batch_x.shape)
    logging.info("Dry-run batch label shape: %s", batch_y.shape)
    logging.info("Dry-run prediction shape: %s", predictions.shape)


def run_training(config: ArchitectureConfig) -> None:
    configure_logging()
    args = build_parser(config).parse_args()
    configure_reproducibility(args.seed, deterministic=not args.non_deterministic)
    configure_device(args.device)

    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    validate_dataset(train_dir, val_dir)

    preprocess_input = load_object(config.preprocess_function)
    train_loader, val_loader = make_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        seed=args.seed,
        preprocess_input=preprocess_input,
    )
    class_names = list(train_loader.class_indices.keys())
    class_count = len(class_names)

    model = build_model(
        config=config,
        input_size=args.input_size,
        class_count=class_count,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weights=args.weights,
    )
    params = count_params(model)
    if args.show_model_summary:
        model.summary(print_fn=logging.info)
    logging.info("Backbone is frozen; trainable parameters: %s", f"{params['trainable']:,}")
    logging.info("Non-trainable parameters: %s", f"{params['non_trainable']:,}")

    if args.dry_run:
        run_dry_check(model, train_loader, class_count)
        logging.info("Dry run completed; no training artifacts were written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{config.key}_best.h5"
    checkpoint_path = out_dir / f"{config.key}_checkpoint.h5"
    final_path = out_dir / f"{config.key}.h5"
    best_weights_path = out_dir / f"{config.key}_best.weights.h5"
    final_weights_path = out_dir / f"{config.key}.weights.h5"

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode="min",
            verbose=1,
        ),
        ModelCheckpoint(
            str(best_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,
        ),
        ModelCheckpoint(
            str(best_weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=0,
        ),
    ]

    class_weight = None if args.no_class_weights else compute_weights(train_loader)
    if class_weight is not None:
        logging.info("Using balanced class weights: %s", class_weight)

    history = model.fit(
        train_loader,
        epochs=args.epochs,
        validation_data=val_loader,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    model.save(final_path)
    model.save_weights(final_weights_path)

    train_loss, train_acc = model.evaluate(train_loader, verbose=1)
    val_loss, val_acc = model.evaluate(val_loader, verbose=1)
    eval_artifacts = save_evaluation_artifacts(model, val_loader, out_dir, config.key)
    history_json_path, history_plot_path = save_history(history, out_dir, config.key)
    speed = measure_inference_speed(model, val_loader, args.inference_samples)

    report = eval_artifacts["classification_report"]
    metadata = {
        "model_family": config.key,
        "display_name": config.display_name,
        "training_method": "feature_extraction",
        "backbone_weights": args.weights,
        "backbone_trainable": False,
        "timestamp": dt.datetime.now().isoformat(),
        "seed": args.seed,
        "deterministic_ops_requested": not args.non_deterministic,
        "device": args.device,
        "class_names": class_names,
        "class_indices": train_loader.class_indices,
        "train_dir": str(train_dir),
        "validation_dir": str(val_dir),
        "test_dir": str(val_dir),
        "legacy_protocol": True,
        "epochs_trained": len(history.history["accuracy"]),
        "train_accuracy": float(train_acc),
        "train_loss": float(train_loss),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "accuracy": float(val_acc),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "parameters": params,
        "model_size_mb": round(final_path.stat().st_size / (1024 * 1024), 3),
        "inference_speed": speed,
        "preprocessing": {
            "input_size": [args.input_size, args.input_size],
            "color_mode": "rgb",
            "input_value_range": "0_255_before_architecture_preprocess",
            "preprocessing_function": config.preprocess_function,
            "source_pipeline": "FER2013 directory images loaded as RGB by ImageDataGenerator",
        },
        "model_path": str(final_path),
        "best_model_path": str(best_path),
        "checkpoint_path": str(checkpoint_path),
        "best_weights_path": str(best_weights_path),
        "final_weights_path": str(final_weights_path),
        "classification_report_path": eval_artifacts["classification_report_path"],
        "confusion_matrix_path": eval_artifacts["confusion_matrix_path"],
        "training_history_json_path": history_json_path,
        "training_history_plot_path": history_plot_path,
    }
    meta_path = out_dir / f"{config.key}_model_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logging.info("Training complete.")
    logging.info("Best checkpoint: %s", best_path)
    logging.info("Final model: %s", final_path)
    logging.info("Metadata: %s", meta_path)
    logging.info("Validation accuracy: %.4f", val_acc)
