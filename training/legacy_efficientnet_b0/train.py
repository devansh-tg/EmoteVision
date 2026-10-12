"""Train and compare EmoteVision emotion models.

The original custom CNN remains the baseline. EfficientNet-B0 is added as a
separate ImageNet transfer-learning model with its own preprocessing and model
artifacts so the baseline is never removed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


SEED = 42
NUM_CLASSES = 7
CNN_IMG_SIZE = (48, 48)
CNN_ARTIFACTS = {
    "final": "model.h5",
    "best": "model_best.h5",
    "checkpoint": "model_checkpoint.h5",
    "meta": "model_meta.json",
    "confusion_matrix": "confusion_matrix.png",
    "classification_report": "classification_report.txt",
    "history": "training_history.png",
}
EFFICIENTNET_ARTIFACTS = {
    "final": "efficientnet_b0.h5",
    "best": "efficientnet_b0_best.h5",
    "checkpoint": "efficientnet_b0_checkpoint.h5",
    "meta": "efficientnet_b0_model_meta.json",
    "confusion_matrix": "efficientnet_b0_confusion_matrix.png",
    "classification_report": "efficientnet_b0_classification_report.txt",
    "history": "efficientnet_b0_training_history.png",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN and/or EfficientNet-B0 transfer model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-family",
        choices=["cnn", "efficientnet_b0", "both"],
        default="cnn",
        help="Which model to train. 'cnn' preserves the original baseline behavior.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip training and compare saved CNN/EfficientNet models on the same test set.",
    )
    parser.add_argument(
        "--train-dir",
        default="/kaggle/input/emotion-detection-dataset/train",
        help="Path to training data directory.",
    )
    parser.add_argument(
        "--val-dir",
        default="/kaggle/input/emotion-detection-dataset/test",
        help="Path to validation data directory.",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Path to held-out test data. Defaults to --val-dir for this FER-style project.",
    )
    parser.add_argument("--output-dir", default=".", help="Directory to save model files.")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--efficientnet-img-size",
        type=int,
        default=96,
        help="Square RGB input size for EfficientNet-B0.",
    )
    parser.add_argument(
        "--efficientnet-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet weights for transfer learning or initialize from scratch.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=20,
        help="Extra low-LR epochs after unfreezing the tail of EfficientNet-B0.",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=30,
        help="Number of final EfficientNet layers to unfreeze during fine-tuning.",
    )
    parser.add_argument(
        "--skip-fine-tune",
        action="store_true",
        help="Train only the EfficientNet classifier head.",
    )
    parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=200,
        help="Single-image predictions used to estimate real-time latency.",
    )
    parser.add_argument("--baseline-model", default=None, help="Saved CNN model for comparison.")
    parser.add_argument(
        "--efficientnet-model",
        default=None,
        help="Saved EfficientNet-B0 model for comparison.",
    )
    return parser


def set_reproducible_seeds() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def build_train_generator(model_family: str) -> ImageDataGenerator:
    kwargs = {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "horizontal_flip": True,
        "zoom_range": 0.15,
        "shear_range": 0.15,
        "brightness_range": [0.8, 1.2],
        "channel_shift_range": 15.0,
        "fill_mode": "nearest",
    }
    if model_family == "cnn":
        kwargs["rescale"] = 1.0 / 255
    return ImageDataGenerator(**kwargs)


def build_eval_generator(model_family: str) -> ImageDataGenerator:
    if model_family == "cnn":
        return ImageDataGenerator(rescale=1.0 / 255)
    return ImageDataGenerator()


def data_settings(model_family: str, efficientnet_size: int) -> dict:
    if model_family == "efficientnet_b0":
        return {
            "target_size": (efficientnet_size, efficientnet_size),
            "color_mode": "rgb",
            "input_shape": (efficientnet_size, efficientnet_size, 3),
            "input_scale": "efficientnet_0_255",
        }
    return {
        "target_size": CNN_IMG_SIZE,
        "color_mode": "grayscale",
        "input_shape": (48, 48, 1),
        "input_scale": "zero_one",
    }


def make_loaders(
    model_family: str,
    train_dir: str | Path,
    eval_dir: str | Path,
    batch_size: int,
    efficientnet_size: int,
    shuffle_train: bool = True,
):
    settings = data_settings(model_family, efficientnet_size)
    train_loader = build_train_generator(model_family).flow_from_directory(
        train_dir,
        target_size=settings["target_size"],
        batch_size=batch_size,
        color_mode=settings["color_mode"],
        class_mode="categorical",
        shuffle=shuffle_train,
        seed=SEED,
    )
    eval_loader = build_eval_generator(model_family).flow_from_directory(
        eval_dir,
        target_size=settings["target_size"],
        batch_size=batch_size,
        color_mode=settings["color_mode"],
        class_mode="categorical",
        shuffle=False,
    )
    return train_loader, eval_loader


def make_eval_loader(
    model_family: str,
    eval_dir: str | Path,
    batch_size: int,
    efficientnet_size: int,
):
    settings = data_settings(model_family, efficientnet_size)
    return build_eval_generator(model_family).flow_from_directory(
        eval_dir,
        target_size=settings["target_size"],
        batch_size=batch_size,
        color_mode=settings["color_mode"],
        class_mode="categorical",
        shuffle=False,
    )


def compute_weights(train_loader) -> dict[int, float]:
    class_weights_array = compute_class_weight(
        "balanced",
        classes=np.unique(train_loader.classes),
        y=train_loader.classes,
    )
    return dict(enumerate(class_weights_array))


def build_baseline_cnn(num_classes: int = NUM_CLASSES) -> Model:
    l2_strength = 0.0005
    model = Sequential(
        [
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(l2_strength),
                input_shape=(48, 48, 1),
            ),
            BatchNormalization(),
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(l2_strength),
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.20),
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(l2_strength),
            ),
            BatchNormalization(),
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(l2_strength),
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(l2_strength),
            ),
            BatchNormalization(),
            Conv2D(
                256,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(l2_strength),
            ),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.30),
            Conv2D(
                512,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(l2_strength),
            ),
            BatchNormalization(),
            Dropout(0.35),
            GlobalAveragePooling2D(),
            Dense(512, activation="relu", kernel_regularizer=l2(l2_strength)),
            BatchNormalization(),
            Dropout(0.40),
            Dense(256, activation="relu", kernel_regularizer=l2(l2_strength)),
            BatchNormalization(),
            Dropout(0.40),
            Dense(num_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=2e-4),
        metrics=["accuracy"],
    )
    return model


def build_efficientnet_b0(
    input_size: int,
    num_classes: int = NUM_CLASSES,
    weights: str | None = "imagenet",
) -> Model:
    inputs = Input(shape=(input_size, input_size, 3), name="efficientnet_rgb_input")
    base = EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
    )
    base.trainable = False

    x = base(inputs, training=False)
    x = GlobalAveragePooling2D(name="efficientnet_gap")(x)
    x = BatchNormalization(name="head_bn_1")(x)
    x = Dropout(0.35, name="head_dropout_1")(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.0005), name="head_dense")(x)
    x = BatchNormalization(name="head_bn_2")(x)
    x = Dropout(0.40, name="head_dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="emotion_logits")(x)

    model = Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_transfer")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model


def fine_tune_efficientnet(
    model: Model,
    train_loader,
    val_loader,
    callbacks,
    class_weights: dict[int, float],
    initial_epoch: int,
    fine_tune_epochs: int,
    fine_tune_layers: int,
):
    if fine_tune_epochs <= 0:
        return None

    base_model = next(
        (layer for layer in model.layers if layer.name.startswith("efficientnetb0")),
        None,
    )
    if base_model is None:
        print("Could not find EfficientNet-B0 backbone; skipping fine-tune.")
        return None

    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-5),
        metrics=["accuracy"],
    )

    print("\n" + "=" * 70)
    print("Fine-tuning EfficientNet-B0 tail at low learning rate")
    print("=" * 70)
    return model.fit(
        train_loader,
        epochs=initial_epoch + fine_tune_epochs,
        initial_epoch=initial_epoch,
        validation_data=val_loader,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )


def callbacks_for(output_dir: Path, artifacts: dict) -> list:
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=18,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1,
            mode="min",
        ),
        ModelCheckpoint(
            output_dir / artifacts["best"],
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        ModelCheckpoint(
            output_dir / artifacts["checkpoint"],
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
            mode="min",
        ),
    ]


def merge_histories(*histories) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def model_file_size_mb(path: Path) -> float | None:
    if path.exists():
        return round(path.stat().st_size / (1024 * 1024), 3)
    return None


def benchmark_single_image_latency(model: Model, eval_loader, samples: int) -> dict:
    if samples <= 0:
        return {"samples": 0, "mean_ms": None, "p95_ms": None, "fps": None}

    eval_loader.reset()
    batch_x, _ = next(eval_loader)
    sample = batch_x[:1]

    for _ in range(5):
        model.predict(sample, verbose=0)

    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        model.predict(sample, verbose=0)
        timings.append((time.perf_counter() - start) * 1000)

    arr = np.asarray(timings, dtype=np.float32)
    mean_ms = float(np.mean(arr))
    return {
        "samples": int(samples),
        "mean_ms": round(mean_ms, 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else None,
    }


def evaluate_model(
    model: Model,
    eval_loader,
    class_names: list[str],
    output_dir: Path,
    artifacts: dict,
    model_family: str,
    benchmark_samples: int,
    model_path: Path | None = None,
) -> dict:
    eval_loader.reset()
    loss, keras_acc = model.evaluate(eval_loader, verbose=0)
    eval_loader.reset()
    y_prob = model.predict(eval_loader, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = eval_loader.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    text_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    dict_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    save_confusion_matrix(cm, class_names, output_dir / artifacts["confusion_matrix"], model_family)
    report_path = output_dir / artifacts["classification_report"]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text_report)

    speed = benchmark_single_image_latency(model, eval_loader, benchmark_samples)
    return {
        "model_family": model_family,
        "test_loss": float(loss),
        "keras_accuracy": float(keras_acc),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": dict_report,
        "classification_report_path": str(report_path),
        "confusion_matrix_path": str(output_dir / artifacts["confusion_matrix"]),
        "model_size_mb": model_file_size_mb(model_path) if model_path else None,
        "inference_speed": speed,
    }


def save_confusion_matrix(cm, class_names: list[str], path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"{title} Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to: {path}")
    except Exception as exc:
        print(f"Could not save confusion matrix plot: {exc}")


def save_history_plot(history: dict[str, list[float]], output_path: Path) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(history.get("accuracy", []), label="Train", linewidth=2)
        axes[0].plot(history.get("val_accuracy", []), label="Validation", linewidth=2)
        axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history.get("loss", []), label="Train", linewidth=2)
        axes[1].plot(history.get("val_loss", []), label="Validation", linewidth=2)
        axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if "accuracy" in history and "val_accuracy" in history:
            gap_hist = [
                (train_acc - val_acc) * 100
                for train_acc, val_acc in zip(history["accuracy"], history["val_accuracy"])
            ]
            axes[2].plot(gap_hist, linewidth=2, color="red")
        axes[2].axhline(y=5, color="orange", linestyle="--", label="5%", alpha=0.7)
        axes[2].axhline(y=10, color="red", linestyle="--", label="10%", alpha=0.7)
        axes[2].set(xlabel="Epoch", ylabel="Gap (%)", title="Overfitting Gap")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to: {output_path}")
    except Exception as exc:
        print(f"Could not save training curves: {exc}")


def print_loader_summary(train_loader, eval_loader) -> None:
    print(f"Found {train_loader.samples} training images")
    print(f"Found {eval_loader.samples} validation/evaluation images")
    print(f"Classes: {list(train_loader.class_indices.keys())}\n")


def print_class_weights(class_weights: dict[int, float], class_names: list[str]) -> None:
    print("Class weights:")
    for idx, emotion in enumerate(class_names):
        print(f"   {emotion:10s}: {class_weights[idx]:.2f}")
    print()


def metadata_for(
    model_family: str,
    class_names: list[str],
    history: dict[str, list[float]],
    train_metrics: dict,
    eval_metrics: dict,
    args,
    model_path: Path,
) -> dict:
    settings = data_settings(model_family, args.efficientnet_img_size)
    return {
        "model_family": model_family,
        "timestamp": dt.datetime.now().isoformat(),
        "class_names": class_names,
        "train_dir": str(args.train_dir),
        "validation_dir": str(args.val_dir),
        "test_dir": str(args.test_dir or args.val_dir),
        "epochs_trained": len(history.get("accuracy", [])),
        "train_accuracy": train_metrics.get("accuracy"),
        "train_loss": train_metrics.get("loss"),
        "val_accuracy": eval_metrics["accuracy"],
        "val_loss": eval_metrics["test_loss"],
        "accuracy": eval_metrics["accuracy"],
        "precision_macro": eval_metrics["precision_macro"],
        "recall_macro": eval_metrics["recall_macro"],
        "f1_macro": eval_metrics["f1_macro"],
        "precision_weighted": eval_metrics["precision_weighted"],
        "recall_weighted": eval_metrics["recall_weighted"],
        "f1_weighted": eval_metrics["f1_weighted"],
        "model_size_mb": eval_metrics["model_size_mb"],
        "inference_speed": eval_metrics["inference_speed"],
        "preprocessing": {
            "input_size": list(settings["target_size"]),
            "color_mode": settings["color_mode"],
            "input_scale": settings["input_scale"],
            "source_pipeline": "48x48 grayscale FER faces adapted at loader/inference time",
        },
        "model_path": str(model_path),
        "confusion_matrix_path": eval_metrics["confusion_matrix_path"],
        "classification_report_path": eval_metrics["classification_report_path"],
    }


def evaluate_train_subset(model: Model, train_loader) -> dict:
    loss, accuracy = model.evaluate(train_loader, verbose=0, steps=min(100, len(train_loader)))
    return {"loss": float(loss), "accuracy": float(accuracy)}


def train_one(model_family: str, args, output_dir: Path) -> dict:
    artifacts = CNN_ARTIFACTS if model_family == "cnn" else EFFICIENTNET_ARTIFACTS
    train_loader, val_loader = make_loaders(
        model_family,
        args.train_dir,
        args.val_dir,
        args.batch_size,
        args.efficientnet_img_size,
    )
    print_loader_summary(train_loader, val_loader)
    class_names = list(train_loader.class_indices.keys())
    class_weights = compute_weights(train_loader)
    print_class_weights(class_weights, class_names)

    if model_family == "cnn":
        model = build_baseline_cnn(len(class_names))
        print("=" * 70)
        print("Training baseline CNN")
        print("=" * 70)
    else:
        weights = None if args.efficientnet_weights == "none" else "imagenet"
        model = build_efficientnet_b0(args.efficientnet_img_size, len(class_names), weights)
        print("=" * 70)
        print("Training EfficientNet-B0 transfer model")
        print("Input adaptation: 48x48 grayscale source -> RGB resized face tensor")
        print("=" * 70)

    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}\n")

    callbacks = callbacks_for(output_dir, artifacts)
    head_history = model.fit(
        train_loader,
        epochs=args.epochs,
        validation_data=val_loader,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    fine_history = None
    if model_family == "efficientnet_b0" and not args.skip_fine_tune:
        fine_history = fine_tune_efficientnet(
            model,
            train_loader,
            val_loader,
            callbacks,
            class_weights,
            initial_epoch=len(head_history.history.get("accuracy", [])),
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_layers=args.fine_tune_layers,
        )

    history = merge_histories(head_history, fine_history)
    final_model_path = output_dir / artifacts["final"]
    model.save(final_model_path)
    best_model_path = output_dir / artifacts["best"]
    eval_model = load_model(best_model_path if best_model_path.exists() else final_model_path)

    test_dir = args.test_dir or args.val_dir
    test_loader = make_eval_loader(
        model_family,
        test_dir,
        args.batch_size,
        args.efficientnet_img_size,
    )
    train_metrics = evaluate_train_subset(eval_model, train_loader)
    eval_metrics = evaluate_model(
        eval_model,
        test_loader,
        class_names,
        output_dir,
        artifacts,
        model_family,
        args.benchmark_samples,
        best_model_path if best_model_path.exists() else final_model_path,
    )
    save_history_plot(history, output_dir / artifacts["history"])

    meta = metadata_for(
        model_family,
        class_names,
        history,
        train_metrics,
        eval_metrics,
        args,
        best_model_path if best_model_path.exists() else final_model_path,
    )
    with open(output_dir / artifacts["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print_metrics_summary(model_family, eval_metrics)
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Metadata saved to: {output_dir / artifacts['meta']}")
    return meta


def print_metrics_summary(model_family: str, metrics: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{model_family} evaluation")
    print("=" * 70)
    print(f"Accuracy:           {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro precision:    {metrics['precision_macro'] * 100:.2f}%")
    print(f"Macro recall:       {metrics['recall_macro'] * 100:.2f}%")
    print(f"Macro F1:           {metrics['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1:        {metrics['f1_weighted'] * 100:.2f}%")
    print(f"Model size:         {metrics['model_size_mb']} MB")
    speed = metrics["inference_speed"]
    print(f"Mean latency:       {speed['mean_ms']} ms")
    print(f"P95 latency:        {speed['p95_ms']} ms")
    print(f"Estimated FPS:      {speed['fps']}")
    print("=" * 70)


def default_model_path(output_dir: Path, artifacts: dict) -> Path:
    best = output_dir / artifacts["best"]
    if best.exists():
        return best
    return output_dir / artifacts["final"]


def compare_saved_models(args, output_dir: Path) -> dict:
    test_dir = args.test_dir or args.val_dir
    baseline_path = Path(args.baseline_model) if args.baseline_model else default_model_path(output_dir, CNN_ARTIFACTS)
    efficientnet_path = (
        Path(args.efficientnet_model)
        if args.efficientnet_model
        else default_model_path(output_dir, EFFICIENTNET_ARTIFACTS)
    )

    missing = [str(path) for path in [baseline_path, efficientnet_path] if not path.exists()]
    if missing:
        print("Skipping comparison because these model files are missing:")
        for path in missing:
            print(f"  {path}")
        return {}

    cnn_loader = make_eval_loader("cnn", test_dir, args.batch_size, args.efficientnet_img_size)
    class_names = list(cnn_loader.class_indices.keys())
    eff_loader = make_eval_loader(
        "efficientnet_b0",
        test_dir,
        args.batch_size,
        args.efficientnet_img_size,
    )

    print("\nLoading saved baseline CNN for comparison...")
    cnn_model = load_model(baseline_path)
    cnn_metrics = evaluate_model(
        cnn_model,
        cnn_loader,
        class_names,
        output_dir,
        {
            **CNN_ARTIFACTS,
            "confusion_matrix": "comparison_cnn_confusion_matrix.png",
            "classification_report": "comparison_cnn_classification_report.txt",
        },
        "cnn",
        args.benchmark_samples,
        baseline_path,
    )

    print("\nLoading saved EfficientNet-B0 for comparison...")
    eff_model = load_model(efficientnet_path)
    eff_metrics = evaluate_model(
        eff_model,
        eff_loader,
        class_names,
        output_dir,
        {
            **EFFICIENTNET_ARTIFACTS,
            "confusion_matrix": "comparison_efficientnet_b0_confusion_matrix.png",
            "classification_report": "comparison_efficientnet_b0_classification_report.txt",
        },
        "efficientnet_b0",
        args.benchmark_samples,
        efficientnet_path,
    )

    comparison = {
        "timestamp": dt.datetime.now().isoformat(),
        "test_dir": str(test_dir),
        "baseline_model": str(baseline_path),
        "efficientnet_model": str(efficientnet_path),
        "cnn": compact_metrics(cnn_metrics),
        "efficientnet_b0": compact_metrics(eff_metrics),
        "delta_efficientnet_minus_cnn": {
            "accuracy": eff_metrics["accuracy"] - cnn_metrics["accuracy"],
            "precision_macro": eff_metrics["precision_macro"] - cnn_metrics["precision_macro"],
            "recall_macro": eff_metrics["recall_macro"] - cnn_metrics["recall_macro"],
            "f1_macro": eff_metrics["f1_macro"] - cnn_metrics["f1_macro"],
            "mean_latency_ms": (
                eff_metrics["inference_speed"]["mean_ms"]
                - cnn_metrics["inference_speed"]["mean_ms"]
            ),
            "model_size_mb": eff_metrics["model_size_mb"] - cnn_metrics["model_size_mb"],
        },
    }

    json_path = output_dir / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    save_comparison_csv(comparison, output_dir / "model_comparison.csv")
    print_comparison_summary(comparison)
    return comparison


def compact_metrics(metrics: dict) -> dict:
    return {
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "model_size_mb": metrics["model_size_mb"],
        "inference_speed": metrics["inference_speed"],
    }


def save_comparison_csv(comparison: dict, path: Path) -> None:
    rows = []
    for name in ["cnn", "efficientnet_b0"]:
        metric = comparison[name]
        rows.append(
            {
                "model": name,
                "accuracy": metric["accuracy"],
                "precision_macro": metric["precision_macro"],
                "recall_macro": metric["recall_macro"],
                "f1_macro": metric["f1_macro"],
                "model_size_mb": metric["model_size_mb"],
                "mean_latency_ms": metric["inference_speed"]["mean_ms"],
                "p95_latency_ms": metric["inference_speed"]["p95_ms"],
                "estimated_fps": metric["inference_speed"]["fps"],
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison CSV saved to: {path}")


def print_comparison_summary(comparison: dict) -> None:
    delta = comparison["delta_efficientnet_minus_cnn"]
    print("\n" + "=" * 70)
    print("CNN vs EfficientNet-B0 comparison on the same test set")
    print("=" * 70)
    print(f"CNN accuracy:              {comparison['cnn']['accuracy'] * 100:.2f}%")
    print(f"EfficientNet-B0 accuracy:  {comparison['efficientnet_b0']['accuracy'] * 100:.2f}%")
    print(f"Accuracy delta:            {delta['accuracy'] * 100:+.2f} points")
    print(f"Macro F1 delta:            {delta['f1_macro'] * 100:+.2f} points")
    print(f"Mean latency delta:        {delta['mean_latency_ms']:+.2f} ms")
    print(f"Model size delta:          {delta['model_size_mb']:+.2f} MB")
    print("=" * 70)


def main() -> None:
    args = build_parser().parse_args()
    set_reproducible_seeds()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        compare_saved_models(args, output_dir)
        return

    families = ["cnn", "efficientnet_b0"] if args.model_family == "both" else [args.model_family]
    for family in families:
        train_one(family, args, output_dir)

    if args.model_family in {"efficientnet_b0", "both"}:
        compare_saved_models(args, output_dir)

    print("\nTraining run complete.")