"""
export_tflite.py — Convert model.h5 to a quantized TFLite model.

Usage:
    python export_tflite.py [--model path/to/model.h5] [--out path/to/model.tflite]

Options:
    --model   Path to the Keras .h5 model  (default: MODEL_PATH from config)
    --out     Output .tflite path           (default: TFLITE_PATH from config)
    --quant   Quantization mode: none | dynamic | float16 | int8  (default: dynamic)
    --rep-dir Directory of representative images for int8 calibration (required for --quant int8)
"""

import argparse
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Parse args before heavy TF import so --help is fast.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Keras .h5 model to TFLite with optional quantization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   default=None, help="Path to .h5 model file")
    p.add_argument("--out",     default=None, help="Output .tflite file path")
    p.add_argument(
        "--quant",
        default="dynamic",
        choices=["none", "dynamic", "float16", "int8"],
        help="Quantization strategy",
    )
    p.add_argument(
        "--rep-dir",
        default=None,
        metavar="DIR",
        help="Folder of 48×48 grayscale images for int8 representative dataset",
    )
    return p


def representative_dataset_gen(image_dir: str):
    """Yield representative 48×48 grayscale samples for int8 calibration."""
    import cv2  # local import — only needed for int8

    supported = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [
        os.path.join(root, f)
        for root, _, files in os.walk(image_dir)
        for f in files
        if os.path.splitext(f)[1].lower() in supported
    ]
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir!r}")

    print(f"  Found {len(files)} calibration images in {image_dir!r}")
    for path in files[:500]:  # cap at 500 samples
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (48, 48)).astype(np.float32) / 255.0
        img = img.reshape(1, 48, 48, 1)
        yield [img]


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve paths via config if not provided.
    from config import MODEL_PATH, TFLITE_PATH  # noqa: E402

    model_path  = args.model or MODEL_PATH
    output_path = args.out   or TFLITE_PATH

    if not os.path.isfile(model_path):
        sys.exit(f"[ERROR] Model file not found: {model_path!r}")

    print("[1/4] Importing TensorFlow …")
    import tensorflow as tf

    print(f"[2/4] Loading Keras model from {model_path!r} …")
    model = tf.keras.models.load_model(model_path)
    model.summary(print_fn=lambda s: None)  # suppress verbose summary

    print("[3/4] Building TFLite converter …")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quant == "none":
        print("  Quantization: none (float32)")

    elif args.quant == "dynamic":
        print("  Quantization: dynamic range")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif args.quant == "float16":
        print("  Quantization: float16")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif args.quant == "int8":
        if not args.rep_dir:
            sys.exit("[ERROR] --rep-dir is required for int8 quantization.")
        print(f"  Quantization: int8 (calibrating with images from {args.rep_dir!r})")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset_gen(args.rep_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.uint8
        converter.inference_output_type = tf.uint8

    print("[4/4] Converting and saving …")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✓ TFLite model saved to {output_path!r}  ({size_kb:.1f} KB)")
    print("  To use it, set  USE_TFLITE=true  (or =auto) in your environment.")


if __name__ == "__main__":
    main()
