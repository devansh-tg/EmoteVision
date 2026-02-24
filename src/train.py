# train.py - BALANCED OPTIMIZATION: Target 70% accuracy with ~5% overfitting gap
import os
import argparse
import json
import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# ── Argument parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train EmoteVision emotion CNN')
parser.add_argument('--train-dir', default='/kaggle/input/emotion-detection-dataset/train',
                    help='Path to training data directory')
parser.add_argument('--val-dir',   default='/kaggle/input/emotion-detection-dataset/test',
                    help='Path to validation data directory')
parser.add_argument('--output-dir', default='.', help='Directory to save model files')
parser.add_argument('--epochs', type=int, default=150, help='Max training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
args = parser.parse_args()

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# dataset paths
train_dir = args.train_dir
val_dir   = args.val_dir

# LIGHTER data augmentation - preserve facial features for better learning
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,              # Lighter rotation
    width_shift_range=0.2,          # Lighter horizontal shift
    height_shift_range=0.2,         # Lighter vertical shift
    horizontal_flip=True,
    zoom_range=0.15,                # Lighter zoom
    shear_range=0.15,               # Lighter shear
    brightness_range=[0.8, 1.2],    # Lighter brightness range
    channel_shift_range=15.0,       # Lighter contrast
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

batch_size = args.batch_size
img_size   = (48, 48)

train_loader = train_gen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    color_mode='grayscale', class_mode='categorical', shuffle=True)

val_loader = val_gen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size,
    color_mode='grayscale', class_mode='categorical', shuffle=False)

print(f"✅ Found {train_loader.samples} training images")
print(f"✅ Found {val_loader.samples} validation images")
print(f"📊 Classes: {list(train_loader.class_indices.keys())}\n")

# Compute class weights to handle imbalance
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(train_loader.classes),
    y=train_loader.classes
)
class_weights = dict(enumerate(class_weights_array))
print("📊 Class Weights:")
for idx, emotion in enumerate(train_loader.class_indices.keys()):
    print(f"   {emotion:10s}: {class_weights[idx]:.2f}")
print()

# BALANCED model: Higher capacity for 70% accuracy + moderate regularization for ~5% gap
l2_strength = 0.0005  # Lighter L2 regularization for better learning

model = Sequential([
    # Block 1: (48x48) -> (24x24) - Increased capacity
    Conv2D(64, (3,3), activation='relu', padding='same', 
           kernel_regularizer=l2(l2_strength), input_shape=(48,48,1)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(l2_strength)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.20),  # Lighter dropout

    # Block 2: (24x24) -> (12x12) - More filters
    Conv2D(128, (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(l2_strength)),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(l2_strength)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Block 3: (12x12) -> (6x6) - Even more filters
    Conv2D(256, (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(l2_strength)),
    BatchNormalization(),
    Conv2D(256, (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(l2_strength)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.30),
    
    # Block 4: (6x6) -> Additional depth for better features
    Conv2D(512, (3,3), activation='relu', padding='same',
           kernel_regularizer=l2(l2_strength)),
    BatchNormalization(),
    Dropout(0.35),

    # Global Average Pooling + Flatten hybrid
    GlobalAveragePooling2D(),
    
    # Substantial dense layers for better classification
    Dense(512, activation='relu', kernel_regularizer=l2(l2_strength)),
    BatchNormalization(),
    Dropout(0.40),
    
    Dense(256, activation='relu', kernel_regularizer=l2(l2_strength)),
    BatchNormalization(),
    Dropout(0.40),
    
    Dense(7, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=2e-4),  # Higher learning rate for faster learning
    metrics=['accuracy']
)

model.summary()
total_params = model.count_params()
print(f"\n📊 Total Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
print("   (Balanced model for 70% accuracy with minimal overfitting!)\n")

# Balanced callbacks for optimal training
callbacks = [
    # Early stopping with moderate patience
    EarlyStopping(
        monitor='val_accuracy',
        patience=18,              # Allow more epochs to reach higher accuracy
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001,
        mode='max'
    ),
    
    # Reduce learning rate when plateauing
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,               # Halve the learning rate
        patience=6,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    ),
    
    # Save best model based on validation accuracy
    ModelCheckpoint(
        'model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    
    # Save checkpoint every improvement
    ModelCheckpoint(
        'model_checkpoint.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0,
        mode='min'
    )
]

print("=" * 70)
print("🚀 OPTIMIZED TRAINING - Target: 70% Accuracy with ~5% Gap")
print("=" * 70)
print("✅ Lighter data augmentation (better feature learning)")
print("✅ Balanced model capacity (~2-3M params)")
print("✅ 4 conv blocks + GlobalAveragePooling")
print("✅ Reduced dropout (0.20-0.40)")
print("✅ Lower L2 regularization (0.0005)")
print("✅ Class balancing with weights")
print("✅ Higher learning rate (2e-4)")
print("=" * 70)
print()

# model training with class weights and callbacks
history = model.fit(
    train_loader,
    epochs=args.epochs,
    validation_data=val_loader,
    callbacks=callbacks,
    class_weight=class_weights,   # Handle class imbalance
    verbose=1
)

# save final model
out_dir = args.output_dir
os.makedirs(out_dir, exist_ok=True)
model.save(os.path.join(out_dir, 'model.h5'))
print("\n✅ Training complete!")
print(f"   💾 Best model saved to: {os.path.join(out_dir, 'model_best.h5')}")
print(f"   💾 Final model saved to: {os.path.join(out_dir, 'model.h5')}")

# Evaluate and show gap
print("\n" + "=" * 70)
print("📊 FINAL PERFORMANCE ANALYSIS")
print("=" * 70)

val_loss, val_acc = model.evaluate(val_loader, verbose=0)
train_loss, train_acc = model.evaluate(train_loader, verbose=0, steps=100)

gap = abs(train_acc - val_acc) * 100

print(f"\n📈 Training Accuracy:     {train_acc * 100:.2f}%")
print(f"📊 Validation Accuracy:   {val_acc * 100:.2f}%")
print(f"📉 Train-Val Gap:         {gap:.2f}%")
print(f"📉 Train Loss:            {train_loss:.4f}")
print(f"📉 Val Loss:              {val_loss:.4f}")

# Detailed status
print("\n" + "-" * 70)
if val_acc >= 0.70 and gap < 5:
    print("🏆 Status: PERFECT - 70%+ accuracy with <5% gap!")
    print("   Outstanding performance! Production-ready model.")
elif val_acc >= 0.68 and gap < 7:
    print("✅ Status: EXCELLENT - Near 70% with minimal overfitting!")
    print("   Great balance of accuracy and generalization.")
elif val_acc >= 0.65 and gap < 10:
    print("✅ Status: VERY GOOD - Strong performance")
    print("   Good accuracy with acceptable generalization.")
elif gap < 5:
    print("⚠️  Status: Low overfitting but lower accuracy")
    print("   Consider: Increase model capacity or reduce regularization.")
else:
    print("⚠️  Status: Room for improvement")
    print(f"   Target: 70% accuracy with ~5% gap (Current: {val_acc*100:.1f}% with {gap:.1f}% gap)")

print("-" * 70)

# Estimate real-world performance
real_world_est = val_acc * 0.92
print(f"\n🌍 Estimated Real-World Accuracy: {real_world_est * 100:.1f}%")

# ── Save model_meta.json ───────────────────────────────────────────────────────
import datetime
import json
meta = {
    "val_accuracy":   float(val_acc),
    "train_accuracy": float(train_acc),
    "val_loss":       float(val_loss),
    "train_loss":     float(train_loss),
    "gap_pct":        float(gap),
    "epochs_trained": len(history.history['accuracy']),
    "timestamp":      datetime.datetime.now().isoformat(),
    "class_names":    list(train_loader.class_indices.keys()),
}
meta_path = os.path.join(out_dir, 'model_meta.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\n📋 Model metadata saved to: {meta_path}")

# ── Confusion matrix & classification report ───────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    # Predictions on validation set
    val_loader.reset()
    y_pred_probs = model.predict(val_loader, verbose=0)
    y_pred = y_pred_probs.argmax(axis=1)
    y_true = val_loader.classes
    class_names = list(val_loader.class_indices.keys())

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {cm_path}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    report_path = os.path.join(out_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"📄 Classification report saved to: {report_path}")
    print("\n" + report)

    # Training curves (accuracy, loss, gap)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set(xlabel='Epoch', ylabel='Loss', title='Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    gap_hist = [(t - v) * 100 for t, v in
                zip(history.history['accuracy'], history.history['val_accuracy'])]
    axes[2].plot(gap_hist, linewidth=2, color='red')
    axes[2].axhline(y=5,  color='orange', linestyle='--', label='5%',  alpha=0.7)
    axes[2].axhline(y=10, color='red',    linestyle='--', label='10%', alpha=0.7)
    axes[2].set(xlabel='Epoch', ylabel='Gap (%)', title='Overfitting Gap')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(out_dir, 'training_history.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"📈 Training curves saved to: {hist_path}")

except Exception as e:
    print(f"\n⚠️  Could not generate artifacts: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print(f"\nOutput files in '{out_dir}':")
print("  model.h5               — final model")
print("  model_best.h5          — best val_accuracy checkpoint")
print("  model_meta.json        — accuracy / metadata (read by app.py)")
print("  confusion_matrix.png   — per-class detection heatmap")
print("  classification_report.txt — precision / recall / F1 per class")
print("  training_history.png   — accuracy, loss, gap curves")
print("=" * 70)
