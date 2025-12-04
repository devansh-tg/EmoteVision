# train.py - BALANCED OPTIMIZATION: Target 70% accuracy with ~5% overfitting gap
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# dataset paths
train_dir = '/kaggle/input/emotion-detection-dataset/train'
val_dir   = '/kaggle/input/emotion-detection-dataset/test'

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

batch_size = 64
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
    epochs=150,                   # More epochs with aggressive early stopping
    validation_data=val_loader,
    callbacks=callbacks,
    class_weight=class_weights,   # Handle class imbalance
    verbose=1
)

# save final model
model.save('model.h5')
print("\n✅ Training complete!")
print("   💾 Best model saved to: model_best.h5")
print("   💾 Final model saved to: model.h5")

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
real_world_est = val_acc * 0.92  # Typically 8% drop in real conditions
print(f"\n🌍 Estimated Real-World Accuracy: {real_world_est * 100:.1f}%")
print(f"   (Based on validation accuracy with typical degradation)")

# Plot training history
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Gap plot (difference between train and val accuracy)
    plt.subplot(1, 3, 3)
    gap_history = [(t - v) * 100 for t, v in zip(history.history['accuracy'], history.history['val_accuracy'])]
    plt.plot(gap_history, linewidth=2, color='red')
    plt.axhline(y=5, color='orange', linestyle='--', label='5% threshold', alpha=0.7)
    plt.axhline(y=10, color='red', linestyle='--', label='10% threshold', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gap (%)', fontsize=12)
    plt.title('Overfitting Gap Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n📊 Training plots saved as 'training_history.png'")
    print("   (Includes accuracy, loss, and gap analysis)")
except Exception as e:
    print(f"\n⚠️  Could not create plots: {e}")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. Check training_history.png for visual analysis")
print("2. Use model_best.h5 (best validation accuracy)")
print("3. Test with real webcam using gui_advanced.py")
print("=" * 70)
