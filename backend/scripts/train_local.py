#!/usr/bin/env python3
"""
Local Training Script - Driver Drowsiness Detection
Uses all 3 datasets: DDD, NTHUDDD, yawing
Train/Test split: 75% / 25%
Saves checkpoints after every epoch so training can be resumed if interrupted.

Usage:
    cd backend
    venv\\Scripts\\activate
    python scripts/train_local.py

To resume after interruption, just run the same command again.
"""

import os
import sys
import json
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# ── Optimize CPU threads ───────────────────────────────────────────────────
tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all cores
tf.config.threading.set_inter_op_parallelism_threads(0)

print("=" * 60)
print("Driver Drowsiness Detection - Local Training")
print("=" * 60)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU') or 'None (CPU training)'}")
print("=" * 60)

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATASET_ROOT    = BASE_DIR / 'datasets'
PREPARED_DIR    = BASE_DIR / 'datasets' / '_prepared'
CHECKPOINT_DIR  = BASE_DIR / 'models' / 'checkpoints'
MODELS_DIR      = BASE_DIR / 'models'
HISTORY_PATH    = CHECKPOINT_DIR / 'training_history.json'
BEST_MODEL_PATH = str(CHECKPOINT_DIR / 'best_model.keras')

IMAGE_SIZE  = (128, 128)  # Full size for better accuracy
BATCH_SIZE  = 32          # Standard batch size
EPOCHS      = 50

# Dataset folder name mappings
ALERT_DIRS  = ['alert', 'Non Drowsy', 'notdrowsy', 'no yawn', 'nodrowsy']
DROWSY_DIRS = ['drowsy', 'Drowsy', 'yawn', 'drowsiness']

os.makedirs(str(CHECKPOINT_DIR), exist_ok=True)
os.makedirs(str(MODELS_DIR), exist_ok=True)


# ── Step 1: Prepare dataset structure ─────────────────────────────────────
def prepare_dataset():
    """Scan all 3 datasets, symlink into train/test split (75/25)."""

    # If already prepared, skip
    train_alert = PREPARED_DIR / 'train' / 'alert'
    if train_alert.exists() and len(list(train_alert.iterdir())) > 0:
        print("✓ Dataset already prepared — skipping")
        return

    print("\nPreparing datasets (75% train / 25% test)...")
    random.seed(42)

    all_alert  = []
    all_drowsy = []

    for dataset_dir in sorted(DATASET_ROOT.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('_'):
            continue
        print(f"  Scanning: {dataset_dir.name}")

        for name in ALERT_DIRS:
            p = dataset_dir / name
            if p.exists():
                files = list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + \
                        list(p.glob('*.png')) + list(p.glob('*.JPG'))
                all_alert.extend(files)
                print(f"    alert  → {len(files)} images")
                break

        for name in DROWSY_DIRS:
            p = dataset_dir / name
            if p.exists():
                files = list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + \
                        list(p.glob('*.png')) + list(p.glob('*.JPG'))
                all_drowsy.extend(files)
                print(f"    drowsy → {len(files)} images")
                break

    print(f"\n  Total: {len(all_alert)} alert, {len(all_drowsy)} drowsy")

    random.shuffle(all_alert)
    random.shuffle(all_drowsy)

    def split75(lst):
        n = int(len(lst) * 0.75)
        return lst[:n], lst[n:]

    train_alert_files,  test_alert_files  = split75(all_alert)
    train_drowsy_files, test_drowsy_files = split75(all_drowsy)

    # Create folders
    for split in ['train', 'test']:
        for cls in ['alert', 'drowsy']:
            (PREPARED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Symlink files
    def link(files, dst):
        for i, src in enumerate(files):
            target = dst / f"{i:06d}_{src.name}"
            if not target.exists():
                try:
                    target.symlink_to(src.resolve())
                except Exception:
                    shutil.copy2(str(src), str(target))  # fallback: copy

    link(train_alert_files,  PREPARED_DIR / 'train' / 'alert')
    link(test_alert_files,   PREPARED_DIR / 'test'  / 'alert')
    link(train_drowsy_files, PREPARED_DIR / 'train' / 'drowsy')
    link(test_drowsy_files,  PREPARED_DIR / 'test'  / 'drowsy')

    print(f"\n✓ Dataset prepared:")
    print(f"  train/alert:  {len(train_alert_files)}")
    print(f"  train/drowsy: {len(train_drowsy_files)}")
    print(f"  test/alert:   {len(test_alert_files)}")
    print(f"  test/drowsy:  {len(test_drowsy_files)}")


# ── Step 2: Create generators ──────────────────────────────────────────────
def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        str(PREPARED_DIR / 'train'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        str(PREPARED_DIR / 'test'),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    return train_gen, test_gen


# ── Step 3: Build model (Transfer Learning with MobileNetV2) ───────────────
def build_model():
    """
    Uses MobileNetV2 pretrained on ImageNet as base.
    Transfer learning gives significantly better accuracy than training from scratch.
    MobileNetV2 is lightweight — good for CPU.
    """
    # Load pretrained base (no top layers)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base initially — train only top layers first
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    return model, base_model


# ── Step 4: Resume checkpoint callback ────────────────────────────────────
class LocalCheckpointCallback(keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, history_path, saved_history):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history_path   = Path(history_path)
        self.history_data   = {k: list(v) for k, v in saved_history.items()}

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch_num:02d}.weights.h5'
        self.model.save_weights(str(ckpt_path))

        for k, v in (logs or {}).items():
            self.history_data.setdefault(k, []).append(float(v))
        with open(str(self.history_path), 'w') as f:
            json.dump(self.history_data, f)

        print(f"\n  ✓ Epoch {epoch_num} checkpoint saved")

        # Keep only last 3 checkpoints
        all_ckpts = sorted([
            f for f in self.checkpoint_dir.iterdir()
            if f.name.startswith('checkpoint_epoch_') and f.name.endswith('.weights.h5')
        ])
        for old in all_ckpts[:-3]:
            old.unlink()


# ── Step 5: Find resume point ──────────────────────────────────────────────
def find_resume_epoch(checkpoint_dir):
    ckpt_dir = Path(checkpoint_dir)
    existing = sorted([
        f for f in ckpt_dir.iterdir()
        if f.name.startswith('checkpoint_epoch_') and f.name.endswith('.weights.h5')
    ]) if ckpt_dir.exists() else []

    if existing:
        latest = existing[-1]
        epoch_num = int(latest.name.replace('checkpoint_epoch_', '').replace('.weights.h5', ''))
        return epoch_num, str(latest)
    return 0, None


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Prepare dataset
    prepare_dataset()

    # Create generators
    print("\nCreating data generators...")
    train_gen, test_gen = create_generators()
    print(f"  Train: {train_gen.samples} images")
    print(f"  Test:  {test_gen.samples} images")

    # Build model
    print("\nBuilding model (MobileNetV2 transfer learning)...")
    model, base_model = build_model()
    model.summary()

    # Check for resume
    initial_epoch, latest_ckpt = find_resume_epoch(CHECKPOINT_DIR)
    saved_history = {}

    if latest_ckpt:
        print(f"\n{'='*60}")
        print("CHECKPOINT FOUND — RESUMING TRAINING")
        print(f"{'='*60}")
        print(f"  Latest : {Path(latest_ckpt).name}")
        print(f"  Resume : epoch {initial_epoch + 1}")
        print(f"  Left   : {EPOCHS - initial_epoch} epochs")
        model.load_weights(latest_ckpt)
        print("  ✓ Weights loaded")
        if HISTORY_PATH.exists():
            with open(str(HISTORY_PATH)) as f:
                saved_history = json.load(f)
        print(f"{'='*60}")
    else:
        print("\nNo checkpoint found — starting fresh from epoch 1")

    # Callbacks
    callbacks = [
        LocalCheckpointCallback(CHECKPOINT_DIR, HISTORY_PATH, saved_history),
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(f"\n{'='*60}")
    print("PHASE 1: Training top layers (base frozen)")
    print(f"{'='*60}")
    print(f"Total epochs  : {EPOCHS}")
    print(f"Start epoch   : {initial_epoch + 1}  {'(RESUMING)' if initial_epoch > 0 else '(FRESH START)'}")
    print(f"Image size    : {IMAGE_SIZE}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Train samples : {train_gen.samples}")
    print(f"Test samples  : {test_gen.samples}")
    print(f"Checkpoints   : {CHECKPOINT_DIR}")
    print(f"{'='*60}\n")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1
    )

    # ── Phase 2: Fine-tuning — unfreeze top layers of base model ──────────
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-tuning (unfreezing top 30 layers)")
    print(f"{'='*60}")

    base_model.trainable = True
    # Freeze all except last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    fine_tune_epochs = EPOCHS + 20
    fine_tune_callbacks = [
        LocalCheckpointCallback(CHECKPOINT_DIR, HISTORY_PATH, saved_history),
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    history_fine = model.fit(
        train_gen,
        epochs=fine_tune_epochs,
        initial_epoch=EPOCHS,
        validation_data=test_gen,
        callbacks=fine_tune_callbacks,
        verbose=1
    )

    print("\n✓ TRAINING COMPLETE (both phases)!")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\nEvaluating model...")
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = test_gen.classes

    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen, verbose=0)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    EER = (fpr[eer_idx] + fnr[eer_idx]) / 2

    print(f"\n{'='*60}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision : {test_prec:.4f}")
    print(f"Recall    : {test_rec:.4f}")
    print(f"F1-Score  : {2*(test_prec*test_rec)/(test_prec+test_rec):.4f}")
    print(f"Loss      : {test_loss:.4f}")
    print(f"\nERROR RATE METRICS")
    print(f"FAR (False Acceptance Rate) : {FAR:.4f} ({FAR*100:.2f}%)")
    print(f"FRR (False Rejection Rate)  : {FRR:.4f} ({FRR*100:.2f}%)")
    print(f"EER (Equal Error Rate)      : {EER:.4f} ({EER*100:.2f}%)")
    print(f"\n{'✓' if test_acc >= 0.85 else '✗'} Model {'MEETS' if test_acc >= 0.85 else 'does NOT meet'} accuracy requirement (>= 85%)")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Alert', 'Drowsy'])}")

    # ── Save final model ───────────────────────────────────────────────────
    final_model_path = str(MODELS_DIR / 'cnn_drowsiness.h5')
    model.save(final_model_path)
    print(f"✓ Final model saved: {final_model_path}")

    # ── Convert to TFLite ──────────────────────────────────────────────────
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = str(MODELS_DIR / 'cnn_drowsiness.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✓ TFLite model saved: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    # ── Save plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plot_path = str(MODELS_DIR / 'training_history.png')
    plt.savefig(plot_path)
    print(f"✓ Training plot saved: {plot_path}")

    print("\n" + "="*60)
    print("ALL DONE!")
    print(f"  Model    : {final_model_path}")
    print(f"  TFLite   : {tflite_path}")
    print(f"  Plot     : {plot_path}")
    print("="*60)


if __name__ == '__main__':
    main()
