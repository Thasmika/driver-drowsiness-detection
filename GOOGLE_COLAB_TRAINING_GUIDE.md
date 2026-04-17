# Complete Guide: Training Drowsiness Detection Model on Google Colab

This guide will walk you through training your CNN drowsiness detection model on Google Colab's free GPU.

## Why Google Colab?

- **Free GPU**: NVIDIA T4 (15GB VRAM)
- **12-hour sessions**: Long enough for training
- **15GB Google Drive**: For dataset storage
- **No verification needed**: Just a Google account
- **Easy to use**: Browser-based, no setup

---

## Step 1: Open Google Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **"New Notebook"** or **"File" → "New notebook"**
4. Rename it: "Drowsiness_Detection_Training"

---

## Step 2: Enable GPU

**IMPORTANT: Do this FIRST before running any code!**

1. Click **"Runtime"** in the top menu
2. Select **"Change runtime type"**
3. Under **"Hardware accelerator"**, select **"T4 GPU"** (or "GPU")
4. Click **"Save"**

---

## Step 3: Upload Your Dataset ZIP to Google Drive

Your dataset is ~5GB, so **Google Drive is the only practical option**. Direct upload to Colab won't work for files this large.

### Step 3.1: Prepare Your ZIP File

Zip all 3 datasets into a single file on your PC. The ZIP must have this internal structure:

```
drowsiness-datasets.zip
└── drowsiness-datasets/
    ├── DDD/              ← Driver Drowsiness Dataset
    │   ├── alert/
    │   └── drowsy/
    ├── NTHUDDD/          ← NTHU Driver Drowsiness Dataset
    │   ├── alert/
    │   └── drowsy/
    └── yawning/          ← Yawning Detection Dataset
        ├── alert/
        └── drowsy/
```

All 3 datasets will be combined and used for training.

### Step 3.2: Upload ZIP to Google Drive

1. Go to [https://drive.google.com](https://drive.google.com)
2. Sign in with your Google account
3. Upload `drowsiness-datasets.zip` to **My Drive** (root level)
4. Wait for upload to finish — 5GB takes ~15–30 mins depending on your internet speed
5. The file stays in Drive permanently — if Colab disconnects, you don't need to re-upload

### Step 3.3: Mount Drive and Extract ZIP in Colab

Run this as your **first cell** in Colab:

```python
from google.colab import drive
import zipfile
import os

# Mount Google Drive
drive.mount('/content/drive')

# Verify ZIP exists
zip_path = '/content/drive/MyDrive/drowsiness-datasets.zip'
if os.path.exists(zip_path):
    print(f"✓ ZIP found: {zip_path}")
    print(f"  Size: {os.path.getsize(zip_path) / (1024**3):.2f} GB")
else:
    print(f"❌ ZIP not found at {zip_path}")
    print("   Make sure you uploaded drowsiness-datasets.zip to My Drive root")

# Extract ZIP (takes 5-10 minutes for 5GB)
print("\nExtracting ZIP... please wait...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/datasets')

print("✓ Extraction complete!")

# Verify folder structure
print("\nExtracted structure:")
for item in sorted(os.listdir('/content/datasets/drowsiness-datasets')):
    item_path = f'/content/datasets/drowsiness-datasets/{item}'
    if os.path.isdir(item_path):
        subfolders = os.listdir(item_path)
        print(f"  {item}/ → {subfolders}")
```

> **Note:** After extraction, set `DATASET_PATH = "/content/datasets/drowsiness-datasets"` in Cell 4.

---

## Step 4: Training Code - Copy These Cells

### Cell 1: Install Dependencies

```python
# Install required packages
!pip install -q opencv-python-headless
!pip install -q pillow
!pip install -q scikit-learn

print("✓ Dependencies installed")
```

### Cell 2: Import Libraries

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Verify GPU is working
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("✓ GPU is enabled and ready!")
else:
    print("⚠ WARNING: No GPU detected. Training will be slow.")
    print("   Go to Runtime → Change runtime type → Select GPU")
```

### Cell 3: Prepare Dataset Directory Structure

```python
import os
import shutil
from pathlib import Path

DATASET_PATH = "/content/datasets/drowsiness-datasets"
PREPARED_PATH = "/content/prepared"

# Mapping of possible folder names to standard names
ALERT_DIRS  = ['alert', 'Non Drowsy', 'notdrowsy', 'no yawn', 'nodrowsy']
DROWSY_DIRS = ['drowsy', 'Drowsy', 'yawn', 'drowsiness']

def prepare_dataset(src_root, dst_root):
    """
    Reorganise all 3 datasets into a single flat structure:
      dst_root/train/alert/   (75%)
      dst_root/train/drowsy/  (75%)
      dst_root/test/alert/    (25%)
      dst_root/test/drowsy/   (25%)
    Images are symlinked (not copied) to save disk space.
    """
    import random
    random.seed(42)

    all_alert  = []
    all_drowsy = []

    src_root = Path(src_root)
    for dataset_dir in sorted(src_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        print(f"Scanning: {dataset_dir.name}")

        for name in ALERT_DIRS:
            p = dataset_dir / name
            if p.exists():
                files = list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + \
                        list(p.glob("*.png")) + list(p.glob("*.JPG"))
                all_alert.extend(files)
                print(f"  alert  → {len(files)} images")
                break

        for name in DROWSY_DIRS:
            p = dataset_dir / name
            if p.exists():
                files = list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + \
                        list(p.glob("*.png")) + list(p.glob("*.JPG"))
                all_drowsy.extend(files)
                print(f"  drowsy → {len(files)} images")
                break

    print(f"\nTotal: {len(all_alert)} alert, {len(all_drowsy)} drowsy")

    # Shuffle and split 75/25
    random.shuffle(all_alert)
    random.shuffle(all_drowsy)

    def split(lst):
        n = int(len(lst) * 0.75)
        return lst[:n], lst[n:]

    train_alert,  test_alert  = split(all_alert)
    train_drowsy, test_drowsy = split(all_drowsy)

    # Create destination folders
    for split_name in ['train', 'test']:
        for cls in ['alert', 'drowsy']:
            os.makedirs(f"{dst_root}/{split_name}/{cls}", exist_ok=True)

    # Symlink files (saves disk space, no copying needed)
    def link_files(files, dst_folder):
        for i, src in enumerate(files):
            dst = Path(dst_folder) / f"{i:06d}_{src.name}"
            if not dst.exists():
                os.symlink(src, dst)

    link_files(train_alert,  f"{dst_root}/train/alert")
    link_files(test_alert,   f"{dst_root}/test/alert")
    link_files(train_drowsy, f"{dst_root}/train/drowsy")
    link_files(test_drowsy,  f"{dst_root}/test/drowsy")

    print(f"\n✓ Prepared dataset at: {dst_root}")
    print(f"  train/alert:  {len(train_alert)}")
    print(f"  train/drowsy: {len(train_drowsy)}")
    print(f"  test/alert:   {len(test_alert)}")
    print(f"  test/drowsy:  {len(test_drowsy)}")
    return len(train_alert) + len(train_drowsy), len(test_alert) + len(test_drowsy)

train_count, test_count = prepare_dataset(DATASET_PATH, PREPARED_PATH)
print(f"\nTotal train: {train_count}  |  Total test: {test_count}")
```

### Cell 4: Create Memory-Efficient Data Generators

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_SIZE  = (128, 128)
BATCH_SIZE  = 32

# Training generator — reads from disk, augments on the fly
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

# Test generator — only rescale, no augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    f"{PREPARED_PATH}/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    f"{PREPARED_PATH}/test",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False   # Keep order for evaluation
)

print(f"\n✓ Generators created (images read from disk — no RAM overload)")
print(f"  Train batches: {len(train_generator)}")
print(f"  Test batches:  {len(test_generator)}")
print(f"  Class indices: {train_generator.class_indices}")
```

### Cell 5: Build CNN Model

```python
def build_cnn_model(input_shape=(128, 128, 3)):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        # Block 4
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        # Dense
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_cnn_model(input_shape=(*IMAGE_SIZE, 3))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)
model.summary()
```

### Cell 6: Train Model (with Resume Support)

> **This cell saves a checkpoint to Google Drive after every epoch.**
> If Colab disconnects, just reconnect, re-run Cells 1–5, then re-run this cell — it will automatically resume from the last saved epoch.

```python
import os
import json
from google.colab import drive

# ── Step 1: Make sure Google Drive is mounted ──────────────────────────────
if not os.path.exists('/content/drive/MyDrive'):
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
else:
    print("✓ Google Drive already mounted")

# ── Step 2: Config ─────────────────────────────────────────────────────────
EPOCHS               = 50
DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/drowsiness_checkpoints'
BEST_MODEL_PATH      = f'{DRIVE_CHECKPOINT_DIR}/best_model.h5'
HISTORY_PATH         = f'{DRIVE_CHECKPOINT_DIR}/training_history.json'

os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
print(f"✓ Checkpoint folder: {DRIVE_CHECKPOINT_DIR}")

# ── Step 3: Find last saved checkpoint ────────────────────────────────────
initial_epoch = 0
saved_history = {}

existing_ckpts = sorted([
    f for f in os.listdir(DRIVE_CHECKPOINT_DIR)
    if f.startswith('checkpoint_epoch_') and f.endswith('.h5')
])

if existing_ckpts:
    latest_file = existing_ckpts[-1]
    latest_path = f'{DRIVE_CHECKPOINT_DIR}/{latest_file}'
    initial_epoch = int(latest_file.replace('checkpoint_epoch_', '').replace('.h5', ''))
    
    print(f"\n{'='*60}")
    print(f"CHECKPOINT FOUND — RESUMING TRAINING")
    print(f"{'='*60}")
    print(f"  Latest checkpoint : {latest_file}")
    print(f"  Resuming from     : epoch {initial_epoch + 1}")
    print(f"  Remaining epochs  : {EPOCHS - initial_epoch}")
    
    # Load weights into model
    model.load_weights(latest_path)
    print(f"  ✓ Weights loaded successfully")
    
    # Load previous history
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            saved_history = json.load(f)
        print(f"  ✓ History loaded ({initial_epoch} epochs of data)")
    print(f"{'='*60}")
else:
    print("\nNo checkpoint found — starting fresh from epoch 1")

# ── Step 4: Custom callback — saves to Drive after every epoch ─────────────
class DriveCheckpointCallback(keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, history_path, saved_history):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.history_path   = history_path
        self.history_data   = {k: list(v) for k, v in saved_history.items()}

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        ckpt_path = f'{self.checkpoint_dir}/checkpoint_epoch_{epoch_num:02d}.h5'
        
        # Save weights to Drive
        self.model.save_weights(ckpt_path)
        
        # Append metrics
        for k, v in (logs or {}).items():
            self.history_data.setdefault(k, []).append(float(v))
        
        # Save history JSON to Drive
        with open(self.history_path, 'w') as f:
            json.dump(self.history_data, f)
        
        print(f"\n  ✓ Epoch {epoch_num} checkpoint saved to Drive")
        
        # Keep only last 3 checkpoints to save Drive space
        all_ckpts = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('checkpoint_epoch_') and f.endswith('.h5')
        ])
        for old in all_ckpts[:-3]:
            old_path = f'{self.checkpoint_dir}/{old}'
            os.remove(old_path)

# ── Step 5: Callbacks ──────────────────────────────────────────────────────
callbacks = [
    DriveCheckpointCallback(DRIVE_CHECKPOINT_DIR, HISTORY_PATH, saved_history),
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

# ── Step 6: Train ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Total epochs  : {EPOCHS}")
print(f"Start epoch   : {initial_epoch + 1}  {'(RESUMING)' if initial_epoch > 0 else '(FRESH START)'}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Train samples : {train_generator.samples}")
print(f"Test samples  : {test_generator.samples}")
print(f"Checkpoints   : {DRIVE_CHECKPOINT_DIR}")
print("="*60 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,   # ← KEY: tells Keras to skip already-done epochs
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ TRAINING COMPLETE!")
print(f"  Best model: {BEST_MODEL_PATH}")
```

> **If Colab disconnects mid-training:**
> 1. Click **"Reconnect"** in Colab
> 2. Re-run **Cells 1–5** (takes ~10 mins — extracts ZIP, builds generators, builds model)
> 3. Re-run **Cell 6** — it mounts Drive automatically, finds the latest checkpoint, and resumes from the correct epoch
> 4. You will see: `CHECKPOINT FOUND — RESUMING TRAINING` with the epoch number



### Cell 7: Evaluate Model

```python
from sklearn.metrics import roc_curve

# Reset generator to start from beginning
test_generator.reset()

# Get all predictions
print("Running predictions on test set...")
y_pred_proba = model.predict(test_generator, verbose=1).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)
y_true = test_generator.classes

# Standard metrics
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator, verbose=0)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()

# Error rate metrics
FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
fnr = 1 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
EER = (fpr[eer_idx] + fnr[eer_idx]) / 2

print(f"\n{'='*60}")
print("TEST SET PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {2*(test_precision*test_recall)/(test_precision+test_recall):.4f}")
print(f"Loss:      {test_loss:.4f}")

print(f"\n{'='*60}")
print("ERROR RATE METRICS")
print(f"{'='*60}")
print(f"FAR  (False Acceptance Rate): {FAR:.4f} ({FAR*100:.2f}%)")
print(f"     → Drowsy detected as Alert (missed drowsiness)")
print(f"FRR  (False Rejection Rate):  {FRR:.4f} ({FRR*100:.2f}%)")
print(f"     → Alert detected as Drowsy (false alarm)")
print(f"EER  (Equal Error Rate):      {EER:.4f} ({EER*100:.2f}%)")
print(f"     → Threshold where FAR ≈ FRR (lower is better)")

if test_acc >= 0.85:
    print(f"\n✓ Model MEETS accuracy requirement (>= 85%)")
else:
    print(f"\n✗ Model does NOT meet accuracy requirement (< 85%)")

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(y_true, y_pred, target_names=['Alert', 'Drowsy']))
print(f"\nConfusion Matrix:")
print(cm)
print(f"  TN={TN}  FP={FP}")
print(f"  FN={FN}  TP={TP}")
```

### Cell 8: Plot Training History

```python
# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Alert', 'Drowsy'],
            yticklabels=['Alert', 'Drowsy'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot FAR / FRR / EER curve
plt.figure(figsize=(8, 6))
plt.plot(thresholds, fpr[:len(thresholds)], label='FAR (False Acceptance Rate)', color='red')
plt.plot(thresholds, fnr[:len(thresholds)], label='FRR (False Rejection Rate)', color='blue')
plt.axvline(x=thresholds[eer_idx], color='green', linestyle='--',
            label=f'EER = {EER*100:.2f}% at threshold {thresholds[eer_idx]:.2f}')
plt.xlabel('Decision Threshold')
plt.ylabel('Error Rate')
plt.title('FAR / FRR / EER Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Cell 9: Convert to TensorFlow Lite

```python
# Convert to TFLite
print("Converting to TensorFlow Lite...")

# Standard conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('drowsiness_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✓ TFLite model saved: drowsiness_model.tflite")
print(f"  Size: {len(tflite_model) / 1024:.2f} KB")

# Quantized conversion (INT8)
print("\nConverting to quantized TFLite (INT8)...")

def representative_dataset():
    for i, (batch_x, _) in enumerate(test_generator):
        yield [batch_x.astype(np.float32)]
        if i >= 100:
            break

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_dataset
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.inference_input_type = tf.uint8
converter_quant.inference_output_type = tf.uint8

tflite_quant_model = converter_quant.convert()

# Save quantized model
with open('drowsiness_model_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print(f"✓ Quantized TFLite model saved: drowsiness_model_quantized.tflite")
print(f"  Size: {len(tflite_quant_model) / 1024:.2f} KB")
print(f"  Size reduction: {(1 - len(tflite_quant_model)/len(tflite_model))*100:.1f}%")
```

### Cell 10: Save and Download Models

```python
# Save Keras model
model.save('drowsiness_cnn_model.h5')
print("✓ Keras model saved: drowsiness_cnn_model.h5")

# Save weights
model.save_weights('model_weights.h5')
print("✓ Model weights saved: model_weights.h5")

print("\n" + "="*60)
print("ALL MODELS SAVED SUCCESSFULLY!")
print("="*60)
print("\nSaved files:")
print("  1. best_model.h5 (best checkpoint)")
print("  2. drowsiness_cnn_model.h5 (final Keras model)")
print("  3. model_weights.h5 (weights only)")
print("  4. drowsiness_model.tflite (TFLite for mobile)")
print("  5. drowsiness_model_quantized.tflite (Quantized TFLite)")

# Download models
print("\n" + "="*60)
print("DOWNLOADING MODELS")
print("="*60)

from google.colab import files

print("\nDownloading TFLite model (for your Flutter app)...")
files.download('drowsiness_model.tflite')

print("\nDownloading quantized TFLite model (smaller, optimized)...")
files.download('drowsiness_model_quantized.tflite')

print("\nDownloading Keras model (full model)...")
files.download('drowsiness_cnn_model.h5')

print("\n✓ All downloads started!")
print("  Check your browser's download folder")
```

---

## Important Notes

### Session Management
- **12-hour limit**: Colab sessions disconnect after 12 hours
- **Idle timeout**: Sessions may disconnect if idle for 90 minutes
- **Keep active**: Click in the notebook occasionally to prevent disconnection
- **Save frequently**: Models are auto-saved, but download them when training completes

### If Session Disconnects
1. Reconnect to runtime (click "Reconnect" button)
2. Re-run Cells 1–5 (mount Drive, extract ZIP, create generators, build model)
3. Re-run Cell 6 — it **automatically detects the latest checkpoint** in Google Drive and resumes from that epoch
4. Your best model is always safe in `drowsiness_checkpoints/best_model.h5` on Drive

### Memory Management
- The new approach reads images from disk per batch — RAM usage stays low (~2–3GB)
- If you still get "Out of Memory" errors:
  - Reduce `BATCH_SIZE` to 16 in Cell 4
  - Use smaller `IMAGE_SIZE = (96, 96)` in Cell 4
- If you get "No space left on device" during extraction:
  - Your 5GB ZIP expands to ~10GB — Colab has 100GB disk, this should be fine

### GPU Quota
- Colab has usage limits (not publicly disclosed)
- If GPU is unavailable, wait a few hours
- Consider Colab Pro ($10/month) for priority access

---

## After Training

1. **Download your models** (Cell 10 does this automatically)
2. **Copy to your project**:
   ```
   backend/models/cnn_drowsiness.tflite
   ```
3. **Test locally**:
   ```bash
   cd backend
   python scripts/test_trained_model.py
   ```

---

## Troubleshooting

### "No GPU detected"
- Go to Runtime → Change runtime type → Select GPU → Save
- Restart runtime and re-run all cells

### "Cannot find dataset"
- Check that `drowsiness-datasets.zip` is in **My Drive root** (not inside a subfolder)
- Make sure Google Drive is mounted (Step 3.3)
- Re-run the extraction cell if Colab was restarted
- Verify the path: `DATASET_PATH = "/content/datasets/drowsiness-datasets"`

### "Out of memory"
- The new disk-based approach should prevent this
- If it still happens: reduce `BATCH_SIZE` to 16 in Cell 4
- Restart runtime: Runtime → Restart runtime, then re-run all cells

### "Session disconnected"
- Colab disconnects after inactivity
- Keep the tab open and click occasionally
- Use Google Drive to save datasets (they persist)

---

## Summary

You now have a complete workflow to:
1. ✓ Upload datasets to Google Colab
2. ✓ Enable free GPU (T4)
3. ✓ Train CNN model (2-4 hours)
4. ✓ Evaluate performance (target: 85%+ accuracy)
5. ✓ Convert to TFLite for mobile
6. ✓ Download and integrate with your app

**Estimated time**: 3-5 hours (mostly training)
**Cost**: $0 (completely free!)

Good luck with your training! 🚀
