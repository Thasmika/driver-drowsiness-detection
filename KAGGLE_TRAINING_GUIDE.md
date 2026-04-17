# Complete Guide: Training Drowsiness Detection Model on Kaggle

This guide will walk you through training your CNN drowsiness detection model on Kaggle's free GPU platform.

## Why Kaggle?

- **Free GPU**: Tesla P100 (16GB) or Dual T4 (2x 15GB)
- **30 hours/week**: Generous GPU quota
- **20GB storage**: Persistent storage for datasets
- **12-hour sessions**: Long enough for training
- **Background execution**: Can close browser while training

---

## Step 1: Create Kaggle Account

1. Go to [https://www.kaggle.com](https://www.kaggle.com)
2. Click "Register" and create an account
3. Verify your email address
4. Complete your profile (optional but recommended)

---

## Step 2: Prepare Your Datasets

You have two options for uploading datasets to Kaggle:

### Option A: Upload as Kaggle Dataset (Recommended)

1. **Organize your datasets locally** in this structure:
   ```
   drowsiness-datasets/
   ├── DDD/
   │   ├── alert/          (or "Non Drowsy/")
   │   │   ├── image1.jpg
   │   │   └── ...
   │   └── drowsy/         (or "Drowsy/")
   │       ├── image1.jpg
   │       └── ...
   ├── NTHUDDD/
   │   ├── alert/
   │   └── drowsy/
   └── yawning/
       ├── alert/          (or "no yawn/")
       └── drowsy/         (or "yawn/")
   ```

2. **Create a ZIP file** of your datasets folder

3. **Upload to Kaggle**:
   - Go to [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload your ZIP file
   - Title: "Driver Drowsiness Detection Datasets"
   - Make it "Private" (recommended) or "Public"
   - Click "Create"

### Option B: Upload Directly in Notebook

You can upload datasets directly when creating the notebook (smaller datasets only, <500MB).

---

## Step 3: Create Kaggle Notebook

1. Go to [https://www.kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Choose "Notebook" (not Script)
4. Name it: "Drowsiness Detection CNN Training"

---

## Step 4: Configure GPU

1. In your notebook, click "Settings" (right sidebar)
2. Under "Accelerator", select:
   - **GPU P100** (recommended) or
   - **GPU T4 x2** (dual GPUs)
3. Enable "Internet" (to install packages)
4. Enable "Persistence" (to save outputs)

---

## Step 5: Add Your Dataset to Notebook

1. In the right sidebar, click "Add Data"
2. Search for your uploaded dataset
3. Click "Add" to attach it to your notebook
4. Note the path: `/kaggle/input/your-dataset-name/`

---

## Step 6: Create Training Notebook

Copy and paste this code into your Kaggle notebook:

### Cell 1: Install Dependencies

```python
# Install required packages
!pip install -q opencv-python-headless
!pip install -q pillow
!pip install -q scikit-learn
!pip install -q tensorflow

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
```

### Cell 3: Dataset Loading Functions

```python
def load_images_from_folder(folder_path, label, image_size=(128, 128), max_images=None):
    """Load images from a folder with a specific label."""
    images = []
    labels = []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    # Limit number of images if specified
    if max_images:
        image_files = list(image_files)[:max_images]
    
    for img_path in image_files:
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    return images, labels


def load_dataset(dataset_path, image_size=(128, 128), max_per_class=5000):
    """Load a single dataset with alert and drowsy classes."""
    images = []
    labels = []
    
    # Try different directory naming conventions
    alert_dirs = ['alert', 'Non Drowsy', 'notdrowsy', 'no yawn']
    drowsy_dirs = ['drowsy', 'Drowsy', 'yawn']
    
    dataset_path = Path(dataset_path)
    
    # Load alert images (label = 0)
    for alert_dir in alert_dirs:
        alert_path = dataset_path / alert_dir
        if alert_path.exists():
            print(f"  Loading alert images from: {alert_path}")
            imgs, lbls = load_images_from_folder(alert_path, 0, image_size, max_per_class)
            images.extend(imgs)
            labels.extend(lbls)
            print(f"    Loaded {len(imgs)} alert images")
            break
    
    # Load drowsy images (label = 1)
    for drowsy_dir in drowsy_dirs:
        drowsy_path = dataset_path / drowsy_dir
        if drowsy_path.exists():
            print(f"  Loading drowsy images from: {drowsy_path}")
            imgs, lbls = load_images_from_folder(drowsy_path, 1, image_size, max_per_class)
            images.extend(imgs)
            labels.extend(lbls)
            print(f"    Loaded {len(imgs)} drowsy images")
            break
    
    return images, labels


def load_all_datasets(root_path, image_size=(128, 128), max_per_class=5000):
    """Load all datasets from root directory."""
    all_images = []
    all_labels = []
    
    root_path = Path(root_path)
    
    # Find all dataset directories
    dataset_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(dataset_dirs)} datasets")
    
    for dataset_dir in dataset_dirs:
        print(f"\nLoading dataset: {dataset_dir.name}")
        images, labels = load_dataset(dataset_dir, image_size, max_per_class)
        all_images.extend(images)
        all_labels.extend(labels)
    
    # Convert to numpy arrays
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    print(f"\n{'='*60}")
    print(f"Total images loaded: {len(X)}")
    print(f"  Alert (0): {np.sum(y == 0)}")
    print(f"  Drowsy (1): {np.sum(y == 1)}")
    print(f"  Image shape: {X.shape[1:]}")
    print(f"{'='*60}\n")
    
    return X, y

print("✓ Dataset loading functions defined")
```

### Cell 4: Load Your Datasets

```python
# IMPORTANT: Update this path to match your dataset location
DATASET_PATH = "/kaggle/input/your-dataset-name/drowsiness-datasets"

# Image size for training
IMAGE_SIZE = (128, 128)

# Maximum images per class (to avoid memory issues)
MAX_PER_CLASS = 5000

# Load datasets
print("Loading datasets...")
X, y = load_all_datasets(DATASET_PATH, IMAGE_SIZE, MAX_PER_CLASS)

print(f"Dataset loaded successfully!")
print(f"Shape: {X.shape}")
print(f"Labels: {y.shape}")
```

### Cell 5: Split Data

```python
# Split into train, validation, and test sets
print("Splitting data...")

# First split: 85% train+val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Second split: 82% train, 18% val (of the 85%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
)

print(f"\nDataset splits:")
print(f"  Training:   {len(X_train)} images ({np.sum(y_train==0)} alert, {np.sum(y_train==1)} drowsy)")
print(f"  Validation: {len(X_val)} images ({np.sum(y_val==0)} alert, {np.sum(y_val==1)} drowsy)")
print(f"  Test:       {len(X_test)} images ({np.sum(y_test==0)} alert, {np.sum(y_test==1)} drowsy)")
```

### Cell 6: Data Augmentation

```python
# Create data generators with augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator()  # No augmentation for validation

# Create generators
batch_size = 32

train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size,
    shuffle=True
)

val_generator = val_datagen.flow(
    X_val, y_val,
    batch_size=batch_size,
    shuffle=False
)

print(f"✓ Data generators created")
print(f"  Batch size: {batch_size}")
print(f"  Training batches per epoch: {len(train_generator)}")
print(f"  Validation batches per epoch: {len(val_generator)}")
```

### Cell 7: Build CNN Model

```python
def build_cnn_model(input_shape=(128, 128, 3)):
    """Build CNN model for drowsiness detection."""
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
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

# Build model
model = build_cnn_model(input_shape=X_train.shape[1:])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Print model summary
model.summary()
```

### Cell 8: Train Model

```python
# Training configuration
EPOCHS = 50

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
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

# Train model
print("Starting training...")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {batch_size}")
print("="*60)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training complete!")
```

### Cell 9: Evaluate Model

```python
# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*60}")
print("TEST SET PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"Loss:      {test_loss:.4f}")

# Check if meets requirement
if test_acc >= 0.85:
    print(f"\n✓ Model MEETS accuracy requirement (>= 85%)")
else:
    print(f"\n✗ Model does NOT meet accuracy requirement (< 85%)")

# Predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print(f"\n{'='*60}")
print("CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(y_test, y_pred, target_names=['Alert', 'Drowsy']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)
```

### Cell 10: Plot Training History

```python
# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Validation')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Validation')
axes[1, 0].set_title('Model Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Validation')
axes[1, 1].set_title('Model Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Alert', 'Drowsy'],
            yticklabels=['Alert', 'Drowsy'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### Cell 11: Convert to TensorFlow Lite

```python
# Convert to TFLite for mobile deployment
print("Converting to TensorFlow Lite...")

# Standard conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('drowsiness_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✓ TFLite model saved: drowsiness_model.tflite")
print(f"  Size: {len(tflite_model) / 1024:.2f} KB")

# Quantized conversion (INT8) for smaller size
print("\nConverting to quantized TFLite (INT8)...")

def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1]]

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

### Cell 12: Save Models

```python
# Save Keras model
model.save('drowsiness_cnn_model.h5')
print("✓ Keras model saved: drowsiness_cnn_model.h5")

# Save model weights only
model.save_weights('model_weights.h5')
print("✓ Model weights saved: model_weights.h5")

print("\n" + "="*60)
print("ALL MODELS SAVED SUCCESSFULLY!")
print("="*60)
print("\nSaved files:")
print("  1. best_model.h5 (best checkpoint during training)")
print("  2. drowsiness_cnn_model.h5 (final Keras model)")
print("  3. model_weights.h5 (model weights only)")
print("  4. drowsiness_model.tflite (TFLite for mobile)")
print("  5. drowsiness_model_quantized.tflite (Quantized TFLite)")
```

---

## Step 7: Download Trained Models

After training completes:

1. Click "Output" in the right sidebar
2. You'll see all saved model files
3. Click the download icon next to each file to download:
   - `drowsiness_model.tflite` (for your Flutter app)
   - `drowsiness_model_quantized.tflite` (smaller, optimized)
   - `drowsiness_cnn_model.h5` (full Keras model)

---

## Step 8: Monitor Training

- **Check GPU usage**: Click "Session" → "GPU" to see utilization
- **Monitor time**: You have 12 hours per session
- **Save frequently**: Models are auto-saved in checkpoints
- **Background execution**: You can close the browser tab

---

## Tips for Success

### Memory Management
- Start with `MAX_PER_CLASS = 5000` images
- If you get memory errors, reduce to 3000 or 2000
- Use smaller `IMAGE_SIZE = (96, 96)` if needed

### Training Time
- Expect 2-4 hours for 50 epochs with 10,000 images
- P100 GPU is faster than T4 x2 for this task
- Use early stopping to avoid overtraining

### Improving Accuracy
- Increase epochs to 75-100 if accuracy is low
- Adjust learning rate (try 0.0005 or 0.0001)
- Add more data augmentation
- Try different batch sizes (16 or 64)

### Troubleshooting

**"Dataset not found"**
- Check the path in Cell 4
- Make sure dataset is attached to notebook
- Verify folder structure matches expected format

**"Out of memory"**
- Reduce `MAX_PER_CLASS` to 3000
- Reduce `batch_size` to 16
- Use smaller `IMAGE_SIZE = (96, 96)`

**"GPU quota exceeded"**
- You've used 30 hours this week
- Wait until next week or use CPU (slower)
- Consider Google Colab as backup

---

## Next Steps

After downloading your trained model:

1. Copy `drowsiness_model.tflite` to your project:
   ```
   backend/models/cnn_drowsiness.tflite
   ```

2. Test the model locally:
   ```bash
   cd backend
   python scripts/test_trained_model.py
   ```

3. Integrate with your Flutter app (already configured!)

---

## Additional Resources

- **Kaggle Documentation**: [https://www.kaggle.com/docs](https://www.kaggle.com/docs)
- **TensorFlow Lite Guide**: [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)
- **Your Project README**: See `backend/README.md` for local testing

---

## Summary

You now have a complete workflow to:
1. ✓ Upload datasets to Kaggle
2. ✓ Train CNN model on free GPU
3. ✓ Evaluate performance (target: 85%+ accuracy)
4. ✓ Convert to TFLite for mobile
5. ✓ Download and integrate with your app

**Estimated total time**: 3-5 hours (mostly training)
**Cost**: $0 (completely free!)

Good luck with your training! 🚀
