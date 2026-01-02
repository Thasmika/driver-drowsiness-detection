# CNN Model Training Guide

Complete step-by-step guide to train the CNN drowsiness detection model.

---

## Prerequisites

### 1. Install Required Packages

Make sure you have TensorFlow installed:

```powershell
cd backend
pip install tensorflow keras pillow scikit-learn tqdm
```

### 2. Verify Installation

```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

---

## Step 1: Prepare Dataset Structure

### Required Folder Structure:

```
backend/
├── datasets/
│   ├── dataset1/
│   │   ├── alert/      ← Put alert/awake images here
│   │   └── drowsy/     ← Put drowsy/sleepy images here
│   ├── dataset2/       (optional - for more data)
│   │   ├── alert/
│   │   └── drowsy/
│   └── dataset3/       (optional - for more data)
│       ├── alert/
│       └── drowsy/
└── models/             (will be created automatically)
```

### Create the Folders:

```powershell
cd backend
mkdir -p datasets/dataset1/alert
mkdir -p datasets/dataset1/drowsy
mkdir -p models
```

---

## Step 2: Download Dataset

### Option A: MRL Eye Dataset (Recommended - Small & Fast)

**Size:** ~100MB  
**Images:** ~5,000 images  
**Download:** https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection

1. Go to the Kaggle link
2. Click "Download" (requires free Kaggle account)
3. Extract the ZIP file
4. You'll get folders with images

**Organize the images:**
- Images with **open eyes** → Copy to `datasets/dataset1/alert/`
- Images with **closed eyes** → Copy to `datasets/dataset1/drowsy/`

### Option B: Create Your Own Mini Dataset (Fastest - For Testing)

**Size:** ~10MB  
**Images:** 100-200 images  
**Time:** 30 minutes

1. **Record Alert Images (50-100 images):**
   - Sit in front of your webcam
   - Keep eyes open, look at camera
   - Take photos every 2 seconds
   - Save to `datasets/dataset1/alert/`

2. **Record Drowsy Images (50-100 images):**
   - Close your eyes
   - Yawn
   - Look tired/sleepy
   - Take photos every 2 seconds
   - Save to `datasets/dataset1/drowsy/`

**Quick recording script:**

```python
# Save as: backend/scripts/record_dataset.py
import cv2
import os
from datetime import datetime

def record_images(output_folder, num_images=50, delay=2):
    """Record images from webcam"""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, 1 for external
    
    print(f"Recording {num_images} images to {output_folder}")
    print(f"Press SPACE to capture, ESC to quit")
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, f"Images: {count}/{num_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Record Dataset', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space bar
            filename = f"{output_folder}/img_{count:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            count += 1
        elif key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nRecorded {count} images!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python record_dataset.py <alert|drowsy>")
        sys.exit(1)
    
    category = sys.argv[1]
    if category not in ['alert', 'drowsy']:
        print("Category must be 'alert' or 'drowsy'")
        sys.exit(1)
    
    output_folder = f"datasets/dataset1/{category}"
    
    print(f"\n{'='*60}")
    print(f"Recording {category.upper()} images")
    print(f"{'='*60}")
    
    if category == 'alert':
        print("\nInstructions:")
        print("- Keep your eyes OPEN")
        print("- Look at the camera")
        print("- Press SPACE to capture each image")
        print("- Try different angles and expressions")
    else:
        print("\nInstructions:")
        print("- CLOSE your eyes")
        print("- Yawn, look tired")
        print("- Press SPACE to capture each image")
        print("- Try different drowsy expressions")
    
    input("\nPress ENTER to start recording...")
    
    record_images(output_folder, num_images=50)
```

**To use the recording script:**

```powershell
# Record alert images
cd backend
python scripts/record_dataset.py alert

# Record drowsy images
python scripts/record_dataset.py drowsy
```

---

## Step 3: Verify Dataset

Check if your dataset is properly organized:

```powershell
cd backend
python -c "from src.utils import DatasetLoader; loader = DatasetLoader('datasets'); print('Datasets found:', loader.discover_datasets()); [print(f'{ds}: {loader.get_dataset_statistics(ds)}') for ds in loader.discover_datasets()]"
```

**Expected output:**
```
Datasets found: ['dataset1']
dataset1: {'total_images': 100, 'alert_images': 50, 'drowsy_images': 50}
```

---

## Step 4: Train the CNN Model

### Basic Training (Default Settings):

```powershell
cd backend
python scripts/train_cnn.py
```

### Advanced Training (Custom Settings):

```powershell
# Train with custom parameters
python scripts/train_cnn.py --epochs 30 --batch-size 16 --input-size 128 128

# Options:
#   --epochs: Number of training epochs (default: 50)
#   --batch-size: Batch size (default: 32, use 16 if low memory)
#   --input-size: Image size (default: 224 224, use 128 128 for faster training)
#   --dataset-root: Dataset folder (default: datasets)
#   --model-path: Where to save model (default: models/cnn_drowsiness.h5)
```

### What Happens During Training:

1. **Loading datasets** - Reads all images from datasets folder
2. **Splitting data** - 70% train, 15% validation, 15% test
3. **Creating generators** - Prepares data with augmentation
4. **Building CNN** - Creates the neural network
5. **Training** - Learns patterns (this takes time!)
6. **Evaluation** - Tests accuracy on test set
7. **Saving models** - Saves both Keras (.h5) and TFLite (.tflite) versions

### Training Output Example:

```
================================================================================
CNN Drowsiness Classifier Training
================================================================================

1. Loading datasets...
Found 3 datasets: ['dataset1', 'dataset2', 'dataset3']
Loaded 2500 images total

2. Splitting data into train/val/test sets...
Dataset splits:
  Train:
    Total: 1750
    Alert: 875
    Drowsy: 875
  Val:
    Total: 375
    Alert: 188
    Drowsy: 187
  Test:
    Total: 375
    Alert: 188
    Drowsy: 187

3. Creating data generators with augmentation...
  Training batches per epoch: 55
  Validation batches per epoch: 12

4. Building and training CNN model...
Epoch 1/50
55/55 [==============================] - 45s 820ms/step - loss: 0.6234 - accuracy: 0.6543 - val_loss: 0.5123 - val_accuracy: 0.7467
Epoch 2/50
55/55 [==============================] - 42s 763ms/step - loss: 0.4567 - accuracy: 0.7823 - val_loss: 0.3891 - val_accuracy: 0.8267
...
Epoch 50/50
55/55 [==============================] - 41s 745ms/step - loss: 0.1234 - accuracy: 0.9543 - val_loss: 0.1567 - val_accuracy: 0.9333

5. Evaluating model on test set...
Test Set Performance:
  accuracy: 0.9280
  precision: 0.9156
  recall: 0.9412
  f1_score: 0.9282

✓ Model meets accuracy requirement (>= 85%): 92.80%

6. Saving Keras model to models/cnn_drowsiness.h5...
Model saved successfully!

7. Converting to TensorFlow Lite format...
TFLite model saved to models/cnn_drowsiness.tflite

================================================================================
Training Complete!
================================================================================

Saved models:
  Keras model: models/cnn_drowsiness.h5
  TFLite model: models/cnn_drowsiness.tflite
```

---

## Step 5: Verify Trained Model

Test if the model works:

```powershell
cd backend
python scripts/validate_models.py
```

This will:
- Load the trained model
- Test it on sample images
- Show accuracy metrics
- Verify it meets requirements (>85% accuracy)

---

## Step 6: Use Trained Model in Your App

The HTTP server will automatically use the trained model!

1. **Restart the HTTP server:**
   ```powershell
   cd backend
   python src/http_server.py
   ```

2. **Check the startup message:**
   ```
   ✓ CNN model loaded
   ```

3. **Run your Flutter app:**
   ```powershell
   cd mobile_app
   flutter run
   ```

4. **Test drowsiness detection** - It should now be much more accurate!

---

## Troubleshooting

### Problem: "No datasets found"

**Solution:**
```powershell
# Check if folders exist
cd backend
dir datasets\dataset1\alert
dir datasets\dataset1\drowsy

# Should show image files (.jpg, .png)
```

### Problem: "Out of memory" during training

**Solution 1 - Reduce batch size:**
```powershell
python scripts/train_cnn.py --batch-size 8
```

**Solution 2 - Reduce image size:**
```powershell
python scripts/train_cnn.py --input-size 128 128 --batch-size 16
```

**Solution 3 - Use fewer images:**
- Start with just 100-200 images per class
- Train successfully
- Add more images later

### Problem: "TensorFlow not found"

**Solution:**
```powershell
pip install tensorflow
# Or for CPU-only (smaller):
pip install tensorflow-cpu
```

### Problem: Training is very slow

**Expected times:**
- **With GPU:** 30-60 minutes for 50 epochs
- **With CPU:** 1-3 hours for 50 epochs

**Speed up training:**
```powershell
# Reduce epochs
python scripts/train_cnn.py --epochs 20

# Reduce image size
python scripts/train_cnn.py --input-size 128 128 --epochs 20
```

### Problem: Low accuracy (<85%)

**Solutions:**
1. **Add more images** - Need at least 500 per class
2. **Balance dataset** - Equal alert and drowsy images
3. **Check image quality** - Clear faces, good lighting
4. **Train longer** - Increase epochs to 100
5. **Use data augmentation** - Already enabled by default

---

## CNN Model Architecture

The CNN model structure:

```
Input (224x224x3 RGB image)
    ↓
Conv2D (32 filters, 3x3) + ReLU + MaxPool
    ↓
Conv2D (64 filters, 3x3) + ReLU + MaxPool
    ↓
Conv2D (128 filters, 3x3) + ReLU + MaxPool
    ↓
Flatten
    ↓
Dense (128 units) + ReLU + Dropout(0.5)
    ↓
Dense (2 units) + Softmax
    ↓
Output (Alert or Drowsy)
```

**Key Features:**
- Uses transfer learning (pre-trained weights)
- Data augmentation (rotation, flip, zoom, brightness)
- Dropout for regularization
- Binary classification (alert vs drowsy)

---

## Dataset Recommendations

### Minimum Dataset Size:
- **Alert images:** 100 minimum, 500+ recommended
- **Drowsy images:** 100 minimum, 500+ recommended
- **Total:** 200 minimum, 1000+ recommended

### Image Quality Guidelines:
- ✅ Clear face visible
- ✅ Good lighting
- ✅ Various angles (front, slight left/right)
- ✅ Different expressions
- ✅ Different people (for generalization)
- ❌ Avoid blurry images
- ❌ Avoid extreme angles
- ❌ Avoid very dark images

### Recommended Public Datasets:

1. **MRL Eye Dataset**
   - Link: https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection
   - Size: ~100MB
   - Images: ~5,000
   - Quality: High

2. **Drowsiness Detection Dataset**
   - Link: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
   - Size: ~50MB
   - Images: ~2,500
   - Quality: Good

3. **YawDD Dataset**
   - Link: http://www.site.uottawa.ca/~shervin/yawning/
   - Size: ~200MB
   - Images: ~10,000
   - Quality: Excellent (video frames)

---

## Quick Start Summary

**Fastest way to train (30 minutes):**

1. Create mini dataset (100 images):
   ```powershell
   cd backend
   python scripts/record_dataset.py alert    # Record 50 alert images
   python scripts/record_dataset.py drowsy   # Record 50 drowsy images
   ```

2. Train with fast settings:
   ```powershell
   python scripts/train_cnn.py --epochs 20 --input-size 128 128 --batch-size 16
   ```

3. Test the model:
   ```powershell
   python scripts/validate_models.py
   ```

4. Use in your app:
   ```powershell
   python src/http_server.py
   ```

Done! 🎉

---

## Next Steps After Training

1. ✅ **Validate model accuracy** - Run validation script
2. ✅ **Test with real camera** - Use webcam test script
3. ✅ **Integrate with Flutter** - HTTP server auto-loads model
4. ✅ **Test end-to-end** - Run full system
5. ✅ **Collect more data** - Improve accuracy over time
6. ✅ **Fine-tune parameters** - Adjust thresholds if needed

---

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify dataset structure
3. Check Python package versions
4. Try with smaller dataset first
5. Reduce batch size if memory issues

Good luck with training! 🚀
