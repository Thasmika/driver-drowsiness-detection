# CNN Training Quick Start

**3 Simple Steps to Train Your Drowsiness Detection Model**

---

## Step 1: Prepare Dataset (Choose One)

### Option A: Record Your Own (Fastest - 30 min)

```powershell
cd backend

# Record 50 alert images (eyes open)
python scripts/record_dataset.py alert --num-images 50 --camera 0

# Record 50 drowsy images (eyes closed, yawning)
python scripts/record_dataset.py drowsy --num-images 50 --camera 0
```

**Tips:**
- Use `--camera 1` for external USB camera
- Increase `--num-images` for more data (recommended: 100-200 per category)
- Press SPACE to capture each image
- Press ESC to quit

### Option B: Download Public Dataset (Better Accuracy)

1. Go to: https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection
2. Download the dataset (~100MB)
3. Extract and organize:
   - Open eyes images → `backend/datasets/dataset1/alert/`
   - Closed eyes images → `backend/datasets/dataset1/drowsy/`

---

## Step 2: Train the Model

### Quick Training (20 epochs, ~30 min):

```powershell
cd backend
python scripts/train_cnn.py --epochs 20 --input-size 128 128 --batch-size 16
```

### Full Training (50 epochs, ~1-2 hours):

```powershell
python scripts/train_cnn.py
```

**What to expect:**
- Training will show progress for each epoch
- Accuracy should improve over time
- Target: >85% accuracy
- Models saved to `backend/models/`

---

## Step 3: Test the Model

### Validate the trained model:

```powershell
cd backend
python scripts/validate_models.py
```

### Use in your app:

```powershell
# Start HTTP server (will auto-load trained model)
python src/http_server.py

# In another terminal, run Flutter app
cd mobile_app
flutter run
```

**You should see:**
```
✓ CNN model loaded
```

Now your drowsiness detection will be much more accurate! 🎉

---

## Troubleshooting

### "Out of memory"
```powershell
python scripts/train_cnn.py --batch-size 8 --input-size 128 128
```

### "No datasets found"
```powershell
# Check if images exist
dir backend\datasets\dataset1\alert
dir backend\datasets\dataset1\drowsy
```

### "Training too slow"
```powershell
# Reduce epochs and image size
python scripts/train_cnn.py --epochs 10 --input-size 96 96
```

---

## Full Documentation

For detailed instructions, see: `CNN_TRAINING_GUIDE.md`

---

## Minimum Requirements

- **Images:** 100 total (50 alert + 50 drowsy)
- **Time:** 30 minutes to 2 hours
- **Disk Space:** 100MB-500MB
- **RAM:** 4GB minimum, 8GB recommended
- **GPU:** Optional (speeds up training 5-10x)

---

## Expected Results

After training with 100-200 images per category:
- **Accuracy:** 85-95%
- **Precision:** 80-90%
- **Recall:** 80-90%

With more data (500+ per category):
- **Accuracy:** 90-98%
- **Precision:** 90-95%
- **Recall:** 90-95%

---

**Ready to train? Start with Step 1!** 🚀
