# Testing CNN Model with HTTP Server

## Quick Verification Steps

### Step 1: Verify Model File Exists
```bash
cd backend
dir models\cnn_drowsiness.h5
```
**Expected:** File should be ~302 MB ✓

---

### Step 2: Start HTTP Server
```bash
cd backend
python src/http_server.py
```

**Expected Output:**
```
Initializing detection service...
✓ CNN model loaded          <-- IMPORTANT: Should see this!
✓ Service initialized successfully
 * Running on http://127.0.0.1:5000
```

**If you see:**
- ✓ "CNN model loaded" → SUCCESS! CNN is active
- ⚠ "CNN model not found" → Problem with model path
- ⚠ "CNN model failed to load" → Check error message

---

### Step 3: Test Server (Optional - in another terminal)
```bash
cd backend
python scripts/test_http_server.py
```

This will send a test image to the server and verify it responds correctly.

---

### Step 4: Run Flutter App
```bash
cd mobile_app
flutter run
```

---

## What to Watch For

### In HTTP Server Terminal:
- Should see "✓ CNN model loaded" on startup
- When Flutter app connects, you'll see processing logs
- Look for "model_type: cnn" in responses

### In Flutter App:
- **Drowsiness Score:** Should update in real-time
- **Confidence:** CNN confidence score (0-100%)
- **Model Type:** Should show "cnn" (not "feature_based")
- **FPS:** Should be 10-15 FPS
- **Latency:** Should be 50-100ms

---

## Comparison: Rule-Based vs CNN

### Previous (Rule-Based):
- Used only EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio)
- Simple threshold-based detection
- Less accurate, more false positives

### Now (CNN):
- Uses trained deep learning model
- Analyzes entire face image
- More accurate drowsiness detection
- Better confidence scores

---

## Troubleshooting

### Problem: "CNN model not found"
**Solution:**
```bash
cd backend
python -c "import os; print(os.path.exists('models/cnn_drowsiness.h5'))"
```
Should print `True`

### Problem: Server loads but Flutter can't connect
**Check:**
1. Server running on port 5000
2. Flutter backend URL is `http://10.0.2.2:5000` (for Android emulator)
3. No firewall blocking port 5000

### Problem: Slow performance
**Expected:**
- First prediction: 1-2 seconds (model loading)
- Subsequent predictions: 50-100ms
- If slower, check CPU usage

---

## Success Indicators

✓ Server shows "✓ CNN model loaded"  
✓ Flutter app connects successfully  
✓ Real-time drowsiness detection working  
✓ Confidence scores displayed  
✓ Model type shows "cnn"  

---

## Commands Summary

**Terminal 1 (HTTP Server):**
```bash
cd backend
python src/http_server.py
```

**Terminal 2 (Flutter App):**
```bash
cd mobile_app
flutter run
```

**Terminal 3 (Optional Test):**
```bash
cd backend
python scripts/test_http_server.py
```

---

Good luck! Let me know if you see any errors.
