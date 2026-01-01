# Flutter-Backend Integration Quick Start

## What You Have Now

✅ **Python Backend** - Complete ML pipeline for drowsiness detection  
✅ **Flutter App** - Complete mobile UI with state management  
✅ **Mobile Service** - Bridge between Flutter and Python (`backend/src/mobile_service.py`)  
✅ **Integration Guide** - Detailed documentation (`FLUTTER_BACKEND_INTEGRATION.md`)

## How They Connect

```
Flutter App (Dart)
    ↓ Platform Channel
Native Code (Kotlin/Swift)
    ↓ Process/Socket
Python Backend (mobile_service.py)
    ↓ ML Pipeline
Drowsiness Detection Result
    ↑ JSON Response
Flutter UI Updates
```

## Quick Test (Without Mobile Device)

### 1. Test the Mobile Service

```bash
cd backend

# Activate virtual environment
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Run the test
python scripts/test_mobile_service.py
```

This will test:
- Service initialization
- Ping/pong communication
- Frame processing
- Shutdown

### 2. Manual Test (Interactive)

```bash
cd backend
python src/mobile_service.py
```

Then type commands (one per line):
```json
{"command": "ping"}
{"command": "get_status"}
{"command": "shutdown"}
```

## Integration Options

### Option 1: Platform Channels (Current Setup)
**Best for**: Full integration with native features  
**Complexity**: Medium  
**Performance**: Good  

**What you need to do:**
1. Add native code (Kotlin for Android, Swift for iOS)
2. Package Python with app
3. Handle process management

**Files to create:**
- `mobile_app/android/app/src/main/kotlin/.../MainActivity.kt`
- `mobile_app/ios/Runner/AppDelegate.swift`

### Option 2: TensorFlow Lite (Recommended for Production)
**Best for**: Production deployment  
**Complexity**: Medium  
**Performance**: Excellent  

**What you need to do:**
1. Convert models to TFLite format
2. Use `tflite_flutter` package
3. Implement preprocessing in Dart

**Advantages:**
- No Python runtime needed
- Faster inference
- Smaller app size
- Better battery life

### Option 3: HTTP Server (Easiest for Testing)
**Best for**: Quick testing and development  
**Complexity**: Low  
**Performance**: Depends on network  

**What you need to do:**
1. Run Python backend as Flask/FastAPI server
2. Make HTTP requests from Flutter
3. Can run on device or computer

## Recommended Path Forward

### Phase 1: Local Testing (Now)
```bash
# Test Python backend independently
cd backend
python scripts/test_mobile_service.py
```

### Phase 2: HTTP Server Testing (Next)
Create a simple Flask server for testing:

```python
# backend/src/http_server.py
from flask import Flask, request, jsonify
from mobile_service import MobileDetectionService
import base64

app = Flask(__name__)
service = MobileDetectionService()

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    image_data = data.get('image_data')
    result = service.process_frame(image_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Then update Flutter to use HTTP:
```dart
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> processFrame(Uint8List imageData) async {
  final base64Image = base64Encode(imageData);
  
  final response = await http.post(
    Uri.parse('http://YOUR_IP:5000/process'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'image_data': base64Image}),
  );
  
  return jsonDecode(response.body);
}
```

### Phase 3: TensorFlow Lite Conversion (Production)
Convert your models:

```python
# Convert CNN model
import tensorflow as tf

model = tf.keras.models.load_model('models/cnn_drowsiness_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/drowsiness_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

Use in Flutter:
```dart
import 'package:tflite_flutter/tflite_flutter.dart';

final interpreter = await Interpreter.fromAsset('models/drowsiness_model.tflite');
```

### Phase 4: Full Platform Channel Integration
Follow the detailed guide in `FLUTTER_BACKEND_INTEGRATION.md`

## Testing Checklist

- [ ] Python backend runs independently
- [ ] Mobile service responds to commands
- [ ] Can process test images
- [ ] Flutter app UI works
- [ ] Camera captures frames
- [ ] HTTP server integration (optional)
- [ ] Platform channels work (Android)
- [ ] Platform channels work (iOS)
- [ ] TFLite models converted
- [ ] End-to-end testing on device

## Common Issues & Solutions

### Issue: Python dependencies not found
**Solution**: Ensure virtual environment is activated
```bash
cd backend
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Models not loading
**Solution**: Train models first or use feature-based classifier
```bash
cd backend
python scripts/train_cnn.py
python scripts/train_traditional_ml.py
```

### Issue: Camera not working in Flutter
**Solution**: Check permissions in AndroidManifest.xml / Info.plist
```xml
<!-- Android -->
<uses-permission android:name="android.permission.CAMERA"/>

<!-- iOS -->
<key>NSCameraUsageDescription</key>
<string>Camera access for drowsiness detection</string>
```

### Issue: Slow performance
**Solution**: 
1. Reduce image size before processing
2. Process every 3-5 frames, not every frame
3. Use TFLite instead of full Python backend

## Next Steps

1. **Test the mobile service** (5 minutes)
   ```bash
   cd backend
   python scripts/test_mobile_service.py
   ```

2. **Choose integration approach** (Decision)
   - HTTP Server: Quick testing
   - Platform Channels: Full features
   - TFLite: Production ready

3. **Implement chosen approach** (1-2 days)
   - Follow relevant section in FLUTTER_BACKEND_INTEGRATION.md

4. **Test on device** (1 day)
   - Deploy to Android/iOS
   - Test real-time detection
   - Optimize performance

5. **Polish and deploy** (2-3 days)
   - Add error handling
   - Optimize battery usage
   - Prepare for app stores

## Resources

- **Full Integration Guide**: `FLUTTER_BACKEND_INTEGRATION.md`
- **Mobile Service**: `backend/src/mobile_service.py`
- **Test Script**: `backend/scripts/test_mobile_service.py`
- **Flutter Backend Service**: `mobile_app/lib/services/backend_service.dart`

## Questions?

Common questions answered in `FLUTTER_BACKEND_INTEGRATION.md`:
- How to package Python for mobile?
- How to optimize performance?
- How to handle errors?
- How to test on physical devices?

## Summary

You have three main options:

1. **HTTP Server** (Easiest) - Run Python on computer, Flutter makes HTTP calls
2. **Platform Channels** (Full-featured) - Embed Python in mobile app
3. **TensorFlow Lite** (Best) - Convert models, run natively in Flutter

**Recommendation**: Start with HTTP server for testing, then move to TFLite for production.

The mobile service (`backend/src/mobile_service.py`) is ready to use with any of these approaches!
