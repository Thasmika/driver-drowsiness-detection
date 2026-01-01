# How to Connect Flutter to Your Drowsiness Detection System

## Quick Answer

You have **3 options** to connect Flutter to the Python backend:

### 1. HTTP Server (Easiest - Start Here!) ⭐
Run Python as a web server, Flutter makes HTTP requests.

**Time to setup**: 10 minutes  
**Best for**: Testing and development

### 2. Platform Channels (Full Integration)
Embed Python in the mobile app using native code.

**Time to setup**: 2-3 days  
**Best for**: Full-featured mobile app

### 3. TensorFlow Lite (Production Ready)
Convert models to TFLite, run directly in Flutter.

**Time to setup**: 1-2 days  
**Best for**: Production deployment

---

## Option 1: HTTP Server (Recommended to Start)

### Step 1: Install Flask

```bash
cd backend
.\venv\Scripts\activate
pip install flask flask-cors
```

### Step 2: Start the HTTP Server

```bash
python src/http_server.py
```

You should see:
```
Drowsiness Detection HTTP Server
Server starting...
Endpoints:
  - http://localhost:5000/
  - http://localhost:5000/health
  - http://localhost:5000/status
  - http://localhost:5000/process (POST)
```

### Step 3: Test the Server

Open a new terminal:
```bash
cd backend
.\venv\Scripts\activate
python scripts/test_http_server.py
```

### Step 4: Update Flutter to Use HTTP

Add dependency to `mobile_app/pubspec.yaml`:
```yaml
dependencies:
  http: ^1.1.0
```

Update `mobile_app/lib/services/backend_service.dart`:

```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class BackendService {
  // Change this to your computer's IP address
  // Find it with: ipconfig (Windows) or ifconfig (Mac/Linux)
  static const String baseUrl = 'http://192.168.1.100:5000';
  
  bool _isInitialized = false;
  
  bool get isInitialized => _isInitialized;
  
  Future<bool> initialize() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      _isInitialized = response.statusCode == 200;
      return _isInitialized;
    } catch (e) {
      print('Error initializing backend: $e');
      return false;
    }
  }
  
  Future<Map<String, dynamic>> processFrame(Uint8List imageData) async {
    if (!_isInitialized) {
      throw Exception('Backend not initialized');
    }
    
    try {
      // Convert image to base64
      final base64Image = base64Encode(imageData);
      
      // Send request
      final response = await http.post(
        Uri.parse('$baseUrl/process'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image_data': base64Image}),
      );
      
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        return {
          'success': false,
          'error': 'Server returned ${response.statusCode}',
        };
      }
    } catch (e) {
      print('Error processing frame: $e');
      return {
        'success': false,
        'error': e.toString(),
      };
    }
  }
  
  Future<void> shutdown() async {
    _isInitialized = false;
  }
}
```

### Step 5: Find Your Computer's IP Address

**Windows:**
```bash
ipconfig
```
Look for "IPv4 Address" (e.g., 192.168.1.100)

**Mac/Linux:**
```bash
ifconfig
```
Look for "inet" address

**Update the baseUrl in backend_service.dart with your IP!**

### Step 6: Run Flutter App

```bash
cd mobile_app
flutter run
```

**Important**: Your phone and computer must be on the same WiFi network!

---

## Testing the Connection

### 1. Start Python Server
```bash
cd backend
.\venv\Scripts\activate
python src/http_server.py
```

### 2. Test from Browser
Open: http://localhost:5000/

You should see:
```json
{
  "name": "Drowsiness Detection API",
  "version": "1.0.0",
  "endpoints": { ... }
}
```

### 3. Test from Flutter
Run the app and tap "Start Monitoring"

---

## Troubleshooting

### Problem: "Connection refused"
**Solution**: 
- Check if Python server is running
- Verify IP address in Flutter code
- Make sure phone and computer are on same WiFi
- Try http://localhost:5000 if testing on emulator

### Problem: "No face detected"
**Solution**:
- This is normal! The camera needs good lighting
- Make sure face is visible and well-lit
- Try with a test image first

### Problem: Server is slow
**Solution**:
- Reduce image size before sending
- Process every 3-5 frames, not every frame
- Consider using TensorFlow Lite instead

### Problem: "Module not found"
**Solution**:
```bash
cd backend
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## Performance Tips

### 1. Reduce Image Size
Before sending to server:
```dart
// Resize image to 640x480
final resized = await FlutterImageCompress.compressWithList(
  imageData,
  minWidth: 640,
  minHeight: 480,
  quality: 85,
);
```

### 2. Process Fewer Frames
Don't process every frame:
```dart
int frameCount = 0;

void onCameraFrame(CameraImage image) {
  frameCount++;
  
  // Process every 5th frame
  if (frameCount % 5 == 0) {
    processFrame(image);
  }
}
```

### 3. Use Async Processing
Don't block the UI:
```dart
Future<void> processFrame(Uint8List imageData) async {
  // Process in background
  final result = await backendService.processFrame(imageData);
  
  // Update UI
  setState(() {
    drowsinessScore = result['drowsiness_score'];
  });
}
```

---

## Next Steps

### For Development
✅ Use HTTP server (you're done!)

### For Production
1. **Convert to TensorFlow Lite**
   - Smaller app size
   - Faster inference
   - No network needed
   - Better battery life

2. **Or use Platform Channels**
   - Full Python integration
   - All features available
   - More complex setup

See `FLUTTER_BACKEND_INTEGRATION.md` for detailed guides.

---

## File Reference

- **HTTP Server**: `backend/src/http_server.py`
- **Mobile Service**: `backend/src/mobile_service.py`
- **Test Script**: `backend/scripts/test_http_server.py`
- **Flutter Service**: `mobile_app/lib/services/backend_service.dart`
- **Full Guide**: `FLUTTER_BACKEND_INTEGRATION.md`
- **Quick Start**: `INTEGRATION_QUICKSTART.md`

---

## Summary

**To connect Flutter to Python:**

1. Install Flask: `pip install flask flask-cors`
2. Start server: `python backend/src/http_server.py`
3. Update Flutter with your IP address
4. Run Flutter app: `flutter run`

**That's it!** Your Flutter app can now communicate with the Python backend.

For production, consider converting to TensorFlow Lite for better performance and offline capability.

---

## Questions?

- **How do I deploy this?** See `FLUTTER_BACKEND_INTEGRATION.md` → TensorFlow Lite section
- **How do I improve performance?** Process fewer frames, reduce image size
- **Can I run Python on the phone?** Yes, but TensorFlow Lite is better
- **Do I need internet?** Only for HTTP server approach. TFLite works offline.

**Need help?** Check the detailed guides:
- `FLUTTER_BACKEND_INTEGRATION.md` - Complete integration guide
- `INTEGRATION_QUICKSTART.md` - Quick reference
- `HOW_TO_CONNECT_FLUTTER.md` - This file
