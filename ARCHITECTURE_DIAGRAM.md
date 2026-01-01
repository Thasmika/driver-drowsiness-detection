# System Architecture Diagram

## Current System (HTTP Server Approach)

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR PHONE                              │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Flutter Mobile App                           │ │
│  │                                                           │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │ │
│  │  │   Camera    │  │  UI Screens  │  │   Providers     │ │ │
│  │  │   Service   │  │  - Home      │  │  - Drowsiness   │ │ │
│  │  │             │  │  - Monitoring│  │  - Settings     │ │ │
│  │  └──────┬──────┘  └──────────────┘  └─────────────────┘ │ │
│  │         │                                                 │ │
│  │         │ Captures Frame                                  │ │
│  │         ▼                                                 │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │         Backend Service (Dart)                      │ │ │
│  │  │  - Converts image to Base64                         │ │ │
│  │  │  - Makes HTTP POST request                          │ │ │
│  │  │  - Receives JSON response                           │ │ │
│  │  └──────────────────┬──────────────────────────────────┘ │ │
│  └────────────────────│────────────────────────────────────┘ │
└────────────────────────│──────────────────────────────────────┘
                         │
                         │ HTTP Request
                         │ POST /process
                         │ { "image_data": "base64..." }
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR COMPUTER                                │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │         Python HTTP Server (Flask)                        │ │
│  │         Port: 5000                                        │ │
│  │                                                           │ │
│  │  Endpoints:                                               │ │
│  │  - GET  /health  → Check if alive                        │ │
│  │  - GET  /status  → Get service info                      │ │
│  │  - POST /process → Process frame                         │ │
│  │                                                           │ │
│  └──────────────────────┬────────────────────────────────────┘ │
│                         │                                       │
│                         │ Decodes Base64                        │
│                         │ Converts to numpy array               │
│                         ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              ML Detection Pipeline                        │ │
│  │                                                           │ │
│  │  1. Face Detection (MediaPipe)                           │ │
│  │     ↓                                                     │ │
│  │  2. Landmark Extraction (68 points)                      │ │
│  │     ↓                                                     │ │
│  │  3. Feature Calculation (EAR, MAR)                       │ │
│  │     ↓                                                     │ │
│  │  4. ML Classification (CNN or Feature-based)             │ │
│  │     ↓                                                     │ │
│  │  5. Decision Engine (Drowsiness Score)                   │ │
│  │     ↓                                                     │ │
│  │  6. Alert Manager (Check thresholds)                     │ │
│  │                                                           │ │
│  └──────────────────────┬────────────────────────────────────┘ │
│                         │                                       │
│                         │ Returns JSON                          │
│                         ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Response:                                                │ │
│  │  {                                                        │ │
│  │    "success": true,                                       │ │
│  │    "face_detected": true,                                 │ │
│  │    "drowsiness_score": 0.75,                              │ │
│  │    "confidence": 0.92,                                    │ │
│  │    "ear": 0.25,                                           │ │
│  │    "mar": 0.45,                                           │ │
│  │    "alert_level": "warning",                              │ │
│  │    "alert_message": "Drowsiness detected",                │ │
│  │    "face_bbox": [100, 150, 300, 400],                     │ │
│  │    "landmarks": [[x1,y1], [x2,y2], ...]                   │ │
│  │  }                                                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                         │
                         │ HTTP Response
                         │ JSON data
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR PHONE                              │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Flutter Mobile App                           │ │
│  │                                                           │ │
│  │  Updates UI:                                              │ │
│  │  - Drowsiness score indicator                             │ │
│  │  - Alert status (normal/warning/critical)                 │ │
│  │  - Face detection overlay                                 │ │
│  │  - Eye landmarks visualization                            │ │
│  │  - Performance metrics (FPS, latency)                     │ │
│  │                                                           │ │
│  │  Triggers Alerts:                                         │ │
│  │  - Visual (color changes)                                 │ │
│  │  - Audio (alarm sound)                                    │ │
│  │  - Haptic (vibration)                                     │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Camera Frame → Base64 Encoding → HTTP POST → Python Server
                                                    ↓
                                            Face Detection
                                                    ↓
                                            Landmark Extraction
                                                    ↓
                                            Feature Calculation
                                                    ↓
                                            ML Classification
                                                    ↓
                                            Decision Engine
                                                    ↓
                                            Alert Check
                                                    ↓
JSON Response ← HTTP Response ← Python Server ← Result
      ↓
Flutter UI Update
      ↓
User Sees Alert
```

## Component Breakdown

### Flutter App Components
```
mobile_app/
├── lib/
│   ├── main.dart                    # App entry point
│   ├── providers/
│   │   ├── drowsiness_provider.dart # State management
│   │   └── settings_provider.dart   # Settings state
│   ├── screens/
│   │   ├── home_screen.dart         # Welcome screen
│   │   ├── monitoring_screen.dart   # Real-time monitoring
│   │   └── settings_screen.dart     # Configuration
│   ├── services/
│   │   ├── camera_service.dart      # Camera management
│   │   └── backend_service.dart     # HTTP communication ⭐
│   └── widgets/
│       └── camera_preview_widget.dart # Camera UI
```

### Python Backend Components
```
backend/
├── src/
│   ├── http_server.py              # Flask HTTP server ⭐
│   ├── mobile_service.py           # Stdin/stdout service
│   ├── face_detection/
│   │   ├── face_detector.py        # MediaPipe face detection
│   │   └── landmark_detector.py    # 68-point landmarks
│   ├── feature_extraction/
│   │   ├── ear_calculator.py       # Eye Aspect Ratio
│   │   └── mar_calculator.py       # Mouth Aspect Ratio
│   ├── ml_models/
│   │   ├── cnn_classifier.py       # CNN model
│   │   └── feature_based_classifier.py # Traditional ML
│   └── decision_logic/
│       ├── decision_engine.py      # Drowsiness scoring
│       └── alert_manager.py        # Alert generation
```

## Network Communication

### Request Format
```json
POST http://192.168.1.100:5000/process
Content-Type: application/json

{
  "image_data": "iVBORw0KGgoAAAANSUhEUgAA..."  // Base64 encoded image
}
```

### Response Format
```json
{
  "success": true,
  "face_detected": true,
  "drowsiness_score": 0.75,      // 0.0 = alert, 1.0 = drowsy
  "confidence": 0.92,             // ML model confidence
  "ear": 0.25,                    // Eye Aspect Ratio
  "mar": 0.45,                    // Mouth Aspect Ratio
  "alert_level": "warning",       // normal/warning/critical
  "alert_message": "Drowsiness detected",
  "face_bbox": [100, 150, 300, 400],  // [x1, y1, x2, y2]
  "landmarks": [[x1,y1], [x2,y2], ...],  // 68 facial points
  "model_type": "cnn"             // cnn or feature_based
}
```

## Alternative Architectures

### Option 2: Platform Channels (Embedded Python)
```
Flutter App
    ↓ MethodChannel
Native Code (Kotlin/Swift)
    ↓ Process/Pipe
Python Backend (on device)
    ↓
ML Pipeline
    ↓
Result
```

### Option 3: TensorFlow Lite (No Python)
```
Flutter App
    ↓
TFLite Interpreter (Dart)
    ↓
TFLite Model (.tflite file)
    ↓
Result
```

## Performance Comparison

| Approach | Latency | Battery | Offline | Complexity |
|----------|---------|---------|---------|------------|
| HTTP Server | 100-500ms | Medium | ❌ No | Low |
| Platform Channels | 50-200ms | Medium | ✅ Yes | High |
| TensorFlow Lite | 10-50ms | Low | ✅ Yes | Medium |

## Recommended Path

1. **Start**: HTTP Server (easy testing)
2. **Develop**: Keep using HTTP Server
3. **Production**: Convert to TensorFlow Lite

## Key Files for Integration

1. **Flutter Side**:
   - `mobile_app/lib/services/backend_service.dart` - Update with HTTP code

2. **Python Side**:
   - `backend/src/http_server.py` - Run this server

3. **Testing**:
   - `backend/scripts/test_http_server.py` - Test the server

## Connection Checklist

- [ ] Python server running (`python src/http_server.py`)
- [ ] Server responds to health check (`http://localhost:5000/health`)
- [ ] Found computer's IP address (`ipconfig` or `ifconfig`)
- [ ] Updated Flutter with correct IP
- [ ] Phone and computer on same WiFi
- [ ] Flutter app can reach server
- [ ] Camera permissions granted
- [ ] Frames being processed successfully

## Troubleshooting Flow

```
Connection Failed?
    ↓
Is server running? → No → Start server
    ↓ Yes
Can access from browser? → No → Check firewall
    ↓ Yes
Same WiFi network? → No → Connect to same network
    ↓ Yes
Correct IP in Flutter? → No → Update IP address
    ↓ Yes
Check server logs for errors
```

---

**Ready to connect?** Follow the steps in `HOW_TO_CONNECT_FLUTTER.md`!
