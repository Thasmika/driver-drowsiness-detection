# Flutter-Python Backend Integration Guide

## Overview

This guide explains how to connect the Flutter mobile app to the Python backend for real-time drowsiness detection. The integration uses **Platform Channels** to communicate between Flutter (Dart) and Python.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Flutter Mobile App                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  UI Layer (Dart)                                       │ │
│  │  - Screens, Widgets, Providers                         │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │  Platform Channel (MethodChannel)                      │ │
│  │  - Dart ↔ Native Bridge                               │ │
│  └────────────────┬───────────────────────────────────────┘ │
└───────────────────┼──────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────┐
│              Native Platform Layer                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Android (Kotlin/Java) or iOS (Swift/Objective-C)     │ │
│  │  - Handles Platform Channel calls                      │ │
│  │  - Manages Python process                              │ │
│  └────────────────┬───────────────────────────────────────┘ │
└───────────────────┼──────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────┐
│              Python Backend Process                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ML Pipeline                                           │ │
│  │  - Face Detection → Landmarks → Features → ML → Alert │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Integration Approaches

### Option 1: Platform Channels with Python Subprocess (Recommended)
**Best for**: Development and testing
- Flutter calls native code via Platform Channels
- Native code spawns Python subprocess
- Communication via stdin/stdout or sockets

### Option 2: Convert Python to TensorFlow Lite
**Best for**: Production deployment
- Convert ML models to TFLite format
- Run inference directly in Flutter using `tflite_flutter`
- No Python runtime needed on device

### Option 3: REST API Server
**Best for**: Distributed systems
- Python backend runs as HTTP server
- Flutter makes HTTP requests
- Can run on device or remote server

## Recommended Implementation: Platform Channels + Python

### Step 1: Set Up Python Backend as a Service

Create a Python service that can be called from native code:

**File: `backend/src/mobile_service.py`**

```python
#!/usr/bin/env python3
"""
Mobile service for Flutter integration
Provides a simple interface for drowsiness detection
"""

import sys
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image

# Import your existing modules
from face_detection.face_detector import FaceDetector
from face_detection.landmark_detector import LandmarkDetector
from feature_extraction.ear_calculator import EARCalculator
from feature_extraction.mar_calculator import MARCalculator
from ml_models.cnn_classifier import CNNDrowsinessClassifier
from decision_logic.decision_engine import DecisionEngine
from decision_logic.alert_manager import AlertManager


class MobileDetectionService:
    """Service for mobile drowsiness detection"""
    
    def __init__(self):
        """Initialize all components"""
        self.face_detector = FaceDetector()
        self.landmark_detector = LandmarkDetector()
        self.ear_calculator = EARCalculator()
        self.mar_calculator = MARCalculator()
        self.cnn_model = CNNDrowsinessClassifier()
        self.decision_engine = DecisionEngine()
        self.alert_manager = AlertManager()
        
        # Load models
        self.cnn_model.load_model('models/cnn_drowsiness_model.h5')
        
    def process_frame(self, image_data):
        """
        Process a single frame from mobile camera
        
        Args:
            image_data: Base64 encoded image or numpy array
            
        Returns:
            dict: Detection results
        """
        try:
            # Decode image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                frame = np.array(image)
            else:
                frame = image_data
            
            # Detect face
            face_result = self.face_detector.detectFace(frame)
            if not face_result['success']:
                return {
                    'success': False,
                    'face_detected': False,
                    'message': 'No face detected'
                }
            
            # Extract landmarks
            landmarks = self.landmark_detector.detectLandmarks(frame)
            if landmarks is None:
                return {
                    'success': False,
                    'face_detected': True,
                    'message': 'Could not extract landmarks'
                }
            
            # Calculate features
            ear = self.ear_calculator.calculateEAR(landmarks)
            mar = self.mar_calculator.calculateMAR(landmarks)
            
            # Get ML prediction
            face_bbox = face_result['bbox']
            face_img = frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            ml_score = self.cnn_model.predict(face_img)
            
            # Decision engine
            drowsiness_score = self.decision_engine.calculate_drowsiness_score(
                ear=ear,
                mar=mar,
                ml_score=ml_score
            )
            
            # Check for alerts
            alert = self.alert_manager.check_alert(drowsiness_score)
            
            return {
                'success': True,
                'face_detected': True,
                'drowsiness_score': float(drowsiness_score),
                'confidence': float(ml_score),
                'ear': float(ear),
                'mar': float(mar),
                'alert_level': alert['level'],
                'alert_message': alert['message'],
                'face_bbox': face_bbox,
                'landmarks': landmarks.tolist() if landmarks is not None else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Main service loop - reads from stdin, writes to stdout"""
    service = MobileDetectionService()
    
    # Signal ready
    print(json.dumps({'status': 'ready'}), flush=True)
    
    # Process requests
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            command = request.get('command')
            
            if command == 'process_frame':
                image_data = request.get('image_data')
                result = service.process_frame(image_data)
                print(json.dumps(result), flush=True)
                
            elif command == 'ping':
                print(json.dumps({'status': 'alive'}), flush=True)
                
            elif command == 'shutdown':
                print(json.dumps({'status': 'shutdown'}), flush=True)
                break
                
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            print(json.dumps(error_response), flush=True)


if __name__ == '__main__':
    main()
```

### Step 2: Android Native Integration

**File: `mobile_app/android/app/src/main/kotlin/com/example/drowsiness_detection/MainActivity.kt`**

```kotlin
package com.example.drowsiness_detection

import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.*
import android.util.Base64
import kotlinx.coroutines.*
import org.json.JSONObject

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.drowsiness_detection/backend"
    private var pythonProcess: Process? = null
    private var processWriter: BufferedWriter? = null
    private var processReader: BufferedReader? = null
    
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "initializeBackend" -> {
                    initializePythonBackend(result)
                }
                "processFrame" -> {
                    val imageData = call.argument<ByteArray>("imageData")
                    if (imageData != null) {
                        processFrame(imageData, result)
                    } else {
                        result.error("INVALID_ARGUMENT", "Image data is null", null)
                    }
                }
                "getDrowsinessScore" -> {
                    // Return cached score
                    result.success(0.5)
                }
                "shutdown" -> {
                    shutdownBackend(result)
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }
    
    private fun initializePythonBackend(result: MethodChannel.Result) {
        GlobalScope.launch(Dispatchers.IO) {
            try {
                // Path to Python executable and script
                val pythonPath = "${applicationContext.filesDir}/python/bin/python3"
                val scriptPath = "${applicationContext.filesDir}/backend/mobile_service.py"
                
                // Start Python process
                val processBuilder = ProcessBuilder(pythonPath, scriptPath)
                processBuilder.redirectErrorStream(true)
                pythonProcess = processBuilder.start()
                
                processWriter = BufferedWriter(OutputStreamWriter(pythonProcess!!.outputStream))
                processReader = BufferedReader(InputStreamReader(pythonProcess!!.inputStream))
                
                // Wait for ready signal
                val readyResponse = processReader!!.readLine()
                val json = JSONObject(readyResponse)
                
                if (json.getString("status") == "ready") {
                    withContext(Dispatchers.Main) {
                        result.success(true)
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        result.error("INIT_FAILED", "Backend not ready", null)
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    result.error("INIT_ERROR", e.message, null)
                }
            }
        }
    }
    
    private fun processFrame(imageData: ByteArray, result: MethodChannel.Result) {
        GlobalScope.launch(Dispatchers.IO) {
            try {
                // Encode image to base64
                val base64Image = Base64.encodeToString(imageData, Base64.NO_WRAP)
                
                // Create request
                val request = JSONObject()
                request.put("command", "process_frame")
                request.put("image_data", base64Image)
                
                // Send to Python
                processWriter!!.write(request.toString() + "\n")
                processWriter!!.flush()
                
                // Read response
                val responseLine = processReader!!.readLine()
                val response = JSONObject(responseLine)
                
                // Convert to Map for Flutter
                val resultMap = mutableMapOf<String, Any>()
                resultMap["success"] = response.getBoolean("success")
                resultMap["face_detected"] = response.optBoolean("face_detected", false)
                
                if (response.getBoolean("success")) {
                    resultMap["drowsiness_score"] = response.getDouble("drowsiness_score")
                    resultMap["confidence"] = response.getDouble("confidence")
                    resultMap["ear"] = response.getDouble("ear")
                    resultMap["mar"] = response.getDouble("mar")
                    resultMap["alert_level"] = response.getString("alert_level")
                    resultMap["alert_message"] = response.getString("alert_message")
                }
                
                withContext(Dispatchers.Main) {
                    result.success(resultMap)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    result.error("PROCESS_ERROR", e.message, null)
                }
            }
        }
    }
    
    private fun shutdownBackend(result: MethodChannel.Result) {
        try {
            if (pythonProcess != null) {
                val request = JSONObject()
                request.put("command", "shutdown")
                processWriter!!.write(request.toString() + "\n")
                processWriter!!.flush()
                
                pythonProcess!!.destroy()
                pythonProcess = null
            }
            result.success(true)
        } catch (e: Exception) {
            result.error("SHUTDOWN_ERROR", e.message, null)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        pythonProcess?.destroy()
    }
}
```

### Step 3: iOS Native Integration

**File: `mobile_app/ios/Runner/AppDelegate.swift`**

```swift
import UIKit
import Flutter

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    private var pythonProcess: Process?
    private var processInput: FileHandle?
    private var processOutput: FileHandle?
    
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        let controller : FlutterViewController = window?.rootViewController as! FlutterViewController
        let channel = FlutterMethodChannel(name: "com.drowsiness_detection/backend",
                                          binaryMessenger: controller.binaryMessenger)
        
        channel.setMethodCallHandler({
            [weak self] (call: FlutterMethodCall, result: @escaping FlutterResult) -> Void in
            guard let self = self else { return }
            
            switch call.method {
            case "initializeBackend":
                self.initializePythonBackend(result: result)
            case "processFrame":
                if let args = call.arguments as? [String: Any],
                   let imageData = args["imageData"] as? FlutterStandardTypedData {
                    self.processFrame(imageData: imageData.data, result: result)
                } else {
                    result(FlutterError(code: "INVALID_ARGUMENT",
                                      message: "Image data is null",
                                      details: nil))
                }
            case "getDrowsinessScore":
                result(0.5)
            case "shutdown":
                self.shutdownBackend(result: result)
            default:
                result(FlutterMethodNotImplemented)
            }
        })
        
        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
    
    private func initializePythonBackend(result: @escaping FlutterResult) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Path to Python and script
                let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let pythonPath = documentsPath.appendingPathComponent("python/bin/python3").path
                let scriptPath = documentsPath.appendingPathComponent("backend/mobile_service.py").path
                
                // Create process
                self.pythonProcess = Process()
                self.pythonProcess?.executableURL = URL(fileURLWithPath: pythonPath)
                self.pythonProcess?.arguments = [scriptPath]
                
                let inputPipe = Pipe()
                let outputPipe = Pipe()
                
                self.pythonProcess?.standardInput = inputPipe
                self.pythonProcess?.standardOutput = outputPipe
                
                self.processInput = inputPipe.fileHandleForWriting
                self.processOutput = outputPipe.fileHandleForReading
                
                try self.pythonProcess?.run()
                
                // Wait for ready signal
                if let data = self.processOutput?.availableData,
                   let response = String(data: data, encoding: .utf8),
                   let jsonData = response.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                   json["status"] as? String == "ready" {
                    DispatchQueue.main.async {
                        result(true)
                    }
                } else {
                    DispatchQueue.main.async {
                        result(FlutterError(code: "INIT_FAILED",
                                          message: "Backend not ready",
                                          details: nil))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    result(FlutterError(code: "INIT_ERROR",
                                      message: error.localizedDescription,
                                      details: nil))
                }
            }
        }
    }
    
    private func processFrame(imageData: Data, result: @escaping FlutterResult) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Encode to base64
                let base64Image = imageData.base64EncodedString()
                
                // Create request
                let request: [String: Any] = [
                    "command": "process_frame",
                    "image_data": base64Image
                ]
                
                let jsonData = try JSONSerialization.data(withJSONObject: request)
                var jsonString = String(data: jsonData, encoding: .utf8)!
                jsonString += "\n"
                
                // Send to Python
                self.processInput?.write(jsonString.data(using: .utf8)!)
                
                // Read response
                if let data = self.processOutput?.availableData,
                   let responseLine = String(data: data, encoding: .utf8),
                   let responseData = responseLine.data(using: .utf8),
                   let response = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any] {
                    DispatchQueue.main.async {
                        result(response)
                    }
                } else {
                    DispatchQueue.main.async {
                        result(FlutterError(code: "PROCESS_ERROR",
                                          message: "Failed to parse response",
                                          details: nil))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    result(FlutterError(code: "PROCESS_ERROR",
                                      message: error.localizedDescription,
                                      details: nil))
                }
            }
        }
    }
    
    private func shutdownBackend(result: @escaping FlutterResult) {
        do {
            if let process = pythonProcess {
                let request: [String: Any] = ["command": "shutdown"]
                let jsonData = try JSONSerialization.data(withJSONObject: request)
                var jsonString = String(data: jsonData, encoding: .utf8)!
                jsonString += "\n"
                
                processInput?.write(jsonString.data(using: .utf8)!)
                process.terminate()
                pythonProcess = nil
            }
            result(true)
        } catch {
            result(FlutterError(code: "SHUTDOWN_ERROR",
                              message: error.localizedDescription,
                              details: nil))
        }
    }
}
```

### Step 4: Update Flutter Backend Service

Update the existing `backend_service.dart` to use the platform channel:

```dart
// This is already in your code, but here's the complete implementation
import 'dart:typed_data';
import 'package:flutter/services.dart';

class BackendService {
  static const platform = MethodChannel('com.drowsiness_detection/backend');
  
  bool _isInitialized = false;
  
  bool get isInitialized => _isInitialized;
  
  /// Initialize the Python backend
  Future<bool> initialize() async {
    try {
      final result = await platform.invokeMethod('initializeBackend');
      _isInitialized = result == true;
      return _isInitialized;
    } catch (e) {
      print('Error initializing backend: $e');
      return false;
    }
  }
  
  /// Process a camera frame
  Future<Map<String, dynamic>> processFrame(Uint8List imageData) async {
    if (!_isInitialized) {
      throw Exception('Backend not initialized');
    }
    
    try {
      final result = await platform.invokeMethod('processFrame', {
        'imageData': imageData,
      });
      
      return Map<String, dynamic>.from(result);
    } catch (e) {
      print('Error processing frame: $e');
      return {
        'success': false,
        'error': e.toString(),
      };
    }
  }
  
  /// Shutdown the backend
  Future<void> shutdown() async {
    try {
      await platform.invokeMethod('shutdown');
      _isInitialized = false;
    } catch (e) {
      print('Error shutting down backend: $e');
    }
  }
}
```

## Deployment Steps

### 1. Package Python Backend for Mobile

**For Android:**
```bash
# Use Chaquopy or similar to package Python
# Add to android/app/build.gradle:
plugins {
    id 'com.chaquo.python' version '14.0.2'
}

chaquopy {
    defaultConfig {
        version "3.8"
        pip {
            install "opencv-python-headless"
            install "mediapipe"
            install "tensorflow-lite"
            install "numpy"
            install "pillow"
        }
    }
}
```

**For iOS:**
```bash
# Use Python-iOS or Kivy-iOS
# Package Python runtime and dependencies
```

### 2. Copy Backend Files to Mobile Assets

```bash
# Copy Python backend to mobile app
cp -r backend/src mobile_app/assets/backend/
```

### 3. Update pubspec.yaml

```yaml
flutter:
  assets:
    - assets/backend/
    - assets/models/
```

## Testing the Integration

```dart
// Test in your Flutter app
final backendService = BackendService();

// Initialize
await backendService.initialize();

// Process frame from camera
final imageData = await cameraController.takePicture();
final bytes = await imageData.readAsBytes();

final result = await backendService.processFrame(bytes);

if (result['success']) {
  print('Drowsiness Score: ${result['drowsiness_score']}');
  print('Alert Level: ${result['alert_level']}');
}
```

## Alternative: TensorFlow Lite (Production Recommended)

For production, convert your models to TFLite and run directly in Flutter:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteService {
  Interpreter? _interpreter;
  
  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('models/drowsiness_model.tflite');
  }
  
  Future<double> predict(Uint8List imageData) async {
    // Preprocess image
    var input = preprocessImage(imageData);
    
    // Run inference
    var output = List.filled(1, 0.0).reshape([1, 1]);
    _interpreter!.run(input, output);
    
    return output[0][0];
  }
}
```

## Performance Considerations

1. **Frame Rate**: Process every 3-5 frames (not every frame)
2. **Image Size**: Resize to 224x224 before sending to backend
3. **Threading**: Use isolates in Flutter for heavy processing
4. **Battery**: Monitor CPU usage and adjust processing rate

## Troubleshooting

### Common Issues:

1. **Python not found**: Ensure Python runtime is packaged correctly
2. **Import errors**: Check all dependencies are included
3. **Slow performance**: Reduce image size, process fewer frames
4. **Memory issues**: Clear old frames, limit buffer size

## Next Steps

1. Create the `mobile_service.py` file
2. Implement native platform code (Kotlin/Swift)
3. Test on physical devices
4. Optimize for battery life
5. Consider TFLite conversion for production

Would you like me to create any of these files for you?
