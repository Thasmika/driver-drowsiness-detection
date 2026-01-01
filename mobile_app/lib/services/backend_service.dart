import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

/// Backend service for communicating with Python ML pipeline
/// 
/// Uses HTTP to communicate with Python backend server.
/// For production, consider using Platform Channels or TensorFlow Lite.
/// 
/// Validates: Requirements 4.1, 5.4
class BackendService {
  // For Android Emulator, use 10.0.2.2 to access localhost
  // For physical device (iPhone/Android), use your computer's IP (e.g., 192.168.1.1)
  // For iOS simulator, use localhost
  static const String baseUrl = 'http://10.0.2.2:5000';
  
  bool _isInitialized = false;
  
  bool get isInitialized => _isInitialized;
  
  /// Initialize the Python backend
  /// 
  /// Checks if the HTTP server is running and accessible
  Future<bool> initialize() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
      ).timeout(const Duration(seconds: 5));
      
      _isInitialized = response.statusCode == 200;
      
      if (_isInitialized) {
        print('✓ Backend initialized successfully');
        
        // Get status info
        final statusResponse = await http.get(Uri.parse('$baseUrl/status'));
        if (statusResponse.statusCode == 200) {
          final status = jsonDecode(statusResponse.body);
          print('Backend status: ${jsonEncode(status)}');
        }
      } else {
        print('❌ Backend health check failed: ${response.statusCode}');
      }
      
      return _isInitialized;
    } catch (e) {
      print('❌ Error initializing backend: $e');
      print('Make sure Python server is running: python backend/src/http_server.py');
      print('And update baseUrl with your computer\'s IP address');
      return false;
    }
  }
  
  /// Process a camera frame
  /// 
  /// Sends the frame to the Python backend for drowsiness detection
  /// Returns detection results including drowsiness score, features, etc.
  Future<Map<String, dynamic>> processFrame(CameraImage image) async {
    if (!_isInitialized) {
      return {
        'success': false,
        'error': 'Backend not initialized',
        'face_detected': false,
        'drowsiness_score': 0.0,
        'confidence': 0.0,
      };
    }
    
    try {
      // Convert CameraImage to JPEG bytes
      final jpegBytes = await _convertCameraImageToJpeg(image);
      
      // Convert to base64
      final base64Image = base64Encode(jpegBytes);
      
      // Send request to Python backend
      final response = await http.post(
        Uri.parse('$baseUrl/process'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'image_data': base64Image}),
      ).timeout(const Duration(seconds: 3));
      
      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        return Map<String, dynamic>.from(result);
      } else {
        print('Server error: ${response.statusCode}');
        return {
          'success': false,
          'error': 'Server returned ${response.statusCode}',
          'face_detected': false,
          'drowsiness_score': 0.0,
          'confidence': 0.0,
        };
      }
    } catch (e) {
      print('Error processing frame: $e');
      return {
        'success': false,
        'error': e.toString(),
        'face_detected': false,
        'drowsiness_score': 0.0,
        'confidence': 0.0,
      };
    }
  }
  
  /// Convert CameraImage to JPEG bytes
  Future<Uint8List> _convertCameraImageToJpeg(CameraImage image) async {
    try {
      // Convert YUV420 to RGB
      final int width = image.width;
      final int height = image.height;
      
      // Create image from YUV420
      final img.Image rgbImage = img.Image(width: width, height: height);
      
      // Get Y, U, V planes
      final yPlane = image.planes[0];
      final uPlane = image.planes[1];
      final vPlane = image.planes[2];
      
      // Convert YUV to RGB (simplified conversion)
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * yPlane.bytesPerRow + x;
          final int uvIndex = (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2);
          
          if (yIndex < yPlane.bytes.length && 
              uvIndex < uPlane.bytes.length && 
              uvIndex < vPlane.bytes.length) {
            final int yValue = yPlane.bytes[yIndex];
            final int uValue = uPlane.bytes[uvIndex];
            final int vValue = vPlane.bytes[uvIndex];
            
            // YUV to RGB conversion
            int r = (yValue + 1.370705 * (vValue - 128)).round().clamp(0, 255);
            int g = (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128)).round().clamp(0, 255);
            int b = (yValue + 1.732446 * (uValue - 128)).round().clamp(0, 255);
            
            rgbImage.setPixelRgba(x, y, r, g, b, 255);
          }
        }
      }
      
      // Encode to JPEG
      final jpegBytes = img.encodeJpg(rgbImage, quality: 85);
      return Uint8List.fromList(jpegBytes);
      
    } catch (e) {
      print('Error converting image: $e');
      // Return empty image on error
      return Uint8List(0);
    }
  }
  
  /// Get current drowsiness score (cached from last processFrame call)
  Future<double> getDrowsinessScore() async {
    // This is now handled by processFrame
    // Return 0.0 as placeholder
    return 0.0;
  }
  
  /// Update settings
  Future<void> updateSettings(Map<String, dynamic> settings) async {
    // Settings are handled locally in Flutter
    // No need to send to backend for HTTP approach
    print('Settings updated locally: $settings');
  }
  
  /// Shutdown the backend
  Future<void> dispose() async {
    _isInitialized = false;
    print('Backend connection closed');
  }
  
  /// Check if backend is alive
  Future<bool> ping() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
      ).timeout(const Duration(seconds: 2));
      
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}
