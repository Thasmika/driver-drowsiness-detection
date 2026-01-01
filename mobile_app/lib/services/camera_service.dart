import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';

/// Camera service for managing camera initialization and frame capture
/// 
/// Validates: Requirements 4.4, 10.1
class CameraService {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  
  bool get isInitialized => _isInitialized;
  CameraController? get controller => _controller;
  
  /// Initialize camera service
  Future<bool> initialize() async {
    try {
      // Get available cameras
      _cameras = await availableCameras();
      
      if (_cameras == null || _cameras!.isEmpty) {
        debugPrint('No cameras available');
        return false;
      }
      
      // Use front camera for driver monitoring
      final frontCamera = _cameras!.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => _cameras!.first,
      );
      
      // Initialize camera controller
      _controller = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      
      await _controller!.initialize();
      _isInitialized = true;
      
      debugPrint('Camera initialized successfully');
      return true;
    } catch (e) {
      debugPrint('Error initializing camera: $e');
      return false;
    }
  }
  
  /// Start image stream for real-time processing
  Future<void> startImageStream(Function(CameraImage) onImage) async {
    if (_controller == null || !_controller!.value.isInitialized) {
      throw Exception('Camera not initialized');
    }
    
    await _controller!.startImageStream(onImage);
  }
  
  /// Stop image stream
  Future<void> stopImageStream() async {
    if (_controller != null && _controller!.value.isStreamingImages) {
      await _controller!.stopImageStream();
    }
  }
  
  /// Dispose camera resources
  Future<void> dispose() async {
    if (_controller != null) {
      if (_controller!.value.isStreamingImages) {
        await _controller!.stopImageStream();
      }
      await _controller!.dispose();
      _controller = null;
      _isInitialized = false;
    }
  }
  
  /// Get camera preview widget
  Widget? getPreviewWidget() {
    if (_controller == null || !_controller!.value.isInitialized) {
      return null;
    }
    return CameraPreview(_controller!);
  }
}
