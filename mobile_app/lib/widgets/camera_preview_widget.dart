import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import '../services/camera_service.dart';

/// Camera preview widget with face detection overlay
/// 
/// Validates: Requirements 4.4, 10.4
class CameraPreviewWidget extends StatefulWidget {
  final CameraService cameraService;
  final bool showOverlay;
  final Map<String, dynamic>? detectionData;
  
  const CameraPreviewWidget({
    super.key,
    required this.cameraService,
    this.showOverlay = true,
    this.detectionData,
  });

  @override
  State<CameraPreviewWidget> createState() => _CameraPreviewWidgetState();
}

class _CameraPreviewWidgetState extends State<CameraPreviewWidget> {
  @override
  Widget build(BuildContext context) {
    if (!widget.cameraService.isInitialized) {
      return Container(
        color: Colors.black,
        child: const Center(
          child: CircularProgressIndicator(),
        ),
      );
    }
    
    final controller = widget.cameraService.controller;
    if (controller == null) {
      return Container(
        color: Colors.black,
        child: const Center(
          child: Text(
            'Camera not available',
            style: TextStyle(color: Colors.white),
          ),
        ),
      );
    }
    
    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera preview
        CameraPreview(controller),
        
        // Face detection overlay
        if (widget.showOverlay && widget.detectionData != null)
          CustomPaint(
            painter: FaceOverlayPainter(
              detectionData: widget.detectionData!,
            ),
          ),
        
        // Detection status indicator
        if (widget.detectionData != null)
          Positioned(
            top: 16,
            left: 16,
            child: _buildStatusIndicator(),
          ),
      ],
    );
  }
  
  Widget _buildStatusIndicator() {
    final faceDetected = widget.detectionData?['face_detected'] ?? false;
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: faceDetected ? Colors.green.withOpacity(0.8) : Colors.red.withOpacity(0.8),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            faceDetected ? Icons.check_circle : Icons.warning,
            color: Colors.white,
            size: 16,
          ),
          const SizedBox(width: 4),
          Text(
            faceDetected ? 'Face Detected' : 'No Face',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}

/// Custom painter for face detection overlay
class FaceOverlayPainter extends CustomPainter {
  final Map<String, dynamic> detectionData;
  
  FaceOverlayPainter({required this.detectionData});
  
  @override
  void paint(Canvas canvas, Size size) {
    final faceDetected = detectionData['face_detected'] ?? false;
    
    if (!faceDetected) return;
    
    // Draw face bounding box if available
    final faceBounds = detectionData['face_bounds'];
    if (faceBounds != null) {
      final paint = Paint()
        ..color = Colors.green
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;
      
      final rect = Rect.fromLTWH(
        faceBounds['x'] * size.width,
        faceBounds['y'] * size.height,
        faceBounds['width'] * size.width,
        faceBounds['height'] * size.height,
      );
      
      canvas.drawRect(rect, paint);
    }
    
    // Draw eye landmarks if available
    final landmarks = detectionData['landmarks'];
    if (landmarks != null) {
      final landmarkPaint = Paint()
        ..color = Colors.blue
        ..style = PaintingStyle.fill;
      
      // Draw eye points
      final leftEye = landmarks['left_eye'];
      final rightEye = landmarks['right_eye'];
      
      if (leftEye != null) {
        for (var point in leftEye) {
          canvas.drawCircle(
            Offset(point['x'] * size.width, point['y'] * size.height),
            2,
            landmarkPaint,
          );
        }
      }
      
      if (rightEye != null) {
        for (var point in rightEye) {
          canvas.drawCircle(
            Offset(point['x'] * size.width, point['y'] * size.height),
            2,
            landmarkPaint,
          );
        }
      }
    }
    
    // Draw drowsiness indicator
    final drowsinessScore = detectionData['drowsiness_score'];
    if (drowsinessScore != null) {
      final score = (drowsinessScore as num).toDouble();
      final color = _getScoreColor(score);
      
      final indicatorPaint = Paint()
        ..color = color.withOpacity(0.3)
        ..style = PaintingStyle.fill;
      
      // Draw semi-transparent overlay based on drowsiness level
      if (score > 0.6) {
        canvas.drawRect(
          Rect.fromLTWH(0, 0, size.width, size.height),
          indicatorPaint,
        );
      }
    }
  }
  
  Color _getScoreColor(double score) {
    if (score >= 0.8) return Colors.red;
    if (score >= 0.6) return Colors.orange;
    return Colors.green;
  }
  
  @override
  bool shouldRepaint(FaceOverlayPainter oldDelegate) {
    return detectionData != oldDelegate.detectionData;
  }
}
