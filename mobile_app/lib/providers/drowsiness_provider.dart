import 'package:flutter/foundation.dart';

/// Provider for drowsiness detection state management
/// 
/// Manages the state of drowsiness monitoring including:
/// - Monitoring status (active/inactive)
/// - Drowsiness score and confidence
/// - Alert status
/// - Performance metrics
class DrowsinessProvider with ChangeNotifier {
  bool _isMonitoring = false;
  double _drowsinessScore = 0.0;
  double _confidence = 0.0;
  String _alertStatus = 'none';
  bool _faceDetected = false;
  
  // Performance metrics
  double _fps = 0.0;
  double _latencyMs = 0.0;
  
  // Getters
  bool get isMonitoring => _isMonitoring;
  double get drowsinessScore => _drowsinessScore;
  double get confidence => _confidence;
  String get alertStatus => _alertStatus;
  bool get faceDetected => _faceDetected;
  double get fps => _fps;
  double get latencyMs => _latencyMs;
  
  /// Start drowsiness monitoring
  void startMonitoring() {
    _isMonitoring = true;
    notifyListeners();
  }
  
  /// Stop drowsiness monitoring
  void stopMonitoring() {
    _isMonitoring = false;
    _drowsinessScore = 0.0;
    _confidence = 0.0;
    _alertStatus = 'none';
    _faceDetected = false;
    notifyListeners();
  }
  
  /// Update drowsiness score from backend
  void updateDrowsinessScore(double score, double conf) {
    _drowsinessScore = score;
    _confidence = conf;
    
    // Update alert status based on score
    if (score >= 0.8) {
      _alertStatus = 'critical';
    } else if (score >= 0.6) {
      _alertStatus = 'warning';
    } else {
      _alertStatus = 'normal';
    }
    
    notifyListeners();
  }
  
  /// Update face detection status
  void updateFaceDetection(bool detected) {
    _faceDetected = detected;
    notifyListeners();
  }
  
  /// Update performance metrics
  void updatePerformanceMetrics(double fps, double latency) {
    _fps = fps;
    _latencyMs = latency;
    notifyListeners();
  }
  
  /// Reset all state
  void reset() {
    _isMonitoring = false;
    _drowsinessScore = 0.0;
    _confidence = 0.0;
    _alertStatus = 'none';
    _faceDetected = false;
    _fps = 0.0;
    _latencyMs = 0.0;
    notifyListeners();
  }
}
