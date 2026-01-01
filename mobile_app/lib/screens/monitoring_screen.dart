import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:permission_handler/permission_handler.dart';
import '../providers/drowsiness_provider.dart';
import '../services/camera_service.dart';
import '../services/backend_service.dart';
import '../widgets/camera_preview_widget.dart';
import 'dart:async';

/// Monitoring screen with camera preview and real-time detection
/// 
/// Validates: Requirements 4.4, 10.1, 10.4
class MonitoringScreen extends StatefulWidget {
  const MonitoringScreen({super.key});

  @override
  State<MonitoringScreen> createState() => _MonitoringScreenState();
}

class _MonitoringScreenState extends State<MonitoringScreen> {
  final CameraService _cameraService = CameraService();
  final BackendService _backendService = BackendService();
  
  bool _isInitializing = true;
  bool _hasPermission = false;
  String _errorMessage = '';
  
  Map<String, dynamic>? _detectionData;
  Timer? _processingTimer;
  
  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  Future<void> _initializeServices() async {
    setState(() {
      _isInitializing = true;
      _errorMessage = '';
    });
    
    // Request camera permission
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() {
        _isInitializing = false;
        _hasPermission = false;
        _errorMessage = 'Camera permission denied';
      });
      return;
    }
    
    setState(() {
      _hasPermission = true;
    });
    
    // Initialize camera
    final cameraInitialized = await _cameraService.initialize();
    if (!cameraInitialized) {
      setState(() {
        _isInitializing = false;
        _errorMessage = 'Failed to initialize camera';
      });
      return;
    }
    
    // Initialize backend (would connect to Python backend in production)
    // For now, we'll simulate backend responses
    await _backendService.initialize();
    
    // Start monitoring
    if (mounted) {
      context.read<DrowsinessProvider>().startMonitoring();
      
      // Start simulated processing (in production, this would process real frames)
      _startSimulatedProcessing();
    }
    
    setState(() {
      _isInitializing = false;
    });
  }
  
  void _startSimulatedProcessing() {
    // Simulate frame processing every 100ms
    _processingTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      
      // Simulate detection data (in production, this comes from backend)
      final simulatedData = _generateSimulatedData();
      
      setState(() {
        _detectionData = simulatedData;
      });
      
      // Update provider
      final provider = context.read<DrowsinessProvider>();
      provider.updateDrowsinessScore(
        simulatedData['drowsiness_score'],
        simulatedData['confidence'],
      );
      provider.updateFaceDetection(simulatedData['face_detected']);
      provider.updatePerformanceMetrics(
        simulatedData['fps'],
        simulatedData['latency_ms'],
      );
    });
  }
  
  Map<String, dynamic> _generateSimulatedData() {
    // Simulate varying drowsiness levels
    final time = DateTime.now().millisecondsSinceEpoch / 1000;
    final drowsinessScore = (0.3 + 0.3 * (time % 10) / 10).clamp(0.0, 1.0);
    
    return {
      'face_detected': true,
      'drowsiness_score': drowsinessScore,
      'confidence': 0.85,
      'fps': 15.0,
      'latency_ms': 45.0,
      'face_bounds': {
        'x': 0.2,
        'y': 0.2,
        'width': 0.6,
        'height': 0.6,
      },
      'landmarks': {
        'left_eye': [
          {'x': 0.35, 'y': 0.4},
          {'x': 0.38, 'y': 0.4},
        ],
        'right_eye': [
          {'x': 0.62, 'y': 0.4},
          {'x': 0.65, 'y': 0.4},
        ],
      },
    };
  }

  @override
  void dispose() {
    _processingTimer?.cancel();
    _cameraService.dispose();
    _backendService.dispose();
    context.read<DrowsinessProvider>().stopMonitoring();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Monitoring'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () {
              _showInfoDialog(context);
            },
          ),
        ],
      ),
      body: SafeArea(
        child: _buildBody(),
      ),
    );
  }
  
  Widget _buildBody() {
    if (_isInitializing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing camera...'),
          ],
        ),
      );
    }
    
    if (!_hasPermission || _errorMessage.isNotEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.error_outline,
              size: 64,
              color: Colors.red,
            ),
            const SizedBox(height: 16),
            Text(
              _errorMessage.isEmpty ? 'Camera permission required' : _errorMessage,
              style: const TextStyle(fontSize: 16),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                _initializeServices();
              },
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }
    
    return Column(
      children: [
        // Camera preview with overlay
        Expanded(
          flex: 3,
          child: CameraPreviewWidget(
            cameraService: _cameraService,
            showOverlay: true,
            detectionData: _detectionData,
          ),
        ),
        
        // Status panel
        Expanded(
          flex: 2,
          child: Consumer<DrowsinessProvider>(
            builder: (context, provider, child) {
              return Container(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Drowsiness indicator
                    Card(
                      color: _getAlertColor(provider.alertStatus),
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          children: [
                            Text(
                              _getAlertText(provider.alertStatus),
                              style: const TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Drowsiness: ${(provider.drowsinessScore * 100).toStringAsFixed(0)}%',
                              style: const TextStyle(
                                fontSize: 18,
                                color: Colors.white,
                              ),
                            ),
                            Text(
                              'Confidence: ${(provider.confidence * 100).toStringAsFixed(0)}%',
                              style: const TextStyle(
                                fontSize: 14,
                                color: Colors.white70,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    
                    // System status
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildStatusItem(
                          'Face',
                          provider.faceDetected ? 'Detected' : 'Not Found',
                          provider.faceDetected ? Colors.green : Colors.red,
                        ),
                        _buildStatusItem(
                          'FPS',
                          provider.fps.toStringAsFixed(0),
                          Colors.blue,
                        ),
                        _buildStatusItem(
                          'Latency',
                          '${provider.latencyMs.toStringAsFixed(0)}ms',
                          Colors.orange,
                        ),
                      ],
                    ),
                  ],
                ),
              );
            },
          ),
        ),
      ],
    );
  }
  
  Widget _buildStatusItem(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            color: Colors.grey,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }
  
  Color _getAlertColor(String status) {
    switch (status) {
      case 'critical':
        return Colors.red;
      case 'warning':
        return Colors.orange;
      default:
        return Colors.green;
    }
  }
  
  String _getAlertText(String status) {
    switch (status) {
      case 'critical':
        return 'CRITICAL - PULL OVER!';
      case 'warning':
        return 'Warning - Stay Alert';
      default:
        return 'Normal - Driving Safe';
    }
  }
  
  void _showInfoDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Monitoring Info'),
        content: const Text(
          'The system is monitoring your face for signs of drowsiness.\n\n'
          '• Green: Alert and safe\n'
          '• Orange: Showing signs of fatigue\n'
          '• Red: Critical - pull over immediately\n\n'
          'Keep your face visible to the camera for accurate detection.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }
}
