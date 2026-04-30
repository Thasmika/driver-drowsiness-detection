import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:camera/camera.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:geolocator/geolocator.dart';
import 'package:geocoding/geocoding.dart';
import 'package:url_launcher/url_launcher.dart';
import '../providers/drowsiness_provider.dart';
import '../services/camera_service.dart';
import '../services/backend_service.dart';
import '../services/location_service.dart';
import '../widgets/camera_preview_widget.dart';

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
  final LocationService _locationService = LocationService();
  final AudioPlayer _audioPlayer = AudioPlayer();
  
  bool _isInitializing = true;
  bool _hasPermission = false;
  String _errorMessage = '';
  
  Map<String, dynamic>? _detectionData;
  
  // Location tracking
  Position? _currentPosition;
  String _currentAddress = 'Loading address...';
  DateTime? _lastLocationUpdate;
  
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
    final cameraStatus = await Permission.camera.request();
    if (!cameraStatus.isGranted) {
      setState(() {
        _isInitializing = false;
        _hasPermission = false;
        _errorMessage = 'Camera permission denied';
      });
      return;
    }
    
    // Request location permission - CRITICAL FOR REAL LOCATION
    final locationStatus = await Permission.location.request();
    debugPrint('Location permission status: $locationStatus');
    
    // Check if location services are enabled
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    debugPrint('Location service enabled: $serviceEnabled');
    
    if (!serviceEnabled) {
      debugPrint('WARNING: Location services are disabled!');
      // Continue anyway, but location won't work
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
    
    // Initialize location tracking - FORCE IMMEDIATE UPDATE
    debugPrint('Starting location tracking...');
    _startLocationTracking();
    
    // Force immediate location update
    _forceLocationUpdate();
    
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
  
  // Force immediate location update
  Future<void> _forceLocationUpdate() async {
    debugPrint('Forcing location update...');
    try {
      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        forceAndroidLocationManager: true,
      );
      debugPrint('Got position: ${position.latitude}, ${position.longitude}');
      if (mounted) {
        await _updateLocationFromPosition(position);
      }
    } catch (e) {
      debugPrint('Error forcing location update: $e');
    }
  }
  
  bool _isProcessing = false;
  DateTime _lastProcessTime = DateTime.now();

  void _startLocationTracking() {
    // Get initial position
    _updateLocation();
    
    // Listen to position updates
    _locationService.getPositionStream().listen((Position position) {
      if (!mounted) return;
      _updateLocationFromPosition(position);
    }, onError: (error) {
      debugPrint('Location stream error: $error');
    });
  }
  
  Future<void> _updateLocation() async {
    try {
      final position = await _locationService.getCurrentPosition();
      if (position != null && mounted) {
        _updateLocationFromPosition(position);
      }
    } catch (e) {
      debugPrint('Error updating location: $e');
    }
  }
  
  Future<void> _updateLocationFromPosition(Position position) async {
    setState(() {
      _currentPosition = position;
      _lastLocationUpdate = DateTime.now();
    });
    
    // Get address from coordinates
    try {
      List<Placemark> placemarks = await placemarkFromCoordinates(
        position.latitude,
        position.longitude,
      );
      
      if (placemarks.isNotEmpty && mounted) {
        final place = placemarks.first;
        setState(() {
          _currentAddress = _formatAddress(place);
        });
      }
    } catch (e) {
      debugPrint('Error getting address: $e');
      setState(() {
        _currentAddress = 'Address unavailable';
      });
    }
  }
  
  String _formatAddress(Placemark place) {
    List<String> parts = [];
    
    if (place.street != null && place.street!.isNotEmpty) {
      parts.add(place.street!);
    }
    if (place.locality != null && place.locality!.isNotEmpty) {
      parts.add(place.locality!);
    }
    if (place.administrativeArea != null && place.administrativeArea!.isNotEmpty) {
      parts.add(place.administrativeArea!);
    }
    if (place.country != null && place.country!.isNotEmpty) {
      parts.add(place.country!);
    }
    
    return parts.isEmpty ? 'Address unavailable' : parts.join(', ');
  }


  void _startSimulatedProcessing() {
    // Start real camera image stream and send to backend
    _cameraService.startImageStream((CameraImage image) async {
      if (!mounted) return;

      // Throttle to 1 frame per second to avoid overwhelming the backend
      final now = DateTime.now();
      if (_isProcessing || now.difference(_lastProcessTime).inMilliseconds < 1000) {
        return;
      }
      _isProcessing = true;
      _lastProcessTime = now;

      try {
        debugPrint('Sending frame to backend...');
        final result = await _backendService.processFrame(image);
        debugPrint('Backend result: $result');

        if (!mounted) return;

        setState(() {
          _detectionData = result;
        });

        final provider = context.read<DrowsinessProvider>();
        final score = (result['drowsiness_score'] as num?)?.toDouble() ?? 0.0;
        final conf  = (result['confidence'] as num?)?.toDouble() ?? 0.0;
        provider.updateDrowsinessScore(score, conf);
        provider.updateFaceDetection(result['face_detected'] == true);
        provider.updatePerformanceMetrics(1.0, 1000.0);

        // Show alert if critical
        if (score >= 0.8 && mounted) {
          _showDrowsinessAlert();
        }
      } catch (e) {
        debugPrint('Frame processing error: $e');
      } finally {
        _isProcessing = false;
      }
    }).catchError((e) {
      debugPrint('Image stream error: $e');
    });
  }

  void _showDrowsinessAlert() {
    // Play beep sound
    _audioPlayer.play(AssetSource('sounds/alert_beep.wav'));
    // Vibrate
    HapticFeedback.heavyImpact();
    // Show snackbar
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('DROWSINESS DETECTED! Please pull over!',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
        backgroundColor: Colors.red,
        duration: Duration(seconds: 3),
      ),
    );
  }

  // Make emergency call to 119
  Future<void> _makeEmergencyCall() async {
    final Uri phoneUri = Uri(scheme: 'tel', path: '119');
    
    try {
      if (await canLaunchUrl(phoneUri)) {
        await launchUrl(phoneUri);
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Unable to make phone call'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  // Show emergency confirmation dialog
  void _showEmergencyDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.grey[900],
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: BorderSide(
            color: Colors.red.withOpacity(0.5),
            width: 2,
          ),
        ),
        title: const Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.red, size: 32),
            SizedBox(width: 12),
            Text(
              'Emergency Call',
              style: TextStyle(color: Colors.white, fontSize: 20),
            ),
          ],
        ),
        content: const Text(
          'Do you want to call 119 Emergency Services?',
          style: TextStyle(color: Colors.white70, fontSize: 16),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text(
              'Cancel',
              style: TextStyle(color: Colors.white.withOpacity(0.7)),
            ),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              _makeEmergencyCall();
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            child: const Text('Call 119'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraService.stopImageStream();
    _cameraService.dispose();
    _backendService.dispose();
    _audioPlayer.dispose();
    context.read<DrowsinessProvider>().stopMonitoring();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Monitoring'),
        actions: [
          // Emergency 119 button in AppBar
          Container(
            margin: const EdgeInsets.only(right: 8),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [
                  Color(0xFFFF1744),
                  Color(0xFFD50000),
                ],
              ),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                onTap: _showEmergencyDialog,
                borderRadius: BorderRadius.circular(8),
                child: Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.phone_in_talk,
                        color: Colors.white,
                        size: 18,
                      ),
                      const SizedBox(width: 4),
                      const Text(
                        '119',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
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
        
        // Location display
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          color: Colors.black87,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.location_on, color: Colors.blue, size: 20),
                  const SizedBox(width: 8),
                  const Text(
                    'Current Location',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                  const Spacer(),
                  if (_lastLocationUpdate != null)
                    Text(
                      _formatLastUpdate(_lastLocationUpdate!),
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 4),
              if (_currentPosition != null)
                Text(
                  'Lat: ${_currentPosition!.latitude.toStringAsFixed(6)}, '
                  'Lon: ${_currentPosition!.longitude.toStringAsFixed(6)}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontFamily: 'monospace',
                  ),
                )
              else
                const Text(
                  'Getting location...',
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 12,
                  ),
                ),
              const SizedBox(height: 2),
              Text(
                _currentAddress,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
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
  
  String _formatLastUpdate(DateTime time) {
    final now = DateTime.now();
    final difference = now.difference(time);
    
    if (difference.inSeconds < 60) {
      return '${difference.inSeconds}s ago';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else {
      return '${difference.inHours}h ago';
    }
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
