import 'package:geolocator/geolocator.dart';
import 'package:flutter/foundation.dart';

/// Service for handling GPS location tracking
/// 
/// Provides current position and address resolution
class LocationService {
  Position? _currentPosition;
  DateTime? _lastUpdate;
  
  /// Get current GPS position
  Future<Position?> getCurrentPosition() async {
    try {
      // Check if location services are enabled
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        debugPrint('Location services are disabled');
        // Try to open location settings
        await Geolocator.openLocationSettings();
        return null;
      }

      // Check location permissions
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          debugPrint('Location permissions are denied');
          return null;
        }
      }
      
      if (permission == LocationPermission.deniedForever) {
        debugPrint('Location permissions are permanently denied');
        // Try to open app settings
        await Geolocator.openAppSettings();
        return null;
      }

      // Get current position with high accuracy and force Android location manager
      debugPrint('Requesting current position...');
      _currentPosition = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        forceAndroidLocationManager: true,
        timeLimit: const Duration(seconds: 10),
      );
      _lastUpdate = DateTime.now();
      
      debugPrint('Position obtained: ${_currentPosition!.latitude}, ${_currentPosition!.longitude}');
      return _currentPosition;
    } catch (e) {
      debugPrint('Error getting location: $e');
      return null;
    }
  }
  
  /// Start listening to position updates
  Stream<Position> getPositionStream() {
    const LocationSettings locationSettings = LocationSettings(
      accuracy: LocationAccuracy.high,
      distanceFilter: 10, // Update every 10 meters
    );
    
    return Geolocator.getPositionStream(locationSettings: locationSettings);
  }
  
  /// Format coordinates as string
  String formatCoordinates(Position position) {
    return '${position.latitude.toStringAsFixed(6)}, ${position.longitude.toStringAsFixed(6)}';
  }
  
  /// Get last known position
  Position? get lastPosition => _currentPosition;
  
  /// Get last update time
  DateTime? get lastUpdateTime => _lastUpdate;
  
  /// Calculate distance between two positions in meters
  double calculateDistance(Position start, Position end) {
    return Geolocator.distanceBetween(
      start.latitude,
      start.longitude,
      end.latitude,
      end.longitude,
    );
  }
}
