import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Provider for app settings and user preferences
/// 
/// Manages:
/// - Alert settings (sensitivity, types)
/// - Emergency contacts
/// - Privacy settings
/// - User preferences
class SettingsProvider with ChangeNotifier {
  // Alert settings
  double _alertSensitivity = 0.7;
  bool _visualAlerts = true;
  bool _audioAlerts = true;
  bool _hapticAlerts = true;
  
  // Emergency settings
  List<EmergencyContact> _emergencyContacts = [];
  bool _emergencyResponseEnabled = false;
  
  // Privacy settings
  bool _dataCollectionEnabled = false;
  bool _locationTrackingEnabled = false;
  
  // Getters
  double get alertSensitivity => _alertSensitivity;
  bool get visualAlerts => _visualAlerts;
  bool get audioAlerts => _audioAlerts;
  bool get hapticAlerts => _hapticAlerts;
  List<EmergencyContact> get emergencyContacts => _emergencyContacts;
  bool get emergencyResponseEnabled => _emergencyResponseEnabled;
  bool get dataCollectionEnabled => _dataCollectionEnabled;
  bool get locationTrackingEnabled => _locationTrackingEnabled;
  
  /// Load settings from local storage
  Future<void> loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    
    _alertSensitivity = prefs.getDouble('alert_sensitivity') ?? 0.7;
    _visualAlerts = prefs.getBool('visual_alerts') ?? true;
    _audioAlerts = prefs.getBool('audio_alerts') ?? true;
    _hapticAlerts = prefs.getBool('haptic_alerts') ?? true;
    _emergencyResponseEnabled = prefs.getBool('emergency_enabled') ?? false;
    _dataCollectionEnabled = prefs.getBool('data_collection') ?? false;
    _locationTrackingEnabled = prefs.getBool('location_tracking') ?? false;
    
    notifyListeners();
  }
  
  /// Save settings to local storage
  Future<void> saveSettings() async {
    final prefs = await SharedPreferences.getInstance();
    
    await prefs.setDouble('alert_sensitivity', _alertSensitivity);
    await prefs.setBool('visual_alerts', _visualAlerts);
    await prefs.setBool('audio_alerts', _audioAlerts);
    await prefs.setBool('haptic_alerts', _hapticAlerts);
    await prefs.setBool('emergency_enabled', _emergencyResponseEnabled);
    await prefs.setBool('data_collection', _dataCollectionEnabled);
    await prefs.setBool('location_tracking', _locationTrackingEnabled);
  }
  
  /// Update alert sensitivity
  void setAlertSensitivity(double value) {
    _alertSensitivity = value;
    notifyListeners();
    saveSettings();
  }
  
  /// Toggle visual alerts
  void toggleVisualAlerts(bool value) {
    _visualAlerts = value;
    notifyListeners();
    saveSettings();
  }
  
  /// Toggle audio alerts
  void toggleAudioAlerts(bool value) {
    _audioAlerts = value;
    notifyListeners();
    saveSettings();
  }
  
  /// Toggle haptic alerts
  void toggleHapticAlerts(bool value) {
    _hapticAlerts = value;
    notifyListeners();
    saveSettings();
  }
  
  /// Toggle emergency response
  void toggleEmergencyResponse(bool value) {
    _emergencyResponseEnabled = value;
    notifyListeners();
    saveSettings();
  }
  
  /// Toggle data collection
  void toggleDataCollection(bool value) {
    _dataCollectionEnabled = value;
    notifyListeners();
    saveSettings();
  }
  
  /// Toggle location tracking
  void toggleLocationTracking(bool value) {
    _locationTrackingEnabled = value;
    notifyListeners();
    saveSettings();
  }
  
  /// Add emergency contact
  void addEmergencyContact(EmergencyContact contact) {
    _emergencyContacts.add(contact);
    notifyListeners();
  }
  
  /// Remove emergency contact
  void removeEmergencyContact(int index) {
    if (index >= 0 && index < _emergencyContacts.length) {
      _emergencyContacts.removeAt(index);
      notifyListeners();
    }
  }
}

/// Emergency contact model
class EmergencyContact {
  final String name;
  final String phoneNumber;
  final String relationship;
  
  EmergencyContact({
    required this.name,
    required this.phoneNumber,
    required this.relationship,
  });
  
  Map<String, dynamic> toJson() => {
    'name': name,
    'phoneNumber': phoneNumber,
    'relationship': relationship,
  };
  
  factory EmergencyContact.fromJson(Map<String, dynamic> json) {
    return EmergencyContact(
      name: json['name'],
      phoneNumber: json['phoneNumber'],
      relationship: json['relationship'],
    );
  }
}
