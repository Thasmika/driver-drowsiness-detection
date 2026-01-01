import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/settings_provider.dart';
import 'emergency_contacts_screen.dart';
import 'data_management_screen.dart';

/// Settings screen for app configuration
/// 
/// Validates: Requirements 3.4, 7.4, 10.3
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
      ),
      body: Consumer<SettingsProvider>(
        builder: (context, settings, child) {
          return ListView(
            children: [
              // Alert Settings Section
              const ListTile(
                title: Text(
                  'Alert Settings',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ),
              ListTile(
                title: const Text('Alert Sensitivity'),
                subtitle: Text('${(settings.alertSensitivity * 100).toStringAsFixed(0)}%'),
                trailing: SizedBox(
                  width: 200,
                  child: Slider(
                    value: settings.alertSensitivity,
                    min: 0.5,
                    max: 0.9,
                    divisions: 8,
                    label: '${(settings.alertSensitivity * 100).toStringAsFixed(0)}%',
                    onChanged: (value) {
                      settings.setAlertSensitivity(value);
                    },
                  ),
                ),
              ),
              SwitchListTile(
                title: const Text('Visual Alerts'),
                subtitle: const Text('Show on-screen alerts'),
                value: settings.visualAlerts,
                onChanged: (value) {
                  settings.toggleVisualAlerts(value);
                },
              ),
              SwitchListTile(
                title: const Text('Audio Alerts'),
                subtitle: const Text('Play alert sounds'),
                value: settings.audioAlerts,
                onChanged: (value) {
                  settings.toggleAudioAlerts(value);
                },
              ),
              SwitchListTile(
                title: const Text('Haptic Alerts'),
                subtitle: const Text('Vibrate on alerts'),
                value: settings.hapticAlerts,
                onChanged: (value) {
                  settings.toggleHapticAlerts(value);
                },
              ),
              const Divider(),
              
              // Emergency Settings Section
              const ListTile(
                title: Text(
                  'Emergency Response',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ),
              SwitchListTile(
                title: const Text('Emergency Response'),
                subtitle: const Text('Enable emergency contact notification'),
                value: settings.emergencyResponseEnabled,
                onChanged: (value) {
                  settings.toggleEmergencyResponse(value);
                },
              ),
              ListTile(
                title: const Text('Emergency Contacts'),
                subtitle: Text('${settings.emergencyContacts.length} contacts'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const EmergencyContactsScreen(),
                    ),
                  );
                },
              ),
              const Divider(),
              
              // Privacy Settings Section
              const ListTile(
                title: Text(
                  'Privacy & Data',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ),
              SwitchListTile(
                title: const Text('Data Collection'),
                subtitle: const Text('Allow anonymous usage data collection'),
                value: settings.dataCollectionEnabled,
                onChanged: (value) {
                  settings.toggleDataCollection(value);
                },
              ),
              SwitchListTile(
                title: const Text('Location Tracking'),
                subtitle: const Text('Enable GPS for emergency response'),
                value: settings.locationTrackingEnabled,
                onChanged: (value) {
                  settings.toggleLocationTracking(value);
                },
              ),
              ListTile(
                title: const Text('Data Management'),
                subtitle: const Text('Export, clear, or delete your data'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const DataManagementScreen(),
                    ),
                  );
                },
              ),
              ListTile(
                title: const Text('Privacy Policy'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () {
                  _showPrivacyPolicy(context);
                },
              ),
              const Divider(),
              
              // About Section
              const ListTile(
                title: Text(
                  'About',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ),
              const ListTile(
                title: Text('Version'),
                subtitle: Text('0.1.0+1'),
              ),
              ListTile(
                title: const Text('About App'),
                trailing: const Icon(Icons.chevron_right),
                onTap: () {
                  _showAboutDialog(context);
                },
              ),
            ],
          );
        },
      ),
    );
  }
  
  void _showPrivacyPolicy(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Privacy Policy'),
        content: const SingleChildScrollView(
          child: Text(
            'Driver Drowsiness Detection Privacy Policy\n\n'
            '1. Data Processing: All facial data is processed locally on your device. '
            'No images or videos are transmitted to external servers.\n\n'
            '2. Data Storage: Temporary data is encrypted and automatically deleted '
            'after processing.\n\n'
            '3. Location Data: GPS location is only collected when emergency response '
            'is enabled and only transmitted in emergency situations.\n\n'
            '4. User Control: You have full control over data collection and can '
            'delete all data at any time.\n\n'
            '5. GDPR Compliance: This app complies with GDPR and other data '
            'protection regulations.',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
  
  void _showAboutDialog(BuildContext context) {
    showAboutDialog(
      context: context,
      applicationName: 'Drowsiness Detection',
      applicationVersion: '0.1.0+1',
      applicationIcon: const Icon(Icons.remove_red_eye, size: 48),
      children: [
        const Text(
          'Real-Time Driver Drowsiness Detection System\n\n'
          'This app uses advanced machine learning to detect signs of '
          'drowsiness and alert drivers to take breaks, improving road safety.\n\n'
          'Features:\n'
          '• Real-time face detection\n'
          '• Drowsiness classification\n'
          '• Multi-modal alerts\n'
          '• Emergency response\n'
          '• Privacy-first design',
        ),
      ],
    );
  }
}
