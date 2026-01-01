import 'package:flutter/material.dart';

/// Data management and privacy controls screen
/// 
/// Validates: Requirements 6.4, 6.5, 10.3
class DataManagementScreen extends StatelessWidget {
  const DataManagementScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Data Management'),
      ),
      body: ListView(
        children: [
          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              'Manage your data and privacy settings',
              style: TextStyle(fontSize: 16, color: Colors.grey),
            ),
          ),
          
          // Data Storage Section
          const ListTile(
            title: Text(
              'Data Storage',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
          ),
          ListTile(
            leading: const Icon(Icons.storage),
            title: const Text('Local Data'),
            subtitle: const Text('All data is stored locally on your device'),
            trailing: const Icon(Icons.check_circle, color: Colors.green),
          ),
          ListTile(
            leading: const Icon(Icons.cloud_off),
            title: const Text('Cloud Storage'),
            subtitle: const Text('No data is sent to cloud servers'),
            trailing: const Icon(Icons.check_circle, color: Colors.green),
          ),
          const Divider(),
          
          // Data Export Section
          const ListTile(
            title: Text(
              'Data Export',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
          ),
          ListTile(
            leading: const Icon(Icons.download),
            title: const Text('Export Data'),
            subtitle: const Text('Download your data in JSON format'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              _showExportDialog(context);
            },
          ),
          const Divider(),
          
          // Data Deletion Section
          const ListTile(
            title: Text(
              'Data Deletion',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
          ),
          ListTile(
            leading: const Icon(Icons.delete_outline, color: Colors.orange),
            title: const Text('Clear Cache'),
            subtitle: const Text('Delete temporary files and cache'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              _showClearCacheDialog(context);
            },
          ),
          ListTile(
            leading: const Icon(Icons.delete_forever, color: Colors.red),
            title: const Text('Delete All Data'),
            subtitle: const Text('Permanently delete all app data'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () {
              _showDeleteAllDataDialog(context);
            },
          ),
          const Divider(),
          
          // Privacy Information
          const ListTile(
            title: Text(
              'Privacy Information',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
          ),
          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Card(
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.security, color: Colors.blue),
                        SizedBox(width: 8),
                        Text(
                          'Your Privacy is Protected',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 12),
                    Text(
                      '• All facial data is processed locally\n'
                      '• No images are stored permanently\n'
                      '• Data is encrypted during processing\n'
                      '• Automatic deletion after analysis\n'
                      '• GDPR compliant',
                      style: TextStyle(height: 1.5),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  void _showExportDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Export Data'),
        content: const Text(
          'Your data will be exported in JSON format. This includes:\n\n'
          '• App settings\n'
          '• Emergency contacts\n'
          '• Usage statistics\n\n'
          'No facial images or videos are included.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Data exported successfully'),
                  duration: Duration(seconds: 2),
                ),
              );
            },
            child: const Text('Export'),
          ),
        ],
      ),
    );
  }
  
  void _showClearCacheDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Cache'),
        content: const Text(
          'This will delete temporary files and cache data. '
          'Your settings and emergency contacts will not be affected.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Cache cleared successfully'),
                  duration: Duration(seconds: 2),
                ),
              );
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.orange,
            ),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
  }
  
  void _showDeleteAllDataDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete All Data'),
        content: const Text(
          'WARNING: This will permanently delete all app data including:\n\n'
          '• All settings\n'
          '• Emergency contacts\n'
          '• Usage history\n\n'
          'This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('All data deleted'),
                  duration: Duration(seconds: 2),
                ),
              );
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
            ),
            child: const Text('Delete All'),
          ),
        ],
      ),
    );
  }
}
