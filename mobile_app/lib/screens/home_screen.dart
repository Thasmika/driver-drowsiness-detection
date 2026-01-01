import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/drowsiness_provider.dart';

/// Home screen with navigation to main features
/// 
/// Validates: Requirements 4.1, 4.2, 10.1
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Drowsiness Detection'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.pushNamed(context, '/settings');
            },
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // App logo/icon
              const Icon(
                Icons.remove_red_eye,
                size: 100,
                color: Colors.blue,
              ),
              const SizedBox(height: 24),
              
              // App title
              const Text(
                'Driver Drowsiness Detection',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              
              // App description
              const Text(
                'Real-time monitoring to keep you safe on the road',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 48),
              
              // Start monitoring button
              ElevatedButton(
                onPressed: () {
                  Navigator.pushNamed(context, '/monitoring');
                },
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text(
                  'Start Monitoring',
                  style: TextStyle(fontSize: 18),
                ),
              ),
              const SizedBox(height: 16),
              
              // Settings button
              OutlinedButton(
                onPressed: () {
                  Navigator.pushNamed(context, '/settings');
                },
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text(
                  'Settings',
                  style: TextStyle(fontSize: 18),
                ),
              ),
              const SizedBox(height: 48),
              
              // Status indicator
              Consumer<DrowsinessProvider>(
                builder: (context, provider, child) {
                  return Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              const Text('Status:'),
                              Text(
                                provider.isMonitoring ? 'Active' : 'Inactive',
                                style: TextStyle(
                                  color: provider.isMonitoring
                                      ? Colors.green
                                      : Colors.grey,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                          if (provider.isMonitoring) ...[
                            const SizedBox(height: 8),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                const Text('Drowsiness:'),
                                Text(
                                  '${(provider.drowsinessScore * 100).toStringAsFixed(0)}%',
                                  style: TextStyle(
                                    color: _getScoreColor(provider.drowsinessScore),
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ],
                      ),
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Color _getScoreColor(double score) {
    if (score >= 0.8) return Colors.red;
    if (score >= 0.6) return Colors.orange;
    return Colors.green;
  }
}
