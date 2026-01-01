import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/home_screen.dart';
import 'screens/monitoring_screen.dart';
import 'screens/settings_screen.dart';
import 'providers/drowsiness_provider.dart';
import 'providers/settings_provider.dart';

void main() {
  runApp(const DrowsinessDetectionApp());
}

class DrowsinessDetectionApp extends StatelessWidget {
  const DrowsinessDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => DrowsinessProvider()),
        ChangeNotifierProvider(create: (_) => SettingsProvider()),
      ],
      child: MaterialApp(
        title: 'Drowsiness Detection',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          useMaterial3: true,
          brightness: Brightness.light,
        ),
        darkTheme: ThemeData(
          primarySwatch: Colors.blue,
          useMaterial3: true,
          brightness: Brightness.dark,
        ),
        themeMode: ThemeMode.system,
        home: const HomeScreen(),
        routes: {
          '/home': (context) => const HomeScreen(),
          '/monitoring': (context) => const MonitoringScreen(),
          '/settings': (context) => const SettingsScreen(),
        },
      ),
    );
  }
}
