import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:math' as math;
import '../providers/drowsiness_provider.dart';

/// Home screen with navigation to main features
/// 
/// Validates: Requirements 4.1, 4.2, 10.1
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _rotateController;
  late Animation<double> _pulseAnimation;
  late Animation<double> _rotateAnimation;

  @override
  void initState() {
    super.initState();
    
    // Pulse animation for the eye icon
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);
    
    _pulseAnimation = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
    
    // Rotate animation for background decoration
    _rotateController = AnimationController(
      duration: const Duration(seconds: 20),
      vsync: this,
    )..repeat();
    
    _rotateAnimation = Tween<double>(begin: 0, end: 2 * math.pi).animate(_rotateController);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _rotateController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              const Color(0xFF1A237E), // Deep blue
              const Color(0xFF0D47A1), // Blue
              const Color(0xFF01579B), // Light blue
              const Color(0xFF006064), // Cyan
            ],
            stops: const [0.0, 0.3, 0.6, 1.0],
          ),
        ),
        child: SafeArea(
          child: Stack(
            children: [
              // Animated background circles
              AnimatedBuilder(
                animation: _rotateAnimation,
                builder: (context, child) {
                  return Positioned(
                    top: -100,
                    right: -100,
                    child: Transform.rotate(
                      angle: _rotateAnimation.value,
                      child: Container(
                        width: 300,
                        height: 300,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: RadialGradient(
                            colors: [
                              Colors.white.withOpacity(0.1),
                              Colors.transparent,
                            ],
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
              AnimatedBuilder(
                animation: _rotateAnimation,
                builder: (context, child) {
                  return Positioned(
                    bottom: -150,
                    left: -150,
                    child: Transform.rotate(
                      angle: -_rotateAnimation.value,
                      child: Container(
                        width: 400,
                        height: 400,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: RadialGradient(
                            colors: [
                              Colors.cyan.withOpacity(0.15),
                              Colors.transparent,
                            ],
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
              
              // Main content
              Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  children: [
                    // Header with settings button
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          'SafeDrive',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                            letterSpacing: 1.2,
                          ),
                        ),
                        Container(
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: Colors.white.withOpacity(0.3),
                              width: 1,
                            ),
                          ),
                          child: IconButton(
                            icon: const Icon(Icons.settings, color: Colors.white),
                            onPressed: () {
                              Navigator.pushNamed(context, '/settings');
                            },
                          ),
                        ),
                      ],
                    ),
                    
                    const Spacer(),
                    
                    // Clean animated car icon with subtle effects
                    AnimatedBuilder(
                      animation: _pulseAnimation,
                      builder: (context, child) {
                        return Transform.scale(
                          scale: _pulseAnimation.value,
                          child: Container(
                            width: 180,
                            height: 180,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              gradient: RadialGradient(
                                colors: [
                                  Colors.cyan.withOpacity(0.25),
                                  Colors.blue.withOpacity(0.1),
                                  Colors.transparent,
                                ],
                              ),
                            ),
                            child: Center(
                              child: Container(
                                width: 130,
                                height: 130,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: Colors.white.withOpacity(0.12),
                                  border: Border.all(
                                    color: Colors.white.withOpacity(0.25),
                                    width: 2,
                                  ),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.cyan.withOpacity(0.25),
                                      blurRadius: 25,
                                      spreadRadius: 8,
                                    ),
                                  ],
                                ),
                                child: Stack(
                                  alignment: Alignment.center,
                                  children: [
                                    // Main car icon
                                    const Icon(
                                      Icons.directions_car_rounded,
                                      size: 65,
                                      color: Colors.white,
                                    ),
                                    // Scanning radar effect
                                    AnimatedBuilder(
                                      animation: _rotateController,
                                      builder: (context, child) {
                                        return Transform.rotate(
                                          angle: _rotateController.value * 2 * math.pi,
                                          child: Container(
                                            width: 100,
                                            height: 100,
                                            decoration: BoxDecoration(
                                              shape: BoxShape.circle,
                                              border: Border.all(
                                                color: Colors.cyanAccent.withOpacity(0.3),
                                                width: 2,
                                              ),
                                            ),
                                            child: CustomPaint(
                                              painter: RadarPainter(
                                                color: Colors.cyanAccent.withOpacity(0.4),
                                              ),
                                            ),
                                          ),
                                        );
                                      },
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                    
                    const SizedBox(height: 32),
                    
                    // App title with gradient text effect
                    ShaderMask(
                      shaderCallback: (bounds) => const LinearGradient(
                        colors: [Colors.white, Colors.cyanAccent],
                      ).createShader(bounds),
                      child: const Text(
                        'Driver Drowsiness\nDetection',
                        style: TextStyle(
                          fontSize: 32,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          height: 1.2,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                    
                    const SizedBox(height: 16),
                    
                    // App description
                    Text(
                      'AI-powered real-time monitoring\nto keep you safe on the road',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.white.withOpacity(0.8),
                        height: 1.5,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    
                    const Spacer(),
                    
                    // Status card with glassmorphism effect
                    Consumer<DrowsinessProvider>(
                      builder: (context, provider, child) {
                        return Container(
                          padding: const EdgeInsets.all(20),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: Colors.white.withOpacity(0.3),
                              width: 1.5,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.1),
                                blurRadius: 20,
                                offset: const Offset(0, 10),
                              ),
                            ],
                          ),
                          child: Column(
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  Row(
                                    children: [
                                      Container(
                                        width: 12,
                                        height: 12,
                                        decoration: BoxDecoration(
                                          shape: BoxShape.circle,
                                          color: provider.isMonitoring
                                              ? Colors.greenAccent
                                              : Colors.grey,
                                          boxShadow: provider.isMonitoring
                                              ? [
                                                  BoxShadow(
                                                    color: Colors.greenAccent.withOpacity(0.5),
                                                    blurRadius: 8,
                                                    spreadRadius: 2,
                                                  ),
                                                ]
                                              : [],
                                        ),
                                      ),
                                      const SizedBox(width: 12),
                                      const Text(
                                        'System Status',
                                        style: TextStyle(
                                          color: Colors.white,
                                          fontSize: 16,
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                    ],
                                  ),
                                  Text(
                                    provider.isMonitoring ? 'Active' : 'Inactive',
                                    style: TextStyle(
                                      color: provider.isMonitoring
                                          ? Colors.greenAccent
                                          : Colors.white70,
                                      fontWeight: FontWeight.bold,
                                      fontSize: 16,
                                    ),
                                  ),
                                ],
                              ),
                              if (provider.isMonitoring) ...[
                                const SizedBox(height: 16),
                                const Divider(color: Colors.white24),
                                const SizedBox(height: 16),
                                Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    const Text(
                                      'Drowsiness Level',
                                      style: TextStyle(
                                        color: Colors.white70,
                                        fontSize: 14,
                                      ),
                                    ),
                                    Text(
                                      '${(provider.drowsinessScore * 100).toStringAsFixed(0)}%',
                                      style: TextStyle(
                                        color: _getScoreColor(provider.drowsinessScore),
                                        fontWeight: FontWeight.bold,
                                        fontSize: 18,
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ],
                          ),
                        );
                      },
                    ),
                    
                    const SizedBox(height: 24),
                    
                    // Start monitoring button with gradient
                    Container(
                      width: double.infinity,
                      height: 60,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [
                            Color(0xFF00E5FF),
                            Color(0xFF00B8D4),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(30),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFF00E5FF).withOpacity(0.4),
                            blurRadius: 20,
                            offset: const Offset(0, 10),
                          ),
                        ],
                      ),
                      child: Material(
                        color: Colors.transparent,
                        child: InkWell(
                          onTap: () {
                            Navigator.pushNamed(context, '/monitoring');
                          },
                          borderRadius: BorderRadius.circular(30),
                          child: const Center(
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.play_circle_filled,
                                  color: Colors.white,
                                  size: 28,
                                ),
                                SizedBox(width: 12),
                                Text(
                                  'Start Monitoring',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white,
                                    letterSpacing: 0.5,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                    
                    const SizedBox(height: 16),
                    
                    // Settings button with glassmorphism
                    Container(
                      width: double.infinity,
                      height: 60,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(30),
                        border: Border.all(
                          color: Colors.white.withOpacity(0.3),
                          width: 1.5,
                        ),
                      ),
                      child: Material(
                        color: Colors.transparent,
                        child: InkWell(
                          onTap: () {
                            Navigator.pushNamed(context, '/settings');
                          },
                          borderRadius: BorderRadius.circular(30),
                          child: Center(
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.tune,
                                  color: Colors.white.withOpacity(0.9),
                                  size: 24,
                                ),
                                const SizedBox(width: 12),
                                Text(
                                  'Settings',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.w600,
                                    color: Colors.white.withOpacity(0.9),
                                    letterSpacing: 0.5,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                    
                    const SizedBox(height: 24),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Color _getScoreColor(double score) {
    if (score >= 0.8) return const Color(0xFFFF5252);
    if (score >= 0.6) return const Color(0xFFFFAB40);
    return const Color(0xFF69F0AE);
  }
}

/// Custom painter for radar scanning effect
class RadarPainter extends CustomPainter {
  final Color color;

  RadarPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final path = Path();
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2;

    // Draw a radar sweep (pie slice)
    path.moveTo(center.dx, center.dy);
    path.arcTo(
      Rect.fromCircle(center: center, radius: radius),
      -math.pi / 2, // Start angle (top)
      math.pi / 3, // Sweep angle (60 degrees)
      false,
    );
    path.close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
