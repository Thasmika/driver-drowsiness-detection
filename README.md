# Real-Time Driver Drowsiness Detection System

A non-intrusive, smartphone-based solution that uses facial analysis and machine learning to detect driver fatigue in real-time.

## Overview

The Driver Drowsiness Detection (DDD) system monitors facial expressions, eye movements, yawning, blinking patterns, and head nodding to identify drowsiness indicators and alert drivers to take breaks, improving road safety through affordable and accessible technology.

## Features

- **Real-time Face Detection**: Continuous monitoring at 15+ FPS
- **Drowsiness Classification**: 85%+ accuracy using ML models
- **Multi-modal Alerts**: Visual, audio, and haptic feedback
- **Cross-platform**: Android and iOS support via Flutter
- **Privacy-first**: All processing happens locally on device
- **Emergency Response**: Optional GPS tracking and emergency contacts

## Project Structure

```
driver-drowsiness-detection/
├── backend/                 # Python ML processing backend
│   ├── src/
│   │   ├── face_detection/
│   │   ├── feature_extraction/
│   │   ├── ml_models/
│   │   ├── decision_logic/
│   │   └── utils/
│   ├── tests/
│   ├── models/             # Trained ML models
│   └── requirements.txt
├── mobile_app/             # Flutter mobile application
│   ├── lib/
│   ├── android/
│   ├── ios/
│   └── pubspec.yaml
├── datasets/               # Training datasets (not included)
└── docs/                   # Documentation
```

## Requirements

### Backend (Python)
- Python 3.8+
- OpenCV
- MediaPipe
- TensorFlow Lite
- NumPy
- scikit-learn

### Mobile App (Flutter)
- Flutter 3.0+
- Dart 2.17+

## Installation

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Mobile App Setup

```bash
cd mobile_app
flutter pub get
flutter run
```

## Development

See the [tasks.md](.kiro/specs/driver-drowsiness-detection/tasks.md) file for the implementation plan.

## License

MIT License
