# Driver Drowsiness Detection System - Complete Setup Instructions

**Last Updated:** March 5, 2026  
**Project:** Real-Time Driver Drowsiness Detection System  
**Version:** 1.0.0

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [System Requirements](#system-requirements)
3. [Initial Setup](#initial-setup)
4. [Backend Setup](#backend-setup)
5. [Mobile App Setup](#mobile-app-setup)
6. [Dataset Preparation](#dataset-preparation)
7. [Model Training](#model-training)
8. [Testing & Validation](#testing--validation)
9. [Integration](#integration)
10. [Troubleshooting](#troubleshooting)
11. [Additional Resources](#additional-resources)

---

## Project Structure

### Complete Directory Tree

```
driver-drowsiness-detection/
├── .git/                           # Git version control
├── .hypothesis/                    # Hypothesis testing cache
├── .kiro/                          # Kiro spec files
│   └── specs/
│       └── driver-drowsiness-detection/
│           ├── requirements.md     # System requirements
│           ├── design.md           # System design
│           └── tasks.md            # Implementation tasks
├── .vscode/                        # VS Code settings
├── backend/                        # Python ML backend
│   ├── .hypothesis/                # Backend test cache
│   ├── .pytest_cache/              # Pytest cache
│   ├── datasets/                   # Training datasets
│   │   ├── DDD/                    # Driver Drowsiness Dataset
│   │   │   ├── alert/              # Alert state images
│   │   │   └── drowsy/             # Drowsy state images
│   │   ├── NTHUDDD/                # NTHU dataset
│   │   │   ├── alert/
│   │   │   └── drowsy/
│   │   └── yawing/                 # Yawning dataset
│   │       ├── alert/
│   │       └── drowsy/
│   ├── models/                     # Trained ML models
│   │   ├── cnn_drowsiness.h5       # Keras model
│   │   └── cnn_drowsiness.tflite   # TensorFlow Lite model
│   ├── scripts/                    # Utility scripts
│   │   ├── train_cnn.py            # CNN training
│   │   ├── train_traditional_ml.py # Traditional ML training
│   │   ├── evaluate_model.py       # Model evaluation
│   │   ├── cross_dataset_evaluation.py
│   │   ├── proper_evaluation.py
│   │   ├── convert_to_tflite.py    # Model conversion
│   │   ├── organize_datasets.py    # Dataset organization
│   │   ├── validate_models.py      # Model validation
│   │   ├── validate_pipeline.py    # Pipeline validation
│   │   ├── benchmark_performance.py
│   │   ├── test_trained_model.py
│   │   ├── test_with_webcam.py     # Webcam testing
│   │   ├── test_http_server.py     # Server testing
│   │   ├── test_mobile_service.py  # Mobile service testing
│   │   └── record_dataset.py       # Dataset recording
│   ├── src/                        # Source code
│   │   ├── __init__.py
│   │   ├── http_server.py          # HTTP API server
│   │   ├── mobile_service.py       # Mobile integration service
│   │   ├── camera/                 # Camera management
│   │   │   ├── __init__.py
│   │   │   ├── camera_manager.py
│   │   │   └── frame_processor.py
│   │   ├── face_detection/         # Face detection
│   │   │   ├── __init__.py
│   │   │   ├── face_detector.py
│   │   │   └── landmark_detector.py
│   │   ├── feature_extraction/     # Feature extraction
│   │   │   ├── __init__.py
│   │   │   ├── ear_calculator.py   # Eye Aspect Ratio
│   │   │   └── mar_calculator.py   # Mouth Aspect Ratio
│   │   ├── ml_models/              # ML models
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py
│   │   │   ├── cnn_classifier.py
│   │   │   ├── feature_based_classifier.py
│   │   │   ├── tflite_model.py
│   │   │   └── device_capabilities.py
│   │   ├── decision_logic/         # Decision engine
│   │   │   ├── __init__.py
│   │   │   ├── decision_engine.py
│   │   │   └── alert_manager.py
│   │   ├── emergency/              # Emergency response
│   │   │   ├── __init__.py
│   │   │   ├── emergency_service.py
│   │   │   └── location_tracker.py
│   │   ├── privacy/                # Privacy features
│   │   │   ├── __init__.py
│   │   │   ├── privacy_manager.py
│   │   │   └── secure_data_handler.py
│   │   ├── monitoring/             # Performance monitoring
│   │   │   ├── __init__.py
│   │   │   ├── metrics_collector.py
│   │   │   └── feedback_manager.py
│   │   └── utils/                  # Utilities
│   │       ├── __init__.py
│   │       ├── data_preprocessing.py
│   │       ├── data_generator.py
│   │       ├── dataset_loader.py
│   │       └── adaptation_manager.py
│   ├── tests/                      # Test suite
│   │   ├── __init__.py
│   │   ├── test_face_detection_properties.py
│   │   ├── test_feature_extraction_properties.py
│   │   ├── test_ml_performance_properties.py
│   │   ├── test_alert_system_properties.py
│   │   ├── test_realtime_performance_properties.py
│   │   ├── test_privacy_properties.py
│   │   ├── test_emergency_properties.py
│   │   ├── test_monitoring_properties.py
│   │   └── test_robustness_properties.py
│   ├── venv/                       # Virtual environment
│   ├── .gitignore
│   ├── requirements.txt            # Python dependencies
│   ├── setup.py                    # Package setup
│   ├── pyproject.toml              # Project configuration
│   ├── pytest.ini                  # Pytest configuration
│   └── README.md                   # Backend documentation
├── mobile_app/                     # Flutter mobile app
│   ├── .dart_tool/                 # Dart build tools
│   ├── .idea/                      # IDE settings
│   ├── android/                    # Android platform
│   │   ├── app/
│   │   │   ├── src/
│   │   │   │   └── main/
│   │   │   │       ├── AndroidManifest.xml
│   │   │   │       ├── kotlin/
│   │   │   │       └── res/
│   │   │   └── build.gradle.kts
│   │   ├── gradle/
│   │   ├── build.gradle.kts
│   │   ├── settings.gradle.kts
│   │   └── gradle.properties
│   ├── ios/                        # iOS platform
│   │   ├── Runner/
│   │   │   ├── Info.plist
│   │   │   ├── AppDelegate.swift
│   │   │   └── Assets.xcassets/
│   │   ├── Runner.xcodeproj/
│   │   └── Runner.xcworkspace/
│   ├── lib/                        # Dart source code
│   │   ├── main.dart               # App entry point
│   │   ├── providers/              # State management
│   │   │   ├── drowsiness_provider.dart
│   │   │   └── settings_provider.dart
│   │   ├── screens/                # UI screens
│   │   │   ├── home_screen.dart
│   │   │   ├── monitoring_screen.dart
│   │   │   ├── settings_screen.dart
│   │   │   ├── emergency_contacts_screen.dart
│   │   │   └── data_management_screen.dart
│   │   ├── services/               # Business logic
│   │   │   ├── camera_service.dart
│   │   │   └── backend_service.dart
│   │   └── widgets/                # Reusable widgets
│   │       └── camera_preview_widget.dart
│   ├── test/                       # Test files
│   │   ├── app_properties_test.dart
│   │   └── widget_test.dart
│   ├── assets/                     # Static assets
│   │   ├── images/
│   │   └── sounds/
│   ├── build/                      # Build output
│   ├── .gitignore
│   ├── pubspec.yaml                # Flutter dependencies
│   ├── pubspec.lock
│   ├── analysis_options.yaml       # Dart analyzer config
│   └── README.md                   # Mobile app documentation

```

### Key Directories Explained

**Backend (`backend/`):**
- `src/`: Core Python source code organized by functionality
- `scripts/`: Standalone scripts for training, testing, and evaluation
- `models/`: Trained machine learning models (`.h5` and `.tflite`)
- `datasets/`: Training data organized by dataset and class
- `tests/`: Property-based tests for correctness validation
- `venv/`: Python virtual environment (created during setup)

**Mobile App (`mobile_app/`):**
- `lib/`: Dart/Flutter source code
  - `screens/`: UI screens for different app views
  - `services/`: Business logic and API integration
  - `providers/`: State management using Provider pattern
  - `widgets/`: Reusable UI components
- `android/`: Android-specific configuration and build files
- `ios/`: iOS-specific configuration and build files
- `test/`: Flutter widget and property-based tests
- `assets/`: Images, sounds, and other static resources

**Documentation:**
- Root-level `.md` files provide setup, training, and integration guides
- `.kiro/specs/`: Formal requirements, design, and task specifications
- `backend/` and `mobile_app/` READMEs: Component-specific documentation

---

## System Requirements

### Hardware Requirements

**Development Machine:**
- Processor: Quad-core 2.0GHz or better
- RAM: 8GB minimum, 16GB recommended
- Storage: 10GB free space
- Webcam: For testing (optional)

**Mobile Device (for testing):**
- Android 8.0+ or iOS 12.0+
- Front-facing camera (5MP minimum)
- 2GB RAM minimum

### Software Requirements

**Required:**
- Python 3.8 or higher
- Flutter SDK 3.0 or higher
- Git
- Code editor (VS Code, PyCharm, or Android Studio)

**Platform-Specific:**
- **Windows:** Visual Studio Build Tools
- **macOS:** Xcode Command Line Tools
- **Linux:** build-essential package

---

## Initial Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd driver-drowsiness-detection
```

### 2. Verify System Prerequisites

**Check Python version:**
```bash
python --version  # Should be 3.8+
```

**Check Flutter installation:**
```bash
flutter --version  # Should be 3.0+
flutter doctor     # Check for any issues
```

**Check Git:**
```bash
git --version
```

---

## Backend Setup

### Step 1: Navigate to Backend Directory

```bash
cd backend
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- OpenCV (computer vision)
- MediaPipe (face detection)
- TensorFlow (machine learning)
- NumPy (numerical computing)
- scikit-learn (traditional ML)
- pytest (testing)

### Step 5: Verify Installation

```bash
python -c "import cv2, mediapipe, tensorflow, numpy, sklearn; print('✓ All backend dependencies installed successfully!')"
```

### Step 6: Create Required Directories

```bash
mkdir -p models datasets/dataset1 datasets/dataset2 datasets/dataset3
```

---

## Mobile App Setup

### Step 1: Navigate to Mobile App Directory

```bash
cd mobile_app
```

### Step 2: Install Flutter Dependencies

```bash
flutter pub get
```

### Step 3: Verify Flutter Setup

```bash
flutter doctor -v
```

Fix any issues reported by `flutter doctor` before proceeding.

### Step 4: Platform-Specific Setup

**For Android:**
1. Install Android Studio
2. Install Android SDK (API level 28+)
3. Create an Android emulator or connect a physical device
4. Enable USB debugging on physical device

**For iOS (macOS only):**
1. Install Xcode from App Store
2. Install Xcode Command Line Tools: `xcode-select --install`
3. Open Xcode and accept license agreements
4. Install CocoaPods: `sudo gem install cocoapods`
5. Run: `cd ios && pod install && cd ..`

### Step 5: Configure Permissions

**Android** - Edit `android/app/src/main/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
```

**iOS** - Edit `ios/Runner/Info.plist`:
```xml
<key>NSCameraUsageDescription</key>
<string>Camera access is required for drowsiness detection</string>
<key>NSLocationWhenInUseUsageDescription</key>
<string>Location access is required for emergency features</string>
```

### Step 6: Test Run

```bash
# List available devices
flutter devices

# Run on connected device
flutter run

# Run on specific device
flutter run -d <device-id>
```

---

## Dataset Preparation

### Required Datasets

The system uses three drowsiness detection datasets:

1. **DDD (Driver Drowsiness Dataset)** - 41,793 images
2. **NTHUDDD** - 66,521 images
3. **YawDD (Yawning Detection Dataset)** - 5,119 images

### Dataset Structure

Organize datasets in the following structure:

```
backend/datasets/
├── DDD/
│   ├── alert/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── drowsy/
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
├── NTHUDDD/
│   ├── alert/
│   └── drowsy/
└── yawing/
    ├── alert/
    └── drowsy/
```

### Organize Datasets

If your datasets are not organized, use the organization script:

```bash
cd backend
python scripts/organize_datasets.py
```

### Verify Dataset Structure

```bash
python scripts/validate_datasets.py
```

---

## Model Training

### Option 1: Local Training (Recommended for Development)

#### Step 1: Activate Backend Environment

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Train CNN Model

```bash
python scripts/train_cnn.py --epochs 50 --batch-size 32
```

Training parameters:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)

Expected training time: 2-4 hours on GPU, 8-12 hours on CPU

#### Step 3: Train Traditional ML Models

```bash
python scripts/train_traditional_ml.py
```

This trains:
- Support Vector Machine (SVM)
- Random Forest Classifier
- Logistic Regression

#### Step 4: Validate Models

```bash
python scripts/validate_models.py
```

### Option 2: Cloud Training (Kaggle)

For faster training with GPU acceleration, see [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md)

### Option 3: Cloud Training (Google Colab)

For free GPU training, see [GOOGLE_COLAB_TRAINING_GUIDE.md](GOOGLE_COLAB_TRAINING_GUIDE.md)

### Model Evaluation

After training, evaluate model performance:

```bash
# Evaluate on test set
python scripts/evaluate_model.py --test-size 1000

# Cross-dataset evaluation
python scripts/cross_dataset_evaluation.py --test-size 2000

# Proper evaluation with all metrics
python scripts/proper_evaluation.py --total-per-class 5000
```

### Convert Models to TensorFlow Lite

For mobile deployment:

```bash
python scripts/convert_to_tflite.py
```

This creates optimized `.tflite` models in `backend/models/` directory.

---

## Testing & Validation

### Backend Testing

#### Run All Tests

```bash
cd backend
pytest
```

#### Run Specific Test Suites

```bash
# Face detection tests
pytest tests/test_face_detection_properties.py

# Feature extraction tests
pytest tests/test_feature_extraction_properties.py

# ML performance tests
pytest tests/test_ml_performance_properties.py

# Privacy tests
pytest tests/test_privacy_properties.py
```

#### Run with Coverage

```bash
pytest --cov=src --cov-report=html tests/
```

View coverage report: `open htmlcov/index.html`

### Mobile App Testing

#### Run Flutter Tests

```bash
cd mobile_app
flutter test
```

#### Run Property-Based Tests

```bash
flutter test test/app_properties_test.dart
```

### Integration Testing

#### Test Backend HTTP Server

```bash
cd backend
python scripts/test_http_server.py
```

#### Test Mobile Service

```bash
python scripts/test_mobile_service.py
```

#### Test with Webcam

```bash
python scripts/test_with_webcam.py
```

---

## Integration

### Backend-Mobile Integration

#### Step 1: Start Backend Server

```bash
cd backend
python src/http_server.py
```

Server runs on `http://localhost:5000`

#### Step 2: Configure Mobile App

Edit `mobile_app/lib/services/backend_service.dart`:

```dart
static const String baseUrl = 'http://YOUR_IP:5000';
```

Replace `YOUR_IP` with your machine's IP address (not localhost).

#### Step 3: Test Connection

```bash
cd mobile_app
flutter run
```

For detailed integration instructions, see [INTEGRATION_QUICKSTART.md](INTEGRATION_QUICKSTART.md)

### Emulator Camera Configuration

For testing with Android emulator, see [configure_emulator_camera.md](configure_emulator_camera.md)

---

## Troubleshooting

### Common Issues

#### Python Dependencies Installation Fails

**Solution:**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

#### TensorFlow Installation Issues

**Windows:**
```bash
pip install tensorflow-cpu  # If no GPU
```

**macOS (M1/M2):**
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

#### Flutter Doctor Issues

**Android License Not Accepted:**
```bash
flutter doctor --android-licenses
```

**Xcode Issues (macOS):**
```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
```

#### MediaPipe Installation Fails

```bash
pip install mediapipe --no-deps
pip install opencv-python numpy protobuf
```

#### Camera Permission Denied

**Android:**
- Go to Settings → Apps → Your App → Permissions
- Enable Camera permission

**iOS:**
- Go to Settings → Privacy → Camera
- Enable for your app

#### Model Loading Errors

Ensure models are in correct location:
```bash
ls backend/models/
# Should show: cnn_drowsiness.h5, cnn_drowsiness.tflite
```

#### Backend Server Connection Failed

1. Check firewall settings
2. Verify IP address is correct
3. Ensure backend server is running
4. Test with: `curl http://YOUR_IP:5000/health`

### Performance Issues

#### Slow Training

- Use GPU acceleration (CUDA)
- Reduce batch size
- Use cloud training (Kaggle/Colab)

#### Low FPS on Mobile

- Reduce input image resolution
- Use TensorFlow Lite quantized models
- Optimize frame processing pipeline

#### High Battery Consumption

- Reduce frame processing rate
- Implement adaptive processing
- Use device-specific optimizations

---

## Additional Resources

### Documentation

- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [Requirements](.kiro/specs/driver-drowsiness-detection/requirements.md) - System requirements
- [Design](.kiro/specs/driver-drowsiness-detection/design.md) - System design
- [Tasks](.kiro/specs/driver-drowsiness-detection/tasks.md) - Implementation tasks

### Training Guides

- [CNN_TRAINING_GUIDE.md](CNN_TRAINING_GUIDE.md) - Detailed CNN training
- [TRAINING_QUICK_START.md](TRAINING_QUICK_START.md) - Quick training guide
- [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md) - Kaggle setup
- [GOOGLE_COLAB_TRAINING_GUIDE.md](GOOGLE_COLAB_TRAINING_GUIDE.md) - Colab setup

### Integration Guides

- [INTEGRATION_QUICKSTART.md](INTEGRATION_QUICKSTART.md) - Integration overview
- [FLUTTER_BACKEND_INTEGRATION.md](FLUTTER_BACKEND_INTEGRATION.md) - Flutter-backend integration
- [HOW_TO_CONNECT_FLUTTER.md](HOW_TO_CONNECT_FLUTTER.md) - Connection instructions

### Architecture

- [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) - System architecture

### Evaluation Reports

- [backend/MODEL_EVALUATION_REPORT.md](backend/MODEL_EVALUATION_REPORT.md) - Model performance
- [backend/FINAL_EVALUATION_REPORT.md](backend/FINAL_EVALUATION_REPORT.md) - Final evaluation

---

## Development Workflow

### Recommended Development Order

1. **Backend Development** (Weeks 1-3)
   - Implement face detection (Task 2)
   - Implement feature extraction (Task 3)
   - Train ML models (Task 4)
   - Implement decision logic (Task 5)

2. **Testing & Validation** (Week 4)
   - Write property-based tests
   - Validate model performance
   - Benchmark real-time performance

3. **Mobile App Development** (Weeks 5-6)
   - Implement UI screens (Task 11)
   - Integrate camera service
   - Implement alert system

4. **Integration** (Week 7)
   - Connect mobile app to backend
   - End-to-end testing
   - Performance optimization

5. **Deployment** (Week 8)
   - Build production APK/IPA
   - Final testing
   - Documentation

---

## Quick Reference Commands

### Backend

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Train models
python scripts/train_cnn.py

# Run tests
pytest

# Start server
python src/http_server.py
```

### Mobile App

```bash
# Get dependencies
flutter pub get

# Run app
flutter run

# Build APK
flutter build apk

# Run tests
flutter test
```

---

## Support & Contact

For issues, questions, or contributions:

1. Check existing documentation
2. Review troubleshooting section
3. Check GitHub issues
4. Contact project maintainer

---

**Setup Complete!** You're ready to develop and test the Driver Drowsiness Detection System.
