# Drowsiness Detection Mobile App

Flutter-based cross-platform mobile application for real-time driver drowsiness detection.

## Features

- Real-time camera monitoring
- Face detection and tracking
- Drowsiness alerts (visual, audio, haptic)
- Customizable sensitivity settings
- Emergency contact configuration
- Privacy-first local processing

## Getting Started

### Prerequisites

- Flutter SDK 3.0+
- Dart 2.17+
- Android Studio / Xcode for platform-specific builds

### Installation

```bash
flutter pub get
```

### Running

```bash
# Run on connected device
flutter run

# Run on specific platform
flutter run -d android
flutter run -d ios
```

### Building

```bash
# Android APK
flutter build apk

# iOS
flutter build ios
```

## Project Structure

```
lib/
├── main.dart              # App entry point
├── screens/               # UI screens
├── widgets/               # Reusable widgets
├── services/              # Business logic
├── models/                # Data models
└── utils/                 # Utilities
```

## Platform Permissions

### Android (android/app/src/main/AndroidManifest.xml)
- Camera
- Location (for emergency features)

### iOS (ios/Runner/Info.plist)
- Camera
- Location (for emergency features)
