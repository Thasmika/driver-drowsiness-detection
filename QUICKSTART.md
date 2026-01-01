# Quick Start Guide

## Prerequisites

- Python 3.8+ installed
- Flutter SDK 3.0+ installed (for mobile app)
- Git installed

## Backend Setup (5 minutes)

### 1. Navigate to backend directory

```bash
cd backend
```

### 2. Create and activate virtual environment

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

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import cv2, mediapipe, tensorflow; print('All dependencies installed successfully!')"
```

## Mobile App Setup (5 minutes)

### 1. Navigate to mobile app directory

```bash
cd mobile_app
```

### 2. Get Flutter dependencies

```bash
flutter pub get
```

### 3. Check Flutter setup

```bash
flutter doctor
```

### 4. Run the app

```bash
# Connect your device or start an emulator, then:
flutter run
```

## Dataset Preparation

Place your 3 drowsiness detection datasets in:

```
backend/datasets/
├── dataset1/
├── dataset2/
└── dataset3/
```

Each dataset should contain:
- Images/videos of drivers
- Labels (drowsy/alert)

## Next Steps

1. **Train Models**: Run `python scripts/train_models.py` to train ML models on your datasets
2. **Test Backend**: Run `pytest` in the backend directory
3. **Develop Features**: Follow the task list in `.kiro/specs/driver-drowsiness-detection/tasks.md`

## Troubleshooting

### Python Dependencies Issues

If you encounter issues installing dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Flutter Issues

```bash
flutter clean
flutter pub get
flutter doctor -v
```

### Camera Permissions

Make sure to grant camera permissions when prompted by the mobile app.

## Development Workflow

1. Start with backend development (Tasks 2-6)
2. Train and validate ML models (Task 4)
3. Develop mobile app UI (Task 11)
4. Integrate backend with mobile app (Task 13)
5. Test end-to-end system (Task 14)

## Support

For detailed implementation guidance, see:
- [Requirements](.kiro/specs/driver-drowsiness-detection/requirements.md)
- [Design](.kiro/specs/driver-drowsiness-detection/design.md)
- [Tasks](.kiro/specs/driver-drowsiness-detection/tasks.md)
