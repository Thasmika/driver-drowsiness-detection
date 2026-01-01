# Driver Drowsiness Detection - Backend

Python-based backend for real-time drowsiness detection using computer vision and machine learning.

## Features

- Face detection using MediaPipe
- Facial landmark extraction
- Eye Aspect Ratio (EAR) calculation
- Mouth Aspect Ratio (MAR) calculation for yawn detection
- CNN-based drowsiness classification
- Traditional ML models (SVM, Random Forest)
- Real-time processing pipeline

## Installation

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Project Structure

```
backend/
├── src/
│   ├── face_detection/      # Face detection and tracking
│   ├── feature_extraction/  # EAR, MAR calculators
│   ├── ml_models/           # ML inference engine
│   ├── decision_logic/      # Alert decision logic
│   └── utils/               # Helper functions
├── tests/                   # Test suite
├── models/                  # Trained models (.tflite)
└── requirements.txt
```

## Usage

```python
from src.face_detection import FaceDetector
from src.feature_extraction import EARCalculator

# Initialize components
detector = FaceDetector()
ear_calc = EARCalculator()

# Process frame
face = detector.detect_face(frame)
landmarks = detector.extract_landmarks(face)
ear = ear_calc.calculate_ear(landmarks)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_face_detection.py
```

## Development

### Code Formatting

```bash
black src/ tests/
```

### Linting

```bash
flake8 src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Model Training

Place your 3 datasets in the `datasets/` directory and run:

```bash
python scripts/train_models.py
```

This will:
1. Load and preprocess the datasets
2. Train CNN and traditional ML models
3. Evaluate performance
4. Convert models to TensorFlow Lite format
5. Save models to `models/` directory
