# Driver Drowsiness Detection System — Final Project Summary & Evaluation Report

**Date:** April 22, 2026  
**Status:** ✅ COMPLETE

---

## 1. Project Overview

A real-time, smartphone-based driver drowsiness detection system that uses computer vision and transfer learning to monitor facial indicators of fatigue and alert drivers without requiring specialised hardware. The system runs entirely on-device, preserving user privacy.

### System Components

| Component | Technology |
|-----------|-----------|
| Mobile App | Flutter (Android & iOS) |
| Backend / ML | Python, TensorFlow, MediaPipe |
| Face Detection | MediaPipe Face Mesh |
| Feature Extraction | Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) |
| ML Model | MobileNetV2 (Transfer Learning) |
| Mobile Deployment | TensorFlow Lite |
| Alert System | Visual, Audio, Haptic |
| Emergency Feature | GPS tracking + Emergency contacts |

---

## 2. System Architecture

```
Smartphone Camera
       ↓
MediaPipe Face Detection & Landmark Extraction
       ↓
Feature Extraction (EAR + MAR)
       ↓
MobileNetV2 CNN Classifier (TFLite — on-device)
       ↓
Decision Engine
       ↓
Multi-modal Alert (Visual / Audio / Haptic)
       ↓
Emergency Response (optional GPS + contacts)
```

---

## 3. ML Model — Architecture & Training

### Model Architecture

- **Base Model:** MobileNetV2 pretrained on ImageNet (1.4 million images)
- **Approach:** Transfer Learning with two-phase fine-tuning
- **Custom layers added on top** for binary drowsiness classification

### Datasets Used

| Dataset | Images | Alert | Drowsy |
|---------|--------|-------|--------|
| DDD (Driver Drowsiness Dataset) | 41,793 | — | — |
| NTHUDDD (NTHU Driver Drowsiness Dataset) | 133,042 | — | — |
| Yawing Dataset | 10,238 | — | — |
| **Total** | **185,073** | **85,609** | **99,464** |

### Training Split

| Split | Percentage | Images |
|-------|-----------|--------|
| Training | 75% | ~138,800 |
| Testing | 25% | ~46,200 |

### Training Process

| Phase | Description | Epochs |
|-------|-------------|--------|
| Phase 1 | Base frozen — only top classification layers trained | 50 |
| Phase 2 | Fine-tuning — top 30 layers of MobileNetV2 unfrozen | 20 |

**Configuration:**
- Image size: 128×128 pixels
- Batch size: 32
- Optimizer: Adam (lr=0.001 → 1e-5 for fine-tuning)
- Loss: Binary cross-entropy

---

## 4. Final Evaluation Results

### Performance Metrics

| Metric | Value | Requirement | Status |
|--------|-------|-------------|--------|
| **Accuracy** | **96.08%** | ≥ 85% | ✅ Pass |
| **Precision** | **95.58%** | ≥ 80% | ✅ Pass |
| **Recall** | **97.19%** | ≥ 80% | ✅ Pass |
| **F1-Score** | **96.38%** | — | ✅ |
| **AUC-ROC** | **0.9952** | — | ✅ |

### Error Rate Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **FAR** (False Acceptance Rate) | 5.22% | Low false alarm rate |
| **FRR** (False Rejection Rate) | 2.81% | Very few missed drowsy states |
| **EER** (Equal Error Rate) | 4.00% | Strong overall discrimination |

### Key Findings

- The model correctly identified drowsy drivers **97.19%** of the time (Recall)
- False alarms (alert driver flagged as drowsy) occurred only **5.22%** of the time
- AUC-ROC of **0.9952** indicates near-perfect class separation
- All performance requirements exceeded

---

## 5. Model Output Files

| File | Description | Size |
|------|-------------|------|
| `backend/models/cnn_drowsiness.h5` | Full Keras model | ~302 MB |
| `backend/models/cnn_drowsiness.tflite` | Mobile-optimised TFLite model | ~10 MB |

TFLite model uses dynamic range quantisation — 92% size reduction from the full Keras model.

---

## 6. System Features Implemented

### Core Detection
- [x] Real-time face detection (MediaPipe)
- [x] Facial landmark extraction (468 landmarks)
- [x] Eye Aspect Ratio (EAR) calculation
- [x] Mouth Aspect Ratio (MAR) / yawn detection
- [x] CNN-based drowsiness classification (MobileNetV2)
- [x] Real-time processing at 15+ FPS

### Alert System
- [x] Visual alerts (on-screen warnings)
- [x] Audio alerts
- [x] Haptic feedback (vibration)
- [x] Graduated alert levels (warning → critical)

### Mobile Application (Flutter)
- [x] Cross-platform (Android & iOS)
- [x] Camera integration
- [x] Monitoring screen
- [x] Settings screen
- [x] Emergency contacts management
- [x] Data management screen
- [x] Backend service integration

### Privacy & Security
- [x] All processing performed locally on-device
- [x] No data transmitted to external servers
- [x] Secure data handling

### Emergency Response
- [x] GPS location tracking
- [x] Emergency contact notification
- [x] Configurable emergency settings

### Robustness & Adaptation
- [x] Adaptation manager for varying conditions
- [x] Metrics collection and monitoring
- [x] Feedback manager

---

## 7. Property-Based Testing

The system was validated using property-based testing (Hypothesis framework) across all major components:

| Test Suite | Component |
|-----------|-----------|
| `test_face_detection_properties.py` | Face detection correctness |
| `test_feature_extraction_properties.py` | EAR/MAR calculation properties |
| `test_ml_performance_properties.py` | ML model performance bounds |
| `test_alert_system_properties.py` | Alert triggering logic |
| `test_realtime_performance_properties.py` | Real-time processing constraints |
| `test_privacy_properties.py` | Privacy preservation |
| `test_emergency_properties.py` | Emergency response correctness |
| `test_monitoring_properties.py` | Metrics and monitoring |
| `test_robustness_properties.py` | System robustness under edge cases |

---

## 8. Requirements vs Achieved

| Requirement | Target | Achieved |
|-------------|--------|----------|
| Detection accuracy | ≥ 85% | **96.08%** ✅ |
| Real-time processing | ≥ 15 FPS | **15+ FPS** ✅ |
| Cross-platform support | Android + iOS | **Both** ✅ |
| On-device processing | Required | **Yes** ✅ |
| Multi-modal alerts | Required | **Visual + Audio + Haptic** ✅ |
| Emergency response | Required | **GPS + Contacts** ✅ |
| Privacy-first | Required | **No external data transmission** ✅ |
| Mobile model size | Compact | **10 MB (TFLite)** ✅ |

---

## 9. Conclusion

The Driver Drowsiness Detection System was successfully developed and evaluated. The MobileNetV2-based transfer learning model, trained on 185,073 images from three diverse public datasets, achieved **96.08% accuracy** with an AUC-ROC of **0.9952**, significantly exceeding the 85% accuracy requirement.

The two-phase training approach (frozen base + fine-tuning) proved effective for learning drowsiness-specific features while leveraging ImageNet pretrained weights. The system is deployed as a cross-platform Flutter mobile application with full on-device inference via TensorFlow Lite, ensuring real-time performance and user privacy.

The system is ready for real-world validation and further deployment.

---

**Report compiled:** April 22, 2026  
**Model version:** Final (MobileNetV2 Transfer Learning)  
**Training completed:** Both phases ✅
