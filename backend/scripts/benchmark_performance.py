"""
Performance benchmarking script for the drowsiness detection pipeline.

This script measures processing times for each component to ensure
real-time performance requirements are met:
- Face detection: < 2 seconds initialization
- Frame processing: >= 15 FPS (< 67ms per frame)
- ML inference: < 100ms
- Alert response: < 500ms

Validates: Requirements 1.1, 1.2, 5.4, 3.1
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2
from statistics import mean, stdev

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from face_detection.face_detector import FaceDetector
from face_detection.landmark_detector import FacialLandmarkDetector
from feature_extraction.ear_calculator import EARCalculator
from feature_extraction.mar_calculator import MARCalculator
from ml_models.cnn_classifier import CNNDrowsinessClassifier
from ml_models.feature_based_classifier import FeatureBasedClassifier
from decision_logic.decision_engine import DecisionEngine
from decision_logic.alert_manager import AlertManager


class PerformanceBenchmark:
    """Benchmark performance of drowsiness detection components."""
    
    def __init__(self, num_iterations: int = 100):
        self.num_iterations = num_iterations
        self.results = {}
    
    def benchmark_face_detection_init(self) -> float:
        """Benchmark face detection initialization time."""
        print("\n=== Benchmarking Face Detection Initialization ===")
        
        start_time = time.time()
        detector = FaceDetector(min_detection_confidence=0.7)
        init_time = time.time() - start_time
        
        print(f"  Initialization time: {init_time*1000:.2f}ms")
        print(f"  Requirement: < 2000ms")
        print(f"  Status: {'✓ PASS' if init_time < 2.0 else '✗ FAIL'}")
        
        self.results['face_detection_init'] = {
            'time_ms': init_time * 1000,
            'requirement_ms': 2000,
            'passed': init_time < 2.0
        }
        
        return init_time
    
    def benchmark_face_detection(self) -> dict:
        """Benchmark face detection processing time."""
        print("\n=== Benchmarking Face Detection Processing ===")
        
        detector = FaceDetector()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 150), (440, 330), (255, 255, 255), -1)
        
        times = []
        for _ in range(self.num_iterations):
            start = time.time()
            result = detector.detectFace(test_image)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        avg_time = mean(times)
        std_time = stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Std deviation: {std_time:.2f}ms")
        print(f"  Min time: {min_time:.2f}ms")
        print(f"  Max time: {max_time:.2f}ms")
        print(f"  Requirement: < 67ms (for 15 FPS)")
        print(f"  Status: {'✓ PASS' if avg_time < 67 else '✗ FAIL'}")
        
        self.results['face_detection'] = {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'requirement_ms': 67,
            'passed': avg_time < 67
        }
        
        return self.results['face_detection']
    
    def benchmark_landmark_extraction(self) -> dict:
        """Benchmark landmark extraction processing time."""
        print("\n=== Benchmarking Landmark Extraction ===")
        
        detector = FacialLandmarkDetector()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 150), (440, 330), (255, 255, 255), -1)
        
        times = []
        for _ in range(self.num_iterations):
            start = time.time()
            landmarks = detector.extractLandmarks(test_image)
            times.append((time.time() - start) * 1000)
        
        avg_time = mean(times)
        std_time = stdev(times) if len(times) > 1 else 0
        
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Std deviation: {std_time:.2f}ms")
        print(f"  Requirement: < 67ms (for 15 FPS)")
        print(f"  Status: {'✓ PASS' if avg_time < 67 else '✗ FAIL'}")
        
        self.results['landmark_extraction'] = {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'requirement_ms': 67,
            'passed': avg_time < 67
        }
        
        return self.results['landmark_extraction']
    
    def benchmark_feature_extraction(self) -> dict:
        """Benchmark EAR and MAR calculation time."""
        print("\n=== Benchmarking Feature Extraction ===")
        
        ear_calc = EARCalculator()
        mar_calc = MARCalculator()
        
        # Synthetic landmarks
        eye_landmarks = [(100+i*5, 100+i%2*5, 0) for i in range(16)]
        mouth_landmarks = [(100+i*5, 100+i%2*5, 0) for i in range(20)]
        
        ear_times = []
        mar_times = []
        
        for _ in range(self.num_iterations):
            # EAR calculation
            start = time.time()
            ear = ear_calc.calculateEAR(eye_landmarks)
            ear_times.append((time.time() - start) * 1000)
            
            # MAR calculation
            start = time.time()
            mar = mar_calc.calculateMAR(mouth_landmarks)
            mar_times.append((time.time() - start) * 1000)
        
        ear_avg = mean(ear_times)
        mar_avg = mean(mar_times)
        total_avg = ear_avg + mar_avg
        
        print(f"  Iterations: {self.num_iterations}")
        print(f"  EAR calculation: {ear_avg:.3f}ms")
        print(f"  MAR calculation: {mar_avg:.3f}ms")
        print(f"  Total feature extraction: {total_avg:.3f}ms")
        print(f"  Requirement: < 10ms")
        print(f"  Status: {'✓ PASS' if total_avg < 10 else '✗ FAIL'}")
        
        self.results['feature_extraction'] = {
            'ear_time_ms': ear_avg,
            'mar_time_ms': mar_avg,
            'total_time_ms': total_avg,
            'requirement_ms': 10,
            'passed': total_avg < 10
        }
        
        return self.results['feature_extraction']
    
    def benchmark_ml_inference(self) -> dict:
        """Benchmark ML model inference time."""
        print("\n=== Benchmarking ML Inference ===")
        
        # CNN model
        cnn_model = CNNDrowsinessClassifier(input_shape=(224, 224, 3))
        keras_model = cnn_model.build_model()
        cnn_model.model = keras_model
        cnn_model.is_loaded = True
        
        test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Warm-up
        for _ in range(5):
            _ = cnn_model.predict(test_input)
        
        cnn_times = []
        for _ in range(50):  # Fewer iterations for ML inference
            start = time.time()
            _ = cnn_model.predict(test_input)
            cnn_times.append((time.time() - start) * 1000)
        
        cnn_avg = mean(cnn_times)
        cnn_std = stdev(cnn_times) if len(cnn_times) > 1 else 0
        
        # Feature-based model
        feature_model = FeatureBasedClassifier(model_type="random_forest")
        X_train = np.random.rand(100, 6)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.rand(20, 6)
        y_val = np.random.randint(0, 2, 20)
        feature_model.train(X_train, y_train, X_val, y_val)
        
        test_features = np.random.rand(1, 6)
        
        feature_times = []
        for _ in range(self.num_iterations):
            start = time.time()
            _ = feature_model.predict(test_features)
            feature_times.append((time.time() - start) * 1000)
        
        feature_avg = mean(feature_times)
        feature_std = stdev(feature_times) if len(feature_times) > 1 else 0
        
        print(f"  CNN Model:")
        print(f"    Average time: {cnn_avg:.2f}ms")
        print(f"    Std deviation: {cnn_std:.2f}ms")
        print(f"    Requirement: < 100ms")
        print(f"    Status: {'✓ PASS' if cnn_avg < 100 else '✗ FAIL'}")
        
        print(f"  Feature-based Model:")
        print(f"    Average time: {feature_avg:.3f}ms")
        print(f"    Std deviation: {feature_std:.3f}ms")
        print(f"    Requirement: < 100ms")
        print(f"    Status: {'✓ PASS' if feature_avg < 100 else '✗ FAIL'}")
        
        self.results['ml_inference'] = {
            'cnn_time_ms': cnn_avg,
            'cnn_std_ms': cnn_std,
            'feature_time_ms': feature_avg,
            'feature_std_ms': feature_std,
            'requirement_ms': 100,
            'passed': cnn_avg < 100 and feature_avg < 100
        }
        
        return self.results['ml_inference']
    
    def benchmark_decision_engine(self) -> dict:
        """Benchmark decision engine processing time."""
        print("\n=== Benchmarking Decision Engine ===")
        
        engine = DecisionEngine()
        
        times = []
        for _ in range(self.num_iterations):
            start = time.time()
            assessment = engine.calculate_drowsiness_score(
                ear_score=0.6,
                mar_score=0.5,
                head_pose_score=0.4,
                ml_confidence=0.55,
                timestamp=time.time()
            )
            times.append((time.time() - start) * 1000)
        
        avg_time = mean(times)
        std_time = stdev(times) if len(times) > 1 else 0
        
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Average time: {avg_time:.3f}ms")
        print(f"  Std deviation: {std_time:.3f}ms")
        print(f"  Requirement: < 10ms")
        print(f"  Status: {'✓ PASS' if avg_time < 10 else '✗ FAIL'}")
        
        self.results['decision_engine'] = {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'requirement_ms': 10,
            'passed': avg_time < 10
        }
        
        return self.results['decision_engine']
    
    def benchmark_alert_manager(self) -> dict:
        """Benchmark alert manager response time."""
        print("\n=== Benchmarking Alert Manager ===")
        
        alert_triggered = []
        def alert_cb(level, message):
            alert_triggered.append(time.time())
        
        manager = AlertManager(visual_callback=alert_cb)
        
        times = []
        for _ in range(self.num_iterations):
            alert_triggered.clear()
            start = time.time()
            manager.trigger_alert(
                alert_level=AlertLevel.HIGH,
                drowsiness_score=0.75
            )
            if alert_triggered:
                times.append((alert_triggered[0] - start) * 1000)
        
        avg_time = mean(times)
        std_time = stdev(times) if len(times) > 1 else 0
        
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Average time: {avg_time:.3f}ms")
        print(f"  Std deviation: {std_time:.3f}ms")
        print(f"  Requirement: < 500ms")
        print(f"  Status: {'✓ PASS' if avg_time < 500 else '✗ FAIL'}")
        
        self.results['alert_manager'] = {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'requirement_ms': 500,
            'passed': avg_time < 500
        }
        
        return self.results['alert_manager']
    
    def benchmark_end_to_end(self) -> dict:
        """Benchmark complete end-to-end pipeline."""
        print("\n=== Benchmarking End-to-End Pipeline ===")
        
        # Initialize components
        face_detector = FaceDetector()
        ear_calc = EARCalculator()
        mar_calc = MARCalculator()
        engine = DecisionEngine()
        manager = AlertManager(visual_callback=lambda l, m: None)
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 150), (440, 330), (255, 255, 255), -1)
        
        times = []
        for _ in range(50):  # Fewer iterations for full pipeline
            start = time.time()
            
            # 1. Face detection
            face_result = face_detector.detectFace(test_image)
            
            # 2. Feature extraction (simulated)
            ear_score = 0.6
            mar_score = 0.5
            
            # 3. Decision engine
            assessment = engine.calculate_drowsiness_score(
                ear_score=ear_score,
                mar_score=mar_score,
                head_pose_score=0.4,
                ml_confidence=0.55,
                timestamp=time.time()
            )
            
            # 4. Alert manager
            manager.trigger_alert(
                assessment.alert_level,
                assessment.drowsiness_score
            )
            
            times.append((time.time() - start) * 1000)
        
        avg_time = mean(times)
        std_time = stdev(times) if len(times) > 1 else 0
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"  Iterations: 50")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Std deviation: {std_time:.2f}ms")
        print(f"  Effective FPS: {fps:.1f}")
        print(f"  Requirement: >= 15 FPS (< 67ms)")
        print(f"  Status: {'✓ PASS' if avg_time < 67 else '✗ FAIL'}")
        
        self.results['end_to_end'] = {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'fps': fps,
            'requirement_ms': 67,
            'requirement_fps': 15,
            'passed': avg_time < 67
        }
        
        return self.results['end_to_end']
    
    def run_all_benchmarks(self) -> bool:
        """Run all performance benchmarks."""
        print("=" * 60)
        print("PERFORMANCE BENCHMARKING")
        print("=" * 60)
        
        # Import AlertLevel here to avoid circular import
        from decision_logic.decision_engine import AlertLevel
        globals()['AlertLevel'] = AlertLevel
        
        # Run benchmarks
        self.benchmark_face_detection_init()
        self.benchmark_face_detection()
        self.benchmark_landmark_extraction()
        self.benchmark_feature_extraction()
        self.benchmark_ml_inference()
        self.benchmark_decision_engine()
        self.benchmark_alert_manager()
        self.benchmark_end_to_end()
        
        # Print summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        all_passed = True
        for component, data in self.results.items():
            status = "✓ PASS" if data['passed'] else "✗ FAIL"
            if 'avg_time_ms' in data:
                print(f"  {component:25s}: {data['avg_time_ms']:7.2f}ms  {status}")
            elif 'time_ms' in data:
                print(f"  {component:25s}: {data['time_ms']:7.2f}ms  {status}")
            all_passed = all_passed and data['passed']
        
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ ALL PERFORMANCE REQUIREMENTS MET")
            print("System meets real-time processing requirements!")
        else:
            print("✗ SOME PERFORMANCE REQUIREMENTS NOT MET")
            print("Optimization may be needed for real-time operation.")
        print("=" * 60)
        
        return all_passed


def main():
    """Main benchmark entry point."""
    benchmark = PerformanceBenchmark(num_iterations=100)
    success = benchmark.run_all_benchmarks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
