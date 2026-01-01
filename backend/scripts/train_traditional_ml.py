"""
Training script for traditional ML classifier using extracted features.

This script loads datasets, extracts features (EAR, MAR, head pose),
trains traditional ML models, and evaluates performance.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from src.utils import DatasetLoader, DataSplitter
from src.ml_models import FeatureBasedClassifier
from src.face_detection import FaceDetector, LandmarkDetector
from src.feature_extraction import EARCalculator, MARCalculator
import argparse
from tqdm import tqdm


def extract_features_from_image(
    image: np.ndarray,
    face_detector: FaceDetector,
    landmark_detector: LandmarkDetector,
    ear_calculator: EARCalculator,
    mar_calculator: MARCalculator
) -> Optional[np.ndarray]:
    """
    Extract drowsiness features from a single image.
    
    Args:
        image: Input image
        face_detector: Face detector instance
        landmark_detector: Landmark detector instance
        ear_calculator: EAR calculator instance
        mar_calculator: MAR calculator instance
        
    Returns:
        Feature vector or None if face not detected
    """
    try:
        # Detect face
        face_result = face_detector.detect_face(image)
        
        if face_result is None or not face_result['face_detected']:
            return None
        
        # Extract landmarks
        landmarks = landmark_detector.extract_landmarks(image, face_result['bbox'])
        
        if landmarks is None:
            return None
        
        # Calculate EAR features
        left_ear = ear_calculator.calculate_ear(landmarks, eye='left')
        right_ear = ear_calculator.calculate_ear(landmarks, eye='right')
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate MAR feature
        mar = mar_calculator.calculate_mar(landmarks)
        
        # Placeholder for head pose (simplified)
        head_pitch = 0.0
        head_yaw = 0.0
        
        features = np.array([left_ear, right_ear, avg_ear, mar, head_pitch, head_yaw])
        
        return features
        
    except Exception as e:
        return None


def extract_features_from_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    face_detector: FaceDetector,
    landmark_detector: LandmarkDetector,
    ear_calculator: EARCalculator,
    mar_calculator: MARCalculator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from all images in a dataset.
    
    Args:
        images: Array of images
        labels: Array of labels
        face_detector: Face detector instance
        landmark_detector: Landmark detector instance
        ear_calculator: EAR calculator instance
        mar_calculator: MAR calculator instance
        
    Returns:
        Tuple of (features, labels) for successfully processed images
    """
    features_list = []
    labels_list = []
    
    print("Extracting features from images...")
    for i, (image, label) in enumerate(tqdm(zip(images, labels), total=len(images))):
        # Convert normalized image back to uint8 for processing
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Extract features
        features = extract_features_from_image(
            image_uint8,
            face_detector,
            landmark_detector,
            ear_calculator,
            mar_calculator
        )
        
        if features is not None:
            features_list.append(features)
            labels_list.append(label)
    
    if len(features_list) == 0:
        raise ValueError("No features could be extracted from the dataset")
    
    print(f"Successfully extracted features from {len(features_list)}/{len(images)} images")
    
    return np.array(features_list), np.array(labels_list)


def train_traditional_ml_model(
    dataset_root: str = "datasets",
    model_save_path: str = "models/traditional_ml_drowsiness.pkl",
    input_size: tuple = (224, 224),
    model_type: str = "ensemble"
):
    """
    Train traditional ML model for drowsiness detection.
    
    Args:
        dataset_root: Root directory containing datasets
        model_save_path: Path to save trained model
        input_size: Input image size for feature extraction
        model_type: Type of classifier ('svm', 'random_forest', or 'ensemble')
    """
    print("=" * 80)
    print("Traditional ML Drowsiness Classifier Training")
    print("=" * 80)
    
    # Initialize feature extractors
    print("\n1. Initializing feature extractors...")
    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    ear_calculator = EARCalculator()
    mar_calculator = MARCalculator()
    
    # Load datasets
    print("\n2. Loading datasets...")
    loader = DatasetLoader(dataset_root)
    
    try:
        images, labels, paths = loader.load_all_datasets(image_size=input_size)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nPlease ensure datasets are placed in '{dataset_root}/' directory")
        return
    
    # Extract features from all images
    print("\n3. Extracting features from images...")
    features, feature_labels = extract_features_from_dataset(
        images, labels,
        face_detector, landmark_detector,
        ear_calculator, mar_calculator
    )
    
    # Split data
    print("\n4. Splitting data into train/val/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.split_data(
        features, feature_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        stratify=True
    )
    
    # Print split statistics
    stats = DataSplitter.get_split_statistics(y_train, y_val, y_test)
    print("\nDataset splits:")
    for split_name, split_stats in stats.items():
        print(f"  {split_name.capitalize()}:")
        print(f"    Total: {split_stats['total']}")
        print(f"    Alert: {split_stats['alert']}")
        print(f"    Drowsy: {split_stats['drowsy']}")
    
    # Train model
    print(f"\n5. Training {model_type} classifier...")
    classifier = FeatureBasedClassifier(model_type=model_type)
    
    val_metrics = classifier.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\n6. Evaluating model on test set...")
    test_metrics = classifier.evaluate(X_test, y_test)
    
    print("\nTest Set Performance:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Check if accuracy meets requirements (85% minimum)
    test_accuracy = test_metrics.get('accuracy', 0)
    if test_accuracy >= 0.85:
        print(f"\n✓ Model meets accuracy requirement (>= 85%): {test_accuracy:.2%}")
    else:
        print(f"\n✗ Model does not meet accuracy requirement: {test_accuracy:.2%} < 85%")
    
    # Print feature importance if available
    feature_importance = classifier.get_feature_importance()
    if feature_importance is not None:
        print("\nFeature Importance:")
        for feature_name, importance in sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {feature_name}: {importance:.4f}")
    
    # Save model
    print(f"\n7. Saving model to {model_save_path}...")
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(model_save_path)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nSaved model: {model_save_path}")
    
    return classifier, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train traditional ML drowsiness classifier")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/traditional_ml_drowsiness.pkl",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size for feature extraction (width height)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['svm', 'random_forest', 'ensemble'],
        default='ensemble',
        help="Type of classifier to train"
    )
    
    args = parser.parse_args()
    
    train_traditional_ml_model(
        dataset_root=args.dataset_root,
        model_save_path=args.model_path,
        input_size=tuple(args.input_size),
        model_type=args.model_type
    )
