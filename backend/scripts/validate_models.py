"""
Model validation script for drowsiness detection models.

This script evaluates both CNN and traditional ML models on test datasets,
compares their performance, and generates detailed performance reports.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import DatasetLoader, DataSplitter
from src.ml_models import CNNDrowsinessClassifier, FeatureBasedClassifier
from src.face_detection import FaceDetector, LandmarkDetector
from src.feature_extraction import EARCalculator, MARCalculator
import argparse
from tqdm import tqdm
import json


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    save_path: str
):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Alert', 'Drowsy'],
        yticklabels=['Alert', 'Drowsy']
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def evaluate_cnn_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> dict:
    """
    Evaluate CNN model on test data.
    
    Args:
        model_path: Path to the CNN model
        X_test: Test images
        y_test: Test labels
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating CNN Model")
    print("=" * 80)
    
    # Load model
    print(f"Loading CNN model from {model_path}...")
    classifier = CNNDrowsinessClassifier()
    
    if not classifier.load_model(model_path):
        print("Failed to load CNN model")
        return {}
    
    # Get predictions
    print("Running predictions on test set...")
    predictions, confidence = classifier.predict_with_confidence(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Print results
    print("\nCNN Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=['Alert', 'Drowsy']
    ))
    
    # Save confusion matrix plot
    plot_confusion_matrix(
        cm,
        "CNN Model - Confusion Matrix",
        output_dir / "cnn_confusion_matrix.png"
    )
    
    # Check requirements
    meets_accuracy = accuracy >= 0.85
    meets_precision = precision >= 0.85
    meets_recall = recall >= 0.80
    
    print("\nRequirement Validation:")
    print(f"  Accuracy >= 85%:  {'✓' if meets_accuracy else '✗'} ({accuracy:.2%})")
    print(f"  Precision >= 85%: {'✓' if meets_precision else '✗'} ({precision:.2%})")
    print(f"  Recall >= 80%:    {'✓' if meets_recall else '✗'} ({recall:.2%})")
    
    return {
        'model_type': 'CNN',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'meets_requirements': meets_accuracy and meets_precision and meets_recall,
        'avg_confidence': float(np.mean(confidence))
    }


def evaluate_traditional_ml_model(
    model_path: str,
    X_test_features: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> dict:
    """
    Evaluate traditional ML model on test data.
    
    Args:
        model_path: Path to the traditional ML model
        X_test_features: Test features
        y_test: Test labels
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating Traditional ML Model")
    print("=" * 80)
    
    # Load model
    print(f"Loading traditional ML model from {model_path}...")
    classifier = FeatureBasedClassifier()
    
    if not classifier.load_model(model_path):
        print("Failed to load traditional ML model")
        return {}
    
    # Get predictions
    print("Running predictions on test set...")
    predictions, confidence = classifier.predict_with_confidence(X_test_features)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Print results
    print("\nTraditional ML Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=['Alert', 'Drowsy']
    ))
    
    # Save confusion matrix plot
    plot_confusion_matrix(
        cm,
        "Traditional ML Model - Confusion Matrix",
        output_dir / "traditional_ml_confusion_matrix.png"
    )
    
    # Feature importance
    feature_importance = classifier.get_feature_importance()
    if feature_importance is not None:
        print("\nFeature Importance:")
        for feature_name, importance in sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {feature_name}: {importance:.4f}")
    
    # Check requirements
    meets_accuracy = accuracy >= 0.85
    meets_precision = precision >= 0.85
    meets_recall = recall >= 0.80
    
    print("\nRequirement Validation:")
    print(f"  Accuracy >= 85%:  {'✓' if meets_accuracy else '✗'} ({accuracy:.2%})")
    print(f"  Precision >= 85%: {'✓' if meets_precision else '✗'} ({precision:.2%})")
    print(f"  Recall >= 80%:    {'✓' if meets_recall else '✗'} ({recall:.2%})")
    
    return {
        'model_type': 'Traditional ML',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'meets_requirements': meets_accuracy and meets_precision and meets_recall,
        'avg_confidence': float(np.mean(confidence)),
        'feature_importance': feature_importance
    }


def compare_models(
    cnn_metrics: dict,
    ml_metrics: dict,
    output_dir: Path
):
    """
    Compare CNN and traditional ML model performance.
    
    Args:
        cnn_metrics: CNN model metrics
        ml_metrics: Traditional ML model metrics
        output_dir: Directory to save comparison
    """
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
    
    print("\nMetric Comparison:")
    print(f"{'Metric':<15} {'CNN':<12} {'Traditional ML':<15} {'Winner':<10}")
    print("-" * 55)
    
    for metric in metrics_to_compare:
        cnn_val = cnn_metrics.get(metric, 0)
        ml_val = ml_metrics.get(metric, 0)
        winner = "CNN" if cnn_val > ml_val else "Traditional ML" if ml_val > cnn_val else "Tie"
        
        print(f"{metric.capitalize():<15} {cnn_val:<12.4f} {ml_val:<15.4f} {winner:<10}")
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    cnn_values = [cnn_metrics.get(m, 0) for m in metrics_to_compare]
    ml_values = [ml_metrics.get(m, 0) for m in metrics_to_compare]
    
    ax.bar(x - width/2, cnn_values, width, label='CNN', alpha=0.8)
    ax.bar(x + width/2, ml_values, width, label='Traditional ML', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_to_compare])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison chart to {output_dir / 'model_comparison.png'}")


def validate_models(
    dataset_root: str = "datasets",
    cnn_model_path: str = "models/cnn_drowsiness.h5",
    ml_model_path: str = "models/traditional_ml_drowsiness.pkl",
    output_dir: str = "validation_results",
    input_size: tuple = (224, 224)
):
    """
    Validate both CNN and traditional ML models.
    
    Args:
        dataset_root: Root directory containing datasets
        cnn_model_path: Path to CNN model
        ml_model_path: Path to traditional ML model
        output_dir: Directory to save validation results
        input_size: Input image size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Model Validation")
    print("=" * 80)
    
    # Load datasets
    print("\n1. Loading datasets...")
    loader = DatasetLoader(dataset_root)
    
    try:
        images, labels, paths = loader.load_all_datasets(image_size=input_size)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Split data (use same split as training)
    print("\n2. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.split_data(
        images, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        stratify=True
    )
    
    print(f"Test set size: {len(X_test)} images")
    
    # Evaluate CNN model
    cnn_metrics = {}
    if Path(cnn_model_path).exists():
        cnn_metrics = evaluate_cnn_model(cnn_model_path, X_test, y_test, output_dir)
    else:
        print(f"\nCNN model not found at {cnn_model_path}, skipping...")
    
    # Extract features for traditional ML model
    ml_metrics = {}
    if Path(ml_model_path).exists():
        print("\n3. Extracting features for traditional ML model...")
        
        face_detector = FaceDetector()
        landmark_detector = LandmarkDetector()
        ear_calculator = EARCalculator()
        mar_calculator = MARCalculator()
        
        # Extract features from test set
        from train_traditional_ml import extract_features_from_dataset
        X_test_features, y_test_features = extract_features_from_dataset(
            X_test, y_test,
            face_detector, landmark_detector,
            ear_calculator, mar_calculator
        )
        
        ml_metrics = evaluate_traditional_ml_model(
            ml_model_path, X_test_features, y_test_features, output_dir
        )
    else:
        print(f"\nTraditional ML model not found at {ml_model_path}, skipping...")
    
    # Compare models
    if cnn_metrics and ml_metrics:
        compare_models(cnn_metrics, ml_metrics, output_dir)
    
    # Save results to JSON
    results = {
        'cnn_model': cnn_metrics,
        'traditional_ml_model': ml_metrics,
        'test_set_size': int(len(y_test))
    }
    
    results_path = output_dir / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nSaved validation results to {results_path}")
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate drowsiness detection models")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--cnn-model",
        type=str,
        default="models/cnn_drowsiness.h5",
        help="Path to CNN model"
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        default="models/traditional_ml_drowsiness.pkl",
        help="Path to traditional ML model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Directory to save validation results"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size (width height)"
    )
    
    args = parser.parse_args()
    
    validate_models(
        dataset_root=args.dataset_root,
        cnn_model_path=args.cnn_model,
        ml_model_path=args.ml_model,
        output_dir=args.output_dir,
        input_size=tuple(args.input_size)
    )
