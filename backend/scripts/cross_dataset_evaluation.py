"""
Cross-Dataset Evaluation
Train on DDD + yawing datasets, test on NTHUDDD dataset
This tests generalization to completely unseen data source
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from src.ml_models import CNNDrowsinessClassifier
from src.utils import DatasetLoader
import argparse


def calculate_far_frr_eer(y_true, y_pred_proba):
    """Calculate FAR, FRR, and EER from predictions."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    frr = 1 - tpr
    far = fpr
    
    eer_threshold_idx = np.nanargmin(np.absolute(far - frr))
    eer = far[eer_threshold_idx]
    eer_threshold = thresholds[eer_threshold_idx]
    
    default_threshold_idx = np.argmin(np.abs(thresholds - 0.5))
    far_at_05 = far[default_threshold_idx]
    frr_at_05 = frr[default_threshold_idx]
    
    return {
        'FAR_at_0.5': far_at_05,
        'FRR_at_0.5': frr_at_05,
        'EER': eer,
        'EER_threshold': eer_threshold,
        'AUC': auc(fpr, tpr)
    }


def load_specific_dataset(dataset_root: str, dataset_name: str, image_size: tuple, max_per_class: int = None):
    """Load a specific dataset by name."""
    loader = DatasetLoader(dataset_root)
    
    # Temporarily modify the dataset root to load only specific dataset
    original_root = loader.dataset_root
    loader.dataset_root = original_root / dataset_name
    
    try:
        images, labels, paths = loader.load_dataset_limited(
            dataset_name=".",  # Current directory
            image_size=image_size,
            max_alert=max_per_class,
            max_drowsy=max_per_class
        )
        return images, labels, paths
    finally:
        loader.dataset_root = original_root


def cross_dataset_evaluation(
    model_path: str = "models/cnn_drowsiness.h5",
    dataset_root: str = "datasets",
    test_dataset: str = "NTHUDDD",
    test_size_per_class: int = 2000
):
    """
    Evaluate model on completely different dataset.
    Model was trained on DDD dataset, now test on NTHUDDD.
    """
    print("=" * 80)
    print("CROSS-DATASET EVALUATION")
    print("=" * 80)
    print(f"\nTraining datasets: DDD + yawing")
    print(f"Test dataset: {test_dataset} (COMPLETELY UNSEEN)")
    print("=" * 80)
    
    # Load the model
    print("\n1. Loading trained model...")
    classifier = CNNDrowsinessClassifier(model_path=model_path)
    
    if not classifier.is_loaded:
        print("Error: Failed to load model")
        return None
    
    print("   ✓ Model loaded successfully")
    
    # Get model input size
    input_shape = classifier.model.input_shape
    expected_height = input_shape[1]
    expected_width = input_shape[2]
    print(f"   Model input size: {expected_width}x{expected_height}")
    
    # Load test dataset (NTHUDDD - completely different from training)
    print(f"\n2. Loading TEST dataset: {test_dataset}...")
    print(f"   Loading up to {test_size_per_class} images per class...")
    
    loader = DatasetLoader(dataset_root)
    
    # Load NTHUDDD dataset
    test_images = []
    test_labels = []
    
    # Try to load from NTHUDDD
    nthuddd_path = Path(dataset_root) / test_dataset
    if nthuddd_path.exists():
        alert_dir = nthuddd_path / "alert"
        drowsy_dir = nthuddd_path / "drowsy"
        
        if alert_dir.exists() and drowsy_dir.exists():
            # Load alert images
            alert_files = loader._get_image_files(alert_dir)
            alert_count = 0
            for img_path in alert_files[:test_size_per_class]:
                img = loader._load_and_preprocess_image(img_path, (expected_width, expected_height))
                if img is not None:
                    test_images.append(img)
                    test_labels.append(0)
                    alert_count += 1
            
            # Load drowsy images
            drowsy_files = loader._get_image_files(drowsy_dir)
            drowsy_count = 0
            for img_path in drowsy_files[:test_size_per_class]:
                img = loader._load_and_preprocess_image(img_path, (expected_width, expected_height))
                if img is not None:
                    test_images.append(img)
                    test_labels.append(1)
                    drowsy_count += 1
            
            print(f"   ✓ Loaded {len(test_images)} test images")
            print(f"     Alert: {alert_count}")
            print(f"     Drowsy: {drowsy_count}")
        else:
            print(f"   Error: Could not find alert/drowsy folders in {nthuddd_path}")
            return None
    else:
        print(f"   Error: Dataset {test_dataset} not found at {nthuddd_path}")
        return None
    
    X_test = np.array(test_images, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int32)
    
    # Evaluate on test dataset
    print(f"\n3. Evaluating on {test_dataset} dataset (UNSEEN)...")
    predictions_proba = classifier.predict(X_test).squeeze()
    predictions = (predictions_proba > 0.5).astype(np.int32)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    far_frr_metrics = calculate_far_frr_eer(y_test, predictions_proba)
    
    # Print results
    print("\n" + "=" * 80)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nModel trained on: DDD + yawing datasets")
    print(f"Model tested on: {test_dataset} dataset (COMPLETELY DIFFERENT SOURCE)")
    print("=" * 80)
    
    print("\n" + "─" * 80)
    print("Table 1: Overall Performance Metrics")
    print("─" * 80)
    print(f"{'Metric':<20} {'Value':<15} {'Percentage':<15}")
    print("─" * 80)
    print(f"{'Accuracy':<20} {accuracy:<15.4f} {accuracy*100:<14.2f}%")
    print(f"{'Precision':<20} {precision:<15.4f} {precision*100:<14.2f}%")
    print(f"{'Recall':<20} {recall:<15.4f} {recall*100:<14.2f}%")
    print(f"{'F1-Score':<20} {f1:<15.4f} {f1*100:<14.2f}%")
    print("─" * 80)
    
    print("\n" + "─" * 80)
    print("Table 2: Error Rates (FAR, FRR, EER)")
    print("─" * 80)
    print(f"{'Metric':<30} {'Value':<15} {'Percentage':<15}")
    print("─" * 80)
    print(f"{'FAR (at threshold 0.5)':<30} {far_frr_metrics['FAR_at_0.5']:<15.4f} {far_frr_metrics['FAR_at_0.5']*100:<14.2f}%")
    print(f"{'FRR (at threshold 0.5)':<30} {far_frr_metrics['FRR_at_0.5']:<15.4f} {far_frr_metrics['FRR_at_0.5']*100:<14.2f}%")
    print(f"{'EER (Equal Error Rate)':<30} {far_frr_metrics['EER']:<15.4f} {far_frr_metrics['EER']*100:<14.2f}%")
    print(f"{'EER Threshold':<30} {far_frr_metrics['EER_threshold']:<15.4f} {'-':<15}")
    print(f"{'AUC (Area Under Curve)':<30} {far_frr_metrics['AUC']:<15.4f} {far_frr_metrics['AUC']*100:<14.2f}%")
    print("─" * 80)
    
    print("\n" + "─" * 80)
    print("Confusion Matrix")
    print("─" * 80)
    print(f"{'Metric':<30} {'Count':<15}")
    print("─" * 80)
    print(f"{'True Negatives (TN)':<30} {tn:<15}")
    print(f"{'False Positives (FP)':<30} {fp:<15}")
    print(f"{'False Negatives (FN)':<30} {fn:<15}")
    print(f"{'True Positives (TP)':<30} {tp:<15}")
    print("─" * 80)
    
    # Analysis
    print("\n" + "─" * 80)
    print("GENERALIZATION ANALYSIS")
    print("─" * 80)
    
    if accuracy >= 0.95:
        print("✓ EXCELLENT: Model generalizes very well to unseen dataset (>95%)")
        print("  The model learned robust features that work across different data sources")
    elif accuracy >= 0.85:
        print("✓ GOOD: Model generalizes well to unseen dataset (>85%)")
        print("  Performance meets requirements on completely different data")
    elif accuracy >= 0.75:
        print("⚠ MODERATE: Model shows some generalization (75-85%)")
        print("  Performance drops on unseen dataset, may need more diverse training data")
    else:
        print("✗ POOR: Model does not generalize well (<75%)")
        print("  Significant performance drop indicates overfitting to training dataset")
    
    print("─" * 80)
    
    print("\n" + "=" * 80)
    print("Cross-Dataset Evaluation Complete!")
    print("=" * 80)
    
    return {
        'test_dataset': test_dataset,
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'FAR': float(far_frr_metrics['FAR_at_0.5']),
        'FRR': float(far_frr_metrics['FRR_at_0.5']),
        'EER': float(far_frr_metrics['EER']),
        'AUC': float(far_frr_metrics['AUC']),
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--model-path", type=str, default="models/cnn_drowsiness.h5")
    parser.add_argument("--dataset-root", type=str, default="datasets")
    parser.add_argument("--test-dataset", type=str, default="NTHUDDD",
                       help="Dataset to use for testing (different from training)")
    parser.add_argument("--test-size", type=int, default=2000,
                       help="Number of test images per class")
    
    args = parser.parse_args()
    
    cross_dataset_evaluation(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        test_dataset=args.test_dataset,
        test_size_per_class=args.test_size
    )
