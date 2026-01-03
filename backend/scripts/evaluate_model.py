"""
Comprehensive Model Evaluation Script
Generates detailed performance metrics including FAR, FRR, and EER
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
    """
    Calculate FAR, FRR, and EER from predictions.
    
    Args:
        y_true: True labels (0=alert, 1=drowsy)
        y_pred_proba: Predicted probabilities for drowsy class
        
    Returns:
        Dictionary with FAR, FRR, EER, and threshold
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # FRR = 1 - TPR (False Rejection Rate = miss rate)
    frr = 1 - tpr
    
    # FAR = FPR (False Acceptance Rate = false positive rate)
    far = fpr
    
    # Find EER (point where FAR = FRR)
    eer_threshold_idx = np.nanargmin(np.absolute(far - frr))
    eer = far[eer_threshold_idx]
    eer_threshold = thresholds[eer_threshold_idx]
    
    # Get FAR and FRR at default threshold (0.5)
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


def evaluate_model(
    model_path: str = "models/cnn_drowsiness.h5",
    dataset_root: str = "datasets",
    test_size: int = 1000
):
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to trained model
        dataset_root: Root directory containing datasets
        test_size: Number of test images per class
    """
    print("=" * 80)
    print("CNN Model Comprehensive Evaluation")
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
    
    # Load test data
    print(f"\n2. Loading test data ({test_size} images per class)...")
    loader = DatasetLoader(dataset_root)
    images, labels, paths = loader.load_all_datasets_limited(
        image_size=(expected_width, expected_height),
        max_per_class=test_size
    )
    
    print(f"   ✓ Loaded {len(images)} test images")
    print(f"     Alert: {np.sum(labels == 0)}")
    print(f"     Drowsy: {np.sum(labels == 1)}")
    
    # Get predictions
    print("\n3. Running predictions...")
    predictions_proba = classifier.predict(images).squeeze()
    predictions = (predictions_proba > 0.5).astype(np.int32)
    
    # Calculate basic metrics
    print("\n4. Calculating performance metrics...")
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate FAR, FRR, EER
    far_frr_metrics = calculate_far_frr_eer(labels, predictions_proba)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Table 1: Overall Performance Metrics
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
    
    # Table 2: Error Rates (FAR, FRR, EER)
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
    
    # Additional details
    print("\n" + "─" * 80)
    print("Confusion Matrix Details")
    print("─" * 80)
    print(f"{'Metric':<30} {'Count':<15}")
    print("─" * 80)
    print(f"{'True Negatives (TN)':<30} {tn:<15}")
    print(f"{'False Positives (FP)':<30} {fp:<15}")
    print(f"{'False Negatives (FN)':<30} {fn:<15}")
    print(f"{'True Positives (TP)':<30} {tp:<15}")
    print("─" * 80)
    
    print("\n" + "─" * 80)
    print("Metric Definitions")
    print("─" * 80)
    print("FAR (False Acceptance Rate): Rate of alert states incorrectly classified as drowsy")
    print("FRR (False Rejection Rate): Rate of drowsy states incorrectly classified as alert")
    print("EER (Equal Error Rate): Point where FAR = FRR (lower is better)")
    print("AUC (Area Under Curve): Overall model discrimination ability (higher is better)")
    print("─" * 80)
    
    # Save results to file
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'FAR': float(far_frr_metrics['FAR_at_0.5']),
        'FRR': float(far_frr_metrics['FRR_at_0.5']),
        'EER': float(far_frr_metrics['EER']),
        'EER_threshold': float(far_frr_metrics['EER_threshold']),
        'AUC': float(far_frr_metrics['AUC']),
        'confusion_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        },
        'test_samples': {
            'total': len(images),
            'alert': int(np.sum(labels == 0)),
            'drowsy': int(np.sum(labels == 1))
        }
    }
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN drowsiness classifier")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/cnn_drowsiness.h5",
        help="Path to trained model"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of test images per class"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        test_size=args.test_size
    )
