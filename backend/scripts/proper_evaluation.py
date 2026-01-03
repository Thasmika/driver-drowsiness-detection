"""
Proper Model Evaluation with Separate Test Set
This ensures no data leakage and realistic performance metrics
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
from src.utils import DatasetLoader, DataSplitter
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


def proper_evaluation(
    model_path: str = "models/cnn_drowsiness.h5",
    dataset_root: str = "datasets",
    total_images_per_class: int = 5000
):
    """
    Proper evaluation using the same train/test split methodology as training.
    This ensures we're testing on truly unseen data.
    """
    print("=" * 80)
    print("PROPER MODEL EVALUATION (No Data Leakage)")
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
    
    # Load ALL data (same as training)
    print(f"\n2. Loading dataset ({total_images_per_class} images per class)...")
    loader = DatasetLoader(dataset_root)
    images, labels, paths = loader.load_all_datasets_limited(
        image_size=(expected_width, expected_height),
        max_per_class=total_images_per_class
    )
    
    print(f"   ✓ Loaded {len(images)} total images")
    
    # Split data THE SAME WAY as training (70/15/15 split with same random_state)
    print("\n3. Splitting data (70% train, 15% val, 15% test)...")
    print("   Using random_state=42 (same as training)")
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.split_data(
        images, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,  # SAME as training
        stratify=True
    )
    
    print(f"   Train: {len(X_train)} images")
    print(f"   Val: {len(X_val)} images")
    print(f"   Test: {len(X_test)} images (UNSEEN during training)")
    
    # Evaluate on TEST SET ONLY (truly unseen data)
    print("\n4. Evaluating on TEST SET (unseen data)...")
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
    
    # Also evaluate on VALIDATION SET
    print("\n5. Evaluating on VALIDATION SET...")
    val_predictions_proba = classifier.predict(X_val).squeeze()
    val_predictions = (val_predictions_proba > 0.5).astype(np.int32)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print("\n" + "─" * 80)
    print("Table 1: Overall Performance Metrics (TEST SET)")
    print("─" * 80)
    print(f"{'Metric':<20} {'Value':<15} {'Percentage':<15}")
    print("─" * 80)
    print(f"{'Accuracy':<20} {accuracy:<15.4f} {accuracy*100:<14.2f}%")
    print(f"{'Precision':<20} {precision:<15.4f} {precision*100:<14.2f}%")
    print(f"{'Recall':<20} {recall:<15.4f} {recall*100:<14.2f}%")
    print(f"{'F1-Score':<20} {f1:<15.4f} {f1*100:<14.2f}%")
    print("─" * 80)
    
    print("\n" + "─" * 80)
    print("Table 2: Error Rates (FAR, FRR, EER) - TEST SET")
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
    print("Confusion Matrix (TEST SET)")
    print("─" * 80)
    print(f"{'Metric':<30} {'Count':<15}")
    print("─" * 80)
    print(f"{'True Negatives (TN)':<30} {tn:<15}")
    print(f"{'False Positives (FP)':<30} {fp:<15}")
    print(f"{'False Negatives (FN)':<30} {fn:<15}")
    print(f"{'True Positives (TP)':<30} {tp:<15}")
    print("─" * 80)
    
    print("\n" + "─" * 80)
    print("Validation Set Accuracy (for comparison)")
    print("─" * 80)
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print("─" * 80)
    
    # Analysis
    print("\n" + "─" * 80)
    print("ANALYSIS")
    print("─" * 80)
    
    if accuracy == 1.0 and val_accuracy == 1.0:
        print("⚠ WARNING: Perfect accuracy on both validation and test sets!")
        print("")
        print("Possible explanations:")
        print("1. Dataset is too easy (images are very distinct)")
        print("2. Small dataset size (only 10,000 images total)")
        print("3. Model is very powerful for this task")
        print("4. Images from same source/conditions (limited diversity)")
        print("")
        print("Recommendations:")
        print("- Test on completely different dataset (different source)")
        print("- Test with real-world camera feed")
        print("- Add more challenging scenarios (lighting, angles, occlusions)")
        print("- Increase dataset diversity")
    elif accuracy >= 0.95:
        print("✓ Excellent performance (>95% accuracy)")
        print("Model is performing very well on unseen data")
    elif accuracy >= 0.85:
        print("✓ Good performance (>85% accuracy)")
        print("Model meets requirements")
    else:
        print("⚠ Performance below target (<85% accuracy)")
        print("Consider retraining with more data or different architecture")
    
    print("─" * 80)
    
    # Check for overfitting
    train_val_gap = abs(val_accuracy - accuracy)
    if train_val_gap > 0.05:
        print("\n⚠ Potential overfitting detected!")
        print(f"   Validation-Test gap: {train_val_gap*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    
    return {
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'val_accuracy': float(val_accuracy),
        'FAR': float(far_frr_metrics['FAR_at_0.5']),
        'FRR': float(far_frr_metrics['FRR_at_0.5']),
        'EER': float(far_frr_metrics['EER']),
        'AUC': float(far_frr_metrics['AUC']),
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proper model evaluation")
    parser.add_argument("--model-path", type=str, default="models/cnn_drowsiness.h5")
    parser.add_argument("--dataset-root", type=str, default="datasets")
    parser.add_argument("--total-per-class", type=int, default=5000,
                       help="Total images per class (same as training)")
    
    args = parser.parse_args()
    
    proper_evaluation(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        total_images_per_class=args.total_per_class
    )
