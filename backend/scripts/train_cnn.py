"""
Training script for CNN-based drowsiness classifier.

This script loads datasets, trains the CNN model, evaluates performance,
and converts the model to TensorFlow Lite format for mobile deployment.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.utils import DatasetLoader, DataSplitter, create_generators
from src.ml_models import CNNDrowsinessClassifier
import argparse


def train_cnn_model(
    dataset_root: str = "datasets",
    model_save_path: str = "models/cnn_drowsiness.h5",
    tflite_save_path: str = "models/cnn_drowsiness.tflite",
    input_size: tuple = (128, 128),
    epochs: int = 50,
    batch_size: int = 32,
    quantize: bool = True,
    max_images_per_class: int = 5000  # Limit images to avoid memory issues
):
    """
    Train CNN model for drowsiness detection.
    
    Args:
        dataset_root: Root directory containing datasets
        model_save_path: Path to save trained Keras model
        tflite_save_path: Path to save TFLite model
        input_size: Input image size (width, height)
        epochs: Number of training epochs
        batch_size: Batch size for training
        quantize: Whether to quantize TFLite model
        max_images_per_class: Maximum images per class to avoid memory issues
    """
    print("=" * 80)
    print("CNN Drowsiness Classifier Training")
    print("=" * 80)
    
    # Load datasets with memory limit
    print("\n1. Loading datasets (limited to avoid memory issues)...")
    print(f"   Max images per class: {max_images_per_class}")
    loader = DatasetLoader(dataset_root)
    
    try:
        images, labels, paths = loader.load_all_datasets_limited(
            image_size=input_size,
            max_per_class=max_images_per_class
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nPlease ensure datasets are placed in '{dataset_root}/' directory")
        print("Expected structure:")
        print(f"  {dataset_root}/")
        print("    DDD/")
        print("      alert/ or Non Drowsy/")
        print("      drowsy/ or Drowsy/")
        print("    NTHUDDD/")
        print("      ...")
        return
    
    # Split data
    print("\n2. Splitting data into train/val/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.split_data(
        images, labels,
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
        print(f"    Drowsy ratio: {split_stats['drowsy_ratio']:.2%}")
    
    # Create data generators
    print("\n3. Creating data generators with augmentation...")
    train_gen, val_gen = create_generators(
        X_train, y_train,
        X_val, y_val,
        batch_size=batch_size,
        augment_train=True
    )
    
    print(f"  Training batches per epoch: {len(train_gen)}")
    print(f"  Validation batches per epoch: {len(val_gen)}")
    
    # Build and train model
    print("\n4. Building and training CNN model...")
    input_shape = X_train.shape[1:]
    classifier = CNNDrowsinessClassifier(input_shape=input_shape)
    
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate on test set
    print("\n5. Evaluating model on test set...")
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
    
    # Save Keras model
    print(f"\n6. Saving Keras model to {model_save_path}...")
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(model_save_path)
    
    # Convert to TFLite
    print(f"\n7. Converting to TensorFlow Lite format...")
    # Use a subset of training data for quantization calibration
    representative_data = X_train[:100] if len(X_train) >= 100 else X_train
    classifier.convert_to_tflite(
        tflite_save_path, 
        quantize=quantize,
        representative_data=representative_data
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nSaved models:")
    print(f"  Keras model: {model_save_path}")
    print(f"  TFLite model: {tflite_save_path}")
    
    return classifier, history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN drowsiness classifier")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/cnn_drowsiness.h5",
        help="Path to save trained Keras model"
    )
    parser.add_argument(
        "--tflite-path",
        type=str,
        default="models/cnn_drowsiness.tflite",
        help="Path to save TFLite model"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Input image size (width height)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable INT8 quantization for TFLite model"
    )
    
    args = parser.parse_args()
    
    train_cnn_model(
        dataset_root=args.dataset_root,
        model_save_path=args.model_path,
        tflite_save_path=args.tflite_path,
        input_size=tuple(args.input_size),
        epochs=args.epochs,
        batch_size=args.batch_size,
        quantize=not args.no_quantize
    )
