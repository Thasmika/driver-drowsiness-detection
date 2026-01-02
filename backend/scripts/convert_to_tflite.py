"""
Convert trained Keras model to TensorFlow Lite format.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.ml_models import CNNDrowsinessClassifier
from src.utils import DatasetLoader
import argparse


def convert_model(
    keras_model_path: str = "models/cnn_drowsiness.h5",
    tflite_save_path: str = "models/cnn_drowsiness.tflite",
    dataset_root: str = "datasets",
    quantize: bool = True
):
    """
    Convert Keras model to TFLite format.
    
    Args:
        keras_model_path: Path to trained Keras model
        tflite_save_path: Path to save TFLite model
        dataset_root: Root directory containing datasets (for representative data)
        quantize: Whether to apply quantization
    """
    print("=" * 80)
    print("Converting Keras Model to TensorFlow Lite")
    print("=" * 80)
    
    # Load the trained model
    print(f"\n1. Loading Keras model from {keras_model_path}...")
    classifier = CNNDrowsinessClassifier(model_path=keras_model_path)
    
    if not classifier.is_loaded:
        print("Error: Failed to load model")
        return False
    
    # Load some representative data for quantization
    representative_data = None
    if quantize:
        print("\n2. Loading representative data for quantization...")
        try:
            loader = DatasetLoader(dataset_root)
            # Load just 100 images for calibration
            images, labels, _ = loader.load_all_datasets_limited(
                image_size=(128, 128),
                max_per_class=50
            )
            representative_data = images[:100]
            print(f"   Loaded {len(representative_data)} images for calibration")
        except Exception as e:
            print(f"   Warning: Could not load representative data: {e}")
            print("   Using dynamic range quantization instead")
    
    # Convert to TFLite
    print(f"\n3. Converting to TensorFlow Lite...")
    success = classifier.convert_to_tflite(
        tflite_save_path,
        quantize=quantize,
        representative_data=representative_data
    )
    
    if success:
        print("\n" + "=" * 80)
        print("Conversion Complete!")
        print("=" * 80)
        print(f"\nTFLite model saved to: {tflite_save_path}")
    else:
        print("\nConversion failed!")
    
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument(
        "--keras-model",
        type=str,
        default="models/cnn_drowsiness.h5",
        help="Path to trained Keras model"
    )
    parser.add_argument(
        "--tflite-path",
        type=str,
        default="models/cnn_drowsiness.tflite",
        help="Path to save TFLite model"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization"
    )
    
    args = parser.parse_args()
    
    convert_model(
        keras_model_path=args.keras_model,
        tflite_save_path=args.tflite_path,
        dataset_root=args.dataset_root,
        quantize=not args.no_quantize
    )
