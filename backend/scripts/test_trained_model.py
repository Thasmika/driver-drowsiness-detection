"""
Test the trained CNN model with sample images.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from src.ml_models import CNNDrowsinessClassifier
from src.utils import DatasetLoader


def test_model():
    """Test the trained CNN model."""
    print("=" * 80)
    print("Testing Trained CNN Model")
    print("=" * 80)
    
    # Load the model
    print("\n1. Loading trained model...")
    model_path = "models/cnn_drowsiness.h5"
    classifier = CNNDrowsinessClassifier(model_path=model_path)
    
    if not classifier.is_loaded:
        print("Error: Failed to load model")
        return False
    
    print("   Model loaded successfully!")
    
    # Load some test images
    print("\n2. Loading test images...")
    loader = DatasetLoader("datasets")
    
    # Get the model's expected input size from the model
    input_shape = classifier.model.input_shape
    expected_height = input_shape[1]
    expected_width = input_shape[2]
    
    print(f"   Model expects input size: {expected_width}x{expected_height}")
    
    images, labels, paths = loader.load_all_datasets_limited(
        image_size=(expected_width, expected_height),
        max_per_class=10
    )
    
    print(f"   Loaded {len(images)} test images")
    
    # Test predictions
    print("\n3. Testing predictions...")
    predictions, confidences = classifier.predict_with_confidence(images)
    
    # Calculate accuracy
    correct = np.sum(predictions == labels)
    accuracy = correct / len(labels) * 100
    
    print(f"\n   Accuracy: {accuracy:.2f}% ({correct}/{len(labels)})")
    
    # Show some examples
    print("\n4. Sample predictions:")
    for i in range(min(5, len(images))):
        true_label = "Drowsy" if labels[i] == 1 else "Alert"
        pred_label = "Drowsy" if predictions[i] == 1 else "Alert"
        confidence = confidences[i] * 100
        status = "✓" if predictions[i] == labels[i] else "✗"
        
        print(f"   {status} Image {i+1}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.1f}%")
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    test_model()
