"""
TensorFlow Lite model implementation for mobile deployment.

This module provides a TensorFlow Lite model wrapper for efficient on-device
inference on mobile platforms.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import tensorflow as tf

from .base_model import MLModel


class TFLiteModel(MLModel):
    """TensorFlow Lite model implementation for mobile deployment."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the TFLite model.
        
        Args:
            model_path: Path to the .tflite model file
        """
        super().__init__(model_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a TensorFlow Lite model from disk.
        
        Args:
            model_path: Path to the .tflite model file
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                print(f"Error: Model file not found at {model_path}")
                return False
            
            # Load TFLite model and allocate tensors
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Store input/output shapes
            self.input_shape = self.input_details[0]['shape']
            self.output_shape = self.output_details[0]['shape']
            
            self.model_path = str(model_path)
            self.is_loaded = True
            
            # Store metadata
            self.model_metadata = {
                "input_dtype": str(self.input_details[0]['dtype']),
                "output_dtype": str(self.output_details[0]['dtype']),
                "quantized": self.input_details[0]['dtype'] != np.float32
            }
            
            print(f"Successfully loaded TFLite model from {model_path}")
            print(f"  Input shape: {self.input_shape}")
            print(f"  Output shape: {self.output_shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input data of shape (batch_size, H, W, C) or (H, W, C)
            
        Returns:
            Model predictions of shape (batch_size, num_classes)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        input_data = self.preprocess_input(input_data)
        
        # Validate input shape
        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input shape. Expected {self.input_shape}, got {input_data.shape}")
        
        # Handle batch processing
        batch_size = input_data.shape[0]
        predictions = []
        
        for i in range(batch_size):
            # Get single sample
            sample = input_data[i:i+1]
            
            # Convert to input dtype if needed
            if self.input_details[0]['dtype'] == np.uint8:
                sample = (sample * 255).astype(np.uint8)
            else:
                sample = sample.astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], sample)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(output[0])
        
        predictions = np.array(predictions)
        return self.postprocess_output(predictions)
    
    def predict_with_confidence(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference and return predictions with confidence scores.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Tuple of (predictions, confidence_scores)
            - predictions: Class predictions (0 or 1)
            - confidence_scores: Confidence for each prediction
        """
        # Get raw predictions (probabilities)
        probabilities = self.predict(input_data)
        
        # Handle both binary classification outputs
        if probabilities.shape[-1] == 1:
            # Single output (sigmoid)
            probs = probabilities.squeeze()
            predictions = (probs > 0.5).astype(np.int32)
            confidence = np.where(predictions == 1, probs, 1 - probs)
        else:
            # Two outputs (softmax)
            predictions = np.argmax(probabilities, axis=-1)
            confidence = np.max(probabilities, axis=-1)
        
        return predictions, confidence
    
    def save_model(self, save_path: str) -> bool:
        """
        Save the TFLite model to disk.
        
        Note: TFLite models are typically already saved. This method
        copies the model to a new location if needed.
        
        Args:
            save_path: Path where the model should be saved
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            if not self.is_loaded or self.model_path is None:
                print("Error: No model loaded to save")
                return False
            
            import shutil
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(self.model_path, save_path)
            print(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def get_input_dtype(self) -> np.dtype:
        """Get the expected input data type."""
        if self.input_details is None:
            return np.float32
        return self.input_details[0]['dtype']
    
    def get_output_dtype(self) -> np.dtype:
        """Get the output data type."""
        if self.output_details is None:
            return np.float32
        return self.output_details[0]['dtype']
