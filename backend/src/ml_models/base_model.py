"""
Base ML model interface for drowsiness detection.

This module provides an abstract base class for all ML models used in the
drowsiness detection system, ensuring consistent interfaces and behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path


class MLModel(ABC):
    """Abstract base class for all ML models in the drowsiness detection system."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.input_shape = None
        self.output_shape = None
        self.model_metadata = {}
        
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if loading was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def predict_with_confidence(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference and return predictions with confidence scores.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        pass
    
    @abstractmethod
    def save_model(self, save_path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            save_path: Path where the model should be saved
            
        Returns:
            True if saving was successful, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_type": self.__class__.__name__,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            **self.model_metadata
        }
    
    def validate_input(self, input_data: np.ndarray) -> bool:
        """
        Validate that input data has the correct shape.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if self.input_shape is None:
            return True  # No validation if input shape not set
        
        # Check if input matches expected shape (ignoring batch dimension)
        expected_shape = self.input_shape[1:] if len(self.input_shape) > 1 else self.input_shape
        actual_shape = input_data.shape[1:] if len(input_data.shape) > 1 else input_data.shape
        
        return actual_shape == expected_shape
    
    def preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data before inference.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Preprocessed input data
        """
        # Default implementation: ensure float32 and add batch dimension if needed
        data = input_data.astype(np.float32)
        
        if len(data.shape) == 3:  # Single image (H, W, C)
            data = np.expand_dims(data, axis=0)  # Add batch dimension
        
        return data
    
    def postprocess_output(self, output_data: np.ndarray) -> np.ndarray:
        """
        Postprocess model output.
        
        Args:
            output_data: Raw model output
            
        Returns:
            Postprocessed output
        """
        # Default implementation: return as-is
        return output_data
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_path={self.model_path}, is_loaded={self.is_loaded})"
