"""
CNN-based drowsiness classifier implementation.

This module provides a CNN model for end-to-end drowsiness classification
from face images, with support for training and TensorFlow Lite conversion.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from .base_model import MLModel


class CNNDrowsinessClassifier(MLModel):
    """CNN-based drowsiness classifier extending MLModel."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        model_path: Optional[str] = None
    ):
        """
        Initialize the CNN classifier.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            model_path: Path to saved model (optional)
        """
        super().__init__(model_path)
        self.input_shape = (None,) + input_shape  # Add batch dimension
        self.output_shape = (None, 1)  # Binary classification
        
        if model_path is not None:
            self.load_model(model_path)
    
    def build_model(self, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> keras.Model:
        """
        Build the CNN architecture.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history dictionary
        """
        # Build model if not already built
        if self.model is None:
            input_shape = X_train.shape[1:]
            self.model = self.build_model(input_shape)
            self.input_shape = (None,) + input_shape
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        
        # Train model
        print(f"Training CNN model...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Input shape: {X_train.shape[1:]}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_loaded = True
        
        return history.history
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained Keras model from disk.
        
        Args:
            model_path: Path to the saved model (.h5 or SavedModel format)
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                print(f"Error: Model file not found at {model_path}")
                return False
            
            self.model = keras.models.load_model(str(model_path))
            self.model_path = str(model_path)
            self.is_loaded = True
            
            # Update input/output shapes
            self.input_shape = self.model.input_shape
            self.output_shape = self.model.output_shape
            
            print(f"Successfully loaded model from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input images of shape (batch_size, H, W, C) or (H, W, C)
            
        Returns:
            Predictions of shape (batch_size, 1) with probabilities
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        # Preprocess input
        input_data = self.preprocess_input(input_data)
        
        # Run prediction
        predictions = self.model.predict(input_data, verbose=0)
        
        return predictions
    
    def predict_with_confidence(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference and return predictions with confidence scores.
        
        Args:
            input_data: Input images for prediction
            
        Returns:
            Tuple of (predictions, confidence_scores)
            - predictions: Class predictions (0=alert, 1=drowsy)
            - confidence_scores: Confidence for each prediction
        """
        # Get probabilities
        probabilities = self.predict(input_data).squeeze()
        
        # Convert to class predictions
        predictions = (probabilities > 0.5).astype(np.int32)
        
        # Confidence is the probability of the predicted class
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
        
        return predictions, confidence
    
    def save_model(self, save_path: str, save_format: str = 'h5') -> bool:
        """
        Save the trained model to disk.
        
        Args:
            save_path: Path where the model should be saved
            save_format: Format to save ('h5' or 'tf' for SavedModel)
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            if not self.is_loaded or self.model is None:
                print("Error: No model to save")
                return False
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_format == 'h5':
                self.model.save(str(save_path), save_format='h5')
            else:
                self.model.save(str(save_path))
            
            print(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def convert_to_tflite(
        self,
        save_path: str,
        quantize: bool = True
    ) -> bool:
        """
        Convert the Keras model to TensorFlow Lite format.
        
        Args:
            save_path: Path to save the .tflite model
            quantize: Whether to apply INT8 quantization
            
        Returns:
            True if conversion was successful, False otherwise
        """
        try:
            if not self.is_loaded or self.model is None:
                print("Error: No model to convert")
                return False
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
            
            tflite_model = converter.convert()
            
            # Save to file
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TFLite model saved to {save_path}")
            print(f"  Quantized: {quantize}")
            print(f"  Size: {len(tflite_model) / 1024:.2f} KB")
            
            return True
            
        except Exception as e:
            print(f"Error converting to TFLite: {e}")
            return False
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get metric names
        metric_names = self.model.metrics_names
        
        # Create results dictionary
        metrics = {name: value for name, value in zip(metric_names, results)}
        
        return metrics
