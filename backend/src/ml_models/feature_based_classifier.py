"""
Feature-based traditional ML classifier for drowsiness detection.

This module provides traditional ML classifiers (SVM, Random Forest) that
use extracted features (EAR, MAR, head pose) for drowsiness classification.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .base_model import MLModel


class FeatureBasedClassifier(MLModel):
    """Traditional ML classifier using extracted features."""
    
    def __init__(
        self,
        model_type: str = "ensemble",
        model_path: Optional[str] = None
    ):
        """
        Initialize the feature-based classifier.
        
        Args:
            model_type: Type of classifier ('svm', 'random_forest', or 'ensemble')
            model_path: Path to saved model (optional)
        """
        super().__init__(model_path)
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_names = ['left_ear', 'right_ear', 'avg_ear', 'mar', 'head_pitch', 'head_yaw']
        self.input_shape = (None, len(self.feature_names))
        self.output_shape = (None, 1)
        
        if model_path is not None:
            self.load_model(model_path)
    
    def build_model(self, model_type: str = "ensemble") -> Any:
        """
        Build the traditional ML model.
        
        Args:
            model_type: Type of classifier to build
            
        Returns:
            Scikit-learn classifier
        """
        if model_type == "svm":
            model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        
        elif model_type == "ensemble":
            # Ensemble of SVM and Random Forest
            svm = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            model = VotingClassifier(
                estimators=[('svm', svm), ('rf', rf)],
                voting='soft'
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Train the traditional ML model.
        
        Args:
            X_train: Training features of shape (N, num_features)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with validation metrics
        """
        print(f"Training {self.model_type} classifier...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Features: {X_train.shape[1]}")
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(self.model_type)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_loaded = True
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred)
        }
        
        print("\nValidation Performance:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model (.pkl file)
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                print(f"Error: Model file not found at {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.model_type = saved_data['model_type']
            self.feature_names = saved_data.get('feature_names', self.feature_names)
            
            self.model_path = str(model_path)
            self.is_loaded = True
            
            print(f"Successfully loaded model from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input features.
        
        Args:
            input_data: Input features of shape (batch_size, num_features) or (num_features,)
            
        Returns:
            Predictions (probabilities for drowsy class)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        # Ensure 2D array
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(input_scaled)
        
        # Return probability of drowsy class (class 1)
        return probabilities[:, 1].reshape(-1, 1)
    
    def predict_with_confidence(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference and return predictions with confidence scores.
        
        Args:
            input_data: Input features for prediction
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        # Get probabilities
        probabilities = self.predict(input_data).squeeze()
        
        # Convert to class predictions
        predictions = (probabilities > 0.5).astype(np.int32)
        
        # Confidence is the probability of the predicted class
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
        
        return predictions, confidence
    
    def save_model(self, save_path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            save_path: Path where the model should be saved (.pkl file)
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            if not self.is_loaded or self.model is None:
                print("Error: No model to save")
                return False
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model, scaler, and metadata
            save_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores (for Random Forest models).
        
        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        if not self.is_loaded or self.model is None:
            return None
        
        # Get the actual model (handle ensemble case)
        if self.model_type == "ensemble":
            # Get Random Forest from ensemble
            rf_model = self.model.named_estimators_.get('rf')
            if rf_model is None:
                return None
            model = rf_model
        elif self.model_type == "random_forest":
            model = self.model
        else:
            return None  # SVM doesn't have feature importance
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return {name: float(imp) for name, imp in zip(self.feature_names, importances)}
        
        return None


def extract_features_from_landmarks(
    landmarks: np.ndarray,
    ear_calculator,
    mar_calculator
) -> np.ndarray:
    """
    Extract features from facial landmarks.
    
    Args:
        landmarks: Facial landmarks array
        ear_calculator: EAR calculator instance
        mar_calculator: MAR calculator instance
        
    Returns:
        Feature vector of shape (6,): [left_ear, right_ear, avg_ear, mar, head_pitch, head_yaw]
    """
    # Extract EAR features
    left_ear = ear_calculator.calculate_ear(landmarks, eye='left')
    right_ear = ear_calculator.calculate_ear(landmarks, eye='right')
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Extract MAR feature
    mar = mar_calculator.calculate_mar(landmarks)
    
    # Placeholder for head pose (would need actual head pose estimation)
    head_pitch = 0.0
    head_yaw = 0.0
    
    features = np.array([left_ear, right_ear, avg_ear, mar, head_pitch, head_yaw])
    
    return features
