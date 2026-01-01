#!/usr/bin/env python3
"""
Mobile Service for Flutter Integration
Provides a simple stdin/stdout interface for drowsiness detection
"""

import sys
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detection.face_detector import FaceDetector
from face_detection.landmark_detector import FacialLandmarkDetector
from feature_extraction.ear_calculator import EARCalculator
from feature_extraction.mar_calculator import MARCalculator
from ml_models.cnn_classifier import CNNDrowsinessClassifier
from ml_models.feature_based_classifier import FeatureBasedClassifier
from decision_logic.decision_engine import DecisionEngine
from decision_logic.alert_manager import AlertManager


class MobileDetectionService:
    """Service for mobile drowsiness detection"""
    
    def __init__(self):
        """Initialize all components"""
        try:
            self.face_detector = FaceDetector()
            self.landmark_detector = FacialLandmarkDetector()
            self.ear_calculator = EARCalculator()
            self.mar_calculator = MARCalculator()
            
            # Try to load CNN model, fallback to feature-based
            try:
                self.cnn_model = CNNDrowsinessClassifier()
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_drowsiness_model.h5')
                if os.path.exists(model_path):
                    self.cnn_model.load_model(model_path)
                    self.use_cnn = True
                else:
                    self.use_cnn = False
                    self.feature_model = FeatureBasedClassifier()
                    feature_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_model.pkl')
                    if os.path.exists(feature_model_path):
                        self.feature_model.load_model(feature_model_path)
            except Exception as e:
                self.use_cnn = False
                self.feature_model = FeatureBasedClassifier()
            
            self.decision_engine = DecisionEngine()
            self.alert_manager = AlertManager()
            
            self.initialized = True
            
        except Exception as e:
            self.initialized = False
            raise Exception(f"Failed to initialize service: {str(e)}")
        
    def process_frame(self, image_data):
        """
        Process a single frame from mobile camera
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            dict: Detection results
        """
        try:
            # Decode image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                frame = np.array(image)
            else:
                frame = image_data
            
            # Convert RGB to BGR if needed (OpenCV uses BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = frame[:, :, ::-1]
            
            # Detect face
            face_result = self.face_detector.detectFace(frame)
            if not face_result['success']:
                return {
                    'success': False,
                    'face_detected': False,
                    'message': 'No face detected',
                    'drowsiness_score': 0.0,
                    'confidence': 0.0
                }
            
            # Extract landmarks
            landmarks = self.landmark_detector.detectLandmarks(frame)
            if landmarks is None:
                return {
                    'success': False,
                    'face_detected': True,
                    'message': 'Could not extract landmarks',
                    'drowsiness_score': 0.0,
                    'confidence': 0.0
                }
            
            # Calculate features
            ear = self.ear_calculator.calculateEAR(landmarks)
            mar = self.mar_calculator.calculateMAR(landmarks)
            
            # Get ML prediction
            face_bbox = face_result['bbox']
            x1, y1, x2, y2 = face_bbox
            
            # Ensure bbox is within image bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return {
                    'success': False,
                    'face_detected': True,
                    'message': 'Invalid face bounding box',
                    'drowsiness_score': 0.0,
                    'confidence': 0.0
                }
            
            face_img = frame[y1:y2, x1:x2]
            
            # Get ML score
            if self.use_cnn:
                ml_score = self.cnn_model.predict(face_img)
            else:
                features = np.array([[ear, mar]])
                ml_score = self.feature_model.predict(features)
            
            # Decision engine
            drowsiness_score = self.decision_engine.calculate_drowsiness_score(
                ear=ear,
                mar=mar,
                ml_score=ml_score
            )
            
            # Check for alerts
            alert = self.alert_manager.check_alert(drowsiness_score)
            
            # Convert landmarks to list format for JSON
            landmarks_list = []
            if landmarks is not None:
                for point in landmarks:
                    landmarks_list.append([int(point[0]), int(point[1])])
            
            return {
                'success': True,
                'face_detected': True,
                'drowsiness_score': float(drowsiness_score),
                'confidence': float(ml_score),
                'ear': float(ear),
                'mar': float(mar),
                'alert_level': alert['level'],
                'alert_message': alert['message'],
                'face_bbox': [int(x1), int(y1), int(x2), int(y2)],
                'landmarks': landmarks_list,
                'model_type': 'cnn' if self.use_cnn else 'feature_based'
            }
            
        except Exception as e:
            return {
                'success': False,
                'face_detected': False,
                'error': str(e),
                'drowsiness_score': 0.0,
                'confidence': 0.0
            }
    
    def get_status(self):
        """Get service status"""
        return {
            'initialized': self.initialized,
            'model_type': 'cnn' if self.use_cnn else 'feature_based',
            'components': {
                'face_detector': self.face_detector is not None,
                'landmark_detector': self.landmark_detector is not None,
                'ear_calculator': self.ear_calculator is not None,
                'mar_calculator': self.mar_calculator is not None,
                'decision_engine': self.decision_engine is not None,
                'alert_manager': self.alert_manager is not None
            }
        }


def main():
    """Main service loop - reads from stdin, writes to stdout"""
    try:
        service = MobileDetectionService()
        
        # Signal ready
        print(json.dumps({'status': 'ready', 'service': service.get_status()}), flush=True)
        
        # Process requests
        for line in sys.stdin:
            try:
                line = line.strip()
                if not line:
                    continue
                    
                request = json.loads(line)
                command = request.get('command')
                
                if command == 'process_frame':
                    image_data = request.get('image_data')
                    result = service.process_frame(image_data)
                    print(json.dumps(result), flush=True)
                    
                elif command == 'ping':
                    print(json.dumps({'status': 'alive'}), flush=True)
                    
                elif command == 'get_status':
                    status = service.get_status()
                    print(json.dumps(status), flush=True)
                    
                elif command == 'shutdown':
                    print(json.dumps({'status': 'shutdown'}), flush=True)
                    break
                    
                else:
                    print(json.dumps({
                        'success': False,
                        'error': f'Unknown command: {command}'
                    }), flush=True)
                    
            except json.JSONDecodeError as e:
                error_response = {
                    'success': False,
                    'error': f'JSON decode error: {str(e)}'
                }
                print(json.dumps(error_response), flush=True)
                
            except Exception as e:
                error_response = {
                    'success': False,
                    'error': str(e)
                }
                print(json.dumps(error_response), flush=True)
                
    except Exception as e:
        error_response = {
            'status': 'error',
            'error': f'Service initialization failed: {str(e)}'
        }
        print(json.dumps(error_response), flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
