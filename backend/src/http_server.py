#!/usr/bin/env python3
"""
HTTP Server for Flutter Integration
Simple REST API for testing drowsiness detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import sys
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

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter web testing

# Global service instance
service = None


class HTTPDetectionService:
    """HTTP-based detection service"""
    
    def __init__(self):
        """Initialize all components"""
        print("Initializing detection service...")
        
        self.face_detector = FaceDetector()
        self.landmark_detector = FacialLandmarkDetector()
        self.ear_calculator = EARCalculator()
        self.mar_calculator = MARCalculator()
        
        # Try to load CNN model, fallback to feature-based
        try:
            self.cnn_model = CNNDrowsinessClassifier()
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_drowsiness.h5')
            if os.path.exists(model_path):
                self.cnn_model.load_model(model_path)
                self.use_cnn = True
                print("✓ CNN model loaded")
            else:
                self.use_cnn = False
                self.feature_model = FeatureBasedClassifier()
                print(f"⚠ CNN model not found at {model_path}, using feature-based classifier")
        except Exception as e:
            self.use_cnn = False
            self.feature_model = FeatureBasedClassifier()
            print(f"⚠ CNN model failed to load: {e}")
        
        self.decision_engine = DecisionEngine()
        self.alert_manager = AlertManager()
        
        print("✓ Service initialized successfully")
    
    def process_frame(self, image_data):
        """Process a frame and return detection results"""
        try:
            # Decode image
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                frame = np.array(image)
            else:
                frame = image_data
            
            # Convert RGB to BGR if needed
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
            
            # Ensure bbox is within bounds
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
            
            # Convert landmarks to list
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
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'face_detected': False,
                'error': str(e),
                'drowsiness_score': 0.0,
                'confidence': 0.0
            }


@app.route('/')
def index():
    """API information"""
    return jsonify({
        'name': 'Drowsiness Detection API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/process': 'Process frame (POST)',
            '/status': 'Service status'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'running'
    })


@app.route('/status')
def status():
    """Get service status"""
    if service is None:
        return jsonify({
            'initialized': False,
            'error': 'Service not initialized'
        }), 500
    
    return jsonify({
        'initialized': True,
        'model_type': 'cnn' if service.use_cnn else 'feature_based',
        'components': {
            'face_detector': service.face_detector is not None,
            'landmark_detector': service.landmark_detector is not None,
            'ear_calculator': service.ear_calculator is not None,
            'mar_calculator': service.mar_calculator is not None,
            'decision_engine': service.decision_engine is not None,
            'alert_manager': service.alert_manager is not None
        }
    })


@app.route('/process', methods=['POST'])
def process():
    """Process a frame"""
    if service is None:
        return jsonify({
            'success': False,
            'error': 'Service not initialized'
        }), 500
    
    try:
        data = request.json
        if not data or 'image_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image_data in request'
            }), 400
        
        image_data = data['image_data']
        result = service.process_frame(image_data)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    """Start the HTTP server"""
    global service
    
    print("="*60)
    print("Drowsiness Detection HTTP Server")
    print("="*60)
    
    # Initialize service
    try:
        service = HTTPDetectionService()
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Start server
    print("\n" + "="*60)
    print("Server starting...")
    print("="*60)
    print("\nEndpoints:")
    print("  - http://localhost:5000/")
    print("  - http://localhost:5000/health")
    print("  - http://localhost:5000/status")
    print("  - http://localhost:5000/process (POST)")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
