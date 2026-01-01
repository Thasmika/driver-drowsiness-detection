#!/usr/bin/env python3
"""
Desktop Test Application with USB Camera
Tests the drowsiness detection system using external USB camera
"""

import cv2
import numpy as np
import sys
import os
import time
import base64
import requests
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detection.face_detector import FaceDetector
from face_detection.landmark_detector import FacialLandmarkDetector
from feature_extraction.ear_calculator import EARCalculator
from feature_extraction.mar_calculator import MARCalculator
from decision_logic.decision_engine import DecisionEngine
from decision_logic.alert_manager import AlertManager


class WebcamTester:
    """Test drowsiness detection with webcam"""
    
    def __init__(self, camera_index=0, use_http=False):
        """
        Initialize webcam tester
        
        Args:
            camera_index: Camera device index (0 for default, 1 for external USB)
            use_http: If True, use HTTP server. If False, use local processing
        """
        self.camera_index = camera_index
        self.use_http = use_http
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if use_http:
            self.http_url = 'http://localhost:5000/process'
            print("Using HTTP server for processing")
        else:
            # Initialize local components
            print("Initializing local processing components...")
            self.face_detector = FaceDetector()
            self.landmark_detector = FacialLandmarkDetector()
            self.ear_calculator = EARCalculator()
            self.mar_calculator = MARCalculator()
            self.decision_engine = DecisionEngine()
            self.alert_manager = AlertManager()
            print("✓ Components initialized")
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
    def process_frame_local(self, frame):
        """Process frame locally"""
        # Detect face
        face_result = self.face_detector.detectFace(frame)
        if not face_result['success']:
            return {
                'success': False,
                'face_detected': False,
                'drowsiness_score': 0.0
            }
        
        # Extract landmarks
        landmarks = self.landmark_detector.detectLandmarks(frame)
        if landmarks is None:
            return {
                'success': False,
                'face_detected': True,
                'drowsiness_score': 0.0
            }
        
        # Calculate features
        ear = self.ear_calculator.calculateEAR(landmarks)
        mar = self.mar_calculator.calculateMAR(landmarks)
        
        # Calculate drowsiness score (using simple rule-based for now)
        drowsiness_score = self.decision_engine.calculate_drowsiness_score(
            ear=ear,
            mar=mar,
            ml_score=0.5  # Placeholder
        )
        
        # Check alert
        alert = self.alert_manager.check_alert(drowsiness_score)
        
        return {
            'success': True,
            'face_detected': True,
            'drowsiness_score': drowsiness_score,
            'ear': ear,
            'mar': mar,
            'alert_level': alert['level'],
            'alert_message': alert['message'],
            'face_bbox': face_result['bbox'],
            'landmarks': landmarks
        }
    
    def process_frame_http(self, frame):
        """Process frame via HTTP server"""
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Send to server
            response = requests.post(
                self.http_url,
                json={'image_data': jpg_as_text},
                timeout=3
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'success': False, 'face_detected': False, 'drowsiness_score': 0.0}
        except Exception as e:
            print(f"HTTP error: {e}")
            return {'success': False, 'face_detected': False, 'drowsiness_score': 0.0}
    
    def draw_results(self, frame, result):
        """Draw detection results on frame"""
        h, w = frame.shape[:2]
        
        # Draw face bounding box
        if result.get('face_detected') and 'face_bbox' in result:
            bbox = result['face_bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Draw landmarks
        if 'landmarks' in result and result['landmarks'] is not None:
            landmarks = result['landmarks']
            if isinstance(landmarks, np.ndarray):
                for point in landmarks:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 255, 255), -1)
        
        # Draw status panel
        drowsiness_score = result.get('drowsiness_score', 0.0)
        alert_level = result.get('alert_level', 'none')
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w-10, 150), (255, 255, 255), 2)
        
        # Status text
        if drowsiness_score < 0.3:
            status = "ALERT - Normal"
            color = (0, 255, 0)
        elif drowsiness_score < 0.6:
            status = "WARNING - Drowsy"
            color = (0, 255, 255)
        else:
            status = "DANGER - Very Drowsy"
            color = (0, 0, 255)
        
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Drowsiness: {drowsiness_score*100:.1f}%", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if 'ear' in result:
            cv2.putText(frame, f"EAR: {result['ear']:.3f}", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if 'mar' in result:
            cv2.putText(frame, f"MAR: {result['mar']:.3f}", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Run the webcam test"""
        print("\n" + "="*60)
        print("Drowsiness Detection - Webcam Test")
        print("="*60)
        print(f"Camera: {self.camera_index}")
        print(f"Processing: {'HTTP Server' if self.use_http else 'Local'}")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("="*60 + "\n")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Process frame
                if self.use_http:
                    result = self.process_frame_http(frame)
                else:
                    result = self.process_frame_local(frame)
                
                # Draw results
                display_frame = self.draw_results(frame.copy(), result)
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    current_time = time.time()
                    self.fps = 10 / (current_time - self.last_time)
                    self.last_time = current_time
                
                # Display
                cv2.imshow('Drowsiness Detection', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Screenshot saved: {filename}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nTest completed")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test drowsiness detection with webcam')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (0=default, 1=external USB)')
    parser.add_argument('--http', action='store_true',
                       help='Use HTTP server instead of local processing')
    
    args = parser.parse_args()
    
    try:
        tester = WebcamTester(camera_index=args.camera, use_http=args.http)
        tester.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
