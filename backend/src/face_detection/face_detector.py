"""
Face Detection Module using MediaPipe

This module provides face detection capabilities optimized for real-time
drowsiness detection. It uses MediaPipe's Face Detection solution for
efficient mobile-optimized face detection.

Validates: Requirements 1.1, 1.3, 1.4
"""

import time
from typing import Optional, Tuple, List
import cv2
import mediapipe as mp
import numpy as np


class FaceDetectionResult:
    """Container for face detection results"""
    
    def __init__(
        self,
        face_detected: bool,
        confidence: float,
        bounding_box: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[List[Tuple[float, float]]] = None,
        timestamp: float = None
    ):
        self.face_detected = face_detected
        self.confidence = confidence
        self.bounding_box = bounding_box  # (x, y, width, height)
        self.landmarks = landmarks
        self.timestamp = timestamp or time.time()
    
    def is_valid(self) -> bool:
        """Check if detection result is valid for further processing"""
        return (
            self.face_detected and
            self.confidence > 0.5 and
            self.bounding_box is not None
        )


class FaceDetector:
    """
    Face detection class using MediaPipe Face Detection.
    
    Optimized for real-time performance on mobile devices with
    quality validation and confidence scoring.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        model_selection: int = 0  # 0 for short-range (< 2m), 1 for full-range
    ):
        """
        Initialize the FaceDetector.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            model_selection: 0 for short-range detection, 1 for full-range
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        
        # Tracking state
        self.previous_detection: Optional[FaceDetectionResult] = None
        self.detection_history: List[FaceDetectionResult] = []
        self.max_history_size = 30  # Keep last 30 detections (~1-2 seconds at 15-30 FPS)
        
        # Performance metrics
        self.initialization_time = time.time()
        self.frame_count = 0
        self.total_processing_time = 0.0
    
    def detectFace(self, frame: np.ndarray) -> FaceDetectionResult:
        """
        Detect face in a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
        
        Returns:
            FaceDetectionResult containing detection information
        
        Validates: Requirements 1.1, 1.4
        """
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            return FaceDetectionResult(
                face_detected=False,
                confidence=0.0,
                timestamp=start_time
            )
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Process the frame
        results = self.face_detection.process(frame_rgb)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.frame_count += 1
        self.total_processing_time += processing_time
        
        if results.detections:
            # Get the first (most confident) detection
            detection = results.detections[0]
            confidence = detection.score[0]
            
            # Extract bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            # Extract key landmarks (6 points from MediaPipe)
            landmarks = []
            if detection.location_data.relative_keypoints:
                for landmark in detection.location_data.relative_keypoints:
                    landmarks.append((landmark.x * width, landmark.y * height))
            
            result = FaceDetectionResult(
                face_detected=True,
                confidence=confidence,
                bounding_box=(x, y, w, h),
                landmarks=landmarks,
                timestamp=start_time
            )
            
            # Update tracking state
            self.previous_detection = result
            self._update_history(result)
            
            return result
        else:
            # No face detected
            result = FaceDetectionResult(
                face_detected=False,
                confidence=0.0,
                timestamp=start_time
            )
            self._update_history(result)
            return result
    
    def trackFace(self, previous_frame: np.ndarray, current_frame: np.ndarray) -> FaceDetectionResult:
        """
        Track face across frames with temporal consistency.
        
        Uses previous detection information to improve tracking stability
        and reduce false negatives during brief occlusions.
        
        Args:
            previous_frame: Previous frame (not currently used but available for optical flow)
            current_frame: Current frame to detect face in
        
        Returns:
            FaceDetectionResult with tracking information
        
        Validates: Requirements 1.3
        """
        # Detect face in current frame
        current_result = self.detectFace(current_frame)
        
        # If face detected, return immediately
        if current_result.face_detected:
            return current_result
        
        # If no face detected but we have recent history, check if it's a brief occlusion
        if self.previous_detection and self.previous_detection.face_detected:
            time_since_last = current_result.timestamp - self.previous_detection.timestamp
            
            # If less than 1 second since last detection, might be brief occlusion
            if time_since_last < 1.0:
                # Return a low-confidence result indicating tracking loss
                return FaceDetectionResult(
                    face_detected=False,
                    confidence=0.3,  # Low confidence to indicate uncertainty
                    bounding_box=self.previous_detection.bounding_box,
                    timestamp=current_result.timestamp
                )
        
        return current_result
    
    def validateFaceQuality(self, detection_result: FaceDetectionResult) -> Tuple[bool, str]:
        """
        Validate if detected face is suitable for drowsiness analysis.
        
        Checks face size, confidence, and position to ensure quality.
        
        Args:
            detection_result: Face detection result to validate
        
        Returns:
            Tuple of (is_valid, reason) where reason explains validation failure
        
        Validates: Requirements 1.4
        """
        if not detection_result.face_detected:
            return False, "No face detected"
        
        if detection_result.confidence < self.min_detection_confidence:
            return False, f"Low confidence: {detection_result.confidence:.2f}"
        
        if detection_result.bounding_box is None:
            return False, "No bounding box available"
        
        x, y, w, h = detection_result.bounding_box
        
        # Check face size (should be at least 80x80 pixels for good feature extraction)
        if w < 80 or h < 80:
            return False, f"Face too small: {w}x{h} pixels"
        
        # Check if face is too close to edges (might be partially out of frame)
        # This is a simplified check - in production, you'd need frame dimensions
        if x < 10 or y < 10:
            return False, "Face too close to frame edge"
        
        # Check aspect ratio (faces should be roughly rectangular)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, f"Unusual face aspect ratio: {aspect_ratio:.2f}"
        
        return True, "Valid face detection"
    
    def handleMultipleFaces(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """
        Handle scenarios with multiple faces in frame.
        
        Returns all detected faces sorted by confidence.
        
        Args:
            frame: Input image
        
        Returns:
            List of FaceDetectionResult objects, sorted by confidence (highest first)
        """
        if frame is None or frame.size == 0:
            return []
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        results = self.face_detection.process(frame_rgb)
        
        if not results.detections:
            return []
        
        all_detections = []
        timestamp = time.time()
        
        for detection in results.detections:
            confidence = detection.score[0]
            
            # Extract bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            # Extract landmarks
            landmarks = []
            if detection.location_data.relative_keypoints:
                for landmark in detection.location_data.relative_keypoints:
                    landmarks.append((landmark.x * width, landmark.y * height))
            
            result = FaceDetectionResult(
                face_detected=True,
                confidence=confidence,
                bounding_box=(x, y, w, h),
                landmarks=landmarks,
                timestamp=timestamp
            )
            
            all_detections.append(result)
        
        # Sort by confidence (highest first)
        all_detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_detections
    
    def adaptToLighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Adapt frame preprocessing for different lighting conditions.
        
        Applies histogram equalization and contrast adjustment to improve
        detection in poor lighting.
        
        Args:
            frame: Input frame
        
        Returns:
            Preprocessed frame optimized for current lighting
        
        Validates: Requirements 1.4
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_frame
    
    def getAverageProcessingTime(self) -> float:
        """Get average processing time per frame in seconds"""
        if self.frame_count == 0:
            return 0.0
        return self.total_processing_time / self.frame_count
    
    def getDetectionRate(self) -> float:
        """Get percentage of frames with successful face detection"""
        if not self.detection_history:
            return 0.0
        
        successful = sum(1 for d in self.detection_history if d.face_detected)
        return successful / len(self.detection_history)
    
    def _update_history(self, result: FaceDetectionResult):
        """Update detection history with size limit"""
        self.detection_history.append(result)
        if len(self.detection_history) > self.max_history_size:
            self.detection_history.pop(0)
    
    def reset(self):
        """Reset detector state and history"""
        self.previous_detection = None
        self.detection_history.clear()
        self.frame_count = 0
        self.total_processing_time = 0.0
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
