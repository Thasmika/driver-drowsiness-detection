"""
Facial Landmark Detection Module using MediaPipe Face Mesh

This module provides facial landmark detection capabilities for extracting
precise facial feature points needed for drowsiness analysis.

Validates: Requirements 1.3, 5.2
"""

import time
from typing import Optional, List, Tuple
import cv2
import mediapipe as mp
import numpy as np


class FacialLandmarks:
    """Container for facial landmark data"""
    
    def __init__(
        self,
        landmarks: List[Tuple[float, float, float]],
        confidence: float,
        timestamp: float = None
    ):
        self.landmarks = landmarks  # List of (x, y, z) coordinates
        self.confidence = confidence
        self.timestamp = timestamp or time.time()
        
        # Pre-compute landmark subsets for common features
        self._left_eye_indices = None
        self._right_eye_indices = None
        self._mouth_indices = None
        self._nose_indices = None
        self._jawline_indices = None
        self._eyebrow_indices = None
    
    def get_left_eye(self) -> List[Tuple[float, float, float]]:
        """Get left eye landmarks"""
        if self._left_eye_indices is None:
            # MediaPipe Face Mesh left eye indices
            self._left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        return [self.landmarks[i] for i in self._left_eye_indices if i < len(self.landmarks)]
    
    def get_right_eye(self) -> List[Tuple[float, float, float]]:
        """Get right eye landmarks"""
        if self._right_eye_indices is None:
            # MediaPipe Face Mesh right eye indices
            self._right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        return [self.landmarks[i] for i in self._right_eye_indices if i < len(self.landmarks)]
    
    def get_mouth(self) -> List[Tuple[float, float, float]]:
        """Get mouth landmarks"""
        if self._mouth_indices is None:
            # MediaPipe Face Mesh mouth indices (outer lip)
            self._mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
        return [self.landmarks[i] for i in self._mouth_indices if i < len(self.landmarks)]
    
    def get_nose(self) -> List[Tuple[float, float, float]]:
        """Get nose landmarks"""
        if self._nose_indices is None:
            # MediaPipe Face Mesh nose indices
            self._nose_indices = [1, 2, 98, 327]
        return [self.landmarks[i] for i in self._nose_indices if i < len(self.landmarks)]
    
    def get_jawline(self) -> List[Tuple[float, float, float]]:
        """Get jawline landmarks"""
        if self._jawline_indices is None:
            # MediaPipe Face Mesh jawline indices
            self._jawline_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        return [self.landmarks[i] for i in self._jawline_indices if i < len(self.landmarks)]
    
    def get_eyebrows(self) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Get left and right eyebrow landmarks"""
        if self._eyebrow_indices is None:
            # MediaPipe Face Mesh eyebrow indices
            left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
            right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
            self._eyebrow_indices = (left_eyebrow, right_eyebrow)
        
        left_indices, right_indices = self._eyebrow_indices
        left_eyebrow = [self.landmarks[i] for i in left_indices if i < len(self.landmarks)]
        right_eyebrow = [self.landmarks[i] for i in right_indices if i < len(self.landmarks)]
        return left_eyebrow, right_eyebrow
    
    def to_68_point_format(self) -> np.ndarray:
        """
        Convert MediaPipe 468-point landmarks to traditional 68-point format.
        
        This provides compatibility with traditional dlib-based methods.
        
        Returns:
            numpy array of shape (68, 2) with (x, y) coordinates
        """
        # Mapping from MediaPipe 468 points to dlib 68 points
        # This is an approximation based on corresponding facial features
        mapping = {
            # Jawline (0-16)
            0: 234, 1: 93, 2: 132, 3: 58, 4: 172, 5: 136, 6: 150, 7: 149,
            8: 152, 9: 377, 10: 400, 11: 378, 12: 379, 13: 365, 14: 397,
            15: 288, 16: 361,
            # Right eyebrow (17-21)
            17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
            # Left eyebrow (22-26)
            22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
            # Nose bridge (27-30)
            27: 168, 28: 6, 29: 197, 30: 195,
            # Nose bottom (31-35)
            31: 5, 32: 4, 33: 1, 34: 19, 35: 94,
            # Right eye (36-41)
            36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
            # Left eye (42-47)
            42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
            # Outer lip (48-59)
            48: 61, 49: 146, 50: 91, 51: 181, 52: 84, 53: 17,
            54: 314, 55: 405, 56: 321, 57: 375, 58: 291, 59: 308,
            # Inner lip (60-67)
            60: 78, 61: 95, 62: 88, 63: 178, 64: 87, 65: 14,
            66: 317, 67: 402
        }
        
        landmarks_68 = np.zeros((68, 2), dtype=np.float32)
        for dlib_idx, mediapipe_idx in mapping.items():
            if mediapipe_idx < len(self.landmarks):
                x, y, _ = self.landmarks[mediapipe_idx]
                landmarks_68[dlib_idx] = [x, y]
        
        return landmarks_68


class FacialLandmarkDetector:
    """
    Facial landmark detection class using MediaPipe Face Mesh.
    
    Extracts 468 facial landmarks with high precision for detailed
    drowsiness analysis.
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the FacialLandmarkDetector.
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.previous_landmarks: Optional[FacialLandmarks] = None
    
    def extractLandmarks(self, face_region: np.ndarray) -> Optional[FacialLandmarks]:
        """
        Extract facial landmarks from a face region.
        
        Args:
            face_region: Face image region (BGR format from OpenCV)
        
        Returns:
            FacialLandmarks object or None if extraction fails
        
        Validates: Requirements 1.3, 5.2
        """
        start_time = time.time()
        
        if face_region is None or face_region.size == 0:
            return None
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        height, width = face_region.shape[:2]
        
        # Process the frame
        results = self.face_mesh.process(frame_rgb)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.frame_count += 1
        self.total_processing_time += processing_time
        
        if results.multi_face_landmarks:
            # Get the first face's landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert normalized landmarks to pixel coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = landmark.x * width
                y = landmark.y * height
                z = landmark.z * width  # Z is relative to width
                landmarks.append((x, y, z))
            
            # Calculate confidence (MediaPipe doesn't provide per-landmark confidence)
            # Use a heuristic based on landmark visibility
            confidence = self._calculate_confidence(landmarks, width, height)
            
            result = FacialLandmarks(
                landmarks=landmarks,
                confidence=confidence,
                timestamp=start_time
            )
            
            self.previous_landmarks = result
            return result
        else:
            return None
    
    def getLandmarkSubset(self, landmarks: FacialLandmarks, landmark_type: str) -> List[Tuple[float, float, float]]:
        """
        Extract specific facial features from landmarks.
        
        Args:
            landmarks: FacialLandmarks object
            landmark_type: Type of feature ('left_eye', 'right_eye', 'mouth', 'nose', 'jawline', 'eyebrows')
        
        Returns:
            List of landmark coordinates for the specified feature
        
        Validates: Requirements 1.3
        """
        if landmark_type == 'left_eye':
            return landmarks.get_left_eye()
        elif landmark_type == 'right_eye':
            return landmarks.get_right_eye()
        elif landmark_type == 'mouth':
            return landmarks.get_mouth()
        elif landmark_type == 'nose':
            return landmarks.get_nose()
        elif landmark_type == 'jawline':
            return landmarks.get_jawline()
        elif landmark_type == 'eyebrows':
            return landmarks.get_eyebrows()
        else:
            raise ValueError(f"Unknown landmark type: {landmark_type}")
    
    def validateLandmarkQuality(self, landmarks: FacialLandmarks) -> Tuple[bool, str]:
        """
        Validate if extracted landmarks are suitable for drowsiness analysis.
        
        Checks landmark completeness, confidence, and spatial consistency.
        
        Args:
            landmarks: FacialLandmarks object to validate
        
        Returns:
            Tuple of (is_valid, reason) where reason explains validation failure
        
        Validates: Requirements 1.3
        """
        if landmarks is None:
            return False, "No landmarks detected"
        
        if landmarks.confidence < self.min_detection_confidence:
            return False, f"Low confidence: {landmarks.confidence:.2f}"
        
        if len(landmarks.landmarks) < 468:
            return False, f"Incomplete landmarks: {len(landmarks.landmarks)}/468"
        
        # Check if key features are present and valid
        left_eye = landmarks.get_left_eye()
        right_eye = landmarks.get_right_eye()
        mouth = landmarks.get_mouth()
        
        if len(left_eye) < 10 or len(right_eye) < 10:
            return False, "Insufficient eye landmarks"
        
        if len(mouth) < 15:
            return False, "Insufficient mouth landmarks"
        
        # Check spatial consistency (eyes should be roughly at same height)
        if left_eye and right_eye:
            left_eye_y = np.mean([p[1] for p in left_eye])
            right_eye_y = np.mean([p[1] for p in right_eye])
            eye_height_diff = abs(left_eye_y - right_eye_y)
            
            # Eyes should be within 20% of average eye height
            avg_eye_height = (left_eye_y + right_eye_y) / 2
            if eye_height_diff > avg_eye_height * 0.2:
                return False, f"Inconsistent eye positions: {eye_height_diff:.1f}px difference"
        
        return True, "Valid landmarks"
    
    def normalizeLandmarks(self, landmarks: FacialLandmarks, reference_width: int = 640, reference_height: int = 480) -> FacialLandmarks:
        """
        Normalize landmarks to a reference resolution.
        
        This ensures consistent feature extraction across different image sizes.
        
        Args:
            landmarks: FacialLandmarks to normalize
            reference_width: Target width for normalization
            reference_height: Target height for normalization
        
        Returns:
            Normalized FacialLandmarks object
        
        Validates: Requirements 5.2
        """
        if landmarks is None or not landmarks.landmarks:
            return landmarks
        
        # Calculate current bounding box
        xs = [p[0] for p in landmarks.landmarks]
        ys = [p[1] for p in landmarks.landmarks]
        
        current_width = max(xs) - min(xs)
        current_height = max(ys) - min(ys)
        
        if current_width == 0 or current_height == 0:
            return landmarks
        
        # Calculate scaling factors
        scale_x = reference_width / current_width
        scale_y = reference_height / current_height
        
        # Normalize landmarks
        normalized = []
        for x, y, z in landmarks.landmarks:
            norm_x = x * scale_x
            norm_y = y * scale_y
            norm_z = z * scale_x  # Z uses same scale as X
            normalized.append((norm_x, norm_y, norm_z))
        
        return FacialLandmarks(
            landmarks=normalized,
            confidence=landmarks.confidence,
            timestamp=landmarks.timestamp
        )
    
    def _calculate_confidence(self, landmarks: List[Tuple[float, float, float]], width: int, height: int) -> float:
        """
        Calculate confidence score based on landmark quality.
        
        Uses heuristics like landmark spread and boundary proximity.
        """
        if not landmarks:
            return 0.0
        
        # Check if landmarks are well-distributed
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        
        # Calculate spread
        x_spread = (max(xs) - min(xs)) / width
        y_spread = (max(ys) - min(ys)) / height
        
        # Good landmarks should cover significant portion of image
        spread_score = min(x_spread * y_spread * 4, 1.0)  # Normalize to 0-1
        
        # Check if landmarks are too close to boundaries
        boundary_margin = 0.05
        boundary_violations = sum(1 for x in xs if x < width * boundary_margin or x > width * (1 - boundary_margin))
        boundary_violations += sum(1 for y in ys if y < height * boundary_margin or y > height * (1 - boundary_margin))
        
        boundary_score = max(0, 1.0 - (boundary_violations / len(landmarks)))
        
        # Combine scores
        confidence = (spread_score * 0.6 + boundary_score * 0.4)
        
        return confidence
    
    def getAverageProcessingTime(self) -> float:
        """Get average processing time per frame in seconds"""
        if self.frame_count == 0:
            return 0.0
        return self.total_processing_time / self.frame_count
    
    def reset(self):
        """Reset detector state"""
        self.previous_landmarks = None
        self.frame_count = 0
        self.total_processing_time = 0.0
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
