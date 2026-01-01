"""
Property-Based Tests for Face Detection Module

Feature: driver-drowsiness-detection
Tests universal properties that should hold across all valid inputs.

Validates: Requirements 1.1, 1.3, 1.4
"""

import time
import numpy as np
import cv2
from hypothesis import given, strategies as st, settings, assume
import pytest

from src.face_detection import FaceDetector, FaceDetectionResult


# Generators for test data
@st.composite
def valid_image_frame(draw):
    """Generate valid image frames with various properties"""
    # Generate reasonable image dimensions
    width = draw(st.integers(min_value=320, max_value=1920))
    height = draw(st.integers(min_value=240, max_value=1080))
    
    # Generate random image data (BGR format)
    # Using uint8 for realistic image data
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    return frame


@st.composite
def image_with_synthetic_face(draw):
    """Generate image frames with synthetic face-like regions"""
    width = draw(st.integers(min_value=640, max_value=1280))
    height = draw(st.integers(min_value=480, max_value=720))
    
    # Create base image
    frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add a face-like ellipse in the center (simplified face simulation)
    center_x = width // 2
    center_y = height // 2
    face_width = draw(st.integers(min_value=100, max_value=min(300, width // 3)))
    face_height = draw(st.integers(min_value=120, max_value=min(400, height // 3)))
    
    # Draw ellipse to simulate face region
    cv2.ellipse(
        frame,
        (center_x, center_y),
        (face_width, face_height),
        0, 0, 360,
        (180, 150, 130),  # Skin-tone color
        -1
    )
    
    # Add eye-like regions
    eye_y = center_y - face_height // 4
    left_eye_x = center_x - face_width // 3
    right_eye_x = center_x + face_width // 3
    
    cv2.circle(frame, (left_eye_x, eye_y), 15, (50, 50, 50), -1)
    cv2.circle(frame, (right_eye_x, eye_y), 15, (50, 50, 50), -1)
    
    # Add mouth-like region
    mouth_y = center_y + face_height // 3
    cv2.ellipse(
        frame,
        (center_x, mouth_y),
        (face_width // 4, face_height // 8),
        0, 0, 180,
        (100, 50, 50),
        -1
    )
    
    return frame


class TestFaceDetectionProperties:
    """Property-based tests for face detection functionality"""
    
    @given(frame=valid_image_frame())
    @settings(max_examples=100, deadline=5000)
    def test_property_1_face_detection_initialization_time(self, frame):
        """
        Property 1: Face Detection Initialization Time
        
        For any mobile device and camera configuration, when the facial analyzer
        is activated, face detection should complete within 2 seconds for valid
        face inputs.
        
        Feature: driver-drowsiness-detection, Property 1: Face Detection Initialization Time
        Validates: Requirements 1.1
        """
        # Initialize detector
        detector = FaceDetector(min_detection_confidence=0.5)
        
        # Measure detection time
        start_time = time.time()
        result = detector.detectFace(frame)
        detection_time = time.time() - start_time
        
        # Property: Detection should complete within 2 seconds
        assert detection_time < 2.0, (
            f"Face detection took {detection_time:.3f}s, exceeding 2s requirement. "
            f"Frame size: {frame.shape}"
        )
        
        # Additional validation: result should be a valid FaceDetectionResult
        assert isinstance(result, FaceDetectionResult)
        assert result.timestamp is not None
        
        # Cleanup
        detector.reset()
    
    @given(
        frame=valid_image_frame(),
        min_confidence=st.floats(min_value=0.3, max_value=0.9)
    )
    @settings(max_examples=100, deadline=5000)
    def test_detection_completes_with_various_confidence_thresholds(self, frame, min_confidence):
        """
        Test that detection completes quickly regardless of confidence threshold.
        
        This validates that the initialization time property holds across
        different detector configurations.
        """
        detector = FaceDetector(min_detection_confidence=min_confidence)
        
        start_time = time.time()
        result = detector.detectFace(frame)
        detection_time = time.time() - start_time
        
        # Should still complete within 2 seconds
        assert detection_time < 2.0
        assert isinstance(result, FaceDetectionResult)
        
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=100, deadline=5000)
    def test_detection_returns_valid_result_structure(self, frame):
        """
        Test that detection always returns a properly structured result.
        
        For any input frame, the detector should return a FaceDetectionResult
        with all required fields populated.
        """
        detector = FaceDetector()
        result = detector.detectFace(frame)
        
        # Validate result structure
        assert isinstance(result, FaceDetectionResult)
        assert isinstance(result.face_detected, bool)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert result.timestamp is not None
        
        # If face detected, bounding box should be present
        if result.face_detected:
            assert result.bounding_box is not None
            assert len(result.bounding_box) == 4
            x, y, w, h = result.bounding_box
            assert w > 0 and h > 0
        
        detector.reset()
    
    @given(frames=st.lists(valid_image_frame(), min_size=2, max_size=5))
    @settings(max_examples=50, deadline=10000)
    def test_tracking_maintains_temporal_consistency(self, frames):
        """
        Test that face tracking maintains consistency across frames.
        
        For any sequence of frames, the trackFace method should handle
        frame transitions smoothly.
        """
        detector = FaceDetector()
        
        for i in range(len(frames) - 1):
            previous_frame = frames[i]
            current_frame = frames[i + 1]
            
            start_time = time.time()
            result = detector.trackFace(previous_frame, current_frame)
            tracking_time = time.time() - start_time
            
            # Tracking should also complete quickly
            assert tracking_time < 2.0
            assert isinstance(result, FaceDetectionResult)
        
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=100, deadline=5000)
    def test_multiple_face_detection_completes_quickly(self, frame):
        """
        Test that multiple face detection completes within time constraints.
        
        Even when detecting multiple faces, the operation should complete
        within the 2-second requirement.
        """
        detector = FaceDetector()
        
        start_time = time.time()
        results = detector.handleMultipleFaces(frame)
        detection_time = time.time() - start_time
        
        assert detection_time < 2.0
        assert isinstance(results, list)
        
        # All results should be valid FaceDetectionResult objects
        for result in results:
            assert isinstance(result, FaceDetectionResult)
            assert result.face_detected is True  # Only detected faces in list
        
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=50, deadline=5000)
    def test_lighting_adaptation_preserves_detection_speed(self, frame):
        """
        Test that lighting adaptation doesn't significantly impact detection time.
        
        Validates: Requirements 1.4 - lighting condition adaptation
        """
        detector = FaceDetector()
        
        # Apply lighting adaptation
        start_adapt = time.time()
        adapted_frame = detector.adaptToLighting(frame)
        adapt_time = time.time() - start_adapt
        
        # Adaptation should be fast
        assert adapt_time < 0.5
        
        # Detection on adapted frame should still be fast
        start_detect = time.time()
        result = detector.detectFace(adapted_frame)
        detect_time = time.time() - start_detect
        
        assert detect_time < 2.0
        assert isinstance(result, FaceDetectionResult)
        
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=100, deadline=5000)
    def test_quality_validation_is_deterministic(self, frame):
        """
        Test that quality validation produces consistent results.
        
        For any detection result, quality validation should be deterministic
        and complete quickly.
        """
        detector = FaceDetector()
        result = detector.detectFace(frame)
        
        # Validate quality multiple times - should be consistent
        is_valid_1, reason_1 = detector.validateFaceQuality(result)
        is_valid_2, reason_2 = detector.validateFaceQuality(result)
        
        assert is_valid_1 == is_valid_2
        assert reason_1 == reason_2
        assert isinstance(is_valid_1, bool)
        assert isinstance(reason_1, str)
        
        detector.reset()
    
    def test_empty_frame_handling(self):
        """
        Test that detector handles empty/invalid frames gracefully.
        
        Edge case: Empty or None frames should not crash the detector.
        """
        detector = FaceDetector()
        
        # Test with None
        result = detector.detectFace(None)
        assert isinstance(result, FaceDetectionResult)
        assert result.face_detected is False
        
        # Test with empty array
        empty_frame = np.array([])
        result = detector.detectFace(empty_frame)
        assert isinstance(result, FaceDetectionResult)
        assert result.face_detected is False
        
        detector.reset()
    
    @given(
        width=st.integers(min_value=320, max_value=1920),
        height=st.integers(min_value=240, max_value=1080)
    )
    @settings(max_examples=50, deadline=5000)
    def test_detection_works_across_resolutions(self, width, height):
        """
        Test that detection works across various image resolutions.
        
        For any valid image resolution, detection should complete successfully.
        """
        detector = FaceDetector()
        
        # Create frame with specified resolution
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        start_time = time.time()
        result = detector.detectFace(frame)
        detection_time = time.time() - start_time
        
        assert detection_time < 2.0
        assert isinstance(result, FaceDetectionResult)
        
        detector.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestFacialLandmarkProperties:
    """Property-based tests for facial landmark detection functionality"""
    
    @given(frame=valid_image_frame())
    @settings(max_examples=100, deadline=5000)
    def test_property_6_facial_feature_tracking_completeness(self, frame):
        """
        Property 6: Facial Feature Tracking Completeness
        
        For any detected face, the facial analyzer should successfully identify
        and track eyes, mouth, and head position continuously.
        
        Feature: driver-drowsiness-detection, Property 6: Facial Feature Tracking Completeness
        Validates: Requirements 1.3
        """
        from src.face_detection import FacialLandmarkDetector
        
        # Initialize landmark detector
        detector = FacialLandmarkDetector(
            min_detection_confidence=0.3,  # Lower threshold for testing
            min_tracking_confidence=0.3
        )
        
        # Extract landmarks
        landmarks = detector.extractLandmarks(frame)
        
        # If landmarks are detected, they should include all key features
        if landmarks and landmarks.confidence > 0.3:
            # Property: Should be able to extract eyes
            left_eye = landmarks.get_left_eye()
            right_eye = landmarks.get_right_eye()
            
            assert isinstance(left_eye, list), "Left eye should be a list"
            assert isinstance(right_eye, list), "Right eye should be a list"
            
            # If eyes are detected, they should have reasonable number of points
            if left_eye:
                assert len(left_eye) > 0, "Left eye should have landmarks"
            if right_eye:
                assert len(right_eye) > 0, "Right eye should have landmarks"
            
            # Property: Should be able to extract mouth
            mouth = landmarks.get_mouth()
            assert isinstance(mouth, list), "Mouth should be a list"
            
            # Property: Should be able to extract head position indicators (nose, jawline)
            nose = landmarks.get_nose()
            jawline = landmarks.get_jawline()
            
            assert isinstance(nose, list), "Nose should be a list"
            assert isinstance(jawline, list), "Jawline should be a list"
            
            # Property: All landmark coordinates should be tuples of 3 values (x, y, z)
            for landmark in landmarks.landmarks:
                assert isinstance(landmark, tuple), "Each landmark should be a tuple"
                assert len(landmark) == 3, "Each landmark should have x, y, z coordinates"
                x, y, z = landmark
                assert isinstance(x, (int, float)), "X coordinate should be numeric"
                assert isinstance(y, (int, float)), "Y coordinate should be numeric"
                assert isinstance(z, (int, float)), "Z coordinate should be numeric"
        
        # Cleanup
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=100, deadline=5000)
    def test_landmark_extraction_returns_valid_structure(self, frame):
        """
        Test that landmark extraction always returns a properly structured result.
        
        For any input frame, the detector should return either None or a valid
        FacialLandmarks object with all required fields.
        """
        from src.face_detection import FacialLandmarkDetector, FacialLandmarks
        
        detector = FacialLandmarkDetector()
        result = detector.extractLandmarks(frame)
        
        # Result should be either None or FacialLandmarks
        assert result is None or isinstance(result, FacialLandmarks)
        
        if result is not None:
            # Validate structure
            assert hasattr(result, 'landmarks'), "Should have landmarks attribute"
            assert hasattr(result, 'confidence'), "Should have confidence attribute"
            assert hasattr(result, 'timestamp'), "Should have timestamp attribute"
            
            assert isinstance(result.landmarks, list), "Landmarks should be a list"
            assert isinstance(result.confidence, float), "Confidence should be a float"
            assert 0.0 <= result.confidence <= 1.0, "Confidence should be between 0 and 1"
            assert result.timestamp is not None, "Timestamp should not be None"
        
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=50, deadline=5000)
    def test_landmark_quality_validation_is_deterministic(self, frame):
        """
        Test that landmark quality validation produces consistent results.
        
        For any landmarks, quality validation should be deterministic and
        return consistent results when called multiple times.
        """
        from src.face_detection import FacialLandmarkDetector
        
        detector = FacialLandmarkDetector()
        landmarks = detector.extractLandmarks(frame)
        
        if landmarks:
            # Validate quality multiple times - should be consistent
            is_valid_1, reason_1 = detector.validateLandmarkQuality(landmarks)
            is_valid_2, reason_2 = detector.validateLandmarkQuality(landmarks)
            
            assert is_valid_1 == is_valid_2, "Quality validation should be deterministic"
            assert reason_1 == reason_2, "Validation reason should be consistent"
            assert isinstance(is_valid_1, bool), "Validation result should be boolean"
            assert isinstance(reason_1, str), "Validation reason should be string"
        
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=50, deadline=5000)
    def test_landmark_subset_extraction_completeness(self, frame):
        """
        Test that all landmark subsets can be extracted successfully.
        
        For any detected landmarks, all feature subsets (eyes, mouth, nose, etc.)
        should be extractable without errors.
        """
        from src.face_detection import FacialLandmarkDetector
        
        detector = FacialLandmarkDetector()
        landmarks = detector.extractLandmarks(frame)
        
        if landmarks:
            # All subset types should be extractable
            feature_types = ['left_eye', 'right_eye', 'mouth', 'nose', 'jawline', 'eyebrows']
            
            for feature_type in feature_types:
                try:
                    subset = detector.getLandmarkSubset(landmarks, feature_type)
                    # Should return a list or tuple
                    assert isinstance(subset, (list, tuple)), f"{feature_type} should return list or tuple"
                except Exception as e:
                    pytest.fail(f"Failed to extract {feature_type}: {e}")
        
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=50, deadline=5000)
    def test_landmark_normalization_preserves_structure(self, frame):
        """
        Test that landmark normalization preserves the landmark structure.
        
        For any landmarks, normalization should preserve the number of landmarks
        and their relative positions.
        """
        from src.face_detection import FacialLandmarkDetector
        
        detector = FacialLandmarkDetector()
        landmarks = detector.extractLandmarks(frame)
        
        if landmarks and len(landmarks.landmarks) > 0:
            original_count = len(landmarks.landmarks)
            
            # Normalize landmarks
            normalized = detector.normalizeLandmarks(landmarks)
            
            # Should preserve landmark count
            assert len(normalized.landmarks) == original_count, \
                "Normalization should preserve landmark count"
            
            # Should preserve confidence
            assert normalized.confidence == landmarks.confidence, \
                "Normalization should preserve confidence"
            
            # All normalized landmarks should still be 3D tuples
            for landmark in normalized.landmarks:
                assert isinstance(landmark, tuple), "Normalized landmark should be tuple"
                assert len(landmark) == 3, "Normalized landmark should have x, y, z"
        
        detector.reset()
    
    @given(frame=valid_image_frame())
    @settings(max_examples=50, deadline=5000)
    def test_68_point_conversion_produces_correct_shape(self, frame):
        """
        Test that 68-point landmark conversion produces correct output shape.
        
        For any landmarks, conversion to 68-point format should produce
        a numpy array of shape (68, 2).
        """
        from src.face_detection import FacialLandmarkDetector
        
        detector = FacialLandmarkDetector()
        landmarks = detector.extractLandmarks(frame)
        
        if landmarks and len(landmarks.landmarks) >= 468:
            # Convert to 68-point format
            landmarks_68 = landmarks.to_68_point_format()
            
            # Should be numpy array
            assert isinstance(landmarks_68, np.ndarray), "Should return numpy array"
            
            # Should have correct shape
            assert landmarks_68.shape == (68, 2), \
                f"Should have shape (68, 2), got {landmarks_68.shape}"
            
            # All values should be numeric
            assert np.all(np.isfinite(landmarks_68)), \
                "All landmark values should be finite numbers"
        
        detector.reset()
    
    def test_empty_frame_handling(self):
        """
        Test that detector handles empty/invalid frames gracefully.
        
        Edge case: Empty or None frames should not crash the detector.
        """
        from src.face_detection import FacialLandmarkDetector
        
        detector = FacialLandmarkDetector()
        
        # Test with None
        result = detector.extractLandmarks(None)
        assert result is None, "Should return None for None input"
        
        # Test with empty array
        empty_frame = np.array([])
        result = detector.extractLandmarks(empty_frame)
        assert result is None, "Should return None for empty array"
        
        detector.reset()
    
    @given(
        width=st.integers(min_value=320, max_value=1920),
        height=st.integers(min_value=240, max_value=1080)
    )
    @settings(max_examples=50, deadline=5000)
    def test_landmark_detection_works_across_resolutions(self, width, height):
        """
        Test that landmark detection works across various image resolutions.
        
        For any valid image resolution, detection should complete successfully.
        """
        from src.face_detection import FacialLandmarkDetector
        
        detector = FacialLandmarkDetector()
        
        # Create frame with specified resolution
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        start_time = time.time()
        result = detector.extractLandmarks(frame)
        detection_time = time.time() - start_time
        
        # Should complete quickly
        assert detection_time < 2.0, f"Detection took {detection_time:.3f}s"
        
        # Result should be valid structure
        assert result is None or hasattr(result, 'landmarks')
        
        detector.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
