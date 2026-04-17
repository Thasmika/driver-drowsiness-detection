"""
Adaptation Manager for System Robustness

This module implements adaptive features for the drowsiness detection system
including lighting condition adaptation, face re-detection after occlusion,
cross-demographic adaptation, and environmental noise handling.

Validates: Requirements 1.4, 1.5, 8.1, 8.2, 8.3, 8.4
"""

import time
from typing import Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np
import cv2


class DemographicProfile(Enum):
    """Demographic profiles for model adaptation"""
    UNKNOWN = "unknown"
    YOUNG_ADULT = "young_adult"
    MIDDLE_AGED = "middle_aged"
    SENIOR = "senior"
    DIVERSE = "diverse"


class NoiseLevel(Enum):
    """Environmental noise levels"""
    QUIET = "quiet"
    MODERATE = "moderate"
    LOUD = "loud"
    VERY_LOUD = "very_loud"


@dataclass
class AdaptationState:
    """Current adaptation state of the system"""
    lighting_adapted: bool = False
    demographic_profile: DemographicProfile = DemographicProfile.UNKNOWN
    noise_level: NoiseLevel = NoiseLevel.MODERATE
    last_face_detection_time: float = 0.0
    occlusion_detected: bool = False
    occlusion_start_time: Optional[float] = None
    re_detection_attempts: int = 0
    model_calibration_factor: float = 1.0


class LightingAdapter:
    """
    Handles automatic camera adjustment for varying lighting conditions.
    
    Validates: Requirements 1.4, 8.1
    """
    
    def __init__(self):
        self.previous_brightness: Optional[float] = None
        self.adaptation_history: list = []
        self.max_history = 30
        self.accuracy_threshold = 0.90  # 90% accuracy requirement
        
    def detect_lighting_change(
        self, current_brightness: float, threshold: float = 0.15
    ) -> bool:
        """
        Detect significant lighting condition changes.
        
        Args:
            current_brightness: Current frame brightness (0-1)
            threshold: Minimum change to trigger adaptation
            
        Returns:
            True if significant lighting change detected
        """
        if self.previous_brightness is None:
            self.previous_brightness = current_brightness
            return False
        
        brightness_change = abs(current_brightness - self.previous_brightness)
        
        if brightness_change > threshold:
            self.previous_brightness = current_brightness
            return True
        
        return False
    
    def adjust_camera_parameters(
        self, frame: np.ndarray, brightness: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal camera parameters for current lighting.
        
        Args:
            frame: Current frame
            brightness: Current brightness level (0-1)
            
        Returns:
            Dictionary of recommended camera adjustments
        """
        adjustments = {
            'brightness': 0.5,
            'contrast': 1.0,
            'exposure': 0.0,
            'gamma': 1.0
        }
        
        # Very dark conditions
        if brightness < 0.2:
            adjustments['brightness'] = 0.7
            adjustments['contrast'] = 1.2
            adjustments['gamma'] = 1.3
            adjustments['exposure'] = 0.3
        
        # Dark conditions
        elif brightness < 0.4:
            adjustments['brightness'] = 0.6
            adjustments['contrast'] = 1.1
            adjustments['gamma'] = 1.2
            adjustments['exposure'] = 0.2
        
        # Very bright conditions
        elif brightness > 0.8:
            adjustments['brightness'] = 0.3
            adjustments['contrast'] = 0.9
            adjustments['gamma'] = 0.9
            adjustments['exposure'] = -0.2
        
        # Bright conditions
        elif brightness > 0.6:
            adjustments['brightness'] = 0.4
            adjustments['contrast'] = 0.95
            adjustments['gamma'] = 0.95
            adjustments['exposure'] = -0.1
        
        return adjustments
    
    def enhance_frame_for_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply image enhancement based on lighting conditions.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        # Apply adaptive histogram equalization for low light
        if brightness < 0.4:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return enhanced_frame
        
        # Apply gamma correction for bright conditions
        elif brightness > 0.7:
            gamma = 0.8
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype("uint8")
            
            return cv2.LUT(frame, table)
        
        return frame
    
    def validate_detection_accuracy(
        self, detections: int, total_frames: int
    ) -> bool:
        """
        Validate that detection accuracy meets requirements.
        
        Args:
            detections: Number of successful detections
            total_frames: Total frames processed
            
        Returns:
            True if accuracy above 90% threshold
        """
        if total_frames == 0:
            return False
        
        accuracy = detections / total_frames
        return accuracy >= self.accuracy_threshold


class OcclusionHandler:
    """
    Handles face re-detection after occlusion events.
    
    Validates: Requirements 1.5
    """
    
    def __init__(self):
        self.occlusion_threshold = 3.0  # 3 seconds requirement
        self.max_re_detection_attempts = 10
        
    def detect_occlusion(
        self,
        face_detected: bool,
        previous_detection_time: float,
        current_time: float
    ) -> bool:
        """
        Detect if face occlusion has occurred.
        
        Args:
            face_detected: Whether face is currently detected
            previous_detection_time: Time of last successful detection
            current_time: Current timestamp
            
        Returns:
            True if occlusion detected
        """
        if face_detected:
            return False
        
        time_since_detection = current_time - previous_detection_time
        
        # Occlusion if no face for more than 0.5 seconds
        return time_since_detection > 0.5
    
    def should_attempt_re_detection(
        self,
        occlusion_start_time: float,
        current_time: float,
        attempts: int
    ) -> bool:
        """
        Determine if re-detection should be attempted.
        
        Args:
            occlusion_start_time: When occlusion started
            current_time: Current timestamp
            attempts: Number of re-detection attempts so far
            
        Returns:
            True if re-detection should be attempted
        """
        time_since_occlusion = current_time - occlusion_start_time
        
        # Must attempt within 3 seconds
        if time_since_occlusion > self.occlusion_threshold:
            return False
        
        # Limit number of attempts
        if attempts >= self.max_re_detection_attempts:
            return False
        
        return True
    
    def apply_re_detection_strategy(
        self, frame: np.ndarray, attempt_number: int
    ) -> np.ndarray:
        """
        Apply different preprocessing strategies for re-detection.
        
        Args:
            frame: Input frame
            attempt_number: Current attempt number
            
        Returns:
            Preprocessed frame
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Try different strategies based on attempt number
        if attempt_number % 3 == 0:
            # Strategy 1: Enhance contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        elif attempt_number % 3 == 1:
            # Strategy 2: Adjust brightness
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, 30)
            enhanced_hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        else:
            # Strategy 3: Denoise
            return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)


class DemographicAdapter:
    """
    Handles cross-demographic adaptation and model calibration.
    
    Validates: Requirements 8.2, 8.4
    """
    
    def __init__(self):
        self.calibration_factors = {
            DemographicProfile.YOUNG_ADULT: 1.0,
            DemographicProfile.MIDDLE_AGED: 1.05,
            DemographicProfile.SENIOR: 1.1,
            DemographicProfile.DIVERSE: 1.0
        }
        
        self.head_pose_tolerance = {
            DemographicProfile.YOUNG_ADULT: 30.0,  # degrees
            DemographicProfile.MIDDLE_AGED: 35.0,
            DemographicProfile.SENIOR: 40.0,
            DemographicProfile.DIVERSE: 35.0
        }
    
    def estimate_demographic_profile(
        self, facial_features: Dict[str, Any]
    ) -> DemographicProfile:
        """
        Estimate demographic profile from facial features.
        
        Args:
            facial_features: Dictionary of facial feature measurements
            
        Returns:
            Estimated demographic profile
        """
        # Simplified demographic estimation
        # In production, this would use more sophisticated analysis
        
        # For now, return diverse as default
        return DemographicProfile.DIVERSE
    
    def get_calibration_factor(
        self, demographic: DemographicProfile
    ) -> float:
        """
        Get model calibration factor for demographic.
        
        Args:
            demographic: Demographic profile
            
        Returns:
            Calibration factor to apply to drowsiness scores
        """
        return self.calibration_factors.get(demographic, 1.0)
    
    def validate_head_pose(
        self,
        head_pose_angles: Tuple[float, float, float],
        demographic: DemographicProfile
    ) -> bool:
        """
        Validate if head pose is within acceptable range.
        
        Args:
            head_pose_angles: (pitch, yaw, roll) in degrees
            demographic: Demographic profile
            
        Returns:
            True if head pose is acceptable
        """
        pitch, yaw, roll = head_pose_angles
        tolerance = self.head_pose_tolerance.get(demographic, 35.0)
        
        # Check if any angle exceeds tolerance
        return (
            abs(pitch) <= tolerance and
            abs(yaw) <= tolerance and
            abs(roll) <= tolerance
        )
    
    def adjust_detection_parameters(
        self, demographic: DemographicProfile
    ) -> Dict[str, Any]:
        """
        Get adjusted detection parameters for demographic.
        
        Args:
            demographic: Demographic profile
            
        Returns:
            Dictionary of adjusted parameters
        """
        base_params = {
            'ear_threshold': 0.25,
            'mar_threshold': 0.6,
            'blink_threshold': 0.3,
            'yawn_threshold': 0.65
        }
        
        # Adjust thresholds based on demographic
        if demographic == DemographicProfile.SENIOR:
            base_params['ear_threshold'] = 0.23
            base_params['blink_threshold'] = 0.28
        elif demographic == DemographicProfile.YOUNG_ADULT:
            base_params['ear_threshold'] = 0.26
            base_params['blink_threshold'] = 0.32
        
        return base_params


class EnvironmentalNoiseAdapter:
    """
    Handles environmental noise adaptation for alert systems.
    
    Validates: Requirements 8.3
    """
    
    def __init__(self):
        self.noise_thresholds = {
            NoiseLevel.QUIET: 40,  # dB
            NoiseLevel.MODERATE: 60,
            NoiseLevel.LOUD: 80,
            NoiseLevel.VERY_LOUD: 100
        }
        
        self.visual_prominence_factors = {
            NoiseLevel.QUIET: 1.0,
            NoiseLevel.MODERATE: 1.2,
            NoiseLevel.LOUD: 1.5,
            NoiseLevel.VERY_LOUD: 2.0
        }
    
    def detect_noise_level(self, audio_level: float) -> NoiseLevel:
        """
        Detect environmental noise level.
        
        Args:
            audio_level: Audio level in decibels
            
        Returns:
            Detected noise level
        """
        if audio_level < self.noise_thresholds[NoiseLevel.QUIET]:
            return NoiseLevel.QUIET
        elif audio_level < self.noise_thresholds[NoiseLevel.MODERATE]:
            return NoiseLevel.MODERATE
        elif audio_level < self.noise_thresholds[NoiseLevel.LOUD]:
            return NoiseLevel.LOUD
        else:
            return NoiseLevel.VERY_LOUD
    
    def adjust_alert_prominence(
        self, noise_level: NoiseLevel
    ) -> Dict[str, float]:
        """
        Adjust alert prominence based on noise level.
        
        Args:
            noise_level: Current environmental noise level
            
        Returns:
            Dictionary with adjusted alert parameters
        """
        visual_factor = self.visual_prominence_factors.get(noise_level, 1.0)
        
        # Increase visual prominence, decrease audio reliance
        adjustments = {
            'visual_brightness': min(1.0 * visual_factor, 1.0),
            'visual_size': min(1.0 * visual_factor, 2.0),
            'audio_volume': max(1.0 - (visual_factor - 1.0) * 0.3, 0.5),
            'haptic_intensity': min(0.7 * visual_factor, 1.0)
        }
        
        return adjustments
    
    def should_increase_visual_alerts(self, noise_level: NoiseLevel) -> bool:
        """
        Determine if visual alerts should be increased.
        
        Args:
            noise_level: Current noise level
            
        Returns:
            True if visual alerts should be more prominent
        """
        return noise_level in [NoiseLevel.LOUD, NoiseLevel.VERY_LOUD]


class AdaptationManager:
    """
    Main adaptation manager coordinating all adaptive features.
    
    Validates: Requirements 1.4, 1.5, 8.1, 8.2, 8.3, 8.4
    """
    
    def __init__(self):
        self.lighting_adapter = LightingAdapter()
        self.occlusion_handler = OcclusionHandler()
        self.demographic_adapter = DemographicAdapter()
        self.noise_adapter = EnvironmentalNoiseAdapter()
        
        self.state = AdaptationState()
        
        # Performance tracking
        self.total_frames = 0
        self.successful_detections = 0
        self.lighting_adaptations = 0
        self.occlusion_recoveries = 0
    
    def process_frame_adaptation(
        self,
        frame: np.ndarray,
        face_detected: bool,
        brightness: float,
        current_time: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process frame with all adaptive features.
        
        Args:
            frame: Input frame
            face_detected: Whether face was detected
            brightness: Frame brightness (0-1)
            current_time: Current timestamp
            
        Returns:
            Tuple of (adapted_frame, adaptation_info)
        """
        adapted_frame = frame.copy()
        adaptation_info = {
            'lighting_adapted': False,
            'occlusion_handled': False,
            'enhancement_applied': False
        }
        
        self.total_frames += 1
        
        # Handle lighting adaptation
        if self.lighting_adapter.detect_lighting_change(brightness):
            adapted_frame = self.lighting_adapter.enhance_frame_for_lighting(
                adapted_frame
            )
            self.state.lighting_adapted = True
            self.lighting_adaptations += 1
            adaptation_info['lighting_adapted'] = True
            adaptation_info['enhancement_applied'] = True
        
        # Handle occlusion
        if face_detected:
            self.state.last_face_detection_time = current_time
            self.successful_detections += 1
            
            if self.state.occlusion_detected:
                # Recovered from occlusion
                self.state.occlusion_detected = False
                self.state.occlusion_start_time = None
                self.state.re_detection_attempts = 0
                self.occlusion_recoveries += 1
                adaptation_info['occlusion_handled'] = True
        else:
            # Check for occlusion
            if self.occlusion_handler.detect_occlusion(
                face_detected,
                self.state.last_face_detection_time,
                current_time
            ):
                if not self.state.occlusion_detected:
                    self.state.occlusion_detected = True
                    self.state.occlusion_start_time = current_time
                
                # Attempt re-detection
                if self.occlusion_handler.should_attempt_re_detection(
                    self.state.occlusion_start_time,
                    current_time,
                    self.state.re_detection_attempts
                ):
                    adapted_frame = self.occlusion_handler.apply_re_detection_strategy(
                        adapted_frame,
                        self.state.re_detection_attempts
                    )
                    self.state.re_detection_attempts += 1
                    adaptation_info['occlusion_handled'] = True
                    adaptation_info['enhancement_applied'] = True
        
        return adapted_frame, adaptation_info
    
    def adapt_for_demographic(
        self,
        facial_features: Dict[str, Any],
        drowsiness_score: float
    ) -> float:
        """
        Adapt drowsiness score for demographic characteristics.
        
        Args:
            facial_features: Facial feature measurements
            drowsiness_score: Raw drowsiness score
            
        Returns:
            Calibrated drowsiness score
        """
        # Estimate demographic if not already set
        if self.state.demographic_profile == DemographicProfile.UNKNOWN:
            self.state.demographic_profile = (
                self.demographic_adapter.estimate_demographic_profile(
                    facial_features
                )
            )
        
        # Get calibration factor
        calibration = self.demographic_adapter.get_calibration_factor(
            self.state.demographic_profile
        )
        
        # Apply calibration
        calibrated_score = drowsiness_score * calibration
        
        return min(calibrated_score, 1.0)
    
    def adapt_alerts_for_noise(
        self, audio_level: float
    ) -> Dict[str, float]:
        """
        Adapt alert parameters for environmental noise.
        
        Args:
            audio_level: Current audio level in dB
            
        Returns:
            Adjusted alert parameters
        """
        noise_level = self.noise_adapter.detect_noise_level(audio_level)
        self.state.noise_level = noise_level
        
        return self.noise_adapter.adjust_alert_prominence(noise_level)
    
    def get_detection_accuracy(self) -> float:
        """
        Get current detection accuracy.
        
        Returns:
            Detection accuracy (0-1)
        """
        if self.total_frames == 0:
            return 0.0
        
        return self.successful_detections / self.total_frames
    
    def validate_system_robustness(self) -> Dict[str, Any]:
        """
        Validate system meets robustness requirements.
        
        Returns:
            Dictionary with validation results
        """
        accuracy = self.get_detection_accuracy()
        
        return {
            'meets_accuracy_requirement': accuracy >= 0.80,  # 80% requirement
            'current_accuracy': accuracy,
            'total_frames': self.total_frames,
            'successful_detections': self.successful_detections,
            'lighting_adaptations': self.lighting_adaptations,
            'occlusion_recoveries': self.occlusion_recoveries,
            'lighting_adapted': self.state.lighting_adapted,
            'demographic_profile': self.state.demographic_profile.value
        }
    
    def reset(self):
        """Reset adaptation manager state"""
        self.state = AdaptationState()
        self.total_frames = 0
        self.successful_detections = 0
        self.lighting_adaptations = 0
        self.occlusion_recoveries = 0
