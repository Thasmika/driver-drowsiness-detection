"""
Camera Manager Module

This module provides a cross-platform camera interface for capturing
video frames with configurable settings optimized for face detection
and drowsiness monitoring.

Validates: Requirements 1.2, 4.2, 4.5
"""

import time
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass
import cv2
import numpy as np


class CameraStatus(Enum):
    """Camera operational status"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    CAPTURING = "capturing"
    ERROR = "error"
    CLOSED = "closed"


class LightingCondition(Enum):
    """Detected lighting conditions"""
    VERY_DARK = "very_dark"
    DARK = "dark"
    NORMAL = "normal"
    BRIGHT = "bright"
    VERY_BRIGHT = "very_bright"


@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    camera_index: int = 0  # 0 for default camera, typically front camera on mobile
    target_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    auto_exposure: bool = True
    auto_white_balance: bool = True
    buffer_size: int = 1  # Minimal buffer for real-time processing
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.target_fps < 15:
            raise ValueError("Target FPS must be at least 15 for real-time detection")
        if self.frame_width < 320 or self.frame_height < 240:
            raise ValueError("Frame dimensions too small for face detection")


@dataclass
class FrameData:
    """Container for captured frame data and metadata"""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int
    lighting_condition: Optional[LightingCondition] = None
    exposure_time: Optional[float] = None
    brightness: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Check if frame data is valid"""
        return (
            self.frame is not None and
            self.frame.size > 0 and
            len(self.frame.shape) == 3 and
            self.frame.shape[2] == 3  # BGR format
        )


class CameraManager:
    """
    Cross-platform camera manager for video capture and frame processing.
    
    Handles camera initialization, frame capture, quality adjustment,
    and permission management for mobile and desktop platforms.
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        """
        Initialize the CameraManager.
        
        Args:
            config: Camera configuration parameters
        """
        self.config = config or CameraConfig()
        self.camera: Optional[cv2.VideoCapture] = None
        self.status = CameraStatus.UNINITIALIZED
        
        # Frame tracking
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = 0.0
        self.initialization_time = 0.0
        
        # Performance metrics
        self.actual_fps = 0.0
        self.average_capture_time = 0.0
        self.total_capture_time = 0.0
        
        # Camera capabilities
        self.supports_auto_exposure = False
        self.supports_auto_focus = False
        self.max_fps = 30
        
    def initializeCamera(self) -> Tuple[bool, str]:
        """
        Initialize camera with configured settings.
        
        Returns:
            Tuple of (success, message) indicating initialization result
        
        Validates: Requirements 4.2
        """
        start_time = time.time()
        self.status = CameraStatus.INITIALIZING
        
        try:
            # Open camera
            self.camera = cv2.VideoCapture(self.config.camera_index)
            
            if not self.camera.isOpened():
                self.status = CameraStatus.ERROR
                return False, f"Failed to open camera {self.config.camera_index}"
            
            # Configure camera settings
            self._configureCameraSettings()
            
            # Verify camera is working by capturing a test frame
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                self.status = CameraStatus.ERROR
                self.camera.release()
                return False, "Camera opened but failed to capture test frame"
            
            # Query camera capabilities
            self._queryCameraCapabilities()
            
            self.status = CameraStatus.READY
            self.initialization_time = time.time() - start_time
            
            return True, f"Camera initialized successfully in {self.initialization_time:.3f}s"
            
        except Exception as e:
            self.status = CameraStatus.ERROR
            if self.camera:
                self.camera.release()
            return False, f"Camera initialization error: {str(e)}"
    
    def captureFrame(self) -> Optional[FrameData]:
        """
        Capture a single frame from the camera.
        
        Returns:
            FrameData object containing frame and metadata, or None if capture fails
        
        Validates: Requirements 1.2
        """
        if self.status not in [CameraStatus.READY, CameraStatus.CAPTURING]:
            return None
        
        capture_start = time.time()
        
        try:
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                self.dropped_frames += 1
                return None
            
            # Update status
            self.status = CameraStatus.CAPTURING
            
            # Calculate timing metrics
            current_time = time.time()
            capture_time = current_time - capture_start
            self.total_capture_time += capture_time
            self.frame_count += 1
            
            # Calculate actual FPS
            if self.last_frame_time > 0:
                frame_interval = current_time - self.last_frame_time
                self.actual_fps = 1.0 / frame_interval if frame_interval > 0 else 0.0
            self.last_frame_time = current_time
            
            # Detect lighting condition
            lighting = self._detectLightingCondition(frame)
            
            # Calculate brightness
            brightness = self._calculateBrightness(frame)
            
            # Create frame data
            frame_data = FrameData(
                frame=frame,
                timestamp=current_time,
                frame_number=self.frame_count,
                width=frame.shape[1],
                height=frame.shape[0],
                lighting_condition=lighting,
                brightness=brightness
            )
            
            return frame_data
            
        except Exception as e:
            self.dropped_frames += 1
            print(f"Frame capture error: {e}")
            return None
    
    def adjustCameraSettings(self, lighting_condition: LightingCondition) -> bool:
        """
        Adjust camera settings based on lighting conditions.
        
        Args:
            lighting_condition: Current lighting condition
        
        Returns:
            True if settings were adjusted successfully
        
        Validates: Requirements 1.2, 4.5
        """
        if self.camera is None or not self.camera.isOpened():
            return False
        
        try:
            if lighting_condition == LightingCondition.VERY_DARK:
                # Increase exposure for very dark conditions
                if self.supports_auto_exposure:
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)
                
            elif lighting_condition == LightingCondition.DARK:
                # Moderate exposure increase
                if self.supports_auto_exposure:
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
                
            elif lighting_condition == LightingCondition.VERY_BRIGHT:
                # Reduce exposure for very bright conditions
                if self.supports_auto_exposure:
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)
                
            else:
                # Normal conditions - use auto settings
                if self.supports_auto_exposure:
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            
            return True
            
        except Exception as e:
            print(f"Error adjusting camera settings: {e}")
            return False
    
    def setFrameRate(self, target_fps: int) -> bool:
        """
        Set target frame rate for capture.
        
        Args:
            target_fps: Desired frames per second
        
        Returns:
            True if frame rate was set successfully
        """
        if target_fps < 15:
            return False
        
        if target_fps > self.max_fps:
            target_fps = self.max_fps
        
        self.config.target_fps = target_fps
        
        if self.camera and self.camera.isOpened():
            try:
                self.camera.set(cv2.CAP_PROP_FPS, target_fps)
                return True
            except Exception:
                return False
        
        return True
    
    def setResolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        
        Returns:
            True if resolution was set successfully
        """
        if width < 320 or height < 240:
            return False
        
        self.config.frame_width = width
        self.config.frame_height = height
        
        if self.camera and self.camera.isOpened():
            try:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return True
            except Exception:
                return False
        
        return True
    
    def getPerformanceMetrics(self) -> Dict[str, Any]:
        """
        Get camera performance metrics.
        
        Returns:
            Dictionary containing performance statistics
        """
        avg_capture_time = (
            self.total_capture_time / self.frame_count
            if self.frame_count > 0 else 0.0
        )
        
        drop_rate = (
            self.dropped_frames / (self.frame_count + self.dropped_frames)
            if (self.frame_count + self.dropped_frames) > 0 else 0.0
        )
        
        return {
            'status': self.status.value,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'drop_rate': drop_rate,
            'actual_fps': self.actual_fps,
            'target_fps': self.config.target_fps,
            'average_capture_time_ms': avg_capture_time * 1000,
            'initialization_time_s': self.initialization_time
        }
    
    def isReady(self) -> bool:
        """Check if camera is ready for capture"""
        return self.status in [CameraStatus.READY, CameraStatus.CAPTURING]
    
    def close(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None
        self.status = CameraStatus.CLOSED
    
    def reset(self):
        """Reset camera manager state"""
        self.close()
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = 0.0
        self.actual_fps = 0.0
        self.total_capture_time = 0.0
        self.status = CameraStatus.UNINITIALIZED
    
    def _configureCameraSettings(self):
        """Configure camera with optimal settings"""
        if not self.camera:
            return
        
        try:
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            
            # Set FPS
            self.camera.set(cv2.CAP_PROP_FPS, self.config.target_fps)
            
            # Set buffer size (minimal for real-time)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            # Enable auto exposure if supported
            if self.config.auto_exposure:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            # Enable auto white balance if supported
            if self.config.auto_white_balance:
                self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
                
        except Exception as e:
            print(f"Warning: Some camera settings could not be applied: {e}")
    
    def _queryCameraCapabilities(self):
        """Query camera capabilities"""
        if not self.camera:
            return
        
        try:
            # Check auto exposure support
            self.supports_auto_exposure = (
                self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE) >= 0
            )
            
            # Get maximum FPS
            self.max_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            if self.max_fps <= 0:
                self.max_fps = 30  # Default fallback
                
        except Exception:
            # Use defaults if query fails
            self.supports_auto_exposure = False
            self.max_fps = 30
    
    def _detectLightingCondition(self, frame: np.ndarray) -> LightingCondition:
        """
        Detect lighting condition from frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Detected lighting condition
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Classify lighting condition
        if avg_brightness < 40:
            return LightingCondition.VERY_DARK
        elif avg_brightness < 80:
            return LightingCondition.DARK
        elif avg_brightness < 180:
            return LightingCondition.NORMAL
        elif avg_brightness < 220:
            return LightingCondition.BRIGHT
        else:
            return LightingCondition.VERY_BRIGHT
    
    def _calculateBrightness(self, frame: np.ndarray) -> float:
        """
        Calculate normalized brightness value.
        
        Args:
            frame: Input frame
        
        Returns:
            Brightness value between 0.0 and 1.0
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) / 255.0
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
    
    def __enter__(self):
        """Context manager entry"""
        self.initializeCamera()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
