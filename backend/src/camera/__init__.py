"""
Camera Management Module

This module provides camera interface and management capabilities
for cross-platform video capture and frame processing.

Validates: Requirements 1.2, 4.2, 4.5
"""

from .camera_manager import CameraManager, CameraConfig, FrameData

__all__ = ['CameraManager', 'CameraConfig', 'FrameData']
