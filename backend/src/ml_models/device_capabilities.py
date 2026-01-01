"""
Device capability detection for optimal model selection.

This module detects device capabilities and recommends optimal model
configurations for performance and accuracy.
"""

import platform
import psutil
import numpy as np
from typing import Dict, Any, Optional
from enum import Enum


class DeviceType(Enum):
    """Device type classification."""
    HIGH_END = "high_end"
    MID_RANGE = "mid_range"
    LOW_END = "low_end"
    UNKNOWN = "unknown"


class DeviceCapabilities:
    """Detects and reports device capabilities for model selection."""
    
    def __init__(self):
        """Initialize device capability detector."""
        self.capabilities = self._detect_capabilities()
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """
        Detect device capabilities.
        
        Returns:
            Dictionary with device capability information
        """
        capabilities = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
            "architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_gb": self._get_total_memory_gb(),
            "available_memory_gb": self._get_available_memory_gb(),
            "gpu_available": self._check_gpu_availability(),
            "device_type": None  # Will be set by classify_device
        }
        
        capabilities["device_type"] = self._classify_device(capabilities)
        
        return capabilities
    
    def _get_total_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            return psutil.virtual_memory().total / (1024 ** 3)
        except:
            return 0.0
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except:
            return 0.0
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except:
            return False
    
    def _classify_device(self, capabilities: Dict[str, Any]) -> DeviceType:
        """
        Classify device based on capabilities.
        
        Args:
            capabilities: Device capability dictionary
            
        Returns:
            DeviceType classification
        """
        memory_gb = capabilities.get("total_memory_gb", 0)
        cpu_count = capabilities.get("cpu_count", 0)
        has_gpu = capabilities.get("gpu_available", False)
        
        # High-end device criteria
        if (memory_gb >= 6 and cpu_count >= 6) or has_gpu:
            return DeviceType.HIGH_END
        
        # Mid-range device criteria
        elif memory_gb >= 3 and cpu_count >= 4:
            return DeviceType.MID_RANGE
        
        # Low-end device criteria
        elif memory_gb >= 1 and cpu_count >= 2:
            return DeviceType.LOW_END
        
        else:
            return DeviceType.UNKNOWN
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get device capabilities.
        
        Returns:
            Dictionary with device capability information
        """
        return self.capabilities.copy()
    
    def get_device_type(self) -> DeviceType:
        """
        Get device type classification.
        
        Returns:
            DeviceType enum value
        """
        return self.capabilities["device_type"]
    
    def recommend_model_config(self) -> Dict[str, Any]:
        """
        Recommend optimal model configuration based on device capabilities.
        
        Returns:
            Dictionary with recommended model configuration
        """
        device_type = self.get_device_type()
        
        if device_type == DeviceType.HIGH_END:
            return {
                "model_type": "cnn",
                "input_size": (224, 224),
                "batch_size": 8,
                "use_quantization": False,
                "enable_gpu": self.capabilities["gpu_available"],
                "max_fps": 30,
                "description": "High-end device: Use full CNN model with high resolution"
            }
        
        elif device_type == DeviceType.MID_RANGE:
            return {
                "model_type": "cnn",
                "input_size": (160, 160),
                "batch_size": 4,
                "use_quantization": True,
                "enable_gpu": False,
                "max_fps": 20,
                "description": "Mid-range device: Use quantized CNN with medium resolution"
            }
        
        elif device_type == DeviceType.LOW_END:
            return {
                "model_type": "traditional_ml",
                "input_size": (128, 128),
                "batch_size": 1,
                "use_quantization": True,
                "enable_gpu": False,
                "max_fps": 15,
                "description": "Low-end device: Use traditional ML with low resolution"
            }
        
        else:
            return {
                "model_type": "traditional_ml",
                "input_size": (128, 128),
                "batch_size": 1,
                "use_quantization": True,
                "enable_gpu": False,
                "max_fps": 15,
                "description": "Unknown device: Use conservative settings"
            }
    
    def print_capabilities(self):
        """Print device capabilities in a readable format."""
        print("=" * 60)
        print("Device Capabilities")
        print("=" * 60)
        
        for key, value in self.capabilities.items():
            if key == "device_type":
                print(f"{key:25s}: {value.value if value else 'unknown'}")
            else:
                print(f"{key:25s}: {value}")
        
        print("\n" + "=" * 60)
        print("Recommended Model Configuration")
        print("=" * 60)
        
        config = self.recommend_model_config()
        for key, value in config.items():
            print(f"{key:25s}: {value}")
        
        print("=" * 60)


def get_optimal_model_config() -> Dict[str, Any]:
    """
    Get optimal model configuration for the current device.
    
    Returns:
        Dictionary with recommended model configuration
    """
    detector = DeviceCapabilities()
    return detector.recommend_model_config()


def print_device_info():
    """Print device information and recommended configuration."""
    detector = DeviceCapabilities()
    detector.print_capabilities()
