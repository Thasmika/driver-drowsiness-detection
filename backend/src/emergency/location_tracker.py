"""
Location Tracker Module

This module provides GPS tracking and location services for emergency
response with privacy controls and accuracy validation.

Validates: Requirements 7.1, 7.5
"""

import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import random  # For simulation - would use actual GPS in production


class LocationAccuracy(Enum):
    """GPS location accuracy levels"""
    HIGH = "high"  # < 10 meters
    MEDIUM = "medium"  # 10-50 meters
    LOW = "low"  # > 50 meters
    UNKNOWN = "unknown"


@dataclass
class LocationData:
    """Container for GPS location data"""
    latitude: float
    longitude: float
    accuracy_meters: float
    altitude: Optional[float]
    speed: Optional[float]  # meters per second
    heading: Optional[float]  # degrees
    timestamp: float
    
    def get_accuracy_level(self) -> LocationAccuracy:
        """Determine accuracy level from accuracy value"""
        if self.accuracy_meters < 10:
            return LocationAccuracy.HIGH
        elif self.accuracy_meters < 50:
            return LocationAccuracy.MEDIUM
        elif self.accuracy_meters < 1000:
            return LocationAccuracy.LOW
        else:
            return LocationAccuracy.UNKNOWN
    
    def is_valid(self) -> bool:
        """Check if location data is valid"""
        return (
            -90 <= self.latitude <= 90 and
            -180 <= self.longitude <= 180 and
            self.accuracy_meters > 0 and
            self.accuracy_meters < 10000  # Reasonable accuracy limit
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'accuracy_meters': self.accuracy_meters,
            'altitude': self.altitude,
            'speed': self.speed,
            'heading': self.heading,
            'timestamp': self.timestamp,
            'accuracy_level': self.get_accuracy_level().value
        }


class LocationTracker:
    """
    GPS location tracker with continuous monitoring and privacy controls.
    
    Provides location tracking for emergency response while respecting
    user privacy preferences and consent requirements.
    """
    
    def __init__(
        self,
        enable_tracking: bool = False,
        update_interval_seconds: float = 5.0,
        min_accuracy_meters: float = 100.0
    ):
        """
        Initialize location tracker.
        
        Args:
            enable_tracking: Whether tracking is enabled (requires user consent)
            update_interval_seconds: Minimum time between location updates
            min_accuracy_meters: Minimum acceptable accuracy
        """
        self.enable_tracking = enable_tracking
        self.update_interval = update_interval_seconds
        self.min_accuracy = min_accuracy_meters
        
        # Current location
        self.current_location: Optional[LocationData] = None
        self.last_update_time = 0.0
        
        # Location history
        self.location_history: List[LocationData] = []
        self.max_history_size = 100
        
        # Statistics
        self.total_updates = 0
        self.failed_updates = 0
        self.accuracy_errors = 0
        
        # Privacy tracking
        self.user_consent_granted = False
        self.tracking_start_time: Optional[float] = None
    
    def requestLocationPermission(self) -> bool:
        """
        Request location permission from user.
        
        Returns:
            True if permission granted
        
        Validates: Requirements 7.1, 7.5
        """
        # In production, this would trigger OS-level permission request
        # For now, we simulate based on enable_tracking flag
        if self.enable_tracking:
            self.user_consent_granted = True
            return True
        return False
    
    def startTracking(self) -> bool:
        """
        Start continuous GPS tracking.
        
        Returns:
            True if tracking started successfully
        
        Validates: Requirements 7.1
        """
        if not self.user_consent_granted:
            return False
        
        if not self.enable_tracking:
            return False
        
        self.tracking_start_time = time.time()
        return True
    
    def stopTracking(self):
        """Stop GPS tracking"""
        self.tracking_start_time = None
    
    def getCurrentLocation(self) -> Optional[LocationData]:
        """
        Get current GPS location.
        
        Returns:
            LocationData if available, None otherwise
        
        Validates: Requirements 7.1
        """
        if not self.enable_tracking or not self.user_consent_granted:
            return None
        
        current_time = time.time()
        
        # Check if we need to update
        if current_time - self.last_update_time < self.update_interval:
            return self.current_location
        
        # Get new location
        location = self._fetchLocation()
        
        if location:
            # Validate accuracy
            if location.accuracy_meters <= self.min_accuracy:
                self.current_location = location
                self.last_update_time = current_time
                self.total_updates += 1
                
                # Add to history
                self._addToHistory(location)
                
                return location
            else:
                self.accuracy_errors += 1
                return self.current_location  # Return last known good location
        else:
            self.failed_updates += 1
            return self.current_location
    
    def getLocationHistory(
        self,
        max_count: Optional[int] = None,
        time_range_seconds: Optional[float] = None
    ) -> List[LocationData]:
        """
        Get location history with optional filtering.
        
        Args:
            max_count: Maximum number of locations to return
            time_range_seconds: Only return locations within this time range
        
        Returns:
            List of LocationData objects
        """
        history = self.location_history.copy()
        
        # Filter by time range
        if time_range_seconds:
            current_time = time.time()
            history = [
                loc for loc in history
                if current_time - loc.timestamp <= time_range_seconds
            ]
        
        # Limit count
        if max_count:
            history = history[-max_count:]
        
        return history
    
    def validateLocationAccuracy(self, location: LocationData) -> Tuple[bool, str]:
        """
        Validate if location accuracy is acceptable.
        
        Args:
            location: Location data to validate
        
        Returns:
            Tuple of (is_valid, reason)
        
        Validates: Requirements 7.1
        """
        if not location.is_valid():
            return False, "Invalid location coordinates"
        
        if location.accuracy_meters > self.min_accuracy:
            return False, f"Accuracy {location.accuracy_meters}m exceeds limit {self.min_accuracy}m"
        
        if location.get_accuracy_level() == LocationAccuracy.UNKNOWN:
            return False, "Location accuracy is unknown"
        
        return True, "Location accuracy is acceptable"
    
    def getDistanceBetween(
        self,
        loc1: LocationData,
        loc2: LocationData
    ) -> float:
        """
        Calculate distance between two locations using Haversine formula.
        
        Args:
            loc1: First location
            loc2: Second location
        
        Returns:
            Distance in meters
        """
        import math
        
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat1 = math.radians(loc1.latitude)
        lat2 = math.radians(loc2.latitude)
        dlat = math.radians(loc2.latitude - loc1.latitude)
        dlon = math.radians(loc2.longitude - loc1.longitude)
        
        # Haversine formula
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance
    
    def getStatistics(self) -> Dict[str, Any]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with statistics
        """
        success_rate = (
            (self.total_updates / (self.total_updates + self.failed_updates))
            if (self.total_updates + self.failed_updates) > 0
            else 0.0
        )
        
        return {
            'tracking_enabled': self.enable_tracking,
            'consent_granted': self.user_consent_granted,
            'total_updates': self.total_updates,
            'failed_updates': self.failed_updates,
            'accuracy_errors': self.accuracy_errors,
            'success_rate': success_rate,
            'history_size': len(self.location_history),
            'current_location': (
                self.current_location.to_dict()
                if self.current_location else None
            )
        }
    
    def clearHistory(self):
        """Clear location history for privacy"""
        self.location_history.clear()
    
    def _fetchLocation(self) -> Optional[LocationData]:
        """
        Fetch current location from GPS.
        
        In production, this would interface with actual GPS hardware/API.
        For now, we simulate location data.
        
        Returns:
            LocationData if successful, None otherwise
        """
        # Simulate GPS fetch
        # In production, this would use platform-specific GPS APIs
        
        try:
            # Simulate location (San Francisco area for example)
            base_lat = 37.7749
            base_lon = -122.4194
            
            # Add small random variation to simulate movement
            lat = base_lat + random.uniform(-0.01, 0.01)
            lon = base_lon + random.uniform(-0.01, 0.01)
            
            # Simulate accuracy
            accuracy = random.uniform(5.0, 50.0)
            
            location = LocationData(
                latitude=lat,
                longitude=lon,
                accuracy_meters=accuracy,
                altitude=random.uniform(0, 100),
                speed=random.uniform(0, 30),  # 0-30 m/s (0-108 km/h)
                heading=random.uniform(0, 360),
                timestamp=time.time()
            )
            
            return location
            
        except Exception as e:
            print(f"Error fetching location: {e}")
            return None
    
    def _addToHistory(self, location: LocationData):
        """Add location to history with size limit"""
        self.location_history.append(location)
        if len(self.location_history) > self.max_history_size:
            self.location_history.pop(0)
