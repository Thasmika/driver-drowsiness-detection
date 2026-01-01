"""
Emergency Response Module

This module provides emergency response features including GPS tracking,
location services, and emergency contact management.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""

from .location_tracker import LocationTracker, LocationData, LocationAccuracy
from .emergency_service import EmergencyService, EmergencyContact, EmergencyStatus

__all__ = [
    'LocationTracker',
    'LocationData',
    'LocationAccuracy',
    'EmergencyService',
    'EmergencyContact',
    'EmergencyStatus'
]
