"""
Emergency Service Module

This module provides emergency response logic including severe drowsiness
detection, driver response prompting, and emergency contact management.

Validates: Requirements 7.2, 7.3, 7.4, 7.5
"""

import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .location_tracker import LocationTracker, LocationData


class EmergencyStatus(Enum):
    """Emergency response status"""
    INACTIVE = "inactive"
    MONITORING = "monitoring"
    WARNING = "warning"
    PROMPTING = "prompting"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class SeverityLevel(Enum):
    """Drowsiness severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class EmergencyContact:
    """Emergency contact information"""
    name: str
    phone_number: str
    relationship: str
    priority: int = 1  # 1 = primary, 2 = secondary, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'phone_number': self.phone_number,
            'relationship': self.relationship,
            'priority': self.priority
        }


@dataclass
class EmergencyEvent:
    """Record of an emergency event"""
    event_id: str
    timestamp: float
    severity_level: SeverityLevel
    drowsiness_score: float
    location: Optional[LocationData]
    driver_responded: bool
    response_time_seconds: Optional[float]
    contacts_notified: List[str]
    resolution: str


class EmergencyService:
    """
    Emergency response service for severe drowsiness scenarios.
    
    Monitors drowsiness levels, prompts driver for response,
    and initiates emergency protocols when necessary.
    """
    
    def __init__(
        self,
        location_tracker: LocationTracker,
        severe_threshold: float = 0.85,
        critical_threshold: float = 0.95,
        response_timeout_seconds: float = 30.0
    ):
        """
        Initialize emergency service.
        
        Args:
            location_tracker: GPS location tracker
            severe_threshold: Drowsiness score threshold for severe alert
            critical_threshold: Drowsiness score threshold for critical alert
            response_timeout_seconds: Time to wait for driver response
        """
        self.location_tracker = location_tracker
        self.severe_threshold = severe_threshold
        self.critical_threshold = critical_threshold
        self.response_timeout = response_timeout_seconds
        
        # Emergency contacts
        self.emergency_contacts: List[EmergencyContact] = []
        
        # Current status
        self.status = EmergencyStatus.INACTIVE
        self.current_severity = SeverityLevel.NONE
        
        # Monitoring state
        self.severe_drowsiness_count = 0
        self.consecutive_severe_threshold = 3  # Number of consecutive severe detections
        self.last_severe_detection_time = 0.0
        
        # Response tracking
        self.prompt_start_time: Optional[float] = None
        self.driver_response_received = False
        
        # Event history
        self.emergency_events: List[EmergencyEvent] = []
        
        # Callbacks
        self.on_prompt_callback: Optional[Callable[[str], None]] = None
        self.on_escalate_callback: Optional[Callable[[EmergencyEvent], None]] = None
    
    def addEmergencyContact(self, contact: EmergencyContact) -> bool:
        """
        Add emergency contact.
        
        Args:
            contact: Emergency contact information
        
        Returns:
            True if added successfully
        
        Validates: Requirements 7.4
        """
        # Check if contact already exists
        for existing in self.emergency_contacts:
            if existing.phone_number == contact.phone_number:
                return False
        
        self.emergency_contacts.append(contact)
        
        # Sort by priority
        self.emergency_contacts.sort(key=lambda x: x.priority)
        
        return True
    
    def removeEmergencyContact(self, phone_number: str) -> bool:
        """
        Remove emergency contact.
        
        Args:
            phone_number: Phone number of contact to remove
        
        Returns:
            True if removed successfully
        """
        initial_count = len(self.emergency_contacts)
        self.emergency_contacts = [
            c for c in self.emergency_contacts
            if c.phone_number != phone_number
        ]
        return len(self.emergency_contacts) < initial_count
    
    def getEmergencyContacts(self) -> List[EmergencyContact]:
        """Get list of emergency contacts sorted by priority"""
        return self.emergency_contacts.copy()
    
    def detectSevereDrowsiness(
        self,
        drowsiness_score: float,
        timestamp: float
    ) -> bool:
        """
        Detect if drowsiness has reached severe levels.
        
        Args:
            drowsiness_score: Current drowsiness score (0-1)
            timestamp: Current timestamp
        
        Returns:
            True if severe drowsiness detected
        
        Validates: Requirements 7.2
        """
        # Check if score exceeds severe threshold
        if drowsiness_score >= self.severe_threshold:
            # Check if this is consecutive
            time_since_last = timestamp - self.last_severe_detection_time
            
            if time_since_last < 10.0:  # Within 10 seconds
                self.severe_drowsiness_count += 1
            else:
                self.severe_drowsiness_count = 1
            
            self.last_severe_detection_time = timestamp
            
            # Update severity level
            if drowsiness_score >= self.critical_threshold:
                self.current_severity = SeverityLevel.CRITICAL
            else:
                self.current_severity = SeverityLevel.HIGH
            
            # Check if we've reached threshold for emergency response
            if self.severe_drowsiness_count >= self.consecutive_severe_threshold:
                return True
        else:
            # Reset counter if score drops
            if drowsiness_score < self.severe_threshold * 0.8:
                self.severe_drowsiness_count = 0
                self.current_severity = SeverityLevel.NONE
        
        return False
    
    def promptDriverResponse(self, message: str = "Are you okay?") -> bool:
        """
        Prompt driver for response to verify alertness.
        
        Args:
            message: Message to display to driver
        
        Returns:
            True if prompt initiated successfully
        
        Validates: Requirements 7.2, 7.3
        """
        if self.status == EmergencyStatus.PROMPTING:
            return False  # Already prompting
        
        self.status = EmergencyStatus.PROMPTING
        self.prompt_start_time = time.time()
        self.driver_response_received = False
        
        # Trigger callback if set
        if self.on_prompt_callback:
            self.on_prompt_callback(message)
        
        return True
    
    def recordDriverResponse(self, response: bool = True) -> float:
        """
        Record driver's response to prompt.
        
        Args:
            response: Whether driver responded (True) or not (False)
        
        Returns:
            Response time in seconds
        
        Validates: Requirements 7.3
        """
        if self.prompt_start_time is None:
            return 0.0
        
        response_time = time.time() - self.prompt_start_time
        self.driver_response_received = response
        
        if response:
            # Driver responded - resolve emergency
            self.status = EmergencyStatus.RESOLVED
            self.severe_drowsiness_count = 0
        
        return response_time
    
    def checkResponseTimeout(self) -> bool:
        """
        Check if driver response has timed out.
        
        Returns:
            True if timeout occurred
        
        Validates: Requirements 7.3
        """
        if self.status != EmergencyStatus.PROMPTING:
            return False
        
        if self.prompt_start_time is None:
            return False
        
        elapsed = time.time() - self.prompt_start_time
        
        if elapsed >= self.response_timeout:
            if not self.driver_response_received:
                return True
        
        return False
    
    def initiateEmergencyProtocol(
        self,
        drowsiness_score: float
    ) -> EmergencyEvent:
        """
        Initiate emergency protocol when driver fails to respond.
        
        Args:
            drowsiness_score: Current drowsiness score
        
        Returns:
            EmergencyEvent record
        
        Validates: Requirements 7.3, 7.4, 7.5
        """
        self.status = EmergencyStatus.ESCALATED
        
        # Get current location
        location = self.location_tracker.getCurrentLocation()
        
        # Prepare emergency data
        event_id = self._generateEventId()
        timestamp = time.time()
        
        # Notify emergency contacts
        notified_contacts = self._notifyEmergencyContacts(
            drowsiness_score,
            location
        )
        
        # Create event record
        event = EmergencyEvent(
            event_id=event_id,
            timestamp=timestamp,
            severity_level=self.current_severity,
            drowsiness_score=drowsiness_score,
            location=location,
            driver_responded=False,
            response_time_seconds=None,
            contacts_notified=notified_contacts,
            resolution="Emergency protocol initiated"
        )
        
        self.emergency_events.append(event)
        
        # Trigger callback if set
        if self.on_escalate_callback:
            self.on_escalate_callback(event)
        
        return event
    
    def prepareEmergencyData(
        self,
        drowsiness_score: float
    ) -> Dict[str, Any]:
        """
        Prepare emergency data package for transmission.
        
        Args:
            drowsiness_score: Current drowsiness score
        
        Returns:
            Dictionary with emergency data
        
        Validates: Requirements 7.5
        """
        location = self.location_tracker.getCurrentLocation()
        
        data = {
            'timestamp': time.time(),
            'severity_level': self.current_severity.value,
            'drowsiness_score': drowsiness_score,
            'location': location.to_dict() if location else None,
            'driver_status': 'unresponsive' if not self.driver_response_received else 'responsive',
            'emergency_contacts': [c.to_dict() for c in self.emergency_contacts],
            'event_history': [
                {
                    'event_id': e.event_id,
                    'timestamp': e.timestamp,
                    'severity': e.severity_level.value,
                    'responded': e.driver_responded
                }
                for e in self.emergency_events[-5:]  # Last 5 events
            ]
        }
        
        return data
    
    def getStatus(self) -> EmergencyStatus:
        """Get current emergency status"""
        return self.status
    
    def getSeverityLevel(self) -> SeverityLevel:
        """Get current severity level"""
        return self.current_severity
    
    def getEmergencyHistory(
        self,
        max_count: Optional[int] = None
    ) -> List[EmergencyEvent]:
        """
        Get emergency event history.
        
        Args:
            max_count: Maximum number of events to return
        
        Returns:
            List of emergency events
        """
        events = self.emergency_events.copy()
        
        if max_count:
            events = events[-max_count:]
        
        return events
    
    def reset(self):
        """Reset emergency service state"""
        self.status = EmergencyStatus.INACTIVE
        self.current_severity = SeverityLevel.NONE
        self.severe_drowsiness_count = 0
        self.prompt_start_time = None
        self.driver_response_received = False
    
    def setPromptCallback(self, callback: Callable[[str], None]):
        """Set callback for driver prompts"""
        self.on_prompt_callback = callback
    
    def setEscalateCallback(self, callback: Callable[[EmergencyEvent], None]):
        """Set callback for emergency escalation"""
        self.on_escalate_callback = callback
    
    def _notifyEmergencyContacts(
        self,
        drowsiness_score: float,
        location: Optional[LocationData]
    ) -> List[str]:
        """
        Notify emergency contacts.
        
        In production, this would send actual notifications (SMS, call, etc.)
        
        Returns:
            List of contact names notified
        """
        notified = []
        
        for contact in self.emergency_contacts[:2]:  # Notify top 2 contacts
            # In production, send actual notification
            print(f"[EMERGENCY] Notifying {contact.name} at {contact.phone_number}")
            print(f"  Drowsiness Score: {drowsiness_score:.2f}")
            if location:
                print(f"  Location: {location.latitude:.6f}, {location.longitude:.6f}")
            
            notified.append(contact.name)
        
        return notified
    
    def _generateEventId(self) -> str:
        """Generate unique event ID"""
        import hashlib
        timestamp = str(time.time()).encode()
        return hashlib.sha256(timestamp).hexdigest()[:12]
