"""
Property-Based Tests for Emergency Features

Tests correctness properties for GPS tracking, location services,
and emergency response functionality.

Feature: driver-drowsiness-detection
Properties: 32, 33, 34
Validates: Requirements 7.1, 7.2, 7.3
"""

import pytest
import time
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

from src.emergency.location_tracker import (
    LocationTracker,
    LocationData,
    LocationAccuracy
)
from src.emergency.emergency_service import (
    EmergencyService,
    EmergencyContact,
    EmergencyStatus,
    SeverityLevel
)


# ============================================================================
# Test Generators
# ============================================================================

@st.composite
def location_data_strategy(draw):
    """Generate valid location data"""
    return LocationData(
        latitude=draw(st.floats(min_value=-90, max_value=90)),
        longitude=draw(st.floats(min_value=-180, max_value=180)),
        accuracy_meters=draw(st.floats(min_value=1.0, max_value=100.0)),
        altitude=draw(st.floats(min_value=0, max_value=5000)),
        speed=draw(st.floats(min_value=0, max_value=50)),
        heading=draw(st.floats(min_value=0, max_value=360)),
        timestamp=time.time()
    )


@st.composite
def emergency_contact_strategy(draw):
    """Generate emergency contact"""
    return EmergencyContact(
        name=draw(st.text(min_size=3, max_size=50)),
        phone_number=draw(st.text(min_size=10, max_size=15)),
        relationship=draw(st.sampled_from(["spouse", "parent", "sibling", "friend"])),
        priority=draw(st.integers(min_value=1, max_value=5))
    )


# ============================================================================
# Property 32: GPS Location Tracking
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    enable_tracking=st.booleans(),
    update_interval=st.floats(min_value=1.0, max_value=10.0)
)
def test_property_32_gps_location_tracking(enable_tracking, update_interval):
    """
    Property 32: GPS Location Tracking
    
    For any emergency feature activation, the emergency service
    should track vehicle GPS location accurately.
    
    Feature: driver-drowsiness-detection, Property 32
    Validates: Requirements 7.1
    """
    tracker = LocationTracker(
        enable_tracking=enable_tracking,
        update_interval_seconds=update_interval
    )
    
    # Property: Tracking requires consent
    if enable_tracking:
        consent_granted = tracker.requestLocationPermission()
        assert consent_granted is True, (
            "Location permission should be granted when tracking enabled"
        )
        
        # Property: Tracking can be started after consent
        tracking_started = tracker.startTracking()
        assert tracking_started is True, (
            "Tracking should start after consent granted"
        )
        
        # Property: Location should be retrievable
        location = tracker.getCurrentLocation()
        if location:
            # Property: Location data should be valid
            assert location.is_valid(), (
                "Retrieved location should be valid"
            )
            
            # Property: Accuracy should be within reasonable bounds
            assert 0 < location.accuracy_meters < 10000, (
                "Location accuracy should be within reasonable bounds"
            )
    else:
        # Property: Without tracking enabled, location should not be available
        location = tracker.getCurrentLocation()
        assert location is None, (
            "Location should not be available without tracking enabled"
        )


# ============================================================================
# Property 33: Emergency Response Prompting
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    drowsiness_score=st.floats(min_value=0.85, max_value=1.0),
    consecutive_detections=st.integers(min_value=1, max_value=5)
)
def test_property_33_emergency_response_prompting(drowsiness_score, consecutive_detections):
    """
    Property 33: Emergency Response Prompting
    
    For any repeated severe drowsiness detection, the emergency
    service should prompt for driver response.
    
    Feature: driver-drowsiness-detection, Property 33
    Validates: Requirements 7.2
    """
    tracker = LocationTracker(enable_tracking=True)
    tracker.requestLocationPermission()
    
    service = EmergencyService(
        location_tracker=tracker,
        severe_threshold=0.85
    )
    
    # Simulate consecutive severe drowsiness detections
    severe_detected = False
    current_time = time.time()
    
    for i in range(consecutive_detections):
        is_severe = service.detectSevereDrowsiness(
            drowsiness_score,
            current_time + i
        )
        if is_severe:
            severe_detected = True
            break
    
    # Property: After sufficient consecutive detections, severe drowsiness should be detected
    if consecutive_detections >= service.consecutive_severe_threshold:
        assert severe_detected is True, (
            f"Severe drowsiness should be detected after "
            f"{service.consecutive_severe_threshold} consecutive detections"
        )
        
        # Property: Driver should be prompted for response
        prompt_initiated = service.promptDriverResponse("Are you okay?")
        assert prompt_initiated is True, (
            "Driver prompt should be initiated for severe drowsiness"
        )
        
        # Property: Status should be PROMPTING
        assert service.getStatus() == EmergencyStatus.PROMPTING, (
            "Service status should be PROMPTING after prompt initiated"
        )


# ============================================================================
# Property 34: Emergency Escalation Timing
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    response_timeout=st.floats(min_value=10.0, max_value=60.0)
)
def test_property_34_emergency_escalation_timing(response_timeout):
    """
    Property 34: Emergency Escalation Timing
    
    For any driver non-response scenario, the emergency service
    should prepare emergency contact within the specified timeout.
    
    Feature: driver-drowsiness-detection, Property 34
    Validates: Requirements 7.3
    """
    tracker = LocationTracker(enable_tracking=True)
    tracker.requestLocationPermission()
    
    service = EmergencyService(
        location_tracker=tracker,
        response_timeout_seconds=response_timeout
    )
    
    # Add emergency contact
    contact = EmergencyContact(
        name="Test Contact",
        phone_number="1234567890",
        relationship="spouse",
        priority=1
    )
    service.addEmergencyContact(contact)
    
    # Initiate prompt
    service.promptDriverResponse()
    
    # Property: Initially, timeout should not have occurred
    assert not service.checkResponseTimeout(), (
        "Timeout should not occur immediately after prompt"
    )
    
    # Simulate time passing (just beyond timeout)
    service.prompt_start_time = time.time() - (response_timeout + 1)
    
    # Property: After timeout, timeout should be detected
    timeout_occurred = service.checkResponseTimeout()
    assert timeout_occurred is True, (
        f"Timeout should occur after {response_timeout} seconds"
    )
    
    # Property: Emergency protocol should be initiated
    event = service.initiateEmergencyProtocol(drowsiness_score=0.95)
    
    # Property: Event should be created
    assert event is not None, (
        "Emergency event should be created"
    )
    
    # Property: Status should be ESCALATED
    assert service.getStatus() == EmergencyStatus.ESCALATED, (
        "Service status should be ESCALATED after protocol initiated"
    )
    
    # Property: Emergency contacts should be notified
    assert len(event.contacts_notified) > 0, (
        "Emergency contacts should be notified"
    )


# ============================================================================
# Additional Emergency Tests
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    location1=location_data_strategy(),
    location2=location_data_strategy()
)
def test_location_distance_calculation(location1, location2):
    """
    Test distance calculation between locations.
    
    For any two valid locations, distance should be non-negative
    and symmetric.
    """
    assume(location1.is_valid() and location2.is_valid())
    
    tracker = LocationTracker()
    
    # Calculate distance
    distance = tracker.getDistanceBetween(location1, location2)
    
    # Property: Distance should be non-negative
    assert distance >= 0, (
        "Distance between locations should be non-negative"
    )
    
    # Property: Distance should be symmetric
    distance_reverse = tracker.getDistanceBetween(location2, location1)
    assert abs(distance - distance_reverse) < 1.0, (
        "Distance calculation should be symmetric"
    )
    
    # Property: Distance to self should be zero
    distance_self = tracker.getDistanceBetween(location1, location1)
    assert distance_self < 1.0, (
        "Distance from location to itself should be near zero"
    )


@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    contacts=st.lists(emergency_contact_strategy(), min_size=1, max_size=5)
)
def test_emergency_contact_management(contacts):
    """
    Test emergency contact management.
    
    For any list of emergency contacts, they should be properly
    stored and sorted by priority.
    """
    tracker = LocationTracker(enable_tracking=False)
    service = EmergencyService(location_tracker=tracker)
    
    # Add contacts with unique phone numbers
    added_count = 0
    for i, contact in enumerate(contacts):
        # Make phone number unique
        contact.phone_number = f"{contact.phone_number}_{i}"
        if service.addEmergencyContact(contact):
            added_count += 1
    
    # Property: All unique contacts should be stored
    stored_contacts = service.getEmergencyContacts()
    assert len(stored_contacts) == added_count, (
        "All unique contacts should be stored"
    )
    
    # Property: Contacts should be sorted by priority
    priorities = [c.priority for c in stored_contacts]
    assert priorities == sorted(priorities), (
        "Contacts should be sorted by priority"
    )


@pytest.mark.property
def test_location_accuracy_validation():
    """
    Test location accuracy validation.
    
    For any location data, accuracy validation should correctly
    identify acceptable and unacceptable accuracy levels.
    """
    tracker = LocationTracker(min_accuracy_meters=50.0)
    
    # High accuracy location (should pass)
    high_accuracy = LocationData(
        latitude=37.7749,
        longitude=-122.4194,
        accuracy_meters=10.0,
        altitude=0,
        speed=0,
        heading=0,
        timestamp=time.time()
    )
    
    is_valid, reason = tracker.validateLocationAccuracy(high_accuracy)
    assert is_valid is True, (
        "High accuracy location should be valid"
    )
    
    # Low accuracy location (should fail)
    low_accuracy = LocationData(
        latitude=37.7749,
        longitude=-122.4194,
        accuracy_meters=100.0,
        altitude=0,
        speed=0,
        heading=0,
        timestamp=time.time()
    )
    
    is_valid, reason = tracker.validateLocationAccuracy(low_accuracy)
    assert is_valid is False, (
        "Low accuracy location should be invalid"
    )


@pytest.mark.property
def test_emergency_data_preparation():
    """
    Test emergency data preparation.
    
    For any emergency scenario, prepared data should include
    all necessary information for emergency response.
    """
    tracker = LocationTracker(enable_tracking=True)
    tracker.requestLocationPermission()
    
    service = EmergencyService(location_tracker=tracker)
    
    # Add emergency contact
    contact = EmergencyContact(
        name="Emergency Contact",
        phone_number="911",
        relationship="emergency",
        priority=1
    )
    service.addEmergencyContact(contact)
    
    # Prepare emergency data
    data = service.prepareEmergencyData(drowsiness_score=0.95)
    
    # Property: Data should include required fields
    required_fields = [
        'timestamp',
        'severity_level',
        'drowsiness_score',
        'location',
        'driver_status',
        'emergency_contacts'
    ]
    
    for field in required_fields:
        assert field in data, (
            f"Emergency data should include {field}"
        )
    
    # Property: Drowsiness score should match
    assert data['drowsiness_score'] == 0.95, (
        "Drowsiness score should be included in emergency data"
    )
    
    # Property: Emergency contacts should be included
    assert len(data['emergency_contacts']) > 0, (
        "Emergency contacts should be included in data"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
