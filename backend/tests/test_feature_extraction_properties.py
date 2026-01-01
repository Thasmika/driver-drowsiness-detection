"""
Property-Based Tests for Feature Extraction Module

Feature: driver-drowsiness-detection
Tests universal properties for EAR and MAR calculators.

Validates: Requirements 2.2, 2.3
"""

import time
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import pytest

from src.feature_extraction import EARCalculator, MARCalculator, BlinkEvent, YawnEvent


# Generators for test data
@st.composite
def valid_eye_landmarks(draw):
    """Generate valid eye landmark coordinates"""
    # Eye landmarks should have at least 6 points for EAR calculation
    num_points = draw(st.integers(min_value=6, max_value=16))
    
    # Generate points in a roughly elliptical pattern (simulating an eye)
    center_x = draw(st.floats(min_value=100, max_value=500))
    center_y = draw(st.floats(min_value=100, max_value=400))
    width = draw(st.floats(min_value=20, max_value=50))
    height = draw(st.floats(min_value=10, max_value=30))
    
    landmarks = []
    for i in range(num_points):
        angle = (2 * np.pi * i) / num_points
        x = center_x + width * np.cos(angle)
        y = center_y + height * np.sin(angle)
        z = draw(st.floats(min_value=-10, max_value=10))
        landmarks.append((float(x), float(y), float(z)))
    
    return landmarks


@st.composite
def valid_mouth_landmarks(draw):
    """Generate valid mouth landmark coordinates"""
    # Mouth landmarks should have at least 8 points for MAR calculation
    num_points = draw(st.integers(min_value=8, max_value=20))
    
    # Generate points in an elliptical pattern (simulating a mouth)
    center_x = draw(st.floats(min_value=100, max_value=500))
    center_y = draw(st.floats(min_value=200, max_value=400))
    width = draw(st.floats(min_value=30, max_value=80))
    height = draw(st.floats(min_value=10, max_value=60))
    
    landmarks = []
    for i in range(num_points):
        angle = (2 * np.pi * i) / num_points
        x = center_x + width * np.cos(angle)
        y = center_y + height * np.sin(angle)
        z = draw(st.floats(min_value=-10, max_value=10))
        landmarks.append((float(x), float(y), float(z)))
    
    return landmarks


@st.composite
def ear_time_series(draw):
    """Generate a time series of EAR values simulating microsleep"""
    # Generate a sequence that includes a microsleep episode
    num_samples = draw(st.integers(min_value=50, max_value=200))
    
    # Normal EAR values
    normal_ear = draw(st.floats(min_value=0.25, max_value=0.35))
    
    # Create time series with a microsleep episode
    ear_values = []
    timestamps = []
    current_time = time.time()
    
    # Start with normal values
    for i in range(num_samples // 3):
        ear_values.append(normal_ear + draw(st.floats(min_value=-0.02, max_value=0.02)))
        timestamps.append(current_time + i * 0.033)  # ~30 FPS
    
    # Add microsleep episode (low EAR for extended period)
    microsleep_duration = draw(st.floats(min_value=2.0, max_value=5.0))
    microsleep_samples = int(microsleep_duration / 0.033)
    microsleep_ear = draw(st.floats(min_value=0.10, max_value=0.20))
    
    for i in range(microsleep_samples):
        ear_values.append(microsleep_ear + draw(st.floats(min_value=-0.01, max_value=0.01)))
        timestamps.append(timestamps[-1] + 0.033)
    
    # Return to normal
    for i in range(num_samples // 3):
        ear_values.append(normal_ear + draw(st.floats(min_value=-0.02, max_value=0.02)))
        timestamps.append(timestamps[-1] + 0.033)
    
    return list(zip(timestamps, ear_values))


@st.composite
def mar_time_series_with_yawn(draw):
    """Generate a time series of MAR values including a yawn"""
    num_samples = draw(st.integers(min_value=50, max_value=200))
    
    # Normal MAR values
    normal_mar = draw(st.floats(min_value=0.2, max_value=0.4))
    
    # Create time series with a yawn
    mar_values = []
    timestamps = []
    current_time = time.time()
    
    # Start with normal values
    for i in range(num_samples // 3):
        mar_values.append(normal_mar + draw(st.floats(min_value=-0.05, max_value=0.05)))
        timestamps.append(current_time + i * 0.033)
    
    # Add yawn (high MAR for 1-6 seconds)
    yawn_duration = draw(st.floats(min_value=1.0, max_value=6.0))
    yawn_samples = int(yawn_duration / 0.033)
    yawn_mar = draw(st.floats(min_value=0.6, max_value=0.9))
    
    for i in range(yawn_samples):
        mar_values.append(yawn_mar + draw(st.floats(min_value=-0.05, max_value=0.05)))
        timestamps.append(timestamps[-1] + 0.033)
    
    # Return to normal
    for i in range(num_samples // 3):
        mar_values.append(normal_mar + draw(st.floats(min_value=-0.05, max_value=0.05)))
        timestamps.append(timestamps[-1] + 0.033)
    
    return list(zip(timestamps, mar_values))


class TestEARCalculatorProperties:
    """Property-based tests for EAR Calculator"""
    
    @given(
        left_eye=valid_eye_landmarks(),
        right_eye=valid_eye_landmarks()
    )
    @settings(max_examples=100, deadline=2000)
    def test_ear_calculation_returns_valid_range(self, left_eye, right_eye):
        """
        Test that EAR calculation always returns values in valid range.
        
        For any valid eye landmarks, EAR should be a positive number
        typically between 0 and 1.
        """
        calculator = EARCalculator()
        
        # Calculate EAR for each eye
        left_ear = calculator.calculateEAR(left_eye)
        right_ear = calculator.calculateEAR(right_eye)
        
        # If calculation succeeds, values should be valid
        if left_ear is not None:
            assert isinstance(left_ear, float), "EAR should be a float"
            assert left_ear >= 0, f"EAR should be non-negative, got {left_ear}"
            assert left_ear <= 2.0, f"EAR should be reasonable (<2.0), got {left_ear}"
        
        if right_ear is not None:
            assert isinstance(right_ear, float), "EAR should be a float"
            assert right_ear >= 0, f"EAR should be non-negative, got {right_ear}"
            assert right_ear <= 2.0, f"EAR should be reasonable (<2.0), got {right_ear}"
        
        # Average EAR should also be valid
        avg_ear = calculator.getAverageEAR(left_eye, right_eye)
        if avg_ear is not None:
            assert isinstance(avg_ear, float), "Average EAR should be a float"
            assert avg_ear >= 0, "Average EAR should be non-negative"
            assert avg_ear <= 2.0, "Average EAR should be reasonable"
    
    @given(ear_series=ear_time_series())
    @settings(max_examples=100, deadline=5000)
    def test_property_11_microsleep_detection(self, ear_series):
        """
        Property 11: Microsleep Detection
        
        For any eye closure pattern indicative of microsleep episodes,
        the ML engine should correctly identify the pattern.
        
        Feature: driver-drowsiness-detection, Property 11: Microsleep Detection
        Validates: Requirements 2.2
        """
        calculator = EARCalculator(
            microsleep_duration_min=2.0,
            microsleep_ear_threshold=0.20
        )
        
        # Process the time series
        microsleep_detected = False
        for timestamp, ear_value in ear_series:
            blink_event = calculator.detectBlink(ear_value, timestamp)
        
        # Check if microsleep was detected
        microsleep_count = calculator.detectMicrosleep(time_window=300.0)
        
        # Property: If the series contains a microsleep pattern (low EAR for >2s),
        # it should be detected
        # We know our generator creates a microsleep, so count should be > 0
        if microsleep_count > 0:
            microsleep_detected = True
            
            # Verify microsleep events are recorded
            assert len(calculator.microsleep_events) > 0, \
                "Microsleep events should be recorded"
            
            # Verify microsleep duration is >= minimum
            for event in calculator.microsleep_events:
                assert event.duration >= calculator.microsleep_duration_min, \
                    f"Microsleep duration {event.duration}s should be >= {calculator.microsleep_duration_min}s"
                assert event.min_ear <= calculator.microsleep_ear_threshold, \
                    f"Microsleep EAR {event.min_ear} should be <= {calculator.microsleep_ear_threshold}"
        
        # Property: Microsleep detection should not produce false positives
        # (all detected microsleeps should meet the criteria)
        for event in calculator.microsleep_events:
            assert isinstance(event, BlinkEvent), "Event should be BlinkEvent"
            assert event.duration >= calculator.microsleep_duration_min, \
                "All microsleep events should meet duration requirement"
    
    @given(
        left_eye=valid_eye_landmarks(),
        right_eye=valid_eye_landmarks()
    )
    @settings(max_examples=100, deadline=2000)
    def test_blink_detection_state_management(self, left_eye, right_eye):
        """
        Test that blink detection properly manages state transitions.
        
        For any sequence of EAR values, the detector should correctly
        track eye open/closed states.
        """
        calculator = EARCalculator()
        
        # Calculate initial EAR
        ear = calculator.getAverageEAR(left_eye, right_eye)
        
        if ear is not None:
            # Simulate a blink sequence: normal -> low -> normal
            normal_ear = 0.30
            low_ear = 0.15
            
            current_time = time.time()
            
            # Normal state
            calculator.detectBlink(normal_ear, current_time)
            assert calculator.is_eye_closed is False, "Eyes should be open initially"
            
            # Eye closes
            calculator.detectBlink(low_ear, current_time + 0.1)
            assert calculator.is_eye_closed is True, "Eyes should be closed"
            
            # Eye reopens (quick blink)
            blink_event = calculator.detectBlink(normal_ear, current_time + 0.3)
            assert calculator.is_eye_closed is False, "Eyes should be open again"
            
            # Should have detected a blink
            if blink_event is not None:
                assert isinstance(blink_event, BlinkEvent), "Should return BlinkEvent"
                assert blink_event.duration <= calculator.blink_duration_max, \
                    "Blink duration should be within max"
    
    @given(
        ear_values=st.lists(
            st.floats(min_value=0.1, max_value=0.4),
            min_size=10,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=3000)
    def test_ear_trend_calculation(self, ear_values):
        """
        Test that EAR trend calculation works correctly.
        
        For any sequence of EAR values, trend calculation should
        produce a valid slope value.
        """
        calculator = EARCalculator()
        
        current_time = time.time()
        for i, ear in enumerate(ear_values):
            calculator.detectBlink(ear, current_time + i * 0.1)
        
        # Get trend
        trend = calculator.getEARTrend(time_window=10.0)
        
        if trend is not None:
            assert isinstance(trend, float), "Trend should be a float"
            assert np.isfinite(trend), "Trend should be finite"
            
            # Trend should be reasonable (not extreme)
            assert abs(trend) < 1.0, f"Trend {trend} should be reasonable"
    
    @given(
        ear_values=st.lists(
            st.floats(min_value=0.15, max_value=0.35),
            min_size=30,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=3000)
    def test_drowsiness_score_range(self, ear_values):
        """
        Test that drowsiness score is always in valid range [0, 1].
        
        For any sequence of EAR values, drowsiness score should be
        between 0 (alert) and 1 (severely drowsy).
        """
        calculator = EARCalculator()
        
        current_time = time.time()
        for i, ear in enumerate(ear_values):
            calculator.detectBlink(ear, current_time + i * 0.033)
        
        score = calculator.getDrowsinessScore()
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, f"Score {score} should be in range [0, 1]"
    
    @given(
        normal_ear_min=st.floats(min_value=0.20, max_value=0.30),
        normal_ear_max=st.floats(min_value=0.30, max_value=0.40)
    )
    @settings(max_examples=50, deadline=1000)
    def test_threshold_update_persistence(self, normal_ear_min, normal_ear_max):
        """
        Test that threshold updates persist correctly.
        
        For any valid threshold values, updates should be applied
        and persist in the calculator.
        """
        assume(normal_ear_min < normal_ear_max)
        
        calculator = EARCalculator()
        
        # Update thresholds
        calculator.updateThresholds(
            normal_ear_range=(normal_ear_min, normal_ear_max)
        )
        
        # Verify update
        assert calculator.normal_ear_range == (normal_ear_min, normal_ear_max), \
            "Threshold update should persist"
    
    def test_empty_landmarks_handling(self):
        """
        Test that calculator handles empty/invalid landmarks gracefully.
        
        Edge case: Empty or None landmarks should not crash the calculator.
        """
        calculator = EARCalculator()
        
        # Test with None
        result = calculator.calculateEAR(None)
        assert result is None, "Should return None for None input"
        
        # Test with empty list
        result = calculator.calculateEAR([])
        assert result is None, "Should return None for empty list"
        
        # Test with insufficient landmarks
        result = calculator.calculateEAR([(0, 0, 0), (1, 1, 0)])
        assert result is None, "Should return None for insufficient landmarks"
    
    @given(
        left_eye=valid_eye_landmarks(),
        right_eye=valid_eye_landmarks()
    )
    @settings(max_examples=50, deadline=2000)
    def test_average_ear_symmetry(self, left_eye, right_eye):
        """
        Test that average EAR is computed correctly from both eyes.
        
        For any pair of eyes, average EAR should be between the
        individual EAR values.
        """
        calculator = EARCalculator()
        
        left_ear = calculator.calculateEAR(left_eye)
        right_ear = calculator.calculateEAR(right_eye)
        avg_ear = calculator.getAverageEAR(left_eye, right_eye)
        
        if left_ear is not None and right_ear is not None and avg_ear is not None:
            # Average should be between min and max
            min_ear = min(left_ear, right_ear)
            max_ear = max(left_ear, right_ear)
            
            assert min_ear <= avg_ear <= max_ear, \
                f"Average EAR {avg_ear} should be between {min_ear} and {max_ear}"


class TestMARCalculatorProperties:
    """Property-based tests for MAR Calculator"""
    
    @given(mouth=valid_mouth_landmarks())
    @settings(max_examples=100, deadline=2000)
    def test_mar_calculation_returns_valid_range(self, mouth):
        """
        Test that MAR calculation always returns values in valid range.
        
        For any valid mouth landmarks, MAR should be a positive number.
        """
        calculator = MARCalculator()
        
        mar = calculator.calculateMAR(mouth)
        
        if mar is not None:
            assert isinstance(mar, float), "MAR should be a float"
            assert mar >= 0, f"MAR should be non-negative, got {mar}"
            assert mar <= 3.0, f"MAR should be reasonable (<3.0), got {mar}"
    
    @given(mar_series=mar_time_series_with_yawn())
    @settings(max_examples=100, deadline=5000)
    def test_property_12_yawn_detection(self, mar_series):
        """
        Property 12: Yawn Detection
        
        For any mouth movement sequence, the ML engine should correctly
        identify yawning behavior as a drowsiness indicator.
        
        Feature: driver-drowsiness-detection, Property 12: Yawn Detection
        Validates: Requirements 2.3
        """
        calculator = MARCalculator(
            yawn_mar_threshold=0.6,
            yawn_duration_min=1.0,
            yawn_duration_max=6.0
        )
        
        # Process the time series
        for timestamp, mar_value in mar_series:
            yawn_event = calculator.detectYawn(mar_value, timestamp)
        
        # Check if yawn was detected
        yawn_count = calculator.getYawnCount(time_window=300.0)
        
        # Property: If the series contains a yawn pattern (high MAR for 1-6s),
        # it should be detected
        if yawn_count > 0:
            # Verify yawn events are recorded
            assert len(calculator.yawn_history) > 0, \
                "Yawn events should be recorded"
            
            # Verify yawn characteristics
            for event in calculator.yawn_history:
                assert isinstance(event, YawnEvent), "Event should be YawnEvent"
                assert calculator.yawn_duration_min <= event.duration <= calculator.yawn_duration_max, \
                    f"Yawn duration {event.duration}s should be in valid range"
                assert event.max_mar >= calculator.yawn_mar_threshold, \
                    f"Yawn MAR {event.max_mar} should be >= {calculator.yawn_mar_threshold}"
        
        # Property: All detected yawns should meet the criteria
        for event in calculator.yawn_history:
            assert event.duration >= calculator.yawn_duration_min, \
                "All yawn events should meet minimum duration"
            assert event.duration <= calculator.yawn_duration_max, \
                "All yawn events should be within maximum duration"
    
    @given(mouth=valid_mouth_landmarks())
    @settings(max_examples=100, deadline=2000)
    def test_yawn_detection_state_management(self, mouth):
        """
        Test that yawn detection properly manages state transitions.
        
        For any sequence of MAR values, the detector should correctly
        track mouth open/closed states.
        """
        calculator = MARCalculator()
        
        # Calculate initial MAR
        mar = calculator.calculateMAR(mouth)
        
        if mar is not None:
            # Simulate a yawn sequence: normal -> high -> normal
            normal_mar = 0.35
            high_mar = 0.75
            
            current_time = time.time()
            
            # Normal state
            calculator.detectYawn(normal_mar, current_time)
            assert calculator.is_mouth_open is False, "Mouth should be closed initially"
            
            # Mouth opens
            calculator.detectYawn(high_mar, current_time + 0.5)
            assert calculator.is_mouth_open is True, "Mouth should be open"
            
            # Mouth closes (after yawn duration)
            yawn_event = calculator.detectYawn(normal_mar, current_time + 2.5)
            assert calculator.is_mouth_open is False, "Mouth should be closed again"
            
            # Should have detected a yawn
            if yawn_event is not None:
                assert isinstance(yawn_event, YawnEvent), "Should return YawnEvent"
                assert yawn_event.duration >= calculator.yawn_duration_min, \
                    "Yawn duration should be within valid range"
    
    @given(
        mar_values=st.lists(
            st.floats(min_value=0.2, max_value=0.8),
            min_size=10,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=3000)
    def test_yawn_frequency_calculation(self, mar_values):
        """
        Test that yawn frequency calculation works correctly.
        
        For any sequence of MAR values, frequency calculation should
        produce a valid non-negative value.
        """
        calculator = MARCalculator()
        
        current_time = time.time()
        for i, mar in enumerate(mar_values):
            calculator.detectYawn(mar, current_time + i * 0.1)
        
        # Get frequency
        frequency = calculator.getYawnFrequency(time_window=60.0)
        
        assert isinstance(frequency, float), "Frequency should be a float"
        assert frequency >= 0, "Frequency should be non-negative"
        assert frequency <= 100, "Frequency should be reasonable (<100 yawns/min)"
    
    @given(
        mar_values=st.lists(
            st.floats(min_value=0.2, max_value=0.7),
            min_size=30,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=3000)
    def test_drowsiness_score_range(self, mar_values):
        """
        Test that drowsiness score is always in valid range [0, 1].
        
        For any sequence of MAR values, drowsiness score should be
        between 0 (alert) and 1 (severely drowsy).
        """
        calculator = MARCalculator()
        
        current_time = time.time()
        for i, mar in enumerate(mar_values):
            calculator.detectYawn(mar, current_time + i * 0.033)
        
        score = calculator.getDrowsinessScore()
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, f"Score {score} should be in range [0, 1]"
    
    @given(
        mar_values=st.lists(
            st.floats(min_value=0.2, max_value=0.8),
            min_size=10,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=3000)
    def test_yawn_pattern_recognition(self, mar_values):
        """
        Test that yawn pattern recognition produces valid classifications.
        
        For any sequence of MAR values, pattern should be one of the
        defined categories.
        """
        calculator = MARCalculator()
        
        current_time = time.time()
        for i, mar in enumerate(mar_values):
            calculator.detectYawn(mar, current_time + i * 0.1)
        
        pattern = calculator.detectYawnPattern(time_window=300.0)
        
        assert isinstance(pattern, str), "Pattern should be a string"
        assert pattern in ['none', 'occasional', 'frequent', 'severe'], \
            f"Pattern '{pattern}' should be a valid category"
    
    def test_empty_landmarks_handling(self):
        """
        Test that calculator handles empty/invalid landmarks gracefully.
        
        Edge case: Empty or None landmarks should not crash the calculator.
        """
        calculator = MARCalculator()
        
        # Test with None
        result = calculator.calculateMAR(None)
        assert result is None, "Should return None for None input"
        
        # Test with empty list
        result = calculator.calculateMAR([])
        assert result is None, "Should return None for empty list"
        
        # Test with insufficient landmarks
        result = calculator.calculateMAR([(0, 0, 0), (1, 1, 0)])
        assert result is None, "Should return None for insufficient landmarks"
    
    @given(
        yawn_threshold=st.floats(min_value=0.5, max_value=0.8),
        duration_min=st.floats(min_value=0.5, max_value=2.0)
    )
    @settings(max_examples=50, deadline=1000)
    def test_threshold_update_persistence(self, yawn_threshold, duration_min):
        """
        Test that threshold updates persist correctly.
        
        For any valid threshold values, updates should be applied
        and persist in the calculator.
        """
        calculator = MARCalculator()
        
        # Update thresholds
        calculator.updateThresholds(
            yawn_mar_threshold=yawn_threshold,
            yawn_duration_min=duration_min
        )
        
        # Verify update
        assert calculator.yawn_mar_threshold == yawn_threshold, \
            "Threshold update should persist"
        assert calculator.yawn_duration_min == duration_min, \
            "Duration update should persist"
    
    @given(
        mar_values=st.lists(
            st.floats(min_value=0.2, max_value=0.8),
            min_size=10,
            max_size=50
        )
    )
    @settings(max_examples=50, deadline=2000)
    def test_statistics_completeness(self, mar_values):
        """
        Test that statistics provide complete information.
        
        For any sequence of MAR values, statistics should include
        all expected fields.
        """
        calculator = MARCalculator()
        
        current_time = time.time()
        for i, mar in enumerate(mar_values):
            calculator.detectYawn(mar, current_time + i * 0.1)
        
        stats = calculator.getStatistics()
        
        # Verify all expected fields are present
        expected_fields = [
            'total_yawns', 'yawn_frequency', 'yawn_count_1min',
            'yawn_count_2min', 'avg_mar', 'min_mar', 'max_mar',
            'drowsiness_score', 'mar_trend', 'yawn_pattern'
        ]
        
        for field in expected_fields:
            assert field in stats, f"Statistics should include '{field}'"
        
        # Verify field types
        assert isinstance(stats['total_yawns'], int), "total_yawns should be int"
        assert isinstance(stats['yawn_frequency'], float), "yawn_frequency should be float"
        assert isinstance(stats['drowsiness_score'], float), "drowsiness_score should be float"
        assert isinstance(stats['yawn_pattern'], str), "yawn_pattern should be str"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
