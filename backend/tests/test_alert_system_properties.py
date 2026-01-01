"""
Property-based tests for Alert System.

Tests the correctness properties of the decision logic and alert manager
components using Hypothesis for property-based testing.

Feature: driver-drowsiness-detection
Properties tested:
- Property 4: Alert Response Time
- Property 15: Comprehensive Alert Delivery
- Property 16: Alert Sensitivity Customization

Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

import pytest
import time
from hypothesis import given, strategies as st, settings
from typing import List, Optional

from src.decision_logic import (
    DecisionEngine,
    AlertManager,
    AlertLevel,
    AlertType,
    AlertConfiguration
)


# Test data generators
@st.composite
def drowsiness_scores(draw):
    """Generate valid drowsiness score inputs."""
    return {
        'ear_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'mar_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'head_pose_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'ml_confidence': draw(st.floats(min_value=0.0, max_value=1.0)),
        'timestamp': draw(st.floats(min_value=0.0, max_value=1e10))
    }


@st.composite
def high_drowsiness_scores(draw):
    """Generate drowsiness scores that should trigger high-confidence alerts."""
    # Generate scores that are consistently high
    base_score = draw(st.floats(min_value=0.7, max_value=1.0))
    variance = draw(st.floats(min_value=0.0, max_value=0.1))
    
    return {
        'ear_score': min(base_score + draw(st.floats(min_value=-variance, max_value=variance)), 1.0),
        'mar_score': min(base_score + draw(st.floats(min_value=-variance, max_value=variance)), 1.0),
        'head_pose_score': min(base_score + draw(st.floats(min_value=-variance, max_value=variance)), 1.0),
        'ml_confidence': min(base_score + draw(st.floats(min_value=-variance, max_value=variance)), 1.0),
        'timestamp': time.time()
    }


@st.composite
def alert_configurations(draw):
    """Generate valid alert configurations."""
    # Generate subset of alert types
    all_types = [AlertType.VISUAL, AlertType.AUDIO, AlertType.HAPTIC]
    num_types = draw(st.integers(min_value=1, max_value=3))
    enabled_types = draw(st.lists(
        st.sampled_from(all_types),
        min_size=num_types,
        max_size=num_types,
        unique=True
    ))
    
    return AlertConfiguration(
        enabled_alert_types=enabled_types,
        sensitivity=draw(st.floats(min_value=0.0, max_value=1.0)),
        audio_volume=draw(st.floats(min_value=0.0, max_value=1.0)),
        haptic_intensity=draw(st.floats(min_value=0.0, max_value=1.0)),
        escalation_enabled=draw(st.booleans()),
        escalation_interval=draw(st.floats(min_value=1.0, max_value=10.0))
    )


class TestAlertResponseTime:
    """
    Property 4: Alert Response Time
    
    For any high-confidence drowsiness detection, the alert manager should
    trigger an alert within 500 milliseconds.
    
    Validates: Requirements 3.1
    """
    
    @settings(max_examples=100)
    @given(scores=high_drowsiness_scores())
    def test_alert_response_time_under_500ms(self, scores):
        """Test that alerts are triggered within 500ms for high drowsiness."""
        # Track alert trigger time
        alert_triggered = []
        trigger_time = []
        
        def visual_callback(level, message):
            trigger_time.append(time.time())
            alert_triggered.append(True)
        
        # Create alert manager with callback
        alert_manager = AlertManager(
            configuration=AlertConfiguration(
                enabled_alert_types=[AlertType.VISUAL],
                sensitivity=0.7,
                audio_volume=0.8,
                haptic_intensity=0.7,
                escalation_enabled=False,
                escalation_interval=5.0
            ),
            visual_callback=visual_callback
        )
        
        # Create decision engine
        decision_engine = DecisionEngine()
        
        # Calculate drowsiness assessment
        start_time = time.time()
        assessment = decision_engine.calculate_drowsiness_score(**scores)
        
        # Trigger alert if drowsiness detected
        if assessment.alert_level != AlertLevel.NONE:
            alert_manager.trigger_alert(
                assessment.alert_level,
                assessment.drowsiness_score,
                assessment.recommendations
            )
            end_time = time.time()
            
            # Verify alert was triggered
            assert len(alert_triggered) > 0, "Alert should be triggered for high drowsiness"
            
            # Verify response time is under 500ms
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            assert response_time < 500, (
                f"Alert response time {response_time:.2f}ms exceeds 500ms threshold"
            )


class TestComprehensiveAlertDelivery:
    """
    Property 15: Comprehensive Alert Delivery
    
    For any triggered alert, the alert manager should provide both visual
    and audio alert components (when both are enabled).
    
    Validates: Requirements 3.2
    """
    
    @settings(max_examples=100)
    @given(
        scores=high_drowsiness_scores(),
        config=alert_configurations()
    )
    def test_multiple_alert_types_delivered(self, scores, config):
        """Test that all enabled alert types are delivered."""
        # Track which alert types were triggered
        triggered_types = []
        
        def visual_callback(level, message):
            triggered_types.append(AlertType.VISUAL)
        
        def audio_callback(level, volume):
            triggered_types.append(AlertType.AUDIO)
        
        def haptic_callback(level, intensity):
            triggered_types.append(AlertType.HAPTIC)
        
        # Create alert manager with all callbacks
        alert_manager = AlertManager(
            configuration=config,
            visual_callback=visual_callback,
            audio_callback=audio_callback,
            haptic_callback=haptic_callback
        )
        
        # Create decision engine
        decision_engine = DecisionEngine()
        
        # Calculate drowsiness assessment
        assessment = decision_engine.calculate_drowsiness_score(**scores)
        
        # Trigger alert
        if assessment.alert_level != AlertLevel.NONE:
            alert_manager.trigger_alert(
                assessment.alert_level,
                assessment.drowsiness_score
            )
            
            # Verify that all enabled alert types were triggered
            for alert_type in config.enabled_alert_types:
                # For low alerts, only visual is used
                if assessment.alert_level == AlertLevel.LOW:
                    if alert_type == AlertType.VISUAL:
                        assert alert_type in triggered_types, (
                            f"Visual alert should be triggered for LOW level"
                        )
                # For medium alerts, visual and haptic are used
                elif assessment.alert_level == AlertLevel.MEDIUM:
                    if alert_type in [AlertType.VISUAL, AlertType.HAPTIC]:
                        assert alert_type in triggered_types, (
                            f"{alert_type.value} alert should be triggered for MEDIUM level"
                        )
                # For high and critical alerts, all types are used
                else:
                    assert alert_type in triggered_types, (
                        f"{alert_type.value} alert should be triggered for {assessment.alert_level.name} level"
                    )
    
    @settings(max_examples=100)
    @given(scores=high_drowsiness_scores())
    def test_visual_and_audio_both_delivered(self, scores):
        """Test that both visual and audio alerts are delivered when enabled."""
        visual_triggered = []
        audio_triggered = []
        
        def visual_callback(level, message):
            visual_triggered.append(True)
        
        def audio_callback(level, volume):
            audio_triggered.append(True)
        
        # Create alert manager with both visual and audio enabled
        alert_manager = AlertManager(
            configuration=AlertConfiguration(
                enabled_alert_types=[AlertType.VISUAL, AlertType.AUDIO],
                sensitivity=0.5,
                audio_volume=0.8,
                haptic_intensity=0.7,
                escalation_enabled=False,
                escalation_interval=5.0
            ),
            visual_callback=visual_callback,
            audio_callback=audio_callback
        )
        
        # Create decision engine
        decision_engine = DecisionEngine()
        
        # Calculate drowsiness assessment
        assessment = decision_engine.calculate_drowsiness_score(**scores)
        
        # Trigger alert for high/critical levels
        if assessment.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
            alert_manager.trigger_alert(
                assessment.alert_level,
                assessment.drowsiness_score
            )
            
            # Verify both visual and audio were triggered
            assert len(visual_triggered) > 0, "Visual alert should be triggered"
            assert len(audio_triggered) > 0, "Audio alert should be triggered"


class TestAlertSensitivityCustomization:
    """
    Property 16: Alert Sensitivity Customization
    
    For any user-configured sensitivity level, the alert manager should
    adjust alert triggering behavior accordingly.
    
    Validates: Requirements 3.4
    """
    
    @settings(max_examples=100)
    @given(
        scores=drowsiness_scores(),
        sensitivity=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_sensitivity_affects_alert_triggering(self, scores, sensitivity):
        """Test that sensitivity level affects when alerts are triggered."""
        alert_triggered_high_sensitivity = []
        alert_triggered_low_sensitivity = []
        
        def callback_high(level, message):
            alert_triggered_high_sensitivity.append(True)
        
        def callback_low(level, message):
            alert_triggered_low_sensitivity.append(True)
        
        # Create alert manager with high sensitivity
        alert_manager_high = AlertManager(
            configuration=AlertConfiguration(
                enabled_alert_types=[AlertType.VISUAL],
                sensitivity=1.0,  # Maximum sensitivity
                audio_volume=0.8,
                haptic_intensity=0.7,
                escalation_enabled=False,
                escalation_interval=5.0
            ),
            visual_callback=callback_high
        )
        
        # Create alert manager with low sensitivity
        alert_manager_low = AlertManager(
            configuration=AlertConfiguration(
                enabled_alert_types=[AlertType.VISUAL],
                sensitivity=0.0,  # Minimum sensitivity
                audio_volume=0.8,
                haptic_intensity=0.7,
                escalation_enabled=False,
                escalation_interval=5.0
            ),
            visual_callback=callback_low
        )
        
        # Create decision engine
        decision_engine = DecisionEngine()
        
        # Calculate drowsiness assessment
        assessment = decision_engine.calculate_drowsiness_score(**scores)
        
        # Trigger alerts with both sensitivity levels
        if assessment.alert_level != AlertLevel.NONE:
            alert_manager_high.trigger_alert(
                assessment.alert_level,
                assessment.drowsiness_score
            )
            alert_manager_low.trigger_alert(
                assessment.alert_level,
                assessment.drowsiness_score
            )
            
            # High sensitivity should trigger more or equal alerts than low sensitivity
            # (High sensitivity = lower threshold = more alerts)
            high_count = len(alert_triggered_high_sensitivity)
            low_count = len(alert_triggered_low_sensitivity)
            
            assert high_count >= low_count, (
                f"High sensitivity ({high_count} alerts) should trigger at least as many "
                f"alerts as low sensitivity ({low_count} alerts)"
            )
    
    @settings(max_examples=100)
    @given(
        sensitivity=st.floats(min_value=0.0, max_value=1.0),
        audio_volume=st.floats(min_value=0.0, max_value=1.0),
        haptic_intensity=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_customization_parameters_accepted(
        self, sensitivity, audio_volume, haptic_intensity
    ):
        """Test that all customization parameters are accepted and stored."""
        alert_manager = AlertManager()
        
        # Customize alerts
        alert_manager.customize_alerts(
            sensitivity=sensitivity,
            audio_volume=audio_volume,
            haptic_intensity=haptic_intensity
        )
        
        # Verify customization was applied
        assert alert_manager.configuration.sensitivity == sensitivity
        assert alert_manager.configuration.audio_volume == audio_volume
        assert alert_manager.configuration.haptic_intensity == haptic_intensity
    
    @settings(max_examples=100)
    @given(enabled_types=st.lists(
        st.sampled_from([AlertType.VISUAL, AlertType.AUDIO, AlertType.HAPTIC]),
        min_size=1,
        max_size=3,
        unique=True
    ))
    def test_alert_type_customization(self, enabled_types):
        """Test that users can customize which alert types are enabled."""
        triggered_types = []
        
        def visual_callback(level, message):
            triggered_types.append(AlertType.VISUAL)
        
        def audio_callback(level, volume):
            triggered_types.append(AlertType.AUDIO)
        
        def haptic_callback(level, intensity):
            triggered_types.append(AlertType.HAPTIC)
        
        # Create alert manager
        alert_manager = AlertManager(
            visual_callback=visual_callback,
            audio_callback=audio_callback,
            haptic_callback=haptic_callback
        )
        
        # Customize enabled alert types
        alert_manager.customize_alerts(enabled_types=enabled_types)
        
        # Trigger a high alert
        alert_manager.trigger_alert(
            AlertLevel.HIGH,
            drowsiness_score=0.8
        )
        
        # Verify only enabled types were triggered
        for alert_type in triggered_types:
            assert alert_type in enabled_types, (
                f"{alert_type.value} was triggered but not in enabled types"
            )


class TestDecisionEngineProperties:
    """Additional property tests for DecisionEngine."""
    
    @settings(max_examples=100)
    @given(scores=drowsiness_scores())
    def test_drowsiness_score_in_valid_range(self, scores):
        """Test that drowsiness scores are always in valid range [0, 1]."""
        decision_engine = DecisionEngine()
        
        assessment = decision_engine.calculate_drowsiness_score(**scores)
        
        assert 0.0 <= assessment.drowsiness_score <= 1.0, (
            f"Drowsiness score {assessment.drowsiness_score} outside valid range [0, 1]"
        )
        assert 0.0 <= assessment.confidence <= 1.0, (
            f"Confidence {assessment.confidence} outside valid range [0, 1]"
        )
    
    @settings(max_examples=100)
    @given(
        scores=drowsiness_scores(),
        is_false_positive=st.booleans()
    )
    def test_threshold_adaptation_from_feedback(self, scores, is_false_positive):
        """Test that thresholds adapt based on user feedback."""
        decision_engine = DecisionEngine()
        
        # Get initial thresholds
        initial_low = decision_engine.low_threshold
        initial_medium = decision_engine.medium_threshold
        
        # Provide feedback multiple times
        for _ in range(5):
            decision_engine.update_thresholds(is_false_positive)
        
        # Check that thresholds changed appropriately
        if is_false_positive:
            # False positives should increase thresholds
            assert decision_engine.low_threshold >= initial_low
            assert decision_engine.medium_threshold >= initial_medium
        else:
            # True positives might decrease thresholds (if many true positives)
            # or keep them the same
            assert decision_engine.low_threshold <= initial_low + 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
