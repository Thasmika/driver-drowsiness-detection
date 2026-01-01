"""
Property-Based Tests for Monitoring Features

Tests correctness properties for performance metrics collection,
user feedback tracking, and adaptive threshold adjustment.

Feature: driver-drowsiness-detection
Properties: 38, 41
Validates: Requirements 9.1, 9.4
"""

import pytest
import time
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

from src.monitoring.metrics_collector import (
    MetricsCollector,
    MetricType,
    PerformanceMetrics
)
from src.monitoring.feedback_manager import (
    FeedbackManager,
    FeedbackType,
    UserFeedback
)


# ============================================================================
# Test Generators
# ============================================================================

@st.composite
def prediction_strategy(draw):
    """Generate prediction data"""
    states = ['drowsy', 'alert']
    return {
        'predicted': draw(st.sampled_from(states)),
        'actual': draw(st.sampled_from(states)),
        'confidence': draw(st.floats(min_value=0.0, max_value=1.0))
    }


@st.composite
def feedback_strategy(draw):
    """Generate user feedback data"""
    feedback_types = [
        'alert_accurate',
        'alert_false_alarm',
        'alert_missed',
        'alert_too_sensitive',
        'alert_not_sensitive'
    ]
    return {
        'type': draw(st.sampled_from(feedback_types)),
        'drowsiness_score': draw(st.floats(min_value=0.0, max_value=1.0)),
        'alert_id': draw(st.text(min_size=5, max_size=20))
    }


# ============================================================================
# Property 38: Accuracy Metrics Logging
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    predictions=st.lists(prediction_strategy(), min_size=10, max_size=50)
)
def test_property_38_accuracy_metrics_logging(predictions):
    """
    Property 38: Accuracy Metrics Logging
    
    For any sequence of predictions, the metrics collector should
    accurately log and calculate accuracy metrics.
    
    Feature: driver-drowsiness-detection, Property 38
    Validates: Requirements 9.1
    """
    collector = MetricsCollector()
    
    # Track expected counts
    expected_tp = 0
    expected_tn = 0
    expected_fp = 0
    expected_fn = 0
    
    # Log all predictions
    for pred in predictions:
        collector.logAccuracyMetric(
            predicted_state=pred['predicted'],
            actual_state=pred['actual'],
            confidence=pred['confidence']
        )
        
        # Update expected counts
        if pred['predicted'] == 'drowsy' and pred['actual'] == 'drowsy':
            expected_tp += 1
        elif pred['predicted'] == 'alert' and pred['actual'] == 'alert':
            expected_tn += 1
        elif pred['predicted'] == 'drowsy' and pred['actual'] == 'alert':
            expected_fp += 1
        else:
            expected_fn += 1
    
    # Get metrics
    metrics = collector.getAccuracyMetrics()
    
    # Property: Total predictions should match
    assert metrics['total_predictions'] == len(predictions), (
        "Total predictions should match number of logged predictions"
    )
    
    # Property: Confusion matrix should be correct
    assert metrics['true_positives'] == expected_tp, (
        "True positives count should be correct"
    )
    assert metrics['true_negatives'] == expected_tn, (
        "True negatives count should be correct"
    )
    assert metrics['false_positives'] == expected_fp, (
        "False positives count should be correct"
    )
    assert metrics['false_negatives'] == expected_fn, (
        "False negatives count should be correct"
    )
    
    # Property: Accuracy should be in valid range
    assert 0.0 <= metrics['accuracy'] <= 1.0, (
        "Accuracy should be between 0 and 1"
    )
    
    # Property: Precision should be in valid range
    assert 0.0 <= metrics['precision'] <= 1.0, (
        "Precision should be between 0 and 1"
    )
    
    # Property: Recall should be in valid range
    assert 0.0 <= metrics['recall'] <= 1.0, (
        "Recall should be between 0 and 1"
    )
    
    # Property: F1 score should be in valid range
    assert 0.0 <= metrics['f1_score'] <= 1.0, (
        "F1 score should be between 0 and 1"
    )
    
    # Property: Accuracy calculation should be correct
    if len(predictions) > 0:
        expected_accuracy = (expected_tp + expected_tn) / len(predictions)
        assert abs(metrics['accuracy'] - expected_accuracy) < 0.001, (
            "Accuracy calculation should be correct"
        )


# ============================================================================
# Property 41: User Feedback Tracking
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    feedbacks=st.lists(feedback_strategy(), min_size=5, max_size=30)
)
def test_property_41_user_feedback_tracking(feedbacks):
    """
    Property 41: User Feedback Tracking
    
    For any sequence of user feedback, the feedback manager should
    accurately track and aggregate feedback data.
    
    Feature: driver-drowsiness-detection, Property 41
    Validates: Requirements 9.4
    """
    manager = FeedbackManager()
    
    # Track expected counts
    feedback_counts = {}
    
    # Record all feedback
    for fb in feedbacks:
        manager.recordFeedback(
            feedback_type=fb['type'],
            alert_id=fb['alert_id'],
            drowsiness_score=fb['drowsiness_score']
        )
        
        # Update expected counts
        feedback_counts[fb['type']] = feedback_counts.get(fb['type'], 0) + 1
    
    # Get statistics
    stats = manager.getFeedbackStatistics()
    
    # Property: Total feedback should match
    assert stats['total_feedback'] == len(feedbacks), (
        "Total feedback count should match number of recorded feedbacks"
    )
    
    # Property: Feedback counts by type should be correct
    for fb_type, count in feedback_counts.items():
        assert stats['feedback_by_type'].get(fb_type, 0) == count, (
            f"Feedback count for {fb_type} should be correct"
        )
    
    # Property: Rates should be in valid range
    assert 0.0 <= stats['accuracy_rate'] <= 1.0, (
        "Accuracy rate should be between 0 and 1"
    )
    assert 0.0 <= stats['false_alarm_rate'] <= 1.0, (
        "False alarm rate should be between 0 and 1"
    )
    assert 0.0 <= stats['missed_alert_rate'] <= 1.0, (
        "Missed alert rate should be between 0 and 1"
    )
    
    # Property: Rates should sum to at most 1.0
    rate_sum = (
        stats['accuracy_rate'] +
        stats['false_alarm_rate'] +
        stats['missed_alert_rate']
    )
    assert rate_sum <= 1.0, (
        "Sum of rates should not exceed 1.0"
    )


# ============================================================================
# Additional Monitoring Tests
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    latencies=st.lists(
        st.floats(min_value=1.0, max_value=200.0),
        min_size=10,
        max_size=100
    )
)
def test_latency_metrics_calculation(latencies):
    """
    Test latency metrics calculation.
    
    For any sequence of latency measurements, statistics should
    be calculated correctly.
    """
    collector = MetricsCollector()
    
    # Log all latencies
    for latency in latencies:
        collector.logLatencyMetric(latency)
    
    # Get metrics
    metrics = collector.getLatencyMetrics()
    
    # Property: Sample count should match
    assert metrics['samples'] == len(latencies), (
        "Sample count should match number of logged latencies"
    )
    
    # Property: Mean should be in range
    assert min(latencies) <= metrics['mean_ms'] <= max(latencies), (
        "Mean should be within min and max values"
    )
    
    # Property: Median should be in range
    assert min(latencies) <= metrics['median_ms'] <= max(latencies), (
        "Median should be within min and max values"
    )
    
    # Property: Min and max should match
    assert abs(metrics['min_ms'] - min(latencies)) < 0.001, (
        "Min should match minimum latency"
    )
    assert abs(metrics['max_ms'] - max(latencies)) < 0.001, (
        "Max should match maximum latency"
    )
    
    # Property: Percentiles should be ordered
    assert metrics['min_ms'] <= metrics['median_ms'] <= metrics['max_ms'], (
        "Percentiles should be in ascending order"
    )
    assert metrics['median_ms'] <= metrics['p95_ms'] <= metrics['p99_ms'], (
        "Higher percentiles should be greater than or equal to lower ones"
    )


@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    false_alarm_count=st.integers(min_value=0, max_value=10),
    missed_count=st.integers(min_value=0, max_value=10)
)
def test_threshold_adaptation(false_alarm_count, missed_count):
    """
    Test adaptive threshold adjustment based on feedback.
    
    For any pattern of false alarms and missed alerts, threshold
    should adapt appropriately.
    """
    manager = FeedbackManager(initial_threshold=0.7)
    initial_threshold = manager.getCurrentThreshold()
    
    # Record false alarms
    for i in range(false_alarm_count):
        manager.recordFeedback(
            feedback_type='alert_false_alarm',
            alert_id=f'alert_{i}',
            drowsiness_score=0.6
        )
    
    # Record missed alerts
    for i in range(missed_count):
        manager.recordFeedback(
            feedback_type='alert_missed',
            alert_id=f'alert_{i + false_alarm_count}',
            drowsiness_score=0.8
        )
    
    # Trigger adaptation
    new_threshold, reason = manager.adaptThreshold()
    
    # Property: Threshold should be in valid range
    assert manager.min_threshold <= new_threshold <= manager.max_threshold, (
        "Threshold should be within valid range"
    )
    
    # Property: Threshold should adapt based on feedback
    if false_alarm_count > 3:
        # Should increase threshold (less sensitive)
        assert new_threshold >= initial_threshold, (
            "Threshold should increase with many false alarms"
        )
    elif missed_count > 3:
        # Should decrease threshold (more sensitive)
        assert new_threshold <= initial_threshold, (
            "Threshold should decrease with many missed alerts"
        )


@pytest.mark.property
def test_error_event_recording():
    """
    Test error event recording for false positives and negatives.
    
    For any error events, they should be properly recorded with
    all necessary information.
    """
    collector = MetricsCollector()
    
    # Record false positive
    event_id = collector.recordErrorEvent(
        error_type='false_positive',
        predicted_state='drowsy',
        actual_state='alert',
        confidence=0.85,
        features={'ear': 0.25, 'mar': 0.5}
    )
    
    # Property: Event ID should be returned
    assert event_id is not None and len(event_id) > 0, (
        "Event ID should be returned"
    )
    
    # Get error events
    events = collector.getErrorEvents(error_type='false_positive')
    
    # Property: Event should be recorded
    assert len(events) == 1, (
        "Error event should be recorded"
    )
    
    # Property: Event should have correct type
    assert events[0].error_type == MetricType.FALSE_POSITIVE, (
        "Event should have correct error type"
    )
    
    # Property: Event should have features
    assert 'ear' in events[0].features, (
        "Event should include feature data"
    )


@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    accuracy=st.floats(min_value=0.5, max_value=1.0)
)
def test_performance_degradation_detection(accuracy):
    """
    Test performance degradation detection.
    
    For any accuracy level, degradation should be detected when
    accuracy falls below threshold.
    """
    collector = MetricsCollector(degradation_threshold=0.75)
    
    # Generate predictions to achieve target accuracy
    num_predictions = 20
    num_correct = int(num_predictions * accuracy)
    num_incorrect = num_predictions - num_correct
    
    # Log correct predictions
    for i in range(num_correct):
        collector.logAccuracyMetric(
            predicted_state='drowsy',
            actual_state='drowsy',
            confidence=0.9
        )
    
    # Log incorrect predictions
    for i in range(num_incorrect):
        collector.logAccuracyMetric(
            predicted_state='drowsy',
            actual_state='alert',
            confidence=0.9
        )
    
    # Force degradation check by setting last check time to past
    collector.last_degradation_check = time.time() - collector.degradation_check_interval - 1
    
    # Check degradation
    degraded, reason = collector.checkPerformanceDegradation()
    
    # Property: Degradation should be detected when accuracy is low
    if accuracy < collector.degradation_threshold:
        assert degraded is True, (
            f"Degradation should be detected when accuracy {accuracy:.2%} "
            f"is below threshold {collector.degradation_threshold:.2%}"
        )
    else:
        assert degraded is False, (
            f"Degradation should not be detected when accuracy {accuracy:.2%} "
            f"is above threshold {collector.degradation_threshold:.2%}"
        )


@pytest.mark.property
def test_user_preference_learning():
    """
    Test user preference learning from feedback.
    
    For any pattern of feedback, preferences should be learned
    and applied appropriately.
    """
    manager = FeedbackManager()
    
    # Simulate user who finds system too sensitive
    for i in range(15):
        manager.recordFeedback(
            feedback_type='alert_too_sensitive',
            alert_id=f'alert_{i}',
            drowsiness_score=0.6
        )
    
    # Learn preferences
    preferences = manager.learnUserPreferences()
    
    # Property: Sensitivity should be adjusted down
    assert preferences.sensitivity_adjustment < 0, (
        "Sensitivity should be adjusted down for too sensitive feedback"
    )
    
    # Property: Alert frequency preference should be low
    assert preferences.alert_frequency_preference == 'low', (
        "Alert frequency should be set to low"
    )
    
    # Reset and test opposite
    manager.reset()
    
    # Simulate user who finds system not sensitive enough
    for i in range(15):
        manager.recordFeedback(
            feedback_type='alert_not_sensitive',
            alert_id=f'alert_{i}',
            drowsiness_score=0.8
        )
    
    # Learn preferences
    preferences = manager.learnUserPreferences()
    
    # Property: Sensitivity should be adjusted up
    assert preferences.sensitivity_adjustment > 0, (
        "Sensitivity should be adjusted up for not sensitive feedback"
    )
    
    # Property: Alert frequency preference should be high
    assert preferences.alert_frequency_preference == 'high', (
        "Alert frequency should be set to high"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
