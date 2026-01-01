"""
Metrics Collector Module

This module provides performance metrics collection including accuracy tracking,
latency measurement, error event recording, and performance degradation detection.

Validates: Requirements 9.1, 9.2, 9.3, 9.5, 8.5
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import statistics


class MetricType(Enum):
    """Types of metrics collected"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    TRUE_POSITIVE = "true_positive"
    TRUE_NEGATIVE = "true_negative"
    FRAME_PROCESSING = "frame_processing"
    MODEL_INFERENCE = "model_inference"


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float
    metric_type: MetricType
    value: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'metric_type': self.metric_type.value,
            'value': self.value,
            'metadata': self.metadata or {}
        }


@dataclass
class ErrorEvent:
    """Container for error events (false positives/negatives)"""
    event_id: str
    timestamp: float
    error_type: MetricType  # FALSE_POSITIVE or FALSE_NEGATIVE
    predicted_state: str
    actual_state: str
    confidence: float
    features: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'error_type': self.error_type.value,
            'predicted_state': self.predicted_state,
            'actual_state': self.actual_state,
            'confidence': self.confidence,
            'features': self.features
        }


class MetricsCollector:
    """
    Performance metrics collector for drowsiness detection system.
    
    Tracks accuracy, latency, error events, and detects performance degradation.
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        degradation_threshold: float = 0.75,
        latency_threshold_ms: float = 100.0
    ):
        """
        Initialize metrics collector.
        
        Args:
            history_size: Maximum number of metrics to keep in memory
            degradation_threshold: Accuracy threshold below which to flag degradation
            latency_threshold_ms: Latency threshold in milliseconds
        """
        self.history_size = history_size
        self.degradation_threshold = degradation_threshold
        self.latency_threshold_ms = latency_threshold_ms
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.error_events: List[ErrorEvent] = []
        
        # Counters for accuracy calculation
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Latency tracking
        self.latency_samples: deque = deque(maxlen=100)
        
        # Performance degradation tracking
        self.degradation_detected = False
        self.last_degradation_check = time.time()
        self.degradation_check_interval = 60.0  # Check every 60 seconds
        
        # Session statistics
        self.session_start_time = time.time()
        self.total_predictions = 0

    
    def logAccuracyMetric(
        self,
        predicted_state: str,
        actual_state: str,
        confidence: float,
        features: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Log accuracy metric for a prediction.
        
        Args:
            predicted_state: Predicted drowsiness state
            actual_state: Actual drowsiness state (ground truth)
            confidence: Prediction confidence (0-1)
            features: Optional feature values used for prediction
        
        Returns:
            True if logged successfully
        
        Validates: Requirements 9.1
        """
        self.total_predictions += 1
        
        # Determine if prediction was correct
        is_drowsy_prediction = predicted_state.lower() in ['drowsy', 'alert']
        is_drowsy_actual = actual_state.lower() in ['drowsy', 'alert']
        
        # Update confusion matrix
        if predicted_state.lower() == 'drowsy' and actual_state.lower() == 'drowsy':
            self.true_positives += 1
            metric_type = MetricType.TRUE_POSITIVE
        elif predicted_state.lower() == 'alert' and actual_state.lower() == 'alert':
            self.true_negatives += 1
            metric_type = MetricType.TRUE_NEGATIVE
        elif predicted_state.lower() == 'drowsy' and actual_state.lower() == 'alert':
            self.false_positives += 1
            metric_type = MetricType.FALSE_POSITIVE
            self._recordErrorEvent(
                MetricType.FALSE_POSITIVE,
                predicted_state,
                actual_state,
                confidence,
                features or {}
            )
        else:  # predicted alert, actual drowsy
            self.false_negatives += 1
            metric_type = MetricType.FALSE_NEGATIVE
            self._recordErrorEvent(
                MetricType.FALSE_NEGATIVE,
                predicted_state,
                actual_state,
                confidence,
                features or {}
            )
        
        # Log metric
        metric = PerformanceMetrics(
            timestamp=time.time(),
            metric_type=metric_type,
            value=1.0 if predicted_state == actual_state else 0.0,
            metadata={
                'predicted': predicted_state,
                'actual': actual_state,
                'confidence': confidence
            }
        )
        self.metrics_history.append(metric)
        
        return True
    
    def logLatencyMetric(
        self,
        latency_ms: float,
        operation: str = "inference"
    ) -> bool:
        """
        Log latency metric for an operation.
        
        Args:
            latency_ms: Latency in milliseconds
            operation: Operation name (e.g., "inference", "frame_processing")
        
        Returns:
            True if logged successfully
        
        Validates: Requirements 9.3
        """
        self.latency_samples.append(latency_ms)
        
        metric = PerformanceMetrics(
            timestamp=time.time(),
            metric_type=MetricType.LATENCY,
            value=latency_ms,
            metadata={'operation': operation}
        )
        self.metrics_history.append(metric)
        
        return True
    
    def recordErrorEvent(
        self,
        error_type: str,
        predicted_state: str,
        actual_state: str,
        confidence: float,
        features: Dict[str, float]
    ) -> str:
        """
        Record an error event (false positive or false negative).
        
        Args:
            error_type: Type of error ("false_positive" or "false_negative")
            predicted_state: Predicted state
            actual_state: Actual state
            confidence: Prediction confidence
            features: Feature values used for prediction
        
        Returns:
            Event ID
        
        Validates: Requirements 9.2
        """
        metric_type = (
            MetricType.FALSE_POSITIVE if error_type == "false_positive"
            else MetricType.FALSE_NEGATIVE
        )
        
        return self._recordErrorEvent(
            metric_type,
            predicted_state,
            actual_state,
            confidence,
            features
        )
    
    def getAccuracyMetrics(self) -> Dict[str, float]:
        """
        Calculate and return accuracy metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, F1 score
        
        Validates: Requirements 9.1
        """
        total = (
            self.true_positives + self.true_negatives +
            self.false_positives + self.false_negatives
        )
        
        if total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'total_predictions': 0
            }
        
        accuracy = (
            (self.true_positives + self.true_negatives) / total
        )
        
        # Precision: TP / (TP + FP)
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else 0.0
        )
        
        # Recall: TP / (TP + FN)
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else 0.0
        )
        
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_predictions': total,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }
    
    def getLatencyMetrics(self) -> Dict[str, float]:
        """
        Calculate and return latency metrics.
        
        Returns:
            Dictionary with latency statistics
        
        Validates: Requirements 9.3
        """
        if not self.latency_samples:
            return {
                'mean_ms': 0.0,
                'median_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
                'samples': 0
            }
        
        samples = list(self.latency_samples)
        samples.sort()
        
        return {
            'mean_ms': statistics.mean(samples),
            'median_ms': statistics.median(samples),
            'min_ms': min(samples),
            'max_ms': max(samples),
            'p95_ms': samples[int(len(samples) * 0.95)] if len(samples) > 0 else 0.0,
            'p99_ms': samples[int(len(samples) * 0.99)] if len(samples) > 0 else 0.0,
            'samples': len(samples)
        }
    
    def checkPerformanceDegradation(self) -> Tuple[bool, Optional[str]]:
        """
        Check if performance has degraded below acceptable thresholds.
        
        Returns:
            Tuple of (degradation_detected, reason)
        
        Validates: Requirements 9.5, 8.5
        """
        current_time = time.time()
        
        # Only check periodically
        if current_time - self.last_degradation_check < self.degradation_check_interval:
            return self.degradation_detected, None
        
        self.last_degradation_check = current_time
        
        # Check accuracy degradation
        accuracy_metrics = self.getAccuracyMetrics()
        if accuracy_metrics['total_predictions'] >= 10:  # Need minimum samples
            if accuracy_metrics['accuracy'] < self.degradation_threshold:
                self.degradation_detected = True
                return True, f"Accuracy {accuracy_metrics['accuracy']:.2%} below threshold {self.degradation_threshold:.2%}"
        
        # Check latency degradation
        latency_metrics = self.getLatencyMetrics()
        if latency_metrics['samples'] >= 10:
            if latency_metrics['mean_ms'] > self.latency_threshold_ms:
                self.degradation_detected = True
                return True, f"Mean latency {latency_metrics['mean_ms']:.1f}ms exceeds threshold {self.latency_threshold_ms}ms"
        
        self.degradation_detected = False
        return False, None
    
    def getErrorEvents(
        self,
        error_type: Optional[str] = None,
        max_count: Optional[int] = None
    ) -> List[ErrorEvent]:
        """
        Get recorded error events.
        
        Args:
            error_type: Filter by error type ("false_positive" or "false_negative")
            max_count: Maximum number of events to return
        
        Returns:
            List of error events
        
        Validates: Requirements 9.2
        """
        events = self.error_events.copy()
        
        # Filter by type
        if error_type:
            metric_type = (
                MetricType.FALSE_POSITIVE if error_type == "false_positive"
                else MetricType.FALSE_NEGATIVE
            )
            events = [e for e in events if e.error_type == metric_type]
        
        # Limit count
        if max_count:
            events = events[-max_count:]
        
        return events
    
    def exportMetrics(self, filepath: str) -> bool:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to export file
        
        Returns:
            True if exported successfully
        """
        try:
            data = {
                'session_start': self.session_start_time,
                'export_time': time.time(),
                'accuracy_metrics': self.getAccuracyMetrics(),
                'latency_metrics': self.getLatencyMetrics(),
                'error_events': [e.to_dict() for e in self.error_events],
                'degradation_detected': self.degradation_detected
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting metrics: {e}")
            return False
    
    def reset(self):
        """Reset all metrics and counters"""
        self.metrics_history.clear()
        self.error_events.clear()
        self.latency_samples.clear()
        
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        self.total_predictions = 0
        self.degradation_detected = False
        self.session_start_time = time.time()
    
    def _recordErrorEvent(
        self,
        error_type: MetricType,
        predicted_state: str,
        actual_state: str,
        confidence: float,
        features: Dict[str, float]
    ) -> str:
        """Internal method to record error event"""
        import hashlib
        
        event_id = hashlib.sha256(
            f"{time.time()}{predicted_state}{actual_state}".encode()
        ).hexdigest()[:12]
        
        event = ErrorEvent(
            event_id=event_id,
            timestamp=time.time(),
            error_type=error_type,
            predicted_state=predicted_state,
            actual_state=actual_state,
            confidence=confidence,
            features=features
        )
        
        self.error_events.append(event)
        
        return event_id
