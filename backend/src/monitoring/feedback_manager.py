"""
Feedback Manager Module

This module provides user feedback tracking, preference learning,
and adaptive threshold adjustment based on user input.

Validates: Requirements 9.4, 10.5
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import statistics


class FeedbackType(Enum):
    """Types of user feedback"""
    ALERT_ACCURATE = "alert_accurate"
    ALERT_FALSE_ALARM = "alert_false_alarm"
    ALERT_MISSED = "alert_missed"
    ALERT_TOO_SENSITIVE = "alert_too_sensitive"
    ALERT_NOT_SENSITIVE = "alert_not_sensitive"
    ALERT_TIMING_GOOD = "alert_timing_good"
    ALERT_TIMING_LATE = "alert_timing_late"
    ALERT_TIMING_EARLY = "alert_timing_early"


@dataclass
class UserFeedback:
    """Container for user feedback"""
    feedback_id: str
    timestamp: float
    feedback_type: FeedbackType
    alert_id: Optional[str]
    drowsiness_score: float
    threshold_at_time: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'feedback_id': self.feedback_id,
            'timestamp': self.timestamp,
            'feedback_type': self.feedback_type.value,
            'alert_id': self.alert_id,
            'drowsiness_score': self.drowsiness_score,
            'threshold_at_time': self.threshold_at_time,
            'metadata': self.metadata or {}
        }


@dataclass
class UserPreferences:
    """Container for learned user preferences"""
    sensitivity_adjustment: float  # -1.0 to 1.0
    preferred_alert_types: List[str]
    alert_frequency_preference: str  # "low", "medium", "high"
    timing_preference: str  # "early", "normal", "late"
    confidence_threshold: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'sensitivity_adjustment': self.sensitivity_adjustment,
            'preferred_alert_types': self.preferred_alert_types,
            'alert_frequency_preference': self.alert_frequency_preference,
            'timing_preference': self.timing_preference,
            'confidence_threshold': self.confidence_threshold
        }


class FeedbackManager:
    """
    User feedback manager for drowsiness detection system.
    
    Tracks user feedback, learns preferences, and adapts system behavior.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.7,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.5,
        max_threshold: float = 0.9
    ):
        """
        Initialize feedback manager.
        
        Args:
            initial_threshold: Initial drowsiness threshold
            adaptation_rate: Rate of threshold adaptation (0-1)
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Feedback storage
        self.feedback_history: List[UserFeedback] = []
        self.feedback_counts: Dict[FeedbackType, int] = defaultdict(int)
        
        # Recent feedback for adaptation
        self.recent_feedback: deque = deque(maxlen=20)
        
        # User preferences
        self.preferences = UserPreferences(
            sensitivity_adjustment=0.0,
            preferred_alert_types=['visual', 'audio', 'haptic'],
            alert_frequency_preference='medium',
            timing_preference='normal',
            confidence_threshold=0.7
        )
        
        # Adaptation tracking
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 300.0  # Adapt every 5 minutes
        self.total_adaptations = 0
    
    def recordFeedback(
        self,
        feedback_type: str,
        alert_id: Optional[str],
        drowsiness_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record user feedback on alert accuracy or timing.
        
        Args:
            feedback_type: Type of feedback (e.g., "alert_accurate", "alert_false_alarm")
            alert_id: ID of the alert being rated
            drowsiness_score: Drowsiness score at time of alert
            metadata: Additional metadata
        
        Returns:
            Feedback ID
        
        Validates: Requirements 9.4
        """
        import hashlib
        
        # Generate feedback ID
        feedback_id = hashlib.sha256(
            f"{time.time()}{feedback_type}{alert_id}".encode()
        ).hexdigest()[:12]
        
        # Convert string to enum
        try:
            fb_type = FeedbackType(feedback_type)
        except ValueError:
            # Default to accurate if unknown type
            fb_type = FeedbackType.ALERT_ACCURATE
        
        # Create feedback record
        feedback = UserFeedback(
            feedback_id=feedback_id,
            timestamp=time.time(),
            feedback_type=fb_type,
            alert_id=alert_id,
            drowsiness_score=drowsiness_score,
            threshold_at_time=self.current_threshold,
            metadata=metadata
        )
        
        # Store feedback
        self.feedback_history.append(feedback)
        self.feedback_counts[fb_type] += 1
        self.recent_feedback.append(feedback)
        
        # Trigger adaptation if needed
        self._checkAdaptation()
        
        return feedback_id
    
    def getFeedbackStatistics(self) -> Dict[str, Any]:
        """
        Get statistics on user feedback.
        
        Returns:
            Dictionary with feedback statistics
        
        Validates: Requirements 9.4
        """
        total_feedback = len(self.feedback_history)
        
        if total_feedback == 0:
            return {
                'total_feedback': 0,
                'feedback_by_type': {},
                'accuracy_rate': 0.0,
                'false_alarm_rate': 0.0,
                'missed_alert_rate': 0.0
            }
        
        # Calculate rates
        accurate = self.feedback_counts[FeedbackType.ALERT_ACCURATE]
        false_alarms = self.feedback_counts[FeedbackType.ALERT_FALSE_ALARM]
        missed = self.feedback_counts[FeedbackType.ALERT_MISSED]
        
        return {
            'total_feedback': total_feedback,
            'feedback_by_type': {
                k.value: v for k, v in self.feedback_counts.items()
            },
            'accuracy_rate': accurate / total_feedback if total_feedback > 0 else 0.0,
            'false_alarm_rate': false_alarms / total_feedback if total_feedback > 0 else 0.0,
            'missed_alert_rate': missed / total_feedback if total_feedback > 0 else 0.0
        }
    
    def adaptThreshold(self) -> Tuple[float, str]:
        """
        Adapt drowsiness threshold based on user feedback.
        
        Returns:
            Tuple of (new_threshold, reason)
        
        Validates: Requirements 10.5
        """
        if len(self.recent_feedback) < 5:
            return self.current_threshold, "Insufficient feedback for adaptation"
        
        # Count recent feedback types
        false_alarms = sum(
            1 for f in self.recent_feedback
            if f.feedback_type == FeedbackType.ALERT_FALSE_ALARM
        )
        too_sensitive = sum(
            1 for f in self.recent_feedback
            if f.feedback_type == FeedbackType.ALERT_TOO_SENSITIVE
        )
        missed = sum(
            1 for f in self.recent_feedback
            if f.feedback_type == FeedbackType.ALERT_MISSED
        )
        not_sensitive = sum(
            1 for f in self.recent_feedback
            if f.feedback_type == FeedbackType.ALERT_NOT_SENSITIVE
        )
        
        old_threshold = self.current_threshold
        reason = "No adjustment needed"
        
        # Adjust threshold based on feedback
        if false_alarms > 3 or too_sensitive > 3:
            # Too many false alarms - increase threshold (less sensitive)
            adjustment = self.adaptation_rate * 0.1
            self.current_threshold = min(
                self.max_threshold,
                self.current_threshold + adjustment
            )
            reason = f"Increased threshold due to {false_alarms} false alarms, {too_sensitive} too sensitive"
        
        elif missed > 3 or not_sensitive > 3:
            # Missing alerts - decrease threshold (more sensitive)
            adjustment = self.adaptation_rate * 0.1
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold - adjustment
            )
            reason = f"Decreased threshold due to {missed} missed alerts, {not_sensitive} not sensitive"
        
        if self.current_threshold != old_threshold:
            self.total_adaptations += 1
            self.last_adaptation_time = time.time()
        
        return self.current_threshold, reason
    
    def learnUserPreferences(self) -> UserPreferences:
        """
        Learn user preferences from feedback history.
        
        Returns:
            Updated user preferences
        
        Validates: Requirements 10.5
        """
        if len(self.feedback_history) < 10:
            return self.preferences
        
        # Analyze sensitivity preference
        too_sensitive_count = self.feedback_counts[FeedbackType.ALERT_TOO_SENSITIVE]
        not_sensitive_count = self.feedback_counts[FeedbackType.ALERT_NOT_SENSITIVE]
        
        if too_sensitive_count > not_sensitive_count * 2:
            self.preferences.sensitivity_adjustment = -0.5
            self.preferences.alert_frequency_preference = 'low'
        elif not_sensitive_count > too_sensitive_count * 2:
            self.preferences.sensitivity_adjustment = 0.5
            self.preferences.alert_frequency_preference = 'high'
        else:
            self.preferences.sensitivity_adjustment = 0.0
            self.preferences.alert_frequency_preference = 'medium'
        
        # Analyze timing preference
        early_count = self.feedback_counts[FeedbackType.ALERT_TIMING_EARLY]
        late_count = self.feedback_counts[FeedbackType.ALERT_TIMING_LATE]
        
        if early_count > late_count * 2:
            self.preferences.timing_preference = 'late'
        elif late_count > early_count * 2:
            self.preferences.timing_preference = 'early'
        else:
            self.preferences.timing_preference = 'normal'
        
        # Update confidence threshold based on accuracy
        stats = self.getFeedbackStatistics()
        if stats['accuracy_rate'] > 0.9:
            # High accuracy - can be more aggressive
            self.preferences.confidence_threshold = 0.6
        elif stats['accuracy_rate'] < 0.7:
            # Low accuracy - be more conservative
            self.preferences.confidence_threshold = 0.8
        else:
            self.preferences.confidence_threshold = 0.7
        
        return self.preferences
    
    def getUserPreferences(self) -> UserPreferences:
        """Get current user preferences"""
        return self.preferences
    
    def setUserPreferences(self, preferences: UserPreferences):
        """Set user preferences manually"""
        self.preferences = preferences
    
    def getCurrentThreshold(self) -> float:
        """Get current drowsiness threshold"""
        return self.current_threshold
    
    def setThreshold(self, threshold: float) -> bool:
        """
        Manually set drowsiness threshold.
        
        Args:
            threshold: New threshold value (0-1)
        
        Returns:
            True if set successfully
        """
        if self.min_threshold <= threshold <= self.max_threshold:
            self.current_threshold = threshold
            return True
        return False
    
    def getFeedbackHistory(
        self,
        feedback_type: Optional[str] = None,
        max_count: Optional[int] = None
    ) -> List[UserFeedback]:
        """
        Get feedback history with optional filtering.
        
        Args:
            feedback_type: Filter by feedback type
            max_count: Maximum number of records to return
        
        Returns:
            List of user feedback records
        """
        history = self.feedback_history.copy()
        
        # Filter by type
        if feedback_type:
            try:
                fb_type = FeedbackType(feedback_type)
                history = [f for f in history if f.feedback_type == fb_type]
            except ValueError:
                pass
        
        # Limit count
        if max_count:
            history = history[-max_count:]
        
        return history
    
    def exportFeedback(self, filepath: str) -> bool:
        """
        Export feedback data to JSON file.
        
        Args:
            filepath: Path to export file
        
        Returns:
            True if exported successfully
        """
        try:
            data = {
                'export_time': time.time(),
                'current_threshold': self.current_threshold,
                'total_adaptations': self.total_adaptations,
                'preferences': self.preferences.to_dict(),
                'statistics': self.getFeedbackStatistics(),
                'feedback_history': [f.to_dict() for f in self.feedback_history]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting feedback: {e}")
            return False
    
    def reset(self):
        """Reset feedback manager to initial state"""
        self.feedback_history.clear()
        self.feedback_counts.clear()
        self.recent_feedback.clear()
        
        self.current_threshold = self.initial_threshold
        self.total_adaptations = 0
        self.last_adaptation_time = time.time()
        
        self.preferences = UserPreferences(
            sensitivity_adjustment=0.0,
            preferred_alert_types=['visual', 'audio', 'haptic'],
            alert_frequency_preference='medium',
            timing_preference='normal',
            confidence_threshold=0.7
        )
    
    def _checkAdaptation(self):
        """Check if threshold adaptation should be triggered"""
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return
        
        # Check if we have enough recent feedback
        if len(self.recent_feedback) >= 5:
            self.adaptThreshold()
            self.learnUserPreferences()
