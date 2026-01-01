"""
Eye Aspect Ratio (EAR) Calculator Module

This module implements Eye Aspect Ratio calculation for blink detection
and microsleep episode identification in drowsiness detection.

Validates: Requirements 2.2, 2.5
"""

import time
from typing import List, Tuple, Optional, Deque
from collections import deque
import numpy as np


class BlinkEvent:
    """Container for blink event data"""
    
    def __init__(self, start_time: float, end_time: float, min_ear: float):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.min_ear = min_ear
    
    def __repr__(self):
        return f"BlinkEvent(duration={self.duration:.3f}s, min_ear={self.min_ear:.3f})"


class EARCalculator:
    """
    Eye Aspect Ratio (EAR) calculator for drowsiness detection.
    
    EAR is calculated using the formula:
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    
    Where p1-p6 are eye landmark points arranged as:
    p1, p4: horizontal corners
    p2, p3, p5, p6: vertical points
    """
    
    # Configurable thresholds for different drowsiness levels
    DEFAULT_NORMAL_EAR_MIN = 0.25
    DEFAULT_NORMAL_EAR_MAX = 0.35
    DEFAULT_DROWSY_EAR_THRESHOLD = 0.25
    DEFAULT_MICROSLEEP_EAR_THRESHOLD = 0.20
    DEFAULT_BLINK_DURATION_MAX = 0.5  # seconds
    DEFAULT_MICROSLEEP_DURATION_MIN = 2.0  # seconds
    
    def __init__(
        self,
        normal_ear_range: Tuple[float, float] = (DEFAULT_NORMAL_EAR_MIN, DEFAULT_NORMAL_EAR_MAX),
        drowsy_ear_threshold: float = DEFAULT_DROWSY_EAR_THRESHOLD,
        microsleep_ear_threshold: float = DEFAULT_MICROSLEEP_EAR_THRESHOLD,
        blink_duration_max: float = DEFAULT_BLINK_DURATION_MAX,
        microsleep_duration_min: float = DEFAULT_MICROSLEEP_DURATION_MIN,
        history_window: int = 100
    ):
        """
        Initialize the EAR Calculator.
        
        Args:
            normal_ear_range: (min, max) EAR values for normal alertness
            drowsy_ear_threshold: EAR threshold below which indicates drowsiness
            microsleep_ear_threshold: EAR threshold for microsleep detection
            blink_duration_max: Maximum duration for normal blink (seconds)
            microsleep_duration_min: Minimum duration to classify as microsleep (seconds)
            history_window: Number of EAR values to keep in history
        """
        self.normal_ear_range = normal_ear_range
        self.drowsy_ear_threshold = drowsy_ear_threshold
        self.microsleep_ear_threshold = microsleep_ear_threshold
        self.blink_duration_max = blink_duration_max
        self.microsleep_duration_min = microsleep_duration_min
        self.history_window = history_window
        
        # Time series data for EAR analysis
        self.ear_history: Deque[Tuple[float, float]] = deque(maxlen=history_window)  # (timestamp, ear_value)
        self.blink_history: List[BlinkEvent] = []
        self.microsleep_events: List[BlinkEvent] = []
        
        # State tracking for blink detection
        self.is_eye_closed = False
        self.eye_close_start_time: Optional[float] = None
        self.min_ear_during_closure: Optional[float] = None
        
        # Performance metrics
        self.total_blinks = 0
        self.total_microsleeps = 0
    
    def calculateEAR(self, eye_landmarks: List[Tuple[float, float, float]]) -> Optional[float]:
        """
        Calculate Eye Aspect Ratio for a single eye.
        
        The EAR formula uses 6 key points from the eye landmarks:
        - 2 horizontal points (corners)
        - 4 vertical points (top and bottom of eye)
        
        Args:
            eye_landmarks: List of (x, y, z) coordinates for eye landmarks
        
        Returns:
            EAR value or None if calculation fails
        
        Validates: Requirements 2.2
        """
        if not eye_landmarks or len(eye_landmarks) < 6:
            return None
        
        try:
            # Convert to numpy array for easier computation
            points = np.array([(x, y) for x, y, z in eye_landmarks[:6]])
            
            # For MediaPipe landmarks, we need to identify the key points
            # Using a simplified approach with the first 6 points
            # In practice, we select specific indices based on eye shape
            
            # Calculate vertical distances
            # Using points at indices that represent top and bottom of eye
            vertical_1 = np.linalg.norm(points[1] - points[5])
            vertical_2 = np.linalg.norm(points[2] - points[4])
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(points[0] - points[3])
            
            if horizontal == 0:
                return None
            
            # Calculate EAR
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            
            return float(ear)
            
        except (IndexError, ValueError, ZeroDivisionError):
            return None
    
    def getAverageEAR(self, left_eye_landmarks: List[Tuple[float, float, float]], 
                      right_eye_landmarks: List[Tuple[float, float, float]]) -> Optional[float]:
        """
        Calculate average EAR across both eyes.
        
        Args:
            left_eye_landmarks: Landmarks for left eye
            right_eye_landmarks: Landmarks for right eye
        
        Returns:
            Average EAR value or None if calculation fails
        
        Validates: Requirements 2.2
        """
        left_ear = self.calculateEAR(left_eye_landmarks)
        right_ear = self.calculateEAR(right_eye_landmarks)
        
        if left_ear is None and right_ear is None:
            return None
        elif left_ear is None:
            return right_ear
        elif right_ear is None:
            return left_ear
        else:
            return (left_ear + right_ear) / 2.0
    
    def detectBlink(self, ear_value: float, timestamp: Optional[float] = None) -> Optional[BlinkEvent]:
        """
        Detect blink events using EAR time series analysis.
        
        A blink is characterized by:
        - Rapid decrease in EAR below threshold
        - Duration less than blink_duration_max
        - Rapid increase back to normal
        
        Args:
            ear_value: Current EAR value
            timestamp: Timestamp of the measurement (defaults to current time)
        
        Returns:
            BlinkEvent if a blink was completed, None otherwise
        
        Validates: Requirements 2.2
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add to history
        self.ear_history.append((timestamp, ear_value))
        
        blink_event = None
        
        # Check if eye just closed
        if not self.is_eye_closed and ear_value < self.drowsy_ear_threshold:
            self.is_eye_closed = True
            self.eye_close_start_time = timestamp
            self.min_ear_during_closure = ear_value
        
        # Track minimum EAR during closure
        elif self.is_eye_closed:
            if ear_value < self.min_ear_during_closure:
                self.min_ear_during_closure = ear_value
            
            # Check if eye reopened
            if ear_value >= self.drowsy_ear_threshold:
                closure_duration = timestamp - self.eye_close_start_time
                
                # Classify as blink or microsleep based on duration
                if closure_duration <= self.blink_duration_max:
                    # Normal blink
                    blink_event = BlinkEvent(
                        start_time=self.eye_close_start_time,
                        end_time=timestamp,
                        min_ear=self.min_ear_during_closure
                    )
                    self.blink_history.append(blink_event)
                    self.total_blinks += 1
                
                elif closure_duration >= self.microsleep_duration_min:
                    # Microsleep episode
                    microsleep_event = BlinkEvent(
                        start_time=self.eye_close_start_time,
                        end_time=timestamp,
                        min_ear=self.min_ear_during_closure
                    )
                    self.microsleep_events.append(microsleep_event)
                    self.total_microsleeps += 1
                
                # Reset state
                self.is_eye_closed = False
                self.eye_close_start_time = None
                self.min_ear_during_closure = None
        
        return blink_event
    
    def getBlinkRate(self, time_window: float = 60.0) -> float:
        """
        Calculate blink rate over a time window.
        
        Args:
            time_window: Time window in seconds (default: 60 seconds)
        
        Returns:
            Blinks per minute
        
        Validates: Requirements 2.2
        """
        if not self.blink_history:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Count blinks within time window
        recent_blinks = [b for b in self.blink_history if b.end_time >= cutoff_time]
        
        # Convert to blinks per minute
        blinks_per_minute = (len(recent_blinks) / time_window) * 60.0
        
        return blinks_per_minute
    
    def detectMicrosleep(self, time_window: float = 60.0) -> int:
        """
        Count microsleep episodes in a time window.
        
        Microsleep is characterized by:
        - Eye closure duration >= microsleep_duration_min
        - EAR below microsleep_ear_threshold
        
        Args:
            time_window: Time window in seconds (default: 60 seconds)
        
        Returns:
            Number of microsleep episodes in the time window
        
        Validates: Requirements 2.2, 2.5
        """
        if not self.microsleep_events:
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Count microsleep events within time window
        recent_microsleeps = [m for m in self.microsleep_events if m.end_time >= cutoff_time]
        
        return len(recent_microsleeps)
    
    def getEARTrend(self, time_window: float = 10.0) -> Optional[float]:
        """
        Calculate EAR trend over a time window.
        
        Negative trend indicates decreasing alertness.
        
        Args:
            time_window: Time window in seconds
        
        Returns:
            Slope of EAR trend (negative = decreasing alertness)
        
        Validates: Requirements 2.5
        """
        if len(self.ear_history) < 2:
            return None
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Get recent EAR values
        recent_ears = [(t, ear) for t, ear in self.ear_history if t >= cutoff_time]
        
        if len(recent_ears) < 2:
            return None
        
        # Calculate linear regression slope
        times = np.array([t for t, _ in recent_ears])
        ears = np.array([ear for _, ear in recent_ears])
        
        # Normalize times to start from 0
        times = times - times[0]
        
        # Calculate slope using least squares
        if len(times) > 1:
            slope = np.polyfit(times, ears, 1)[0]
            return float(slope)
        
        return None
    
    def getDrowsinessScore(self) -> float:
        """
        Calculate drowsiness score based on EAR analysis.
        
        Score ranges from 0.0 (alert) to 1.0 (severely drowsy).
        
        Returns:
            Drowsiness score
        
        Validates: Requirements 2.5
        """
        if not self.ear_history:
            return 0.0
        
        # Get recent average EAR
        recent_window = 5.0  # seconds
        current_time = time.time()
        cutoff_time = current_time - recent_window
        recent_ears = [ear for t, ear in self.ear_history if t >= cutoff_time]
        
        if not recent_ears:
            return 0.0
        
        avg_ear = np.mean(recent_ears)
        
        # Calculate base score from EAR value
        if avg_ear >= self.normal_ear_range[0]:
            ear_score = 0.0
        elif avg_ear <= self.microsleep_ear_threshold:
            ear_score = 1.0
        else:
            # Linear interpolation between thresholds
            ear_score = 1.0 - ((avg_ear - self.microsleep_ear_threshold) / 
                              (self.normal_ear_range[0] - self.microsleep_ear_threshold))
        
        # Factor in blink rate (abnormally low or high indicates drowsiness)
        blink_rate = self.getBlinkRate(time_window=60.0)
        normal_blink_rate = 15.0  # blinks per minute
        blink_deviation = abs(blink_rate - normal_blink_rate) / normal_blink_rate
        blink_score = min(blink_deviation, 1.0)
        
        # Factor in microsleep events
        microsleep_count = self.detectMicrosleep(time_window=60.0)
        microsleep_score = min(microsleep_count * 0.3, 1.0)
        
        # Factor in EAR trend
        ear_trend = self.getEARTrend(time_window=10.0)
        trend_score = 0.0
        if ear_trend is not None and ear_trend < 0:
            # Negative trend (decreasing EAR) indicates increasing drowsiness
            trend_score = min(abs(ear_trend) * 10.0, 1.0)
        
        # Weighted combination
        drowsiness_score = (
            ear_score * 0.4 +
            blink_score * 0.2 +
            microsleep_score * 0.3 +
            trend_score * 0.1
        )
        
        return min(drowsiness_score, 1.0)
    
    def updateThresholds(self, normal_ear_range: Optional[Tuple[float, float]] = None,
                        drowsy_ear_threshold: Optional[float] = None,
                        microsleep_ear_threshold: Optional[float] = None):
        """
        Update EAR thresholds for personalization.
        
        Args:
            normal_ear_range: New normal EAR range
            drowsy_ear_threshold: New drowsy threshold
            microsleep_ear_threshold: New microsleep threshold
        
        Validates: Requirements 2.5
        """
        if normal_ear_range is not None:
            self.normal_ear_range = normal_ear_range
        if drowsy_ear_threshold is not None:
            self.drowsy_ear_threshold = drowsy_ear_threshold
        if microsleep_ear_threshold is not None:
            self.microsleep_ear_threshold = microsleep_ear_threshold
    
    def reset(self):
        """Reset calculator state"""
        self.ear_history.clear()
        self.blink_history.clear()
        self.microsleep_events.clear()
        self.is_eye_closed = False
        self.eye_close_start_time = None
        self.min_ear_during_closure = None
        self.total_blinks = 0
        self.total_microsleeps = 0
    
    def getStatistics(self) -> dict:
        """
        Get comprehensive statistics about EAR analysis.
        
        Returns:
            Dictionary with statistics
        """
        if not self.ear_history:
            return {
                'total_blinks': 0,
                'total_microsleeps': 0,
                'blink_rate': 0.0,
                'avg_ear': 0.0,
                'drowsiness_score': 0.0
            }
        
        recent_ears = [ear for _, ear in self.ear_history]
        
        return {
            'total_blinks': self.total_blinks,
            'total_microsleeps': self.total_microsleeps,
            'blink_rate': self.getBlinkRate(),
            'avg_ear': float(np.mean(recent_ears)),
            'min_ear': float(np.min(recent_ears)),
            'max_ear': float(np.max(recent_ears)),
            'drowsiness_score': self.getDrowsinessScore(),
            'ear_trend': self.getEARTrend()
        }
