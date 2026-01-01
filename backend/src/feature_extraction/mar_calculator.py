"""
Mouth Aspect Ratio (MAR) Calculator Module

This module implements Mouth Aspect Ratio calculation for yawn detection
in drowsiness detection systems.

Validates: Requirements 2.3, 2.5
"""

import time
from typing import List, Tuple, Optional, Deque
from collections import deque
import numpy as np


class YawnEvent:
    """Container for yawn event data"""
    
    def __init__(self, start_time: float, end_time: float, max_mar: float):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.max_mar = max_mar
    
    def __repr__(self):
        return f"YawnEvent(duration={self.duration:.3f}s, max_mar={self.max_mar:.3f})"


class MARCalculator:
    """
    Mouth Aspect Ratio (MAR) calculator for yawn detection.
    
    MAR is calculated using mouth landmarks to detect mouth opening.
    A yawn is characterized by sustained mouth opening above threshold.
    """
    
    # Configurable thresholds for yawn detection
    DEFAULT_NORMAL_MAR_MAX = 0.5
    DEFAULT_YAWN_MAR_THRESHOLD = 0.6
    DEFAULT_YAWN_DURATION_MIN = 1.0  # seconds
    DEFAULT_YAWN_DURATION_MAX = 6.0  # seconds
    
    def __init__(
        self,
        normal_mar_max: float = DEFAULT_NORMAL_MAR_MAX,
        yawn_mar_threshold: float = DEFAULT_YAWN_MAR_THRESHOLD,
        yawn_duration_min: float = DEFAULT_YAWN_DURATION_MIN,
        yawn_duration_max: float = DEFAULT_YAWN_DURATION_MAX,
        history_window: int = 100
    ):
        """
        Initialize the MAR Calculator.
        
        Args:
            normal_mar_max: Maximum MAR value for normal mouth state
            yawn_mar_threshold: MAR threshold above which indicates yawning
            yawn_duration_min: Minimum duration to classify as yawn (seconds)
            yawn_duration_max: Maximum duration for yawn (seconds)
            history_window: Number of MAR values to keep in history
        """
        self.normal_mar_max = normal_mar_max
        self.yawn_mar_threshold = yawn_mar_threshold
        self.yawn_duration_min = yawn_duration_min
        self.yawn_duration_max = yawn_duration_max
        self.history_window = history_window
        
        # Time series data for MAR analysis
        self.mar_history: Deque[Tuple[float, float]] = deque(maxlen=history_window)  # (timestamp, mar_value)
        self.yawn_history: List[YawnEvent] = []
        
        # State tracking for yawn detection
        self.is_mouth_open = False
        self.mouth_open_start_time: Optional[float] = None
        self.max_mar_during_opening: Optional[float] = None
        
        # Performance metrics
        self.total_yawns = 0
    
    def calculateMAR(self, mouth_landmarks: List[Tuple[float, float, float]]) -> Optional[float]:
        """
        Calculate Mouth Aspect Ratio.
        
        MAR is calculated as the ratio of vertical mouth opening to horizontal width.
        Higher MAR indicates wider mouth opening.
        
        Args:
            mouth_landmarks: List of (x, y, z) coordinates for mouth landmarks
        
        Returns:
            MAR value or None if calculation fails
        
        Validates: Requirements 2.3
        """
        if not mouth_landmarks or len(mouth_landmarks) < 8:
            return None
        
        try:
            # Convert to numpy array for easier computation
            points = np.array([(x, y) for x, y, z in mouth_landmarks])
            
            # Calculate vertical distances (mouth height at different points)
            # Using multiple vertical measurements for robustness
            # Assuming landmarks are ordered around the mouth perimeter
            
            # Find top and bottom points
            y_coords = points[:, 1]
            top_indices = np.argsort(y_coords)[:len(points)//2]
            bottom_indices = np.argsort(y_coords)[len(points)//2:]
            
            top_points = points[top_indices]
            bottom_points = points[bottom_indices]
            
            # Calculate average vertical distance
            # Match top and bottom points by x-coordinate proximity
            vertical_distances = []
            for top_point in top_points:
                # Find closest bottom point
                distances = np.linalg.norm(bottom_points - top_point, axis=1)
                closest_bottom = bottom_points[np.argmin(distances)]
                vertical_dist = abs(closest_bottom[1] - top_point[1])
                vertical_distances.append(vertical_dist)
            
            avg_vertical = np.mean(vertical_distances) if vertical_distances else 0
            
            # Calculate horizontal distance (mouth width)
            x_coords = points[:, 0]
            horizontal = np.max(x_coords) - np.min(x_coords)
            
            if horizontal == 0:
                return None
            
            # Calculate MAR
            mar = avg_vertical / horizontal
            
            return float(mar)
            
        except (IndexError, ValueError, ZeroDivisionError):
            return None
    
    def detectYawn(self, mar_value: float, timestamp: Optional[float] = None) -> Optional[YawnEvent]:
        """
        Detect yawn events using MAR time series analysis.
        
        A yawn is characterized by:
        - MAR exceeding yawn_mar_threshold
        - Duration between yawn_duration_min and yawn_duration_max
        - Sustained mouth opening
        
        Args:
            mar_value: Current MAR value
            timestamp: Timestamp of the measurement (defaults to current time)
        
        Returns:
            YawnEvent if a yawn was completed, None otherwise
        
        Validates: Requirements 2.3
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add to history
        self.mar_history.append((timestamp, mar_value))
        
        yawn_event = None
        
        # Check if mouth just opened
        if not self.is_mouth_open and mar_value > self.yawn_mar_threshold:
            self.is_mouth_open = True
            self.mouth_open_start_time = timestamp
            self.max_mar_during_opening = mar_value
        
        # Track maximum MAR during opening
        elif self.is_mouth_open:
            if mar_value > self.max_mar_during_opening:
                self.max_mar_during_opening = mar_value
            
            # Check if mouth closed
            if mar_value <= self.yawn_mar_threshold:
                opening_duration = timestamp - self.mouth_open_start_time
                
                # Classify as yawn based on duration
                if self.yawn_duration_min <= opening_duration <= self.yawn_duration_max:
                    yawn_event = YawnEvent(
                        start_time=self.mouth_open_start_time,
                        end_time=timestamp,
                        max_mar=self.max_mar_during_opening
                    )
                    self.yawn_history.append(yawn_event)
                    self.total_yawns += 1
                
                # Reset state
                self.is_mouth_open = False
                self.mouth_open_start_time = None
                self.max_mar_during_opening = None
        
        return yawn_event
    
    def getYawnFrequency(self, time_window: float = 60.0) -> float:
        """
        Calculate yawn frequency over a time window.
        
        Args:
            time_window: Time window in seconds (default: 60 seconds)
        
        Returns:
            Yawns per minute
        
        Validates: Requirements 2.3, 2.5
        """
        if not self.yawn_history:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Count yawns within time window
        recent_yawns = [y for y in self.yawn_history if y.end_time >= cutoff_time]
        
        # Convert to yawns per minute
        yawns_per_minute = (len(recent_yawns) / time_window) * 60.0
        
        return yawns_per_minute
    
    def getYawnCount(self, time_window: float = 60.0) -> int:
        """
        Count yawns in a time window.
        
        Args:
            time_window: Time window in seconds (default: 60 seconds)
        
        Returns:
            Number of yawns in the time window
        
        Validates: Requirements 2.3
        """
        if not self.yawn_history:
            return 0
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Count yawns within time window
        recent_yawns = [y for y in self.yawn_history if y.end_time >= cutoff_time]
        
        return len(recent_yawns)
    
    def getMARTrend(self, time_window: float = 10.0) -> Optional[float]:
        """
        Calculate MAR trend over a time window.
        
        Positive trend indicates increasing mouth opening (potential drowsiness).
        
        Args:
            time_window: Time window in seconds
        
        Returns:
            Slope of MAR trend (positive = increasing mouth opening)
        
        Validates: Requirements 2.5
        """
        if len(self.mar_history) < 2:
            return None
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Get recent MAR values
        recent_mars = [(t, mar) for t, mar in self.mar_history if t >= cutoff_time]
        
        if len(recent_mars) < 2:
            return None
        
        # Calculate linear regression slope
        times = np.array([t for t, _ in recent_mars])
        mars = np.array([mar for _, mar in recent_mars])
        
        # Normalize times to start from 0
        times = times - times[0]
        
        # Calculate slope using least squares
        if len(times) > 1:
            slope = np.polyfit(times, mars, 1)[0]
            return float(slope)
        
        return None
    
    def getDrowsinessScore(self) -> float:
        """
        Calculate drowsiness score based on MAR analysis.
        
        Score ranges from 0.0 (alert) to 1.0 (severely drowsy).
        Higher yawn frequency indicates higher drowsiness.
        
        Returns:
            Drowsiness score
        
        Validates: Requirements 2.5
        """
        if not self.mar_history:
            return 0.0
        
        # Factor in yawn frequency (more yawns = higher drowsiness)
        yawn_frequency = self.getYawnFrequency(time_window=60.0)
        # Normal: 0-1 yawns/min, Drowsy: 2+ yawns/min
        frequency_score = min(yawn_frequency / 3.0, 1.0)
        
        # Factor in recent yawn count
        recent_yawn_count = self.getYawnCount(time_window=120.0)  # Last 2 minutes
        # 3+ yawns in 2 minutes is concerning
        count_score = min(recent_yawn_count / 5.0, 1.0)
        
        # Factor in MAR trend
        mar_trend = self.getMARTrend(time_window=10.0)
        trend_score = 0.0
        if mar_trend is not None and mar_trend > 0:
            # Positive trend (increasing MAR) could indicate fatigue
            trend_score = min(mar_trend * 5.0, 1.0)
        
        # Get recent average MAR
        recent_window = 5.0  # seconds
        current_time = time.time()
        cutoff_time = current_time - recent_window
        recent_mars = [mar for t, mar in self.mar_history if t >= cutoff_time]
        
        avg_mar_score = 0.0
        if recent_mars:
            avg_mar = np.mean(recent_mars)
            # Higher average MAR could indicate mouth hanging open (fatigue)
            if avg_mar > self.normal_mar_max:
                avg_mar_score = min((avg_mar - self.normal_mar_max) / self.normal_mar_max, 1.0)
        
        # Weighted combination
        drowsiness_score = (
            frequency_score * 0.4 +
            count_score * 0.3 +
            trend_score * 0.1 +
            avg_mar_score * 0.2
        )
        
        return min(drowsiness_score, 1.0)
    
    def detectYawnPattern(self, time_window: float = 300.0) -> str:
        """
        Analyze yawn patterns to identify drowsiness indicators.
        
        Args:
            time_window: Time window for pattern analysis (default: 5 minutes)
        
        Returns:
            Pattern description: 'none', 'occasional', 'frequent', 'severe'
        
        Validates: Requirements 2.3, 2.5
        """
        yawn_count = self.getYawnCount(time_window)
        yawn_frequency = self.getYawnFrequency(time_window)
        
        if yawn_count == 0:
            return 'none'
        elif yawn_count <= 2 and yawn_frequency < 1.0:
            return 'occasional'
        elif yawn_count <= 5 and yawn_frequency < 2.0:
            return 'frequent'
        else:
            return 'severe'
    
    def updateThresholds(self, normal_mar_max: Optional[float] = None,
                        yawn_mar_threshold: Optional[float] = None,
                        yawn_duration_min: Optional[float] = None,
                        yawn_duration_max: Optional[float] = None):
        """
        Update MAR thresholds for personalization.
        
        Args:
            normal_mar_max: New normal MAR maximum
            yawn_mar_threshold: New yawn threshold
            yawn_duration_min: New minimum yawn duration
            yawn_duration_max: New maximum yawn duration
        
        Validates: Requirements 2.5
        """
        if normal_mar_max is not None:
            self.normal_mar_max = normal_mar_max
        if yawn_mar_threshold is not None:
            self.yawn_mar_threshold = yawn_mar_threshold
        if yawn_duration_min is not None:
            self.yawn_duration_min = yawn_duration_min
        if yawn_duration_max is not None:
            self.yawn_duration_max = yawn_duration_max
    
    def reset(self):
        """Reset calculator state"""
        self.mar_history.clear()
        self.yawn_history.clear()
        self.is_mouth_open = False
        self.mouth_open_start_time = None
        self.max_mar_during_opening = None
        self.total_yawns = 0
    
    def getStatistics(self) -> dict:
        """
        Get comprehensive statistics about MAR analysis.
        
        Returns:
            Dictionary with statistics
        """
        if not self.mar_history:
            return {
                'total_yawns': 0,
                'yawn_frequency': 0.0,
                'avg_mar': 0.0,
                'drowsiness_score': 0.0,
                'yawn_pattern': 'none'
            }
        
        recent_mars = [mar for _, mar in self.mar_history]
        
        return {
            'total_yawns': self.total_yawns,
            'yawn_frequency': self.getYawnFrequency(),
            'yawn_count_1min': self.getYawnCount(60.0),
            'yawn_count_2min': self.getYawnCount(120.0),
            'avg_mar': float(np.mean(recent_mars)),
            'min_mar': float(np.min(recent_mars)),
            'max_mar': float(np.max(recent_mars)),
            'drowsiness_score': self.getDrowsinessScore(),
            'mar_trend': self.getMARTrend(),
            'yawn_pattern': self.detectYawnPattern()
        }
