"""
Decision Logic Engine for Driver Drowsiness Detection System.

This module implements the DecisionEngine class that combines multiple drowsiness
indicators (EAR, MAR, head pose, ML predictions) into a final drowsiness assessment
with adaptive threshold adjustment based on user feedback.

Validates: Requirements 2.5, 3.5
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class AlertLevel(IntEnum):
    """Alert severity levels for drowsiness detection."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DrowsinessFactors:
    """Contributing factors to drowsiness assessment."""
    ear_contribution: float
    mar_contribution: float
    head_pose_contribution: float
    ml_model_contribution: float
    historical_pattern: float


@dataclass
class DrowsinessAssessment:
    """Complete drowsiness assessment result."""
    timestamp: float
    drowsiness_score: float
    confidence: float
    alert_level: AlertLevel
    contributing_factors: DrowsinessFactors
    recommendations: List[str]


class DecisionEngine:
    """
    Decision logic engine that combines multiple drowsiness indicators
    to make final drowsiness determinations with adaptive thresholds.
    """
    
    def __init__(
        self,
        ear_weight: float = 0.3,
        mar_weight: float = 0.2,
        head_pose_weight: float = 0.2,
        ml_weight: float = 0.3,
        low_threshold: float = 0.3,
        medium_threshold: float = 0.5,
        high_threshold: float = 0.7,
        critical_threshold: float = 0.85
    ):
        """
        Initialize the decision engine with configurable weights and thresholds.
        
        Args:
            ear_weight: Weight for Eye Aspect Ratio contribution (0-1)
            mar_weight: Weight for Mouth Aspect Ratio contribution (0-1)
            head_pose_weight: Weight for head pose contribution (0-1)
            ml_weight: Weight for ML model prediction contribution (0-1)
            low_threshold: Threshold for LOW alert level
            medium_threshold: Threshold for MEDIUM alert level
            high_threshold: Threshold for HIGH alert level
            critical_threshold: Threshold for CRITICAL alert level
        """
        # Validate weights sum to 1.0
        total_weight = ear_weight + mar_weight + head_pose_weight + ml_weight
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.ear_weight = ear_weight
        self.mar_weight = mar_weight
        self.head_pose_weight = head_pose_weight
        self.ml_weight = ml_weight
        
        # Alert thresholds
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        
        # Historical data for pattern analysis
        self.score_history: List[float] = []
        self.max_history_length = 100
        
        # Adaptive threshold adjustment parameters
        self.false_positive_count = 0
        self.true_positive_count = 0
        self.threshold_adjustment_rate = 0.05
        
    def calculate_drowsiness_score(
        self,
        ear_score: float,
        mar_score: float,
        head_pose_score: float,
        ml_confidence: float,
        timestamp: float
    ) -> DrowsinessAssessment:
        """
        Calculate weighted drowsiness score from multiple indicators.
        
        Args:
            ear_score: Eye Aspect Ratio drowsiness score (0-1, higher = more drowsy)
            mar_score: Mouth Aspect Ratio drowsiness score (0-1, higher = more drowsy)
            head_pose_score: Head pose drowsiness score (0-1, higher = more drowsy)
            ml_confidence: ML model drowsiness confidence (0-1, higher = more drowsy)
            timestamp: Current timestamp in seconds
            
        Returns:
            DrowsinessAssessment with score, confidence, and alert level
        """
        # Validate input ranges
        for score, name in [
            (ear_score, "ear_score"),
            (mar_score, "mar_score"),
            (head_pose_score, "head_pose_score"),
            (ml_confidence, "ml_confidence")
        ]:
            if not 0 <= score <= 1:
                raise ValueError(f"{name} must be in range [0, 1], got {score}")
        
        # Calculate weighted contributions
        ear_contribution = self.ear_weight * ear_score
        mar_contribution = self.mar_weight * mar_score
        head_pose_contribution = self.head_pose_weight * head_pose_score
        ml_contribution = self.ml_weight * ml_confidence
        
        # Calculate historical pattern contribution
        historical_pattern = self._calculate_historical_pattern()
        
        # Compute final drowsiness score
        drowsiness_score = (
            ear_contribution +
            mar_contribution +
            head_pose_contribution +
            ml_contribution
        )
        
        # Adjust score based on historical pattern
        drowsiness_score = self._apply_historical_adjustment(
            drowsiness_score, historical_pattern
        )
        
        # Update history
        self.score_history.append(drowsiness_score)
        if len(self.score_history) > self.max_history_length:
            self.score_history.pop(0)
        
        # Calculate confidence level
        confidence = self._calculate_confidence(
            ear_score, mar_score, head_pose_score, ml_confidence
        )
        
        # Determine alert level
        alert_level = self._determine_alert_level(drowsiness_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            alert_level, drowsiness_score
        )
        
        # Create contributing factors
        factors = DrowsinessFactors(
            ear_contribution=ear_contribution,
            mar_contribution=mar_contribution,
            head_pose_contribution=head_pose_contribution,
            ml_model_contribution=ml_contribution,
            historical_pattern=historical_pattern
        )
        
        return DrowsinessAssessment(
            timestamp=timestamp,
            drowsiness_score=drowsiness_score,
            confidence=confidence,
            alert_level=alert_level,
            contributing_factors=factors,
            recommendations=recommendations
        )
    
    def _calculate_historical_pattern(self) -> float:
        """
        Calculate historical drowsiness pattern from recent scores.
        
        Returns:
            Historical pattern score (0-1)
        """
        if len(self.score_history) < 5:
            return 0.0
        
        # Calculate trend over recent history
        recent_scores = self.score_history[-10:]
        trend = np.mean(recent_scores)
        
        # Check for increasing trend
        if len(recent_scores) >= 5:
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])
            if second_half > first_half:
                trend *= 1.1  # Amplify if trend is increasing
        
        return min(trend, 1.0)
    
    def _apply_historical_adjustment(
        self, current_score: float, historical_pattern: float
    ) -> float:
        """
        Apply historical pattern adjustment to current score.
        
        Args:
            current_score: Current drowsiness score
            historical_pattern: Historical pattern score
            
        Returns:
            Adjusted drowsiness score
        """
        # If historical pattern shows drowsiness, slightly increase current score
        adjustment = historical_pattern * 0.1
        adjusted_score = current_score + adjustment
        return min(adjusted_score, 1.0)
    
    def _calculate_confidence(
        self,
        ear_score: float,
        mar_score: float,
        head_pose_score: float,
        ml_confidence: float
    ) -> float:
        """
        Calculate confidence in drowsiness assessment based on indicator agreement.
        
        Args:
            ear_score: Eye Aspect Ratio score
            mar_score: Mouth Aspect Ratio score
            head_pose_score: Head pose score
            ml_confidence: ML model confidence
            
        Returns:
            Confidence level (0-1)
        """
        scores = [ear_score, mar_score, head_pose_score, ml_confidence]
        
        # High confidence when indicators agree
        score_variance = np.var(scores)
        
        # Lower variance = higher confidence
        # Normalize variance to confidence (inverse relationship)
        confidence = 1.0 - min(score_variance * 4, 1.0)
        
        # Boost confidence if multiple indicators are high
        high_indicators = sum(1 for s in scores if s > 0.6)
        if high_indicators >= 3:
            confidence = min(confidence * 1.2, 1.0)
        
        return confidence
    
    def _determine_alert_level(self, drowsiness_score: float) -> AlertLevel:
        """
        Determine alert level based on drowsiness score and thresholds.
        
        Args:
            drowsiness_score: Calculated drowsiness score (0-1)
            
        Returns:
            AlertLevel enum value
        """
        if drowsiness_score >= self.critical_threshold:
            return AlertLevel.CRITICAL
        elif drowsiness_score >= self.high_threshold:
            return AlertLevel.HIGH
        elif drowsiness_score >= self.medium_threshold:
            return AlertLevel.MEDIUM
        elif drowsiness_score >= self.low_threshold:
            return AlertLevel.LOW
        else:
            return AlertLevel.NONE
    
    def _generate_recommendations(
        self, alert_level: AlertLevel, drowsiness_score: float
    ) -> List[str]:
        """
        Generate recommendations based on alert level.
        
        Args:
            alert_level: Current alert level
            drowsiness_score: Current drowsiness score
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if alert_level == AlertLevel.CRITICAL:
            recommendations.append("CRITICAL: Pull over immediately and rest")
            recommendations.append("Consider calling for assistance")
        elif alert_level == AlertLevel.HIGH:
            recommendations.append("HIGH ALERT: Find a safe place to stop soon")
            recommendations.append("Take a 15-20 minute break")
        elif alert_level == AlertLevel.MEDIUM:
            recommendations.append("Drowsiness detected: Plan to stop within 10 minutes")
            recommendations.append("Open windows or adjust temperature")
        elif alert_level == AlertLevel.LOW:
            recommendations.append("Early drowsiness signs detected")
            recommendations.append("Stay alert and monitor your condition")
        
        return recommendations
    
    def update_thresholds(self, is_false_positive: bool) -> None:
        """
        Adapt thresholds based on user feedback about alert accuracy.
        
        Args:
            is_false_positive: True if user indicates alert was false positive
        """
        if is_false_positive:
            self.false_positive_count += 1
            
            # Increase thresholds to reduce false positives
            self.low_threshold = min(
                self.low_threshold + self.threshold_adjustment_rate, 0.5
            )
            self.medium_threshold = min(
                self.medium_threshold + self.threshold_adjustment_rate, 0.7
            )
            self.high_threshold = min(
                self.high_threshold + self.threshold_adjustment_rate, 0.85
            )
            self.critical_threshold = min(
                self.critical_threshold + self.threshold_adjustment_rate, 0.95
            )
        else:
            self.true_positive_count += 1
            
            # Optionally decrease thresholds if too many true positives
            # (user is frequently drowsy but not getting early warnings)
            if self.true_positive_count > 10 and self.false_positive_count < 2:
                self.low_threshold = max(
                    self.low_threshold - self.threshold_adjustment_rate * 0.5, 0.2
                )
                self.medium_threshold = max(
                    self.medium_threshold - self.threshold_adjustment_rate * 0.5, 0.4
                )
    
    def get_confidence_level(self) -> float:
        """
        Get current confidence level in drowsiness assessment system.
        
        Returns:
            Confidence level (0-1) based on historical performance
        """
        total_feedback = self.false_positive_count + self.true_positive_count
        
        if total_feedback == 0:
            return 0.5  # Neutral confidence with no feedback
        
        # Confidence based on accuracy
        accuracy = self.true_positive_count / total_feedback
        return accuracy
    
    def reset_history(self) -> None:
        """Reset historical data and feedback counters."""
        self.score_history.clear()
        self.false_positive_count = 0
        self.true_positive_count = 0
