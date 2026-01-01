"""
Alert Manager for Driver Drowsiness Detection System.

This module implements the AlertManager class that handles multiple alert types
(visual, audio, haptic) with progressive escalation and customizable sensitivity.

Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
from .decision_engine import AlertLevel


class AlertType(Enum):
    """Types of alerts that can be triggered."""
    VISUAL = "visual"
    AUDIO = "audio"
    HAPTIC = "haptic"


@dataclass
class AlertConfiguration:
    """Configuration for alert behavior."""
    enabled_alert_types: List[AlertType]
    sensitivity: float  # 0.0 (low) to 1.0 (high)
    audio_volume: float  # 0.0 to 1.0
    haptic_intensity: float  # 0.0 to 1.0
    escalation_enabled: bool
    escalation_interval: float  # seconds between escalations


@dataclass
class AlertEvent:
    """Record of an alert event."""
    timestamp: float
    alert_level: AlertLevel
    alert_types: List[AlertType]
    drowsiness_score: float
    user_responded: bool
    response_time: Optional[float]


class AlertManager:
    """
    Manages driver notifications and alert escalation for drowsiness detection.
    
    Supports multiple alert types (visual, audio, haptic) with progressive
    escalation and customizable user preferences.
    """
    
    def __init__(
        self,
        configuration: Optional[AlertConfiguration] = None,
        visual_callback: Optional[Callable[[AlertLevel, str], None]] = None,
        audio_callback: Optional[Callable[[AlertLevel, float], None]] = None,
        haptic_callback: Optional[Callable[[AlertLevel, float], None]] = None
    ):
        """
        Initialize the alert manager with configuration and callbacks.
        
        Args:
            configuration: Alert configuration settings
            visual_callback: Callback for visual alerts (alert_level, message)
            audio_callback: Callback for audio alerts (alert_level, volume)
            haptic_callback: Callback for haptic alerts (alert_level, intensity)
        """
        # Default configuration
        if configuration is None:
            configuration = AlertConfiguration(
                enabled_alert_types=[AlertType.VISUAL, AlertType.AUDIO, AlertType.HAPTIC],
                sensitivity=0.7,
                audio_volume=0.8,
                haptic_intensity=0.7,
                escalation_enabled=True,
                escalation_interval=5.0
            )
        
        self.configuration = configuration
        self.visual_callback = visual_callback
        self.audio_callback = audio_callback
        self.haptic_callback = haptic_callback
        
        # Alert state tracking
        self.current_alert_level = AlertLevel.NONE
        self.last_alert_time: Optional[float] = None
        self.escalation_count = 0
        self.alert_history: List[AlertEvent] = []
        self.max_history_length = 100
        
        # False positive tracking for adaptation
        self.recent_false_positives = 0
        self.recent_true_positives = 0
        self.adaptation_window = 20
    
    def trigger_alert(
        self,
        alert_level: AlertLevel,
        drowsiness_score: float,
        recommendations: Optional[List[str]] = None
    ) -> bool:
        """
        Trigger an alert based on drowsiness level.
        
        Args:
            alert_level: Severity level of the alert
            drowsiness_score: Current drowsiness score (0-1)
            recommendations: Optional list of recommendations to display
            
        Returns:
            True if alert was triggered, False if suppressed
        """
        current_time = time.time()
        
        # Check if we should suppress alert based on sensitivity
        if not self._should_trigger_alert(alert_level, drowsiness_score):
            return False
        
        # Determine which alert types to use
        alert_types = self._select_alert_types(alert_level)
        
        # Apply escalation if needed
        if self._should_escalate(alert_level, current_time):
            alert_level = self._escalate_alert_level(alert_level)
            self.escalation_count += 1
        else:
            self.escalation_count = 0
        
        # Trigger each enabled alert type
        triggered_types = []
        
        if AlertType.VISUAL in alert_types and self.visual_callback:
            message = self._generate_alert_message(alert_level, recommendations)
            self.visual_callback(alert_level, message)
            triggered_types.append(AlertType.VISUAL)
        
        if AlertType.AUDIO in alert_types and self.audio_callback:
            volume = self._calculate_audio_volume(alert_level)
            self.audio_callback(alert_level, volume)
            triggered_types.append(AlertType.AUDIO)
        
        if AlertType.HAPTIC in alert_types and self.haptic_callback:
            intensity = self._calculate_haptic_intensity(alert_level)
            self.haptic_callback(alert_level, intensity)
            triggered_types.append(AlertType.HAPTIC)
        
        # Record alert event
        alert_event = AlertEvent(
            timestamp=current_time,
            alert_level=alert_level,
            alert_types=triggered_types,
            drowsiness_score=drowsiness_score,
            user_responded=False,
            response_time=None
        )
        self.alert_history.append(alert_event)
        if len(self.alert_history) > self.max_history_length:
            self.alert_history.pop(0)
        
        # Update state
        self.current_alert_level = alert_level
        self.last_alert_time = current_time
        
        return True
    
    def _should_trigger_alert(
        self, alert_level: AlertLevel, drowsiness_score: float
    ) -> bool:
        """
        Determine if alert should be triggered based on sensitivity settings.
        
        Args:
            alert_level: Proposed alert level
            drowsiness_score: Current drowsiness score
            
        Returns:
            True if alert should be triggered
        """
        if alert_level == AlertLevel.NONE:
            return False
        
        # Critical and high alerts always trigger (safety critical)
        if alert_level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
            return True
        
        # For medium and low alerts, apply sensitivity adjustment
        # Higher sensitivity = lower threshold for triggering
        sensitivity_adjustment = (1.0 - self.configuration.sensitivity) * 0.15
        
        # Check if score exceeds adjusted threshold
        base_thresholds = {
            AlertLevel.LOW: 0.3,
            AlertLevel.MEDIUM: 0.5
        }
        
        threshold = base_thresholds.get(alert_level, 0.5) + sensitivity_adjustment
        return drowsiness_score >= threshold
    
    def _select_alert_types(self, alert_level: AlertLevel) -> List[AlertType]:
        """
        Select which alert types to use based on alert level.
        
        Args:
            alert_level: Current alert level
            
        Returns:
            List of alert types to trigger
        """
        enabled = self.configuration.enabled_alert_types
        
        # For low alerts, use only visual
        if alert_level == AlertLevel.LOW:
            return [t for t in enabled if t == AlertType.VISUAL]
        
        # For medium alerts, use visual and haptic
        elif alert_level == AlertLevel.MEDIUM:
            return [t for t in enabled if t in [AlertType.VISUAL, AlertType.HAPTIC]]
        
        # For high and critical alerts, use all enabled types
        else:
            return enabled
    
    def _should_escalate(
        self, alert_level: AlertLevel, current_time: float
    ) -> bool:
        """
        Determine if alert should be escalated.
        
        Args:
            alert_level: Current alert level
            current_time: Current timestamp
            
        Returns:
            True if escalation should occur
        """
        if not self.configuration.escalation_enabled:
            return False
        
        if self.last_alert_time is None:
            return False
        
        # Check if enough time has passed for escalation
        time_since_last = current_time - self.last_alert_time
        if time_since_last < self.configuration.escalation_interval:
            return False
        
        # Don't escalate if already at critical
        if alert_level == AlertLevel.CRITICAL:
            return False
        
        # Escalate if drowsiness persists
        return True
    
    def _escalate_alert_level(self, current_level: AlertLevel) -> AlertLevel:
        """
        Escalate alert to next level.
        
        Args:
            current_level: Current alert level
            
        Returns:
            Escalated alert level
        """
        escalation_map = {
            AlertLevel.LOW: AlertLevel.MEDIUM,
            AlertLevel.MEDIUM: AlertLevel.HIGH,
            AlertLevel.HIGH: AlertLevel.CRITICAL,
            AlertLevel.CRITICAL: AlertLevel.CRITICAL
        }
        return escalation_map.get(current_level, current_level)
    
    def _generate_alert_message(
        self, alert_level: AlertLevel, recommendations: Optional[List[str]]
    ) -> str:
        """
        Generate alert message text.
        
        Args:
            alert_level: Alert severity level
            recommendations: Optional recommendations
            
        Returns:
            Alert message string
        """
        messages = {
            AlertLevel.LOW: "Drowsiness detected - Stay alert",
            AlertLevel.MEDIUM: "DROWSINESS WARNING - Consider taking a break",
            AlertLevel.HIGH: "HIGH DROWSINESS - Find a safe place to stop",
            AlertLevel.CRITICAL: "CRITICAL ALERT - Pull over immediately!"
        }
        
        message = messages.get(alert_level, "Alert")
        
        if recommendations:
            message += "\n" + "\n".join(recommendations)
        
        return message
    
    def _calculate_audio_volume(self, alert_level: AlertLevel) -> float:
        """
        Calculate audio volume based on alert level.
        
        Args:
            alert_level: Alert severity level
            
        Returns:
            Volume level (0-1)
        """
        base_volume = self.configuration.audio_volume
        
        # Increase volume for higher alert levels
        level_multipliers = {
            AlertLevel.LOW: 0.6,
            AlertLevel.MEDIUM: 0.8,
            AlertLevel.HIGH: 1.0,
            AlertLevel.CRITICAL: 1.0
        }
        
        multiplier = level_multipliers.get(alert_level, 0.8)
        
        # Apply escalation boost
        escalation_boost = min(self.escalation_count * 0.1, 0.3)
        
        return min(base_volume * multiplier + escalation_boost, 1.0)
    
    def _calculate_haptic_intensity(self, alert_level: AlertLevel) -> float:
        """
        Calculate haptic feedback intensity based on alert level.
        
        Args:
            alert_level: Alert severity level
            
        Returns:
            Intensity level (0-1)
        """
        base_intensity = self.configuration.haptic_intensity
        
        # Increase intensity for higher alert levels
        level_multipliers = {
            AlertLevel.LOW: 0.5,
            AlertLevel.MEDIUM: 0.7,
            AlertLevel.HIGH: 0.9,
            AlertLevel.CRITICAL: 1.0
        }
        
        multiplier = level_multipliers.get(alert_level, 0.7)
        
        # Apply escalation boost
        escalation_boost = min(self.escalation_count * 0.1, 0.2)
        
        return min(base_intensity * multiplier + escalation_boost, 1.0)
    
    def customize_alerts(
        self,
        enabled_types: Optional[List[AlertType]] = None,
        sensitivity: Optional[float] = None,
        audio_volume: Optional[float] = None,
        haptic_intensity: Optional[float] = None,
        escalation_enabled: Optional[bool] = None
    ) -> None:
        """
        Customize alert settings based on user preferences.
        
        Args:
            enabled_types: List of enabled alert types
            sensitivity: Alert sensitivity (0-1)
            audio_volume: Audio volume (0-1)
            haptic_intensity: Haptic intensity (0-1)
            escalation_enabled: Whether to enable progressive escalation
        """
        if enabled_types is not None:
            self.configuration.enabled_alert_types = enabled_types
        
        if sensitivity is not None:
            if not 0 <= sensitivity <= 1:
                raise ValueError("Sensitivity must be in range [0, 1]")
            self.configuration.sensitivity = sensitivity
        
        if audio_volume is not None:
            if not 0 <= audio_volume <= 1:
                raise ValueError("Audio volume must be in range [0, 1]")
            self.configuration.audio_volume = audio_volume
        
        if haptic_intensity is not None:
            if not 0 <= haptic_intensity <= 1:
                raise ValueError("Haptic intensity must be in range [0, 1]")
            self.configuration.haptic_intensity = haptic_intensity
        
        if escalation_enabled is not None:
            self.configuration.escalation_enabled = escalation_enabled
    
    def log_alert_response(
        self, is_false_positive: bool, response_time: Optional[float] = None
    ) -> None:
        """
        Log user response to alert for system adaptation.
        
        Args:
            is_false_positive: True if user indicates alert was false positive
            response_time: Time taken for user to respond (seconds)
        """
        if not self.alert_history:
            return
        
        # Update most recent alert event
        last_alert = self.alert_history[-1]
        last_alert.user_responded = True
        last_alert.response_time = response_time
        
        # Track false positives for adaptation
        if is_false_positive:
            self.recent_false_positives += 1
        else:
            self.recent_true_positives += 1
        
        # Adapt sensitivity if too many false positives
        total_recent = self.recent_false_positives + self.recent_true_positives
        if total_recent >= self.adaptation_window:
            false_positive_rate = self.recent_false_positives / total_recent
            
            if false_positive_rate > 0.3:  # More than 30% false positives
                # Decrease sensitivity (increase threshold)
                self.configuration.sensitivity = max(
                    self.configuration.sensitivity - 0.1, 0.3
                )
            elif false_positive_rate < 0.1:  # Less than 10% false positives
                # Increase sensitivity (decrease threshold)
                self.configuration.sensitivity = min(
                    self.configuration.sensitivity + 0.05, 1.0
                )
            
            # Reset counters
            self.recent_false_positives = 0
            self.recent_true_positives = 0
    
    def get_alert_statistics(self) -> Dict[str, any]:
        """
        Get statistics about alert history.
        
        Returns:
            Dictionary with alert statistics
        """
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "alerts_by_level": {},
                "average_response_time": None,
                "false_positive_rate": None
            }
        
        # Count alerts by level
        alerts_by_level = {}
        for event in self.alert_history:
            level_name = event.alert_level.name
            alerts_by_level[level_name] = alerts_by_level.get(level_name, 0) + 1
        
        # Calculate average response time
        response_times = [
            e.response_time for e in self.alert_history
            if e.response_time is not None
        ]
        avg_response_time = (
            sum(response_times) / len(response_times)
            if response_times else None
        )
        
        # Calculate false positive rate
        responded_alerts = [e for e in self.alert_history if e.user_responded]
        if responded_alerts:
            false_positives = sum(
                1 for e in responded_alerts
                if e.response_time is not None and e.response_time < 1.0
            )
            false_positive_rate = false_positives / len(responded_alerts)
        else:
            false_positive_rate = None
        
        return {
            "total_alerts": len(self.alert_history),
            "alerts_by_level": alerts_by_level,
            "average_response_time": avg_response_time,
            "false_positive_rate": false_positive_rate,
            "current_sensitivity": self.configuration.sensitivity
        }
    
    def reset_state(self) -> None:
        """Reset alert manager state."""
        self.current_alert_level = AlertLevel.NONE
        self.last_alert_time = None
        self.escalation_count = 0
        self.alert_history.clear()
        self.recent_false_positives = 0
        self.recent_true_positives = 0
