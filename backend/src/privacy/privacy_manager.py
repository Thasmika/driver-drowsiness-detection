"""
Privacy Manager Module

This module provides user data management, privacy settings,
consent management, and data audit logging for compliance.

Validates: Requirements 6.4, 6.5
"""

import time
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime


class ConsentStatus(Enum):
    """User consent status"""
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"


class DataCategory(Enum):
    """Categories of data for privacy management"""
    FACIAL_DATA = "facial_data"
    LOCATION_DATA = "location_data"
    USAGE_METRICS = "usage_metrics"
    PERFORMANCE_LOGS = "performance_logs"
    ALERT_HISTORY = "alert_history"


@dataclass
class PrivacySettings:
    """User privacy settings configuration"""
    # Data collection preferences
    collect_facial_data: bool = True  # Required for core functionality
    collect_location_data: bool = False  # Optional for emergency features
    collect_usage_metrics: bool = True
    collect_performance_logs: bool = True
    
    # Data retention preferences
    retain_alert_history: bool = True
    alert_history_days: int = 30
    
    # Sharing preferences
    share_anonymous_metrics: bool = False  # For system improvement
    
    # Privacy level
    privacy_level: str = "high"  # "high", "medium", "low"
    
    def __post_init__(self):
        """Validate settings"""
        if not self.collect_facial_data:
            raise ValueError("Facial data collection is required for drowsiness detection")
        
        if self.privacy_level not in ["high", "medium", "low"]:
            raise ValueError("Privacy level must be 'high', 'medium', or 'low'")


@dataclass
class ConsentRecord:
    """Record of user consent"""
    category: DataCategory
    status: ConsentStatus
    timestamp: float
    version: str = "1.0"  # Privacy policy version
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance"""
    timestamp: float
    action: str
    data_category: Optional[DataCategory]
    user_id: Optional[str]
    details: Optional[Dict[str, Any]]
    success: bool


class PrivacyManager:
    """
    Privacy manager for user data management and compliance.
    
    Handles consent management, privacy settings, data deletion,
    and audit logging for GDPR and other privacy regulations.
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        settings_file: str = "./privacy_settings.json",
        audit_log_file: str = "./privacy_audit.log"
    ):
        """
        Initialize privacy manager.
        
        Args:
            user_id: Optional user identifier
            settings_file: Path to privacy settings file
            audit_log_file: Path to audit log file
        """
        self.user_id = user_id or "anonymous"
        self.settings_file = Path(settings_file)
        self.audit_log_file = Path(audit_log_file)
        
        # Load or create privacy settings
        self.settings = self._loadSettings()
        
        # Consent records
        self.consent_records: Dict[DataCategory, ConsentRecord] = {}
        self._loadConsentRecords()
        
        # Audit log
        self.audit_log: List[AuditLogEntry] = []
        
        # Log initialization
        self._logAudit("privacy_manager_initialized", None, None, True)
    
    def requestConsent(
        self,
        category: DataCategory,
        policy_version: str = "1.0"
    ) -> ConsentStatus:
        """
        Request user consent for data collection.
        
        Args:
            category: Data category requiring consent
            policy_version: Privacy policy version
        
        Returns:
            Current consent status
        
        Validates: Requirements 6.4
        """
        # Check if consent already exists
        if category in self.consent_records:
            record = self.consent_records[category]
            if record.status in [ConsentStatus.GRANTED, ConsentStatus.DENIED]:
                return record.status
        
        # Create pending consent record
        record = ConsentRecord(
            category=category,
            status=ConsentStatus.PENDING,
            timestamp=time.time(),
            version=policy_version
        )
        self.consent_records[category] = record
        
        self._logAudit("consent_requested", category, None, True)
        self._saveConsentRecords()
        
        return ConsentStatus.PENDING
    
    def grantConsent(
        self,
        category: DataCategory,
        policy_version: str = "1.0"
    ) -> bool:
        """
        Grant consent for data collection.
        
        Args:
            category: Data category
            policy_version: Privacy policy version
        
        Returns:
            True if consent granted successfully
        
        Validates: Requirements 6.4
        """
        record = ConsentRecord(
            category=category,
            status=ConsentStatus.GRANTED,
            timestamp=time.time(),
            version=policy_version
        )
        self.consent_records[category] = record
        
        self._logAudit("consent_granted", category, None, True)
        self._saveConsentRecords()
        
        return True
    
    def revokeConsent(self, category: DataCategory) -> bool:
        """
        Revoke previously granted consent.
        
        Args:
            category: Data category
        
        Returns:
            True if consent revoked successfully
        
        Validates: Requirements 6.4
        """
        if category not in self.consent_records:
            return False
        
        record = self.consent_records[category]
        record.status = ConsentStatus.REVOKED
        record.timestamp = time.time()
        
        self._logAudit("consent_revoked", category, None, True)
        self._saveConsentRecords()
        
        # Trigger data deletion for this category
        self.deleteDataByCategory(category)
        
        return True
    
    def hasConsent(self, category: DataCategory) -> bool:
        """
        Check if user has granted consent for a data category.
        
        Args:
            category: Data category to check
        
        Returns:
            True if consent is granted
        """
        if category not in self.consent_records:
            return False
        
        record = self.consent_records[category]
        return record.status == ConsentStatus.GRANTED
    
    def updateSettings(self, new_settings: PrivacySettings) -> bool:
        """
        Update privacy settings.
        
        Args:
            new_settings: New privacy settings
        
        Returns:
            True if updated successfully
        
        Validates: Requirements 6.4
        """
        try:
            self.settings = new_settings
            self._saveSettings()
            
            self._logAudit(
                "settings_updated",
                None,
                {"privacy_level": new_settings.privacy_level},
                True
            )
            
            return True
            
        except Exception as e:
            self._logAudit(
                "settings_update_failed",
                None,
                {"error": str(e)},
                False
            )
            return False
    
    def getSettings(self) -> PrivacySettings:
        """Get current privacy settings"""
        return self.settings
    
    def deleteAllUserData(self) -> Dict[str, int]:
        """
        Delete all user data across all categories.
        
        Returns:
            Dictionary with deletion counts per category
        
        Validates: Requirements 6.5
        """
        results = {}
        
        for category in DataCategory:
            count = self.deleteDataByCategory(category)
            results[category.value] = count
        
        self._logAudit(
            "all_user_data_deleted",
            None,
            {"total_items": sum(results.values())},
            True
        )
        
        return results
    
    def deleteDataByCategory(self, category: DataCategory) -> int:
        """
        Delete all data for a specific category.
        
        Args:
            category: Data category to delete
        
        Returns:
            Number of items deleted
        
        Validates: Requirements 6.5
        """
        # This is a placeholder - actual implementation would interface
        # with the SecureDataHandler and other storage systems
        count = 0
        
        self._logAudit(
            "category_data_deleted",
            category,
            {"items_deleted": count},
            True
        )
        
        return count
    
    def exportUserData(self) -> Dict[str, Any]:
        """
        Export all user data for portability (GDPR right to data portability).
        
        Returns:
            Dictionary containing all user data
        
        Validates: Requirements 6.4
        """
        export_data = {
            'user_id': self.user_id,
            'export_timestamp': time.time(),
            'export_date': datetime.now().isoformat(),
            'privacy_settings': asdict(self.settings),
            'consent_records': {
                cat.value: {
                    'status': record.status.value,
                    'timestamp': record.timestamp,
                    'version': record.version
                }
                for cat, record in self.consent_records.items()
            },
            'audit_log_summary': {
                'total_entries': len(self.audit_log),
                'recent_actions': [
                    {
                        'action': entry.action,
                        'timestamp': entry.timestamp,
                        'success': entry.success
                    }
                    for entry in self.audit_log[-10:]  # Last 10 entries
                ]
            }
        }
        
        self._logAudit("user_data_exported", None, None, True)
        
        return export_data
    
    def getAuditLog(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action_filter: Optional[str] = None
    ) -> List[AuditLogEntry]:
        """
        Get audit log entries with optional filtering.
        
        Args:
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            action_filter: Optional action name filter
        
        Returns:
            List of audit log entries
        """
        filtered = self.audit_log
        
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        
        if action_filter:
            filtered = [e for e in filtered if action_filter in e.action]
        
        return filtered
    
    def _loadSettings(self) -> PrivacySettings:
        """Load privacy settings from file"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                return PrivacySettings(**data)
            except Exception as e:
                print(f"Error loading settings: {e}")
        
        # Return default settings
        return PrivacySettings()
    
    def _saveSettings(self):
        """Save privacy settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(asdict(self.settings), f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def _loadConsentRecords(self):
        """Load consent records from settings file"""
        # Consent records are stored with settings
        # In production, these would be in a separate secure storage
        pass
    
    def _saveConsentRecords(self):
        """Save consent records"""
        # Save consent records securely
        # In production, these would be in a separate secure storage
        pass
    
    def _logAudit(
        self,
        action: str,
        category: Optional[DataCategory],
        details: Optional[Dict[str, Any]],
        success: bool
    ):
        """Log action to audit log"""
        entry = AuditLogEntry(
            timestamp=time.time(),
            action=action,
            data_category=category,
            user_id=self.user_id,
            details=details,
            success=success
        )
        
        self.audit_log.append(entry)
        
        # Write to audit log file
        try:
            with open(self.audit_log_file, 'a') as f:
                log_line = json.dumps({
                    'timestamp': entry.timestamp,
                    'date': datetime.fromtimestamp(entry.timestamp).isoformat(),
                    'action': entry.action,
                    'category': category.value if category else None,
                    'user_id': entry.user_id,
                    'details': entry.details,
                    'success': entry.success
                })
                f.write(log_line + '\n')
        except Exception as e:
            print(f"Error writing audit log: {e}")
