"""
Privacy and Security Module

This module provides data privacy and security features including
local-only processing, encryption, and automatic data deletion.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""

from .secure_data_handler import SecureDataHandler, DataEncryption, DataRetentionPolicy
from .privacy_manager import PrivacyManager, PrivacySettings, ConsentStatus

__all__ = [
    'SecureDataHandler',
    'DataEncryption',
    'DataRetentionPolicy',
    'PrivacyManager',
    'PrivacySettings',
    'ConsentStatus'
]
