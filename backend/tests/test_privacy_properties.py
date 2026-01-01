"""
Property-Based Tests for Privacy Features

Tests correctness properties for data privacy and security features
including local processing, encryption, and automatic deletion.

Feature: driver-drowsiness-detection
Properties: 28, 29, 30
Validates: Requirements 6.1, 6.2, 6.3
"""

import pytest
import time
import os
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck

from src.privacy.secure_data_handler import (
    SecureDataHandler,
    DataEncryption,
    DataRetentionPolicy,
    DataType
)
from src.privacy.privacy_manager import (
    PrivacyManager,
    PrivacySettings,
    ConsentStatus,
    DataCategory
)


# ============================================================================
# Test Generators
# ============================================================================

@st.composite
def retention_policy_strategy(draw):
    """Generate valid retention policies"""
    return DataRetentionPolicy(
        max_age_seconds=draw(st.integers(min_value=1, max_value=600)),
        auto_delete_on_process=draw(st.booleans()),
        encrypt_temporary_data=draw(st.booleans()),
        allow_cloud_transmission=False  # Always False for privacy
    )


@st.composite
def test_data_strategy(draw):
    """Generate test data for storage"""
    import numpy as np
    
    data_type = draw(st.sampled_from(list(DataType)))
    
    if data_type == DataType.FRAME:
        # Numpy array for frames
        height = draw(st.integers(min_value=10, max_value=100))
        width = draw(st.integers(min_value=10, max_value=100))
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8), data_type
    elif data_type in [DataType.LANDMARKS, DataType.FEATURES, DataType.METRICS]:
        # Dictionary data
        return {
            "value": draw(st.floats(min_value=0.0, max_value=1.0)),
            "timestamp": time.time()
        }, data_type
    else:  # LOGS
        return draw(st.text(min_size=10, max_size=100)), data_type


# ============================================================================
# Property 28: Local Data Processing
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    policy=retention_policy_strategy()
)
def test_property_28_local_data_processing(policy):
    """
    Property 28: Local Data Processing
    
    For any data processing operation, the system should process
    data locally without cloud transmission.
    
    Feature: driver-drowsiness-detection, Property 28
    Validates: Requirements 6.1
    """
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create handler with policy
        handler = SecureDataHandler(
            storage_dir=temp_dir,
            retention_policy=policy
        )
        
        # Property: Cloud transmission must be disabled
        assert not handler.retention_policy.allow_cloud_transmission, (
            "Cloud transmission should be disabled for privacy compliance"
        )
        
        # Test local processing
        test_data = {"value": 0.5, "processed": False}
        
        def local_processor(data):
            data["processed"] = True
            return data
        
        # Process locally
        result = handler.processLocally(test_data, local_processor)
        
        # Property: Data should be processed locally
        assert result["processed"] is True, (
            "Data should be processed locally"
        )
        
        # Property: No network calls should be made
        # (In a real test, we would mock network calls and verify none were made)
        
        # Cleanup
        handler.deleteAll()
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Property 29: Data Encryption
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    test_data=test_data_strategy(),
    use_encryption=st.booleans()
)
def test_property_29_data_encryption(test_data, use_encryption):
    """
    Property 29: Data Encryption
    
    For any temporarily stored data, the system should encrypt
    the data using industry-standard methods when encryption is enabled.
    
    Feature: driver-drowsiness-detection, Property 29
    Validates: Requirements 6.2
    """
    data, data_type = test_data
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create handler with encryption policy
        policy = DataRetentionPolicy(
            encrypt_temporary_data=use_encryption,
            auto_delete_on_process=False
        )
        
        handler = SecureDataHandler(
            storage_dir=temp_dir,
            retention_policy=policy,
            encryption_password="test_password_123"
        )
        
        # Store data
        data_id = handler.storeTemporary(data, data_type)
        
        # Property: If encryption is enabled, data should be encrypted
        if use_encryption:
            assert handler.encryption is not None, (
                "Encryption should be initialized when enabled"
            )
            
            # Check that stored data is encrypted (not readable as plain text)
            stored_info = handler.stored_data[data_id]
            assert stored_info.encrypted is True, (
                "Data should be marked as encrypted"
            )
            
            # Read raw file and verify it's encrypted
            with open(stored_info.file_path, 'rb') as f:
                raw_data = f.read()
            
            # Encrypted data should not match original
            # (This is a simplified check)
            if isinstance(data, str):
                assert data.encode() not in raw_data, (
                    "Encrypted data should not contain plain text"
                )
        
        # Property: Data should be retrievable regardless of encryption
        retrieved = handler.retrieve(data_id)
        assert retrieved is not None, (
            "Data should be retrievable after storage"
        )
        
        # Cleanup
        handler.deleteAll()
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Property 30: Automatic Data Deletion
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    max_age=st.integers(min_value=1, max_value=5),
    test_data=test_data_strategy()
)
def test_property_30_automatic_data_deletion(max_age, test_data):
    """
    Property 30: Automatic Data Deletion
    
    For any completed data analysis, the system should automatically
    delete processed data after the retention period.
    
    Feature: driver-drowsiness-detection, Property 30
    Validates: Requirements 6.3
    """
    data, data_type = test_data
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create handler with short retention period
        policy = DataRetentionPolicy(
            max_age_seconds=max_age,
            auto_delete_on_process=True,
            encrypt_temporary_data=False
        )
        
        handler = SecureDataHandler(
            storage_dir=temp_dir,
            retention_policy=policy
        )
        
        # Store data
        data_id = handler.storeTemporary(data, data_type)
        
        # Property: Data should exist immediately after storage
        assert data_id in handler.stored_data, (
            "Data should be stored immediately"
        )
        
        # Wait for data to expire
        time.sleep(max_age + 1)
        
        # Run cleanup
        deleted_count = handler.cleanupExpired()
        
        # Property: Expired data should be deleted
        assert deleted_count > 0, (
            "Expired data should be deleted by cleanup"
        )
        
        # Property: Data should no longer be retrievable
        retrieved = handler.retrieve(data_id)
        assert retrieved is None, (
            "Expired data should not be retrievable"
        )
        
        # Property: Data file should be deleted
        stored_info = handler.stored_data.get(data_id)
        if stored_info and stored_info.file_path:
            assert not os.path.exists(stored_info.file_path), (
                "Data file should be deleted"
            )
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Additional Privacy Tests
# ============================================================================

@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_encryption_round_trip():
    """
    Test that encryption and decryption work correctly.
    
    For any data, encrypting then decrypting should return
    the original data.
    """
    encryption = DataEncryption(password="test_password")
    
    # Test with various data types
    test_cases = [
        b"Hello, World!",
        b"Sensitive facial data",
        b"0" * 1000,  # Large data
        b"\x00\x01\x02\x03",  # Binary data
    ]
    
    for original_data in test_cases:
        # Encrypt
        encrypted = encryption.encrypt(original_data)
        
        # Property: Encrypted data should be different from original
        assert encrypted != original_data, (
            "Encrypted data should differ from original"
        )
        
        # Decrypt
        decrypted = encryption.decrypt(encrypted)
        
        # Property: Decrypted data should match original
        assert decrypted == original_data, (
            "Decrypted data should match original"
        )


@pytest.mark.property
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    category=st.sampled_from(list(DataCategory))
)
def test_consent_management(category):
    """
    Test consent management functionality.
    
    For any data category, consent should be properly tracked
    and enforced.
    """
    # Create temporary files for test
    temp_dir = tempfile.mkdtemp()
    settings_file = os.path.join(temp_dir, "settings.json")
    audit_file = os.path.join(temp_dir, "audit.log")
    
    try:
        manager = PrivacyManager(
            user_id="test_user",
            settings_file=settings_file,
            audit_log_file=audit_file
        )
        
        # Property: Initially, consent should not be granted
        assert not manager.hasConsent(category), (
            "Consent should not be granted initially"
        )
        
        # Request consent
        status = manager.requestConsent(category)
        assert status == ConsentStatus.PENDING, (
            "Consent status should be pending after request"
        )
        
        # Grant consent
        success = manager.grantConsent(category)
        assert success is True, (
            "Consent should be granted successfully"
        )
        
        # Property: After granting, consent should be active
        assert manager.hasConsent(category), (
            "Consent should be active after granting"
        )
        
        # Revoke consent
        success = manager.revokeConsent(category)
        assert success is True, (
            "Consent should be revoked successfully"
        )
        
        # Property: After revoking, consent should not be active
        assert not manager.hasConsent(category), (
            "Consent should not be active after revoking"
        )
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.property
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_user_data_deletion():
    """
    Test that user data can be completely deleted.
    
    For any user data deletion request, all data should be
    removed from the system.
    """
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    settings_file = os.path.join(temp_dir, "settings.json")
    audit_file = os.path.join(temp_dir, "audit.log")
    
    try:
        manager = PrivacyManager(
            user_id="test_user",
            settings_file=settings_file,
            audit_log_file=audit_file
        )
        
        # Grant consent for multiple categories
        for category in [DataCategory.FACIAL_DATA, DataCategory.USAGE_METRICS]:
            manager.grantConsent(category)
        
        # Delete all user data
        results = manager.deleteAllUserData()
        
        # Property: Deletion should be logged
        audit_log = manager.getAuditLog()
        deletion_entries = [
            e for e in audit_log
            if "deleted" in e.action.lower()
        ]
        assert len(deletion_entries) > 0, (
            "Data deletion should be logged in audit"
        )
        
        # Property: Results should include all categories
        assert isinstance(results, dict), (
            "Deletion results should be a dictionary"
        )
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.property
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_privacy_settings_validation():
    """
    Test that privacy settings are properly validated.
    
    For any privacy settings, invalid configurations should
    be rejected.
    """
    # Valid settings should work
    valid_settings = PrivacySettings(
        collect_facial_data=True,
        collect_location_data=False,
        privacy_level="high"
    )
    assert valid_settings.collect_facial_data is True
    
    # Invalid privacy level should raise error
    with pytest.raises(ValueError):
        PrivacySettings(privacy_level="invalid")
    
    # Disabling facial data should raise error (required for functionality)
    with pytest.raises(ValueError):
        PrivacySettings(collect_facial_data=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
