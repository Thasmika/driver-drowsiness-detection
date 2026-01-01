"""
Secure Data Handler Module

This module provides secure data handling with local-only processing,
encryption for temporary storage, and automatic data deletion.

Validates: Requirements 6.1, 6.2, 6.3
"""

import os
import time
import hashlib
import secrets
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend


class DataType(Enum):
    """Types of data that can be stored"""
    FRAME = "frame"
    LANDMARKS = "landmarks"
    FEATURES = "features"
    METRICS = "metrics"
    LOGS = "logs"


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration"""
    max_age_seconds: int = 300  # 5 minutes default
    auto_delete_on_process: bool = True
    encrypt_temporary_data: bool = True
    allow_cloud_transmission: bool = False  # Always False for privacy
    
    def __post_init__(self):
        """Validate policy settings"""
        if self.allow_cloud_transmission:
            raise ValueError("Cloud transmission is not allowed for privacy compliance")
        if self.max_age_seconds < 0:
            raise ValueError("Max age must be non-negative")


class DataEncryption:
    """
    Handles encryption and decryption of sensitive data using industry-standard methods.
    
    Uses Fernet (symmetric encryption) with AES-128 in CBC mode.
    """
    
    def __init__(self, password: Optional[str] = None):
        """
        Initialize encryption handler.
        
        Args:
            password: Optional password for key derivation. If None, generates random key.
        """
        if password:
            # Derive key from password using PBKDF2
            salt = b'drowsiness_detection_salt_v1'  # Fixed salt for consistency
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = kdf.derive(password.encode())
            # Fernet requires base64-encoded 32-byte key
            import base64
            self.key = base64.urlsafe_b64encode(key)
        else:
            # Generate random key
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data using Fernet encryption.
        
        Args:
            data: Raw bytes to encrypt
        
        Returns:
            Encrypted bytes
        """
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using Fernet encryption.
        
        Args:
            encrypted_data: Encrypted bytes
        
        Returns:
            Decrypted bytes
        """
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_string(self, text: str) -> str:
        """
        Encrypt string data.
        
        Args:
            text: String to encrypt
        
        Returns:
            Base64-encoded encrypted string
        """
        encrypted = self.encrypt(text.encode('utf-8'))
        import base64
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """
        Decrypt string data.
        
        Args:
            encrypted_text: Base64-encoded encrypted string
        
        Returns:
            Decrypted string
        """
        import base64
        encrypted = base64.b64decode(encrypted_text.encode('utf-8'))
        return self.decrypt(encrypted).decode('utf-8')


@dataclass
class StoredData:
    """Container for stored data with metadata"""
    data_id: str
    data_type: DataType
    timestamp: float
    encrypted: bool
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def is_expired(self, max_age_seconds: int) -> bool:
        """Check if data has exceeded retention period"""
        age = time.time() - self.timestamp
        return age > max_age_seconds


class SecureDataHandler:
    """
    Secure data handler for privacy-compliant data management.
    
    Ensures local-only processing, encryption of temporary data,
    and automatic deletion after processing.
    """
    
    def __init__(
        self,
        storage_dir: str = "./temp_data",
        retention_policy: Optional[DataRetentionPolicy] = None,
        encryption_password: Optional[str] = None
    ):
        """
        Initialize secure data handler.
        
        Args:
            storage_dir: Directory for temporary data storage
            retention_policy: Data retention policy configuration
            encryption_password: Optional password for encryption
        """
        self.storage_dir = Path(storage_dir)
        self.retention_policy = retention_policy or DataRetentionPolicy()
        
        # Initialize encryption if enabled
        self.encryption = None
        if self.retention_policy.encrypt_temporary_data:
            self.encryption = DataEncryption(encryption_password)
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Track stored data
        self.stored_data: Dict[str, StoredData] = {}
        
        # Statistics
        self.total_stored = 0
        self.total_deleted = 0
        self.total_encrypted = 0
    
    def storeTemporary(
        self,
        data: Any,
        data_type: DataType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store data temporarily with optional encryption.
        
        Args:
            data: Data to store (numpy array, dict, string, etc.)
            data_type: Type of data being stored
            metadata: Optional metadata about the data
        
        Returns:
            Unique data ID for retrieval
        
        Validates: Requirements 6.1, 6.2
        """
        # Generate unique ID
        data_id = self._generateDataId()
        timestamp = time.time()
        
        # Serialize data
        serialized = self._serializeData(data, data_type)
        
        # Encrypt if enabled
        encrypted = False
        if self.encryption:
            serialized = self.encryption.encrypt(serialized)
            encrypted = True
            self.total_encrypted += 1
        
        # Store to file
        file_path = self.storage_dir / f"{data_id}.dat"
        with open(file_path, 'wb') as f:
            f.write(serialized)
        
        # Track stored data
        stored = StoredData(
            data_id=data_id,
            data_type=data_type,
            timestamp=timestamp,
            encrypted=encrypted,
            file_path=str(file_path),
            metadata=metadata
        )
        self.stored_data[data_id] = stored
        self.total_stored += 1
        
        # Auto-delete if policy requires
        if self.retention_policy.auto_delete_on_process:
            # Schedule for deletion after processing
            pass  # Will be deleted by cleanup
        
        return data_id
    
    def retrieve(self, data_id: str) -> Optional[Any]:
        """
        Retrieve temporarily stored data.
        
        Args:
            data_id: Unique data ID
        
        Returns:
            Retrieved data, or None if not found
        """
        if data_id not in self.stored_data:
            return None
        
        stored = self.stored_data[data_id]
        
        # Check if expired
        if stored.is_expired(self.retention_policy.max_age_seconds):
            self.delete(data_id)
            return None
        
        # Read from file
        try:
            with open(stored.file_path, 'rb') as f:
                data = f.read()
            
            # Decrypt if encrypted
            if stored.encrypted and self.encryption:
                data = self.encryption.decrypt(data)
            
            # Deserialize
            return self._deserializeData(data, stored.data_type)
            
        except Exception as e:
            print(f"Error retrieving data {data_id}: {e}")
            return None
    
    def delete(self, data_id: str) -> bool:
        """
        Delete stored data immediately.
        
        Args:
            data_id: Unique data ID
        
        Returns:
            True if deleted successfully
        
        Validates: Requirements 6.3
        """
        if data_id not in self.stored_data:
            return False
        
        stored = self.stored_data[data_id]
        
        # Delete file
        try:
            if stored.file_path and os.path.exists(stored.file_path):
                os.remove(stored.file_path)
            
            # Remove from tracking
            del self.stored_data[data_id]
            self.total_deleted += 1
            return True
            
        except Exception as e:
            print(f"Error deleting data {data_id}: {e}")
            return False
    
    def deleteAll(self) -> int:
        """
        Delete all stored data.
        
        Returns:
            Number of items deleted
        
        Validates: Requirements 6.3, 6.5
        """
        count = 0
        data_ids = list(self.stored_data.keys())
        
        for data_id in data_ids:
            if self.delete(data_id):
                count += 1
        
        return count
    
    def cleanupExpired(self) -> int:
        """
        Clean up expired data based on retention policy.
        
        Returns:
            Number of items deleted
        
        Validates: Requirements 6.3
        """
        count = 0
        current_time = time.time()
        expired_ids = []
        
        for data_id, stored in self.stored_data.items():
            if stored.is_expired(self.retention_policy.max_age_seconds):
                expired_ids.append(data_id)
        
        for data_id in expired_ids:
            if self.delete(data_id):
                count += 1
        
        return count
    
    def processLocally(self, data: Any, processor_func: callable) -> Any:
        """
        Process data locally without any cloud transmission.
        
        Args:
            data: Data to process
            processor_func: Function to process the data
        
        Returns:
            Processed data
        
        Validates: Requirements 6.1
        """
        # Ensure no cloud transmission
        if self.retention_policy.allow_cloud_transmission:
            raise RuntimeError("Cloud transmission is not allowed")
        
        # Process locally
        result = processor_func(data)
        
        # Auto-delete if policy requires
        if self.retention_policy.auto_delete_on_process:
            # Data is not stored, processed in memory only
            pass
        
        return result
    
    def getStatistics(self) -> Dict[str, Any]:
        """
        Get statistics about data handling.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_stored': self.total_stored,
            'total_deleted': self.total_deleted,
            'total_encrypted': self.total_encrypted,
            'currently_stored': len(self.stored_data),
            'encryption_enabled': self.encryption is not None,
            'retention_policy': {
                'max_age_seconds': self.retention_policy.max_age_seconds,
                'auto_delete': self.retention_policy.auto_delete_on_process,
                'encrypt_temporary': self.retention_policy.encrypt_temporary_data
            }
        }
    
    def _generateDataId(self) -> str:
        """Generate unique data ID"""
        timestamp = str(time.time()).encode()
        random_bytes = secrets.token_bytes(16)
        hash_input = timestamp + random_bytes
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def _serializeData(self, data: Any, data_type: DataType) -> bytes:
        """Serialize data to bytes"""
        if data_type == DataType.FRAME:
            # Numpy array
            if isinstance(data, np.ndarray):
                return data.tobytes()
            else:
                raise ValueError("Frame data must be numpy array")
        
        elif data_type in [DataType.LANDMARKS, DataType.FEATURES, DataType.METRICS]:
            # JSON-serializable data
            json_str = json.dumps(data, default=str)
            return json_str.encode('utf-8')
        
        elif data_type == DataType.LOGS:
            # String data
            if isinstance(data, str):
                return data.encode('utf-8')
            else:
                return str(data).encode('utf-8')
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _deserializeData(self, data: bytes, data_type: DataType) -> Any:
        """Deserialize bytes to data"""
        if data_type == DataType.FRAME:
            # Would need shape information to reconstruct numpy array
            # For now, return raw bytes
            return data
        
        elif data_type in [DataType.LANDMARKS, DataType.FEATURES, DataType.METRICS]:
            # JSON data
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        
        elif data_type == DataType.LOGS:
            # String data
            return data.decode('utf-8')
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def __del__(self):
        """Cleanup on deletion"""
        # Auto-delete all data on handler destruction
        if hasattr(self, 'retention_policy') and self.retention_policy.auto_delete_on_process:
            self.deleteAll()
