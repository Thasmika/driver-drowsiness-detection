"""
Data generator utilities for efficient batch processing during training.

This module provides data generators that yield batches of data for training
and validation, with support for augmentation and shuffling.
"""

import numpy as np
from typing import Tuple, Optional, Generator
from sklearn.model_selection import train_test_split
from .data_preprocessing import DataAugmentor


class DataSplitter:
    """Handles splitting datasets into train, validation, and test sets."""
    
    @staticmethod
    def split_data(
        images: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            images: Input images of shape (N, H, W, C)
            labels: Labels of shape (N,)
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_state: Random seed for reproducibility
            stratify: Whether to maintain class distribution in splits
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        stratify_labels = labels if stratify else None
        
        # First split: separate test set
        test_size = test_ratio
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def get_split_statistics(
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Get statistics about the data splits.
        
        Args:
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
            
        Returns:
            Dictionary with split statistics
        """
        return {
            "train": {
                "total": len(y_train),
                "alert": np.sum(y_train == 0),
                "drowsy": np.sum(y_train == 1),
                "drowsy_ratio": np.mean(y_train == 1)
            },
            "validation": {
                "total": len(y_val),
                "alert": np.sum(y_val == 0),
                "drowsy": np.sum(y_val == 1),
                "drowsy_ratio": np.mean(y_val == 1)
            },
            "test": {
                "total": len(y_test),
                "alert": np.sum(y_test == 0),
                "drowsy": np.sum(y_test == 1),
                "drowsy_ratio": np.mean(y_test == 1)
            }
        }


class BatchGenerator:
    """Generates batches of data for training and validation."""
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augmentor: Optional[DataAugmentor] = None
    ):
        """
        Initialize the batch generator.
        
        Args:
            images: Input images of shape (N, H, W, C)
            labels: Labels of shape (N,)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            augmentor: Optional data augmentor for training
        """
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentor = augmentor
        self.n_samples = len(images)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.n_batches
    
    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of data.
        
        Yields:
            Tuple of (batch_images, batch_labels)
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for batch_idx in range(self.n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            
            batch_indices = self.indices[start_idx:end_idx]
            batch_images = self.images[batch_indices]
            batch_labels = self.labels[batch_indices]
            
            # Apply augmentation if provided
            if self.augmentor is not None:
                batch_images, batch_labels = self.augmentor.augment_batch(
                    batch_images, batch_labels
                )
            
            yield batch_images, batch_labels
    
    def get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a specific batch by index.
        
        Args:
            batch_idx: Index of the batch to retrieve
            
        Returns:
            Tuple of (batch_images, batch_labels)
        """
        if batch_idx >= self.n_batches:
            raise IndexError(f"Batch index {batch_idx} out of range (0-{self.n_batches-1})")
        
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_indices = self.indices[start_idx:end_idx]
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        if self.augmentor is not None:
            batch_images, batch_labels = self.augmentor.augment_batch(
                batch_images, batch_labels
            )
        
        return batch_images, batch_labels
    
    def reset(self):
        """Reset the generator and reshuffle if enabled."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_generators(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    augment_train: bool = True
) -> Tuple[BatchGenerator, BatchGenerator]:
    """
    Create training and validation batch generators.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        batch_size: Batch size for both generators
        augment_train: Whether to apply augmentation to training data
        
    Returns:
        Tuple of (train_generator, val_generator)
    """
    # Create augmentor for training if requested
    augmentor = DataAugmentor(augmentation_probability=0.5) if augment_train else None
    
    train_gen = BatchGenerator(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True,
        augmentor=augmentor
    )
    
    val_gen = BatchGenerator(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False,
        augmentor=None  # No augmentation for validation
    )
    
    return train_gen, val_gen
