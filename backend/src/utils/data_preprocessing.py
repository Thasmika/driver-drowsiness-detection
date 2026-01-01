"""
Data preprocessing and augmentation utilities for drowsiness detection.

This module provides preprocessing and augmentation pipelines to improve
model generalization and robustness.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import random


class DataPreprocessor:
    """Handles data preprocessing for drowsiness detection models."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
    
    def normalize(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize images to [0, 1] range.
        
        Args:
            images: Input images of shape (N, H, W, C)
            
        Returns:
            Normalized images
        """
        if images.max() > 1.0:
            return images.astype(np.float32) / 255.0
        return images.astype(np.float32)
    
    def standardize(self, images: np.ndarray) -> np.ndarray:
        """
        Standardize images to zero mean and unit variance.
        
        Args:
            images: Input images of shape (N, H, W, C)
            
        Returns:
            Standardized images
        """
        mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
        std = np.std(images, axis=(0, 1, 2), keepdims=True)
        return (images - mean) / (std + 1e-7)
    
    def resize_batch(self, images: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize a batch of images.
        
        Args:
            images: Input images of shape (N, H, W, C)
            target_size: Target size (width, height), uses self.target_size if None
            
        Returns:
            Resized images
        """
        if target_size is None:
            target_size = self.target_size
        
        resized = []
        for img in images:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized.append(resized_img)
        
        return np.array(resized, dtype=images.dtype)


class DataAugmentor:
    """Handles data augmentation for training robustness."""
    
    def __init__(self, augmentation_probability: float = 0.5):
        """
        Initialize the augmentor.
        
        Args:
            augmentation_probability: Probability of applying each augmentation
        """
        self.aug_prob = augmentation_probability
    
    def augment(self, image: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
        """
        Apply random augmentations to an image.
        
        Args:
            image: Input image of shape (H, W, C)
            label: Image label
            
        Returns:
            Tuple of (augmented_image, label)
        """
        img = image.copy()
        
        # Random horizontal flip
        if random.random() < self.aug_prob:
            img = self._horizontal_flip(img)
        
        # Random brightness adjustment
        if random.random() < self.aug_prob:
            img = self._adjust_brightness(img)
        
        # Random contrast adjustment
        if random.random() < self.aug_prob:
            img = self._adjust_contrast(img)
        
        # Random rotation
        if random.random() < self.aug_prob:
            img = self._rotate(img)
        
        # Random noise
        if random.random() < self.aug_prob * 0.5:  # Less frequent
            img = self._add_noise(img)
        
        return img, label
    
    def augment_batch(
        self, 
        images: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations to a batch of images.
        
        Args:
            images: Input images of shape (N, H, W, C)
            labels: Image labels of shape (N,)
            
        Returns:
            Tuple of (augmented_images, labels)
        """
        augmented_images = []
        
        for img, label in zip(images, labels):
            aug_img, _ = self.augment(img, label)
            augmented_images.append(aug_img)
        
        return np.array(augmented_images, dtype=images.dtype), labels
    
    def _horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def _adjust_brightness(self, image: np.ndarray, delta_range: float = 0.2) -> np.ndarray:
        """Adjust image brightness."""
        delta = random.uniform(-delta_range, delta_range)
        img = image + delta
        return np.clip(img, 0.0, 1.0)
    
    def _adjust_contrast(self, image: np.ndarray, factor_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Adjust image contrast."""
        factor = random.uniform(*factor_range)
        mean = np.mean(image)
        img = (image - mean) * factor + mean
        return np.clip(img, 0.0, 1.0)
    
    def _rotate(self, image: np.ndarray, angle_range: int = 15) -> np.ndarray:
        """Rotate image by a random angle."""
        angle = random.uniform(-angle_range, angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def _add_noise(self, image: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """Add random Gaussian noise to image."""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_img = image + noise
        return np.clip(noisy_img, 0.0, 1.0)
