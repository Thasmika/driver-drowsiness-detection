"""
Dataset loading utilities for drowsiness detection datasets.

This module provides utilities to load and manage the 3 available datasets
for training and evaluating drowsiness detection models.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import cv2


class DatasetLoader:
    """Loads and manages drowsiness detection datasets."""
    
    def __init__(self, dataset_root: str = "datasets"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_root: Root directory containing the datasets
        """
        self.dataset_root = Path(dataset_root)
        self.datasets = []
        
    def discover_datasets(self) -> List[str]:
        """
        Discover available datasets in the dataset root directory.
        
        Returns:
            List of dataset names found
        """
        if not self.dataset_root.exists():
            return []
        
        datasets = []
        for item in self.dataset_root.iterdir():
            if item.is_dir():
                datasets.append(item.name)
        
        self.datasets = sorted(datasets)
        return self.datasets
    
    def load_dataset(
        self, 
        dataset_name: str,
        image_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load a single dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            image_size: Target size for images (width, height)
            
        Returns:
            Tuple of (images, labels, file_paths)
            - images: numpy array of shape (N, H, W, C)
            - labels: numpy array of shape (N,) with 0=alert, 1=drowsy
            - file_paths: list of original file paths
        """
        dataset_path = self.dataset_root / dataset_name
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found at {dataset_path}")
        
        images = []
        labels = []
        file_paths = []
        
        # Look for standard directory structure: alert/ and drowsy/
        alert_dir = dataset_path / "alert"
        drowsy_dir = dataset_path / "drowsy"
        
        # Load alert images (label = 0)
        if alert_dir.exists():
            for img_path in self._get_image_files(alert_dir):
                img = self._load_and_preprocess_image(img_path, image_size)
                if img is not None:
                    images.append(img)
                    labels.append(0)
                    file_paths.append(str(img_path))
        
        # Load drowsy images (label = 1)
        if drowsy_dir.exists():
            for img_path in self._get_image_files(drowsy_dir):
                img = self._load_and_preprocess_image(img_path, image_size)
                if img is not None:
                    images.append(img)
                    labels.append(1)
                    file_paths.append(str(img_path))
        
        if len(images) == 0:
            raise ValueError(f"No images found in dataset {dataset_name}")
        
        return (
            np.array(images, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            file_paths
        )
    
    def load_all_datasets(
        self,
        image_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all available datasets and combine them.
        
        Args:
            image_size: Target size for images (width, height)
            
        Returns:
            Tuple of (images, labels, file_paths) combining all datasets
        """
        datasets = self.discover_datasets()
        
        if len(datasets) == 0:
            raise ValueError(f"No datasets found in {self.dataset_root}")
        
        all_images = []
        all_labels = []
        all_paths = []
        
        for dataset_name in datasets:
            print(f"Loading dataset: {dataset_name}")
            images, labels, paths = self.load_dataset(dataset_name, image_size)
            all_images.append(images)
            all_labels.append(labels)
            all_paths.extend(paths)
            print(f"  Loaded {len(images)} images ({np.sum(labels == 0)} alert, {np.sum(labels == 1)} drowsy)")
        
        combined_images = np.concatenate(all_images, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\nTotal: {len(combined_images)} images from {len(datasets)} datasets")
        print(f"  Alert: {np.sum(combined_labels == 0)}")
        print(f"  Drowsy: {np.sum(combined_labels == 1)}")
        
        return combined_images, combined_labels, all_paths
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files from a directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _load_and_preprocess_image(
        self, 
        image_path: Path, 
        target_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image as numpy array or None if loading fails
        """
        try:
            # Load image using OpenCV
            img = cv2.imread(str(image_path))
            
            if img is None:
                print(f"Warning: Could not load image {image_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict[str, int]:
        """
        Get statistics about a dataset without loading all images.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset_path = self.dataset_root / dataset_name
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_name} not found")
        
        alert_dir = dataset_path / "alert"
        drowsy_dir = dataset_path / "drowsy"
        
        alert_count = len(self._get_image_files(alert_dir)) if alert_dir.exists() else 0
        drowsy_count = len(self._get_image_files(drowsy_dir)) if drowsy_dir.exists() else 0
        
        return {
            "name": dataset_name,
            "alert_images": alert_count,
            "drowsy_images": drowsy_count,
            "total_images": alert_count + drowsy_count
        }
