"""
Script to organize 10,000 images from source datasets into the training structure.

This script will:
1. Scan all images from your source datasets
2. Randomly distribute them according to the specified structure:
   - dataset1: 5,000 images (2,500 alert + 2,500 drowsy) - High quality
   - dataset2: 3,000 images (1,500 alert + 1,500 drowsy) - Medium quality
   - dataset3: 2,000 images (1,000 alert + 1,000 drowsy) - Your recordings
3. Copy images to maintain the original structure

Usage:
    python scripts/organize_datasets.py --source-dir <path_to_your_images> --dry-run
    python scripts/organize_datasets.py --source-dir <path_to_your_images>
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse


def find_all_images(directory):
    """Find all image files in directory and subdirectories."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = {'alert': [], 'drowsy': []}
    
    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        
        # Determine category based on folder name
        folder_name = root_path.name.lower()
        if 'alert' in folder_name or 'awake' in folder_name or 'open' in folder_name or 'normal' in folder_name:
            category = 'alert'
        elif 'drowsy' in folder_name or 'sleepy' in folder_name or 'closed' in folder_name or 'tired' in folder_name:
            category = 'drowsy'
        else:
            continue
        
        # Add all image files
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                images[category].append(os.path.join(root, file))
    
    return images


def organize_datasets(source_dir, target_dir='datasets', dry_run=False):
    """
    Organize images into the specified structure.
    
    Target structure:
    - dataset1: 5,000 images (2,500 alert + 2,500 drowsy)
    - dataset2: 3,000 images (1,500 alert + 1,500 drowsy)
    - dataset3: 2,000 images (1,000 alert + 1,000 drowsy)
    """
    print("=" * 80)
    print("Dataset Organization Script")
    print("=" * 80)
    
    # Find all images
    print(f"\n1. Scanning source directory: {source_dir}")
    images = find_all_images(source_dir)
    
    alert_count = len(images['alert'])
    drowsy_count = len(images['drowsy'])
    total_count = alert_count + drowsy_count
    
    print(f"\nFound images:")
    print(f"  Alert: {alert_count}")
    print(f"  Drowsy: {drowsy_count}")
    print(f"  Total: {total_count}")
    
    # Check if we have enough images
    if alert_count < 5000:
        print(f"\n⚠️  Warning: Need 5,000 alert images, but only found {alert_count}")
        print("   Will use all available alert images")
    
    if drowsy_count < 5000:
        print(f"\n⚠️  Warning: Need 5,000 drowsy images, but only found {drowsy_count}")
        print("   Will use all available drowsy images")
    
    # Shuffle images for random distribution
    print("\n2. Shuffling images for random distribution...")
    random.shuffle(images['alert'])
    random.shuffle(images['drowsy'])
    
    # Define distribution
    distribution = {
        'dataset1': {'alert': 2500, 'drowsy': 2500},  # High quality
        'dataset2': {'alert': 1500, 'drowsy': 1500},  # Medium quality
        'dataset3': {'alert': 1000, 'drowsy': 1000},  # Your recordings
    }
    
    # Distribute images
    print("\n3. Distributing images to datasets...")
    
    alert_index = 0
    drowsy_index = 0
    
    for dataset_name, counts in distribution.items():
        dataset_path = Path(target_dir) / dataset_name
        
        print(f"\n{dataset_name}:")
        
        for category in ['alert', 'drowsy']:
            target_folder = dataset_path / category
            target_folder.mkdir(parents=True, exist_ok=True)
            
            # Get the slice of images for this dataset
            count = counts[category]
            
            if category == 'alert':
                source_images = images['alert'][alert_index:alert_index + count]
                alert_index += count
            else:
                source_images = images['drowsy'][drowsy_index:drowsy_index + count]
                drowsy_index += count
            
            actual_count = len(source_images)
            print(f"  {category}: {actual_count} images (target: {count})")
            
            # Copy images
            if not dry_run:
                for i, source_path in enumerate(source_images):
                    # Create unique filename
                    ext = Path(source_path).suffix
                    target_filename = f"{dataset_name}_{category}_{i:04d}{ext}"
                    target_path = target_folder / target_filename
                    
                    # Copy file
                    shutil.copy2(source_path, target_path)
                    
                    if (i + 1) % 500 == 0:
                        print(f"    Copied {i + 1}/{actual_count} images...")
                
                print(f"    ✓ Copied all {actual_count} images")
            else:
                print(f"    [DRY RUN] Would copy {actual_count} images")
    
    # Summary
    print("\n" + "=" * 80)
    print("Organization Complete!")
    print("=" * 80)
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN - no files were copied")
        print("   Remove --dry-run flag to actually copy files")
    else:
        print("\nDataset structure created:")
        print(f"  {target_dir}/")
        print(f"    dataset1/ (5,000 images)")
        print(f"      alert/   (2,500 images)")
        print(f"      drowsy/  (2,500 images)")
        print(f"    dataset2/ (3,000 images)")
        print(f"      alert/   (1,500 images)")
        print(f"      drowsy/  (1,500 images)")
        print(f"    dataset3/ (2,000 images)")
        print(f"      alert/   (1,000 images)")
        print(f"      drowsy/  (1,000 images)")
        print(f"\nTotal: 10,000 images")
        print(f"\nYou can now train the model:")
        print(f"  cd backend")
        print(f"  python scripts/train_cnn.py")


def verify_datasets(target_dir='datasets'):
    """Verify the organized datasets."""
    print("\n" + "=" * 80)
    print("Dataset Verification")
    print("=" * 80)
    
    total_images = 0
    
    for dataset_name in ['dataset1', 'dataset2', 'dataset3']:
        dataset_path = Path(target_dir) / dataset_name
        
        if not dataset_path.exists():
            print(f"\n❌ {dataset_name}: Not found")
            continue
        
        print(f"\n{dataset_name}:")
        
        for category in ['alert', 'drowsy']:
            category_path = dataset_path / category
            
            if not category_path.exists():
                print(f"  ❌ {category}: Not found")
                continue
            
            # Count images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            images = [f for f in category_path.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions]
            
            count = len(images)
            total_images += count
            print(f"  ✓ {category}: {count} images")
    
    print(f"\n{'=' * 80}")
    print(f"Total images: {total_images}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize 10,000 images into training dataset structure"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        help="Source directory containing your image datasets"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="datasets",
        help="Target directory for organized datasets (default: datasets)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without copying files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset organization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    if args.verify:
        verify_datasets(args.target_dir)
    elif args.source_dir:
        organize_datasets(args.source_dir, args.target_dir, args.dry_run)
    else:
        print("Error: Please provide --source-dir or use --verify")
        print("\nExamples:")
        print("  # Preview organization (dry run)")
        print("  python scripts/organize_datasets.py --source-dir C:\\MyImages --dry-run")
        print()
        print("  # Actually organize images")
        print("  python scripts/organize_datasets.py --source-dir C:\\MyImages")
        print()
        print("  # Verify existing organization")
        print("  python scripts/organize_datasets.py --verify")
