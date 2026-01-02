#!/usr/bin/env python3
"""
Dataset Recording Script
Record images from webcam for training drowsiness detection model
"""

import cv2
import os
from datetime import datetime
import argparse


def record_images(output_folder, num_images=50, camera_index=0):
    """
    Record images from webcam
    
    Args:
        output_folder: Where to save images
        num_images: Number of images to record
        camera_index: Camera device index (0=default, 1=external USB)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\nRecording {num_images} images to {output_folder}")
    print("Controls:")
    print("  - Press SPACE to capture image")
    print("  - Press ESC to quit")
    print("\nReady? Position yourself and press SPACE to start capturing...\n")
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Draw UI
        display_frame = frame.copy()
        
        # Progress bar
        bar_width = 400
        bar_height = 30
        bar_x = (display_frame.shape[1] - bar_width) // 2
        bar_y = 20
        
        # Background
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress
        progress = int((count / num_images) * bar_width)
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + progress, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Text
        text = f"Images: {count}/{num_images}"
        cv2.putText(display_frame, text, (bar_x + 10, bar_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(display_frame, "Press SPACE to capture", 
                   (bar_x, bar_y + bar_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Record Dataset', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space bar
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{output_folder}/img_{count:04d}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved: {filename}")
            count += 1
        elif key == 27:  # ESC
            print("\nRecording cancelled by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Recording complete! Saved {count} images to {output_folder}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Record dataset images from webcam')
    parser.add_argument(
        'category',
        choices=['alert', 'drowsy'],
        help='Category to record (alert or drowsy)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=50,
        help='Number of images to record (default: 50)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index (0=default, 1=external USB, default: 0)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset1',
        help='Dataset folder name (default: dataset1)'
    )
    
    args = parser.parse_args()
    
    # Determine output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    output_folder = os.path.join(backend_dir, 'datasets', args.dataset, args.category)
    
    print("\n" + "="*60)
    print(f"Recording {args.category.upper()} images")
    print("="*60)
    
    if args.category == 'alert':
        print("\nInstructions for ALERT images:")
        print("  ✓ Keep your eyes OPEN")
        print("  ✓ Look at the camera")
        print("  ✓ Stay alert and focused")
        print("  ✓ Try different angles (front, slight left/right)")
        print("  ✓ Try different expressions (normal, smiling)")
        print("  ✓ Vary your distance from camera")
    else:
        print("\nInstructions for DROWSY images:")
        print("  ✓ CLOSE your eyes")
        print("  ✓ Yawn")
        print("  ✓ Look tired/sleepy")
        print("  ✓ Half-closed eyes")
        print("  ✓ Head tilted (drowsy posture)")
        print("  ✓ Try different drowsy expressions")
    
    print(f"\nCamera: {args.camera}")
    print(f"Output: {output_folder}")
    print(f"Target: {args.num_images} images")
    
    input("\nPress ENTER to start recording...")
    
    record_images(output_folder, args.num_images, args.camera)


if __name__ == "__main__":
    main()
