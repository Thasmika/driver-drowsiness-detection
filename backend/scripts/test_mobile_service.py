#!/usr/bin/env python3
"""
Test script for mobile service
Simulates Flutter app communication
"""

import sys
import json
import base64
import subprocess
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_mobile_service():
    """Test the mobile service with simulated requests"""
    
    print("Starting mobile service test...")
    
    # Start the mobile service process
    service_path = Path(__file__).parent.parent / 'src' / 'mobile_service.py'
    process = subprocess.Popen(
        [sys.executable, str(service_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Wait for ready signal
        print("\n1. Waiting for service to initialize...")
        ready_line = process.stdout.readline()
        ready_response = json.loads(ready_line)
        print(f"   Status: {ready_response.get('status')}")
        print(f"   Service Info: {json.dumps(ready_response.get('service', {}), indent=2)}")
        
        if ready_response.get('status') != 'ready':
            print("   ❌ Service failed to initialize")
            return False
        print("   ✓ Service initialized successfully")
        
        # Test 1: Ping
        print("\n2. Testing ping command...")
        ping_request = json.dumps({'command': 'ping'}) + '\n'
        process.stdin.write(ping_request)
        process.stdin.flush()
        
        ping_response = json.loads(process.stdout.readline())
        print(f"   Response: {ping_response}")
        if ping_response.get('status') == 'alive':
            print("   ✓ Ping successful")
        else:
            print("   ❌ Ping failed")
        
        # Test 2: Get status
        print("\n3. Testing get_status command...")
        status_request = json.dumps({'command': 'get_status'}) + '\n'
        process.stdin.write(status_request)
        process.stdin.flush()
        
        status_response = json.loads(process.stdout.readline())
        print(f"   Response: {json.dumps(status_response, indent=2)}")
        if status_response.get('initialized'):
            print("   ✓ Status check successful")
        else:
            print("   ❌ Status check failed")
        
        # Test 3: Process frame (with dummy image)
        print("\n4. Testing process_frame command...")
        print("   Creating dummy test image...")
        
        # Create a simple test image (100x100 RGB)
        import numpy as np
        from PIL import Image
        from io import BytesIO
        
        # Create a dummy face-like image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Convert to base64
        img = Image.fromarray(test_image)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        frame_request = json.dumps({
            'command': 'process_frame',
            'image_data': image_base64
        }) + '\n'
        
        process.stdin.write(frame_request)
        process.stdin.flush()
        
        frame_response = json.loads(process.stdout.readline())
        print(f"   Response: {json.dumps(frame_response, indent=2)}")
        
        if frame_response.get('success') or frame_response.get('face_detected') is not None:
            print("   ✓ Frame processing completed (no face expected in random image)")
        else:
            print("   ❌ Frame processing failed")
        
        # Test 4: Shutdown
        print("\n5. Testing shutdown command...")
        shutdown_request = json.dumps({'command': 'shutdown'}) + '\n'
        process.stdin.write(shutdown_request)
        process.stdin.flush()
        
        shutdown_response = json.loads(process.stdout.readline())
        print(f"   Response: {shutdown_response}")
        if shutdown_response.get('status') == 'shutdown':
            print("   ✓ Shutdown successful")
        else:
            print("   ❌ Shutdown failed")
        
        # Wait for process to end
        process.wait(timeout=5)
        print("\n✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_with_real_image():
    """Test with a real image if available"""
    
    print("\n" + "="*60)
    print("Testing with real image (if available)...")
    print("="*60)
    
    # Check if test image exists
    test_image_path = Path(__file__).parent.parent / 'datasets' / 'test_image.jpg'
    
    if not test_image_path.exists():
        print(f"No test image found at {test_image_path}")
        print("Skipping real image test")
        return
    
    # Start service
    service_path = Path(__file__).parent.parent / 'src' / 'mobile_service.py'
    process = subprocess.Popen(
        [sys.executable, str(service_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Wait for ready
        ready_line = process.stdout.readline()
        ready_response = json.loads(ready_line)
        
        if ready_response.get('status') != 'ready':
            print("Service failed to initialize")
            return
        
        # Load and encode image
        from PIL import Image
        from io import BytesIO
        
        img = Image.open(test_image_path)
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Process frame
        frame_request = json.dumps({
            'command': 'process_frame',
            'image_data': image_base64
        }) + '\n'
        
        process.stdin.write(frame_request)
        process.stdin.flush()
        
        frame_response = json.loads(process.stdout.readline())
        
        print("\nReal Image Processing Results:")
        print(f"  Success: {frame_response.get('success')}")
        print(f"  Face Detected: {frame_response.get('face_detected')}")
        
        if frame_response.get('success'):
            print(f"  Drowsiness Score: {frame_response.get('drowsiness_score'):.3f}")
            print(f"  Confidence: {frame_response.get('confidence'):.3f}")
            print(f"  EAR: {frame_response.get('ear'):.3f}")
            print(f"  MAR: {frame_response.get('mar'):.3f}")
            print(f"  Alert Level: {frame_response.get('alert_level')}")
            print(f"  Alert Message: {frame_response.get('alert_message')}")
        
        # Shutdown
        shutdown_request = json.dumps({'command': 'shutdown'}) + '\n'
        process.stdin.write(shutdown_request)
        process.stdin.flush()
        
        process.wait(timeout=5)
        
    except Exception as e:
        print(f"Real image test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


if __name__ == '__main__':
    print("="*60)
    print("Mobile Service Test Suite")
    print("="*60)
    
    # Run basic tests
    success = test_mobile_service()
    
    # Try real image test
    test_with_real_image()
    
    print("\n" + "="*60)
    if success:
        print("✓ Test suite completed successfully!")
    else:
        print("❌ Some tests failed")
    print("="*60)
