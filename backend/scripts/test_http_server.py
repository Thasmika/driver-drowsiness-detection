#!/usr/bin/env python3
"""
Test HTTP server with sample requests
"""

import requests
import base64
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from io import BytesIO


def test_health():
    """Test health endpoint"""
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get('http://localhost:5000/health')
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_status():
    """Test status endpoint"""
    print("\n2. Testing /status endpoint...")
    try:
        response = requests.get('http://localhost:5000/status')
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_process_dummy():
    """Test process endpoint with dummy image"""
    print("\n3. Testing /process endpoint with dummy image...")
    try:
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(dummy_image)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Send request
        response = requests.post(
            'http://localhost:5000/process',
            json={'image_data': image_base64},
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"   Status Code: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_process_real_image():
    """Test with real image if available"""
    print("\n4. Testing /process endpoint with real image...")
    
    # Look for test image
    test_image_path = Path(__file__).parent.parent / 'datasets' / 'test_image.jpg'
    
    if not test_image_path.exists():
        print(f"   ⚠ No test image found at {test_image_path}")
        print("   Skipping real image test")
        return True
    
    try:
        # Load and encode image
        img = Image.open(test_image_path)
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Send request
        start_time = time.time()
        response = requests.post(
            'http://localhost:5000/process',
            json={'image_data': image_base64},
            headers={'Content-Type': 'application/json'}
        )
        elapsed = time.time() - start_time
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Processing Time: {elapsed:.3f}s")
        
        result = response.json()
        
        if result.get('success'):
            print(f"\n   ✓ Detection Results:")
            print(f"     Face Detected: {result.get('face_detected')}")
            print(f"     Drowsiness Score: {result.get('drowsiness_score'):.3f}")
            print(f"     Confidence: {result.get('confidence'):.3f}")
            print(f"     EAR: {result.get('ear'):.3f}")
            print(f"     MAR: {result.get('mar'):.3f}")
            print(f"     Alert Level: {result.get('alert_level')}")
            print(f"     Alert Message: {result.get('alert_message')}")
            print(f"     Model Type: {result.get('model_type')}")
        else:
            print(f"\n   ⚠ Detection failed:")
            print(f"     Face Detected: {result.get('face_detected')}")
            print(f"     Message: {result.get('message')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test processing performance"""
    print("\n5. Testing performance (10 requests)...")
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        img = Image.fromarray(test_image)
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        times = []
        for i in range(10):
            start = time.time()
            response = requests.post(
                'http://localhost:5000/process',
                json={'image_data': image_base64},
                headers={'Content-Type': 'application/json'}
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   Request {i+1}: {elapsed:.3f}s")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n   Performance Summary:")
        print(f"     Average: {avg_time:.3f}s")
        print(f"     Min: {min_time:.3f}s")
        print(f"     Max: {max_time:.3f}s")
        print(f"     FPS: {1/avg_time:.1f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("HTTP Server Test Suite")
    print("="*60)
    print("\nMake sure the server is running:")
    print("  python backend/src/http_server.py")
    print("\nWaiting for server to be ready...")
    
    # Wait for server
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:5000/health', timeout=1)
            if response.status_code == 200:
                print("✓ Server is ready!\n")
                break
        except:
            if i < max_retries - 1:
                time.sleep(1)
            else:
                print("\n❌ Server is not responding")
                print("Please start the server first:")
                print("  python backend/src/http_server.py")
                return
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Status Check", test_status()))
    results.append(("Process Dummy Image", test_process_dummy()))
    results.append(("Process Real Image", test_process_real_image()))
    results.append(("Performance Test", test_performance()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)


if __name__ == '__main__':
    main()
