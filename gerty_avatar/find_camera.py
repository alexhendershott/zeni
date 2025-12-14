#!/usr/bin/env python3
"""Quick script to find available camera indices"""
import cv2

print("Scanning for cameras...")
print("-" * 40)

available_cameras = []
for i in range(10):  # Check first 10 indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        # Try to read a frame to verify it works
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f"✓ Camera {i}: Available ({width}x{height})")
            available_cameras.append(i)
        else:
            print(f"✗ Camera {i}: Opened but can't read frames")
        cap.release()
    else:
        print(f"✗ Camera {i}: Not available")

print("-" * 40)
if available_cameras:
    print(f"\nFound {len(available_cameras)} working camera(s): {available_cameras}")
    print(f"\nYour built-in camera is likely: {available_cameras[0]}")
    if len(available_cameras) > 1:
        print(f"Your USB webcam is likely: {available_cameras[1]}")
        print(f"\nTo use USB webcam, run:")
        print(f"  VISION_CAMERA_INDEX={available_cameras[1]} python main.py")
else:
    print("\nNo cameras found!")
