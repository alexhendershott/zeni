#!/usr/bin/env python3
"""Show camera preview to identify which index is which camera"""
import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python3 show_camera.py <camera_index>")
    print("Example: python3 show_camera.py 0")
    sys.exit(1)

camera_index = int(sys.argv[1])

print(f"Opening camera {camera_index}...")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera {camera_index}")
    sys.exit(1)

# Get camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Camera {camera_index} opened successfully!")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print("\nShowing preview... Press 'q' to quit")
print("Look at the preview window to identify which camera this is")

window_name = f"Camera {camera_index} Preview"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Can't receive frame")
        break
    
    # Add text overlay showing camera index
    cv2.putText(frame, f"Camera {camera_index}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow(window_name, frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nCamera {camera_index} closed")
