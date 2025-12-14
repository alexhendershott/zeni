#!/usr/bin/env python3
"""Test face detection on camera to debug why Zeni isn't seeing you"""
import cv2
import mediapipe as mp

camera_index = 0
print(f"Testing face detection on camera {camera_index}...")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera {camera_index}")
    exit(1)

print("Camera opened successfully!")
print("Press 'q' to quit")
print("\nLook for GREEN boxes around detected faces")
print("If you don't see boxes, try:")
print("  - Better lighting")
print("  - Moving closer to camera")
print("  - Facing camera directly")

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Can't receive frame")
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        # Draw detection results
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
            cv2.putText(frame, f"FACE DETECTED ({len(results.detections)} faces)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO FACE DETECTED", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Face Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Test complete")
