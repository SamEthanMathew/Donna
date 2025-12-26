#!/usr/bin/env python3
"""
Person recognition script using face detection and embeddings.

Displays live camera feed with recognized faces labeled.
"""

import cv2
import numpy as np
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.face_detector import YuNetFaceDetector
from vision.face_embedder import FaceEmbedder
from vision.face_store import FaceStore


class FPS:
    """Simple FPS calculator."""
    def __init__(self):
        self.frame_times = []
        self.max_samples = 30
        
    def update(self):
        self.frame_times.append(time.time())
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
    
    def get(self):
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return len(self.frame_times) / elapsed if elapsed > 0 else 0.0


def draw_face_label(frame, face_box, name, confidence, color):
    """
    Draw labeled bounding box around face.
    
    Args:
        frame: Image frame
        face_box: Detection box [x, y, w, h, score, ...]
        name: Person's name or "Unknown"
        confidence: Match confidence (0-1)
        color: Box color (BGR)
    """
    x, y, w, h = face_box[:4].astype(int)
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Create label text
    if name == "Unknown":
        label = "Unknown"
    else:
        label = f"{name} ({confidence*100:.0f}%)"
    
    # Calculate label background size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    
    # Draw label background
    label_y = y - 10 if y - 10 > 20 else y + h + 20
    cv2.rectangle(frame, 
                  (x, label_y - text_size[1] - 8),
                  (x + text_size[0] + 8, label_y + 4),
                  color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x + 4, label_y - 4), 
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_info_panel(frame, store, fps_value):
    """Draw information panel on frame."""
    persons = store.list_persons()
    
    # Panel dimensions
    panel_width = 250
    panel_height = 100
    panel_x = frame.shape[1] - panel_width - 10
    panel_y = 10
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (100, 100, 100), 1)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    
    y_offset = panel_y + 25
    cv2.putText(frame, f"FPS: {fps_value:.1f}", (panel_x + 10, y_offset),
                font, font_scale, color, 1, cv2.LINE_AA)
    
    y_offset += 25
    cv2.putText(frame, f"Registered: {len(persons)} person(s)", 
                (panel_x + 10, y_offset), font, font_scale, color, 1, cv2.LINE_AA)
    
    y_offset += 25
    cv2.putText(frame, "Press 'q' to quit", (panel_x + 10, y_offset),
                font, font_scale, (100, 255, 100), 1, cv2.LINE_AA)


def main():
    print("=" * 60)
    print("Face Recognition System")
    print("=" * 60)
    print()
    
    # Initialize components
    print("Loading models...")
    detector = YuNetFaceDetector("data/models/yunet.onnx", score_thresh=0.7)
    embedder = FaceEmbedder("data/models/sface.onnx")
    store = FaceStore()
    
    # Check if any persons are registered
    persons = store.list_persons()
    print(f"Registered persons: {len(persons)}")
    if persons:
        for p in persons:
            print(f"  - {p['name']} ({p['embedding_count']} embeddings)")
    else:
        print("\n⚠️  No persons registered yet!")
        print("Run 'python vision/register_person.py' first to register someone.\n")
    
    # Open camera
    camera_index = 1  # Try 1 first, fall back to 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"Camera {camera_index} opened successfully!")
    print()
    print("=" * 60)
    print("Recognition running...")
    print("Press 'q' to quit")
    print("=" * 60)
    print()
    
    fps_calc = FPS()
    recognition_threshold = 0.45
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera")
            break
        
        fps_calc.update()
        
        # Detect faces
        faces = detector.detect(frame)
        
        # Process each detected face
        for face in faces:
            try:
                # Generate embedding
                embedding = embedder.embed_from_detection(frame, face)
                
                # Find match
                match = store.find_match(embedding, threshold=recognition_threshold)
                
                if match:
                    name, confidence = match
                    color = (0, 255, 0)  # Green for recognized
                    draw_face_label(frame, face, name, confidence, color)
                else:
                    color = (0, 165, 255)  # Orange for unknown
                    draw_face_label(frame, face, "Unknown", 0.0, color)
                    
            except Exception as e:
                # Draw error box
                x, y, w, h = face[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                print(f"Error processing face: {e}")
        
        # Draw info panel
        draw_info_panel(frame, store, fps_calc.get())
        
        # Display frame
        cv2.imshow("Face Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print("Recognition stopped")
    print("=" * 60)


if __name__ == "__main__":
    main()

