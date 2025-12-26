#!/usr/bin/env python3
"""
Person registration script for face recognition system.

Captures multiple face images and stores embeddings in database.
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.face_detector import YuNetFaceDetector
from vision.face_embedder import FaceEmbedder
from vision.face_store import FaceStore


def draw_instructions(frame, text, color=(255, 255, 255)):
    """Draw instruction text on frame."""
    # Semi-transparent background
    overlay = frame.copy()
    height = 60
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Text
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)


def draw_face_with_status(frame, face_box, status_text, color):
    """Draw face bounding box with status."""
    x, y, w, h = face_box[:4].astype(int)
    score = float(face_box[4])
    
    # Draw box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw status text above box
    text = f"{status_text} ({score:.2f})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x, y - 30), (x + text_size[0] + 10, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    print("=" * 60)
    print("Face Registration System")
    print("=" * 60)
    print()
    
    # Get person's name
    name = input("Enter person's name: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return
    
    # Check if name already exists
    store = FaceStore()
    existing = store.get_person(name)
    if existing:
        print(f"\nWarning: '{name}' is already registered with {existing['embedding_count']} embeddings.")
        response = input("Do you want to add more captures? (y/n): ").strip().lower()
        if response != 'y':
            print("Registration cancelled.")
            return
        # If yes, we'll add to existing person (handled by updating logic)
        print("Note: This will replace the existing registration.")
        store.delete_person(name)
    
    print()
    print("Initializing camera and models...")
    
    # Initialize components
    detector = YuNetFaceDetector("data/models/yunet.onnx")
    embedder = FaceEmbedder("data/models/sface.onnx")
    
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
    print("Instructions:")
    print("  - Position your face in the camera view")
    print("  - Press SPACE to capture (aim for 5-10 captures)")
    print("  - Vary your angle and expression slightly")
    print("  - Press 'q' or ESC to finish")
    print("=" * 60)
    print()
    
    captures = []
    embeddings = []
    target_captures = 10
    min_captures = 5
    
    # Create directory for saving face images
    face_dir = f"data/faces/{name}"
    os.makedirs(face_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera")
            break
        
        # Detect faces
        faces = detector.detect(frame)
        
        # Draw captures counter
        counter_text = f"Captures: {len(captures)}/{target_captures}"
        cv2.putText(frame, counter_text, (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        if len(faces) == 0:
            # No face detected
            draw_instructions(frame, "No face detected - please position yourself in frame", (0, 0, 255))
        elif len(faces) > 1:
            # Multiple faces
            draw_instructions(frame, "Multiple faces detected - only one person at a time", (0, 165, 255))
            for face in faces:
                draw_face_with_status(frame, face, "Multiple", (0, 165, 255))
        else:
            # Single face detected
            face = faces[0]
            x, y, w, h = face[:4].astype(int)
            
            # Check face size (should be reasonable size)
            frame_area = frame.shape[0] * frame.shape[1]
            face_area = w * h
            face_ratio = face_area / frame_area
            
            if face_ratio < 0.02:  # Too small
                status = "Face too small - move closer"
                color = (0, 165, 255)
            elif face_ratio > 0.5:  # Too large
                status = "Face too close - move back"
                color = (0, 165, 255)
            else:
                status = "Good! Press SPACE to capture"
                color = (0, 255, 0)
            
            draw_face_with_status(frame, face, status, color)
            
            if len(captures) < target_captures:
                draw_instructions(frame, f"Press SPACE to capture ({len(captures)}/{target_captures})", (255, 255, 255))
            else:
                draw_instructions(frame, "Press 'q' or ESC to finish registration", (0, 255, 0))
        
        cv2.imshow("Face Registration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            if len(faces) == 1:
                face = faces[0]
                x, y, w, h = face[:4].astype(int)
                
                # Check if face size is acceptable
                frame_area = frame.shape[0] * frame.shape[1]
                face_area = w * h
                face_ratio = face_area / frame_area
                
                if 0.02 <= face_ratio <= 0.5:
                    try:
                        # Generate embedding
                        embedding = embedder.embed_from_detection(frame, face)
                        embeddings.append(embedding)
                        
                        # Save face image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        face_filename = f"{face_dir}/capture_{len(captures)+1}_{timestamp}.jpg"
                        
                        # Crop and save face
                        margin = 20
                        y1 = max(0, y - margin)
                        y2 = min(frame.shape[0], y + h + margin)
                        x1 = max(0, x - margin)
                        x2 = min(frame.shape[1], x + w + margin)
                        face_crop = frame[y1:y2, x1:x2]
                        cv2.imwrite(face_filename, face_crop)
                        
                        captures.append({
                            'frame': frame.copy(),
                            'face_box': face,
                            'embedding': embedding,
                            'filename': face_filename
                        })
                        
                        print(f"✓ Captured {len(captures)}/{target_captures}")
                        
                    except Exception as e:
                        print(f"✗ Error capturing face: {e}")
                else:
                    print("✗ Face size not acceptable (too small or too large)")
            else:
                print("✗ Please ensure exactly one face is visible")
        
        elif key == ord('q') or key == 27:  # 'q' or ESC
            if len(captures) >= min_captures:
                break
            else:
                print(f"✗ Need at least {min_captures} captures (currently have {len(captures)})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save to database
    if len(embeddings) >= min_captures:
        print()
        print(f"Saving {len(embeddings)} embeddings to database...")
        
        try:
            person_id = store.register_person(name, embeddings)
            print(f"✓ Successfully registered '{name}' (ID: {person_id})")
            print(f"✓ Saved {len(embeddings)} face embeddings")
            print(f"✓ Face images saved to: {face_dir}")
            print()
            print("=" * 60)
            print("Registration complete!")
            print("=" * 60)
            
        except Exception as e:
            print(f"✗ Error saving to database: {e}")
            import traceback
            traceback.print_exc()
    else:
        print()
        print(f"✗ Registration cancelled (insufficient captures: {len(captures)}/{min_captures})")


if __name__ == "__main__":
    main()

