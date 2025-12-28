#!/usr/bin/env python3
"""
Face Recognition Web Application

Real-time face recognition using SCRFD detection and ArcFace embeddings.
Displays live video stream with recognized faces annotated.

Usage:
    python recognize_person.py
    Then open http://<jetson-ip>:5001 in your browser
"""

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
from pathlib import Path
import sys
import threading
from collections import deque
from datetime import datetime

# Add vision module to path
sys.path.insert(0, str(Path(__file__).parent))

from vision.opencv_detector import OpenCVFaceDetector
from vision.arcface_embedder import ArcFaceEmbedder
from vision.face_align import align_face_112
from vision.face_store import FaceStore


app = Flask(__name__)

# Global variables
camera = None
detector = None
embedder = None
face_store = None
latest_frame = None
frame_lock = threading.Lock()

# Recognition configuration
RECOGNITION_THRESHOLD = 0.4  # Cosine similarity threshold
SMOOTHING_WINDOW = 5  # Frames to smooth recognition results

# Track recent recognitions for smoothing
recognition_history = {}

# Configuration
CAMERA_INDEX = 0
MODELS_DIR = Path(__file__).parent / "data" / "models"
DB_PATH = Path(__file__).parent / "data" / "db" / "faces.db"


def init_camera():
    """Initialize camera."""
    global camera
    print("Initializing camera...")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    print("Camera initialized successfully!")


def init_models():
    """Initialize face detector and ArcFace embedder."""
    global detector, embedder, face_store
    
    arcface_path = MODELS_DIR / "arcface_r100_ms1mv3.onnx"
    
    if not arcface_path.exists():
        print(f"Error: ArcFace model not found at {arcface_path}")
        print("Please run: python scripts/download_models.py")
        sys.exit(1)
    
    print("\nInitializing models...")
    print("Using OpenCV face detector (Haar Cascade)")
    detector = OpenCVFaceDetector(method='haar')
    embedder = ArcFaceEmbedder(str(arcface_path))
    face_store = FaceStore(str(DB_PATH))
    print("Models initialized successfully!\n")


def smooth_recognition(face_id: int, name: str, score: float) -> str:
    """
    Smooth recognition results over multiple frames.
    
    Args:
        face_id: Unique ID for tracking this face position
        name: Recognized name
        score: Recognition confidence
    
    Returns:
        Smoothed name (most common in recent history)
    """
    if face_id not in recognition_history:
        recognition_history[face_id] = deque(maxlen=SMOOTHING_WINDOW)
    
    recognition_history[face_id].append(name)
    
    # Return most common name in recent history
    if len(recognition_history[face_id]) >= 3:
        names = list(recognition_history[face_id])
        return max(set(names), key=names.count)
    else:
        return name


def capture_frames():
    """Background thread to continuously capture frames."""
    global latest_frame
    
    while True:
        success, frame = camera.read()
        if success:
            with frame_lock:
                latest_frame = frame.copy()


def generate_video_feed():
    """Generate video frames with face recognition annotations."""
    frame_count = 0
    
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        
        frame_count += 1
        
        # Detect faces
        detections = detector.detect(frame, score_thresh=0.5)
        
        # Process each detected face
        for idx, det in enumerate(detections):
            bbox = det['bbox'].astype(int)
            score = det['score']
            kps = det['kps']
            
            # Create face ID based on position (for tracking)
            face_id = idx
            
            try:
                # Align face
                aligned_face = align_face_112(frame, kps)
                
                # Extract embedding
                embedding = embedder.embed_from_aligned(aligned_face)
                
                # Find match in database
                match = face_store.find_match(embedding, threshold=RECOGNITION_THRESHOLD)
                
                if match:
                    name, similarity = match
                    # Smooth recognition over multiple frames
                    name = smooth_recognition(face_id, name, similarity)
                    label = f"{name} ({similarity:.2f})"
                    color = (0, 255, 0)  # Green for recognized
                else:
                    label = "Unknown"
                    color = (0, 165, 255)  # Orange for unknown
                
            except Exception as e:
                print(f"Recognition error: {e}")
                label = "Error"
                color = (0, 0, 255)  # Red for error
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw keypoints
            for x, y in kps:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
        
        # Add info overlay
        info_text = f"Faces: {len(detections)} | Threshold: {RECOGNITION_THRESHOLD}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #ffffff;
                min-height: 100vh;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .video-container {
                background: #2d2d2d;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                margin-bottom: 30px;
            }
            img {
                width: 100%;
                border-radius: 10px;
                display: block;
            }
            .info-panel {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            }
            .info-row {
                display: flex;
                justify-content: space-between;
                padding: 12px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            .info-row:last-child {
                border-bottom: none;
            }
            .info-label {
                font-weight: 600;
                opacity: 0.8;
            }
            .info-value {
                font-weight: 700;
                color: #4CAF50;
            }
            .status {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                background: rgba(76, 175, 80, 0.3);
                border: 2px solid #4CAF50;
                font-weight: 600;
            }
            .legend {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            .legend-title {
                font-weight: 600;
                margin-bottom: 10px;
            }
            .legend-item {
                display: flex;
                align-items: center;
                margin: 8px 0;
            }
            .legend-color {
                width: 30px;
                height: 20px;
                border-radius: 4px;
                margin-right: 10px;
            }
            .btn-link {
                display: inline-block;
                margin-top: 20px;
                padding: 12px 24px;
                background: rgba(255, 255, 255, 0.2);
                color: white;
                text-decoration: none;
                border-radius: 8px;
                transition: all 0.3s;
            }
            .btn-link:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Face Recognition System</h1>
            
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Recognition Feed">
            </div>
            
            <div class="info-panel">
                <div class="info-row">
                    <span class="info-label">Status:</span>
                    <span class="status">● Live Recognition Active</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Registered People:</span>
                    <span class="info-value" id="person-count">Loading...</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Recognition Threshold:</span>
                    <span class="info-value">0.40</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Detection Model:</span>
                    <span class="info-value">SCRFD 2.5G</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Embedding Model:</span>
                    <span class="info-value">ArcFace ResNet</span>
                </div>
                
                <div class="legend">
                    <div class="legend-title">Legend:</div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(0, 255, 0);"></div>
                        <span>Recognized Person</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(0, 165, 255);"></div>
                        <span>Unknown Person</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgb(0, 0, 255);"></div>
                        <span>Processing Error</span>
                    </div>
                </div>
                
                <a href="http://{{ request.host.split(':')[0] }}:5000" class="btn-link">
                    ➕ Register New Person
                </a>
            </div>
        </div>
        
        <script>
            function loadStats() {
                fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('person-count').textContent = 
                        data.person_count + ' person(s)';
                })
                .catch(error => {
                    document.getElementById('person-count').textContent = 'Error';
                });
            }
            
            // Load stats on page load
            loadStats();
            
            // Reload stats every 10 seconds
            setInterval(loadStats, 10000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/video_feed')
def video_feed():
    """Video streaming route with recognition annotations."""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """Get recognition statistics."""
    try:
        person_stats = face_store.get_person_stats()
        return jsonify({
            'person_count': len(person_stats),
            'threshold': RECOGNITION_THRESHOLD,
            'persons': person_stats
        })
    except Exception as e:
        return jsonify({
            'person_count': 0,
            'threshold': RECOGNITION_THRESHOLD,
            'error': str(e)
        })


def main():
    """Main function."""
    print("="*70)
    print("Face Recognition System")
    print("="*70)
    
    # Initialize
    init_camera()
    init_models()
    
    # Check if database has people
    stats = face_store.get_person_stats()
    if len(stats) == 0:
        print("\n⚠️  WARNING: No people registered in database!")
        print("Please run register_person.py first to register faces.\n")
    else:
        print(f"\n✓ Found {len(stats)} registered person(s) in database:")
        for person in stats:
            print(f"  - {person['name']}: {person['embedding_count']} embedding(s)")
        print()
    
    # Start camera capture thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    # Get local IP
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "localhost"
    
    print("\n" + "="*70)
    print("🌐 Recognition Server Started!")
    print("="*70)
    print(f"\nAccess the web interface:")
    print(f"  • Local:   http://localhost:5001")
    print(f"  • Network: http://{local_ip}:5001")
    print(f"\nRecognition threshold: {RECOGNITION_THRESHOLD}")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Run Flask app (note: different port than registration)
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        if camera:
            camera.release()
        if face_store:
            face_store.close()
        print("Goodbye!")


if __name__ == '__main__':
    main()

