#!/usr/bin/env python3
"""
Face Registration Web Application

Captures face images, detects faces with SCRFD, extracts ArcFace embeddings,
and stores them in the database for later recognition.

Usage:
    python register_person.py
    Then open http://<jetson-ip>:5000 in your browser
"""

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from pathlib import Path
import sys
import threading
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

# Configuration
CAMERA_INDEX = 0
MODELS_DIR = Path(__file__).parent / "data" / "models"
DB_PATH = Path(__file__).parent / "data" / "db" / "faces.db"
FACES_DIR = Path(__file__).parent / "data" / "faces"


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


def capture_frames():
    """Background thread to continuously capture frames."""
    global latest_frame
    
    while True:
        success, frame = camera.read()
        if success:
            with frame_lock:
                latest_frame = frame.copy()


def generate_video_feed():
    """Generate video frames for streaming."""
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        
        # Detect faces and draw boxes
        detections = detector.detect(frame, score_thresh=0.5)
        
        for det in detections:
            bbox = det['bbox'].astype(int)
            score = det['score']
            kps = det['kps']
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # Draw score
            label = f"Face: {score:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw keypoints
            for x, y in kps:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # Add instruction text
        cv2.putText(frame, "Enter name and click 'Capture Face' below", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
        <title>Face Registration</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            .controls {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                font-size: 1.1em;
            }
            input[type="text"] {
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 1em;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                padding: 15px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1.2em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            button:hover {
                background: #45a049;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            .message {
                margin-top: 20px;
                padding: 15px;
                border-radius: 8px;
                font-weight: 500;
                display: none;
            }
            .success {
                background: rgba(76, 175, 80, 0.3);
                border: 2px solid #4CAF50;
            }
            .error {
                background: rgba(244, 67, 54, 0.3);
                border: 2px solid #f44336;
            }
            .stats {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                margin-top: 30px;
            }
            .stats h2 {
                margin-top: 0;
            }
            .person-list {
                list-style: none;
                padding: 0;
            }
            .person-item {
                background: rgba(255, 255, 255, 0.05);
                padding: 10px 15px;
                margin: 5px 0;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👤 Face Registration System</h1>
            
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Camera Feed">
            </div>
            
            <div class="controls">
                <div class="form-group">
                    <label for="person_name">👤 Person Name:</label>
                    <input type="text" id="person_name" placeholder="Enter name (e.g., John Doe)" 
                           autocomplete="off">
                </div>
                
                <button onclick="captureFace()">📸 Capture Face</button>
                
                <div id="message" class="message"></div>
            </div>
            
            <div class="stats">
                <h2>📊 Registered People</h2>
                <div id="stats-content">Loading...</div>
            </div>
        </div>
        
        <script>
            function captureFace() {
                const name = document.getElementById('person_name').value.trim();
                const messageDiv = document.getElementById('message');
                
                if (!name) {
                    showMessage('Please enter a name', 'error');
                    return;
                }
                
                // Disable button during capture
                const button = event.target;
                button.disabled = true;
                button.textContent = '⏳ Capturing...';
                
                fetch('/capture', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ name: name })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage(data.message, 'success');
                        document.getElementById('person_name').value = '';
                        loadStats();
                    } else {
                        showMessage(data.message, 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error: ' + error, 'error');
                })
                .finally(() => {
                    button.disabled = false;
                    button.textContent = '📸 Capture Face';
                });
            }
            
            function showMessage(text, type) {
                const messageDiv = document.getElementById('message');
                messageDiv.textContent = text;
                messageDiv.className = 'message ' + type;
                messageDiv.style.display = 'block';
                
                setTimeout(() => {
                    messageDiv.style.display = 'none';
                }, 5000);
            }
            
            function loadStats() {
                fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    const statsDiv = document.getElementById('stats-content');
                    if (data.persons.length === 0) {
                        statsDiv.innerHTML = '<p>No people registered yet.</p>';
                    } else {
                        let html = '<ul class="person-list">';
                        data.persons.forEach(person => {
                            html += `<li class="person-item">
                                <strong>${person.name}</strong> - 
                                ${person.embedding_count} embedding(s)
                            </li>`;
                        });
                        html += '</ul>';
                        statsDiv.innerHTML = html;
                    }
                });
            }
            
            // Load stats on page load
            loadStats();
            
            // Reload stats every 5 seconds
            setInterval(loadStats, 5000);
            
            // Allow Enter key to capture
            document.getElementById('person_name').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    captureFace();
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    """Capture face and store embedding."""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'success': False, 'message': 'Name is required'})
        
        # Get current frame
        with frame_lock:
            if latest_frame is None:
                return jsonify({'success': False, 'message': 'No camera frame available'})
            frame = latest_frame.copy()
        
        # Detect faces
        detections = detector.detect(frame, score_thresh=0.5)
        
        if len(detections) == 0:
            return jsonify({'success': False, 'message': 'No face detected. Please face the camera.'})
        
        if len(detections) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one person is visible.'})
        
        # Get the face detection
        det = detections[0]
        kps = det['kps']
        
        # Align face
        try:
            aligned_face = align_face_112(frame, kps)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Face alignment failed: {str(e)}'})
        
        # Extract embedding
        embedding = embedder.embed_from_aligned(aligned_face)
        
        # Save to database
        person_id = face_store.get_or_create_person(name)
        
        # Save face image
        person_dir = FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"capture_{timestamp}.jpg"
        image_path = person_dir / image_filename
        
        cv2.imwrite(str(image_path), aligned_face)
        
        # Store embedding
        face_store.add_embedding(person_id, embedding, str(image_path))
        
        # Get embedding count
        stats = face_store.get_person_stats()
        person_stats = next((p for p in stats if p['name'] == name), None)
        count = person_stats['embedding_count'] if person_stats else 0
        
        return jsonify({
            'success': True,
            'message': f'✓ Face captured successfully for {name}! ({count} total embeddings)'
        })
        
    except Exception as e:
        print(f"Error during capture: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/stats')
def stats():
    """Get registration statistics."""
    try:
        person_stats = face_store.get_person_stats()
        return jsonify({'persons': person_stats})
    except Exception as e:
        return jsonify({'persons': [], 'error': str(e)})


def main():
    """Main function."""
    print("="*70)
    print("Face Registration System")
    print("="*70)
    
    # Initialize
    init_camera()
    init_models()
    
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
    print("🌐 Registration Server Started!")
    print("="*70)
    print(f"\nAccess the web interface:")
    print(f"  • Local:   http://localhost:5000")
    print(f"  • Network: http://{local_ip}:5000")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
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

