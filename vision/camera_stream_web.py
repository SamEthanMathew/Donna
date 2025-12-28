#!/usr/bin/env python3
"""
USB Camera Web Streaming Program
Streams live video from USB webcam over HTTP (MJPEG format)
Perfect for viewing camera feed over SSH/network connection
"""

import cv2
from flask import Flask, Response, render_template_string
import sys


app = Flask(__name__)

# Global camera object
camera = None


def init_camera():
    """Initialize the camera."""
    global camera
    print("Initializing camera...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        print("Make sure your USB camera is connected and not being used by another application.")
        sys.exit(1)
    
    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera initialized successfully!")


def generate_frames():
    """Generator function to capture and encode frames."""
    global camera
    
    while True:
        success, frame = camera.read()
        
        if not success:
            print("Error: Failed to grab frame.")
            break
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        if not ret:
            continue
        
        # Convert to bytes
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Video streaming home page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>USB Camera Feed</title>
        <style>
            body {
                margin: 0;
                padding: 20px;
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
            }
            h1 {
                margin-bottom: 20px;
            }
            .container {
                text-align: center;
            }
            img {
                max-width: 90%;
                height: auto;
                border: 3px solid #4CAF50;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }
            .info {
                margin-top: 20px;
                padding: 15px;
                background-color: #2d2d2d;
                border-radius: 5px;
                max-width: 600px;
            }
            .status {
                color: #4CAF50;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 USB Camera Live Feed</h1>
            <img src="{{ url_for('video_feed') }}" alt="Camera Feed">
            <div class="info">
                <p class="status">● Live Stream Active</p>
                <p>Press Ctrl+C in the terminal to stop the server</p>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Returns MJPEG stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def cleanup():
    """Release camera resources."""
    global camera
    if camera is not None:
        camera.release()
        print("\nCamera released. Server stopped.")


if __name__ == '__main__':
    try:
        init_camera()
        
        # Get the local IP address for display
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print("\n" + "="*60)
        print("🌐 USB Camera Web Streaming Server Started!")
        print("="*60)
        print(f"\nAccess the camera feed from your browser:")
        print(f"  • Local:   http://localhost:5000")
        print(f"  • Network: http://{local_ip}:5000")
        print("\nPress Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n\nServer interrupted by user.")
    finally:
        cleanup()

