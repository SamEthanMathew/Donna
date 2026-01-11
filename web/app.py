"""
FastAPI web server for Donna voice assistant.
Provides REST API and WebSocket endpoints for text/voice interaction.
"""

import sys
import json
import asyncio
import cv2
import threading
import numpy as np
from pathlib import Path
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

import config
from core.assistant import Assistant
from core.emotion_detector import get_emotion_detector
from core.vision_monitor import VisionMonitor

# Initialize FastAPI app
app = FastAPI(title="Donna Voice Assistant")

# CORS middleware (for localhost development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global assistant instance
assistant: Optional[Assistant] = None
emotion_detector = get_emotion_detector()

# Global camera instance for webcam feed
camera: Optional[cv2.VideoCapture] = None
camera_lock = threading.Lock()
latest_frame: Optional[np.ndarray] = None

# Global vision monitor for person recognition
vision_monitor: Optional[object] = None

# WebSocket connections for broadcasting person detection
active_websockets: set = set()

# Track last greeting time per person
last_greeting_time: dict = {}


def get_assistant() -> Assistant:
    """Get or create assistant instance."""
    global assistant
    if assistant is None:
        assistant = Assistant()
    return assistant


def init_camera():
    """Initialize camera for webcam feed."""
    global camera
    try:
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
        if not camera.isOpened():
            print(f"[Web] Warning: Could not open camera {config.CAMERA_INDEX}")
            return False
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Start frame capture thread
        def capture_frames():
            global latest_frame
            while camera and camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    with camera_lock:
                        latest_frame = frame.copy()
                else:
                    break
        
        capture_thread = threading.Thread(target=capture_frames, daemon=True)
        capture_thread.start()
        
        print(f"[Web] Camera initialized (index {config.CAMERA_INDEX})")
        return True
    except Exception as e:
        print(f"[Web] Camera initialization error: {e}")
        return False


def generate_webcam_frames():
    """Generate MJPEG frames from camera."""
    global latest_frame
    
    while True:
        with camera_lock:
            if latest_frame is None:
                # Send placeholder frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera not available", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = placeholder
            else:
                frame = latest_frame.copy()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


async def check_person_recognition():
    """Background task to check for recognized persons and send greetings."""
    global vision_monitor, last_greeting_time, active_websockets
    
    import time
    from datetime import datetime, timedelta
    
    while True:
        try:
            if vision_monitor and active_websockets:
                persons = vision_monitor.get_detected_persons()
                
                for person in persons:
                    # Check if we should send a greeting (immediate + cooldown)
                    now = datetime.now()
                    last_greet = last_greeting_time.get(person)
                    
                    # Send immediate greeting if first time or >30 seconds since last
                    should_greet = False
                    if last_greet is None:
                        should_greet = True
                    else:
                        delta = now - last_greet
                        if delta.total_seconds() > 30:  # 30 second cooldown
                            should_greet = True
                    
                    if should_greet:
                        last_greeting_time[person] = now
                        
                        # Generate greeting
                        greeting = f"Hello {person}! How can I help you today?"
                        
                        # Broadcast to all connected WebSockets
                        for ws in list(active_websockets):
                            try:
                                asyncio.create_task(ws.send_json({
                                    "type": "person_detected",
                                    "person": person,
                                    "greeting": greeting
                                }))
                            except:
                                pass
                        
                        # Also trigger TTS if assistant is available
                        try:
                            assistant = get_assistant()
                            if assistant:
                                # Use TTS to speak greeting
                                sys.path.insert(0, str(config.BASE_DIR))
                                from tts.core import say
                                say(greeting)
                        except Exception as e:
                            print(f"[Web] Error speaking greeting: {e}")
            
            await asyncio.sleep(2)  # Check every 2 seconds
        except Exception as e:
            print(f"[Web] Error in person recognition check: {e}")
            await asyncio.sleep(2)


@app.on_event("startup")
async def startup():
    """Initialize assistant on startup."""
    global vision_monitor
    
    print(f"[Web] Starting {config.ASSISTANT_NAME} web server...")
    assistant_instance = get_assistant()
    
    # Initialize camera if vision is enabled
    if config.VISION_ENABLED:
        init_camera()
        
        # Initialize vision monitor for person recognition
        try:
            vision_monitor = VisionMonitor(assistant_instance, enabled=True)
            vision_monitor.start()
            print("[Web] Vision monitor started for person recognition")
            
            # Start background task for person recognition
            asyncio.create_task(check_person_recognition())
        except Exception as e:
            print(f"[Web] Warning: Could not start vision monitor: {e}")
            vision_monitor = None
    
    print(f"[Web] Server ready at http://{config.WEB_HOST}:{config.WEB_PORT}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global camera, vision_monitor
    if camera:
        camera.release()
        print("[Web] Camera released")
    if vision_monitor:
        vision_monitor.stop()
        print("[Web] Vision monitor stopped")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Donna Web UI</h1><p>index.html not found</p>")


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/api/webcam/stream")
async def webcam_stream():
    """Stream webcam feed as MJPEG."""
    return StreamingResponse(
        generate_webcam_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/person/register")
async def register_person(request: dict):
    """
    Register a new person in the face recognition database.
    
    Request: {"name": "person name"}
    Response: {"success": true, "message": "..."}
    """
    if "name" not in request:
        raise HTTPException(status_code=400, detail="Missing 'name' field")
    
    name = request["name"].strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    try:
        # Import vision modules
        sys.path.insert(0, str(config.VISION_BASE_DIR))
        from vision.opencv_detector import OpenCVFaceDetector
        from vision.arcface_embedder import ArcFaceEmbedder
        from vision.face_align import align_face_112
        from vision.face_store import FaceStore
        
        # Initialize models if needed
        models_dir = config.VISION_BASE_DIR / "data" / "models"
        db_path = config.VISION_BASE_DIR / "data" / "db" / "faces.db"
        
        arcface_path = models_dir / "arcface_r100_ms1mv3.onnx"
        if not arcface_path.exists():
            raise HTTPException(status_code=500, detail="ArcFace model not found")
        
        detector = OpenCVFaceDetector(method='haar')
        embedder = ArcFaceEmbedder(str(arcface_path))
        face_store = FaceStore(str(db_path))
        
        # Get current frame from camera
        global latest_frame, camera_lock
        with camera_lock:
            if latest_frame is None:
                raise HTTPException(status_code=400, detail="No camera frame available")
            frame = latest_frame.copy()
        
        # Detect faces
        detections = detector.detect(frame, score_thresh=0.5)
        
        if len(detections) == 0:
            raise HTTPException(status_code=400, detail="No face detected. Please face the camera.")
        
        if len(detections) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected. Please ensure only one person is visible.")
        
        # Get the face detection
        det = detections[0]
        kps = det['kps']
        
        # Align face
        try:
            aligned_face = align_face_112(frame, kps)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Face alignment failed: {str(e)}")
        
        # Extract embedding
        embedding = embedder.embed_from_aligned(aligned_face)
        
        # Save to database
        person_id = face_store.get_or_create_person(name)
        
        # Store embedding
        face_store.add_embedding(person_id, embedding, None)
        
        return {
            "success": True,
            "message": f"Person '{name}' registered successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering person: {str(e)}")


@app.get("/api/person/list")
async def list_persons():
    """
    List all registered persons.
    
    Response: {"persons": [{"id": 1, "name": "John", "embeddings": 3}, ...]}
    """
    try:
        sys.path.insert(0, str(config.VISION_BASE_DIR))
        from vision.face_store import FaceStore
        
        db_path = config.VISION_BASE_DIR / "data" / "db" / "faces.db"
        face_store = FaceStore(str(db_path))
        
        stats = face_store.get_person_stats()
        
        return {
            "persons": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "embeddings": p["embeddings"],
                    "created_at": p["created_at"]
                }
                for p in stats
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing persons: {str(e)}")


@app.get("/api/webcam/face-position")
async def get_face_position():
    """
    Get face position in webcam feed for eye tracking.
    Returns: {"x": 0.5, "y": 0.5, "detected": true} (normalized 0-1)
    """
    global latest_frame
    
    try:
        # Simple face detection using OpenCV Haar Cascade
        with camera_lock:
            if latest_frame is None:
                return {"detected": False, "x": 0.5, "y": 0.5}
            frame = latest_frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face detector (Haar Cascade)
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Calculate center (normalized 0-1)
            center_x = (x + w / 2) / frame.shape[1]
            center_y = (y + h / 2) / frame.shape[0]
            
            return {
                "detected": True,
                "x": center_x,
                "y": center_y
            }
        else:
            return {"detected": False, "x": 0.5, "y": 0.5}
            
    except Exception as e:
        return {"detected": False, "x": 0.5, "y": 0.5, "error": str(e)}


@app.post("/api/chat")
async def chat_endpoint(request: dict):
    """
    REST endpoint for text chat.
    
    Request: {"message": "user input text"}
    Response: {"response": "assistant response", "emotion": "joy"}
    """
    if "message" not in request:
        raise HTTPException(status_code=400, detail="Missing 'message' field")
    
    user_input = request["message"]
    if not user_input or not user_input.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    
    try:
        # Process through assistant
        response_text = get_assistant().process(user_input)
        
        # Detect emotion
        emotion = emotion_detector.get_emotion_from_response(response_text)
        
        return {
            "response": response_text,
            "emotion": emotion
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.post("/api/voice")
async def voice_endpoint(audio: UploadFile = File(...)):
    """
    REST endpoint for voice input.
    Accepts audio file, transcribes, and returns response.
    
    Response: {"transcription": "...", "response": "...", "emotion": "joy"}
    """
    try:
        # Save uploaded audio temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = Path(f.name)
            audio_path.write_bytes(await audio.read())
        
        try:
            # Transcribe using STT
            sys.path.insert(0, str(config.BASE_DIR))
            from stt.stt_whisper import transcribe
            
            transcription = transcribe(audio_path)
            
            if not transcription or not transcription.strip():
                return {
                    "transcription": "",
                    "response": "I couldn't understand that. Please try again.",
                    "emotion": None
                }
            
            # Process through assistant
            response_text = get_assistant().process(transcription)
            
            # Detect emotion
            emotion = emotion_detector.get_emotion_from_response(response_text)
            
            return {
                "transcription": transcription,
                "response": response_text,
                "emotion": emotion
            }
        finally:
            # Clean up temp file
            try:
                audio_path.unlink()
            except:
                pass
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming responses with emotions.
    
    Protocol:
    - Client sends: {"type": "message", "content": "user input"}
    - Server sends: {"type": "token", "content": "token text", "emotion": "joy"}
    - Server sends: {"type": "emotion", "emotion": "joy"}
    - Server sends: {"type": "complete", "content": "full response", "emotion": "joy"}
    - Server sends: {"type": "person_detected", "person": "name", "greeting": "..."}
    """
    global active_websockets
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") != "message":
                continue
            
            user_input = message.get("content", "")
            if not user_input or not user_input.strip():
                continue
            
            # Stream response with emotion detection
            full_response = ""
            last_emotion = None
            last_emotion_update = 0
            
            def stream_callback(token: str):
                """Callback for streaming tokens."""
                nonlocal full_response, last_emotion, last_emotion_update
                full_response += token
                
                # Update emotion periodically during streaming
                import time
                current_time = time.time()
                if current_time - last_emotion_update > config.EMOTION_UPDATE_INTERVAL:
                    emotion = emotion_detector.get_emotion_from_response(full_response)
                    if emotion and emotion != last_emotion:
                        last_emotion = emotion
                        asyncio.create_task(websocket.send_json({
                            "type": "emotion",
                            "emotion": emotion
                        }))
                    last_emotion_update = current_time
                
                # Send token
                asyncio.create_task(websocket.send_json({
                    "type": "token",
                    "content": token
                }))
            
            # Process through assistant with streaming
            try:
                response_text = get_assistant().process(user_input, stream_callback=stream_callback)
                
                # Final emotion detection
                final_emotion = emotion_detector.get_emotion_from_response(response_text)
                
                # Send completion message
                await websocket.send_json({
                    "type": "complete",
                    "content": response_text,
                    "emotion": final_emotion
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print("[Web] WebSocket client disconnected")
        active_websockets.discard(websocket)
    except Exception as e:
        print(f"[Web] WebSocket error: {e}")
        active_websockets.discard(websocket)
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        log_level=config.LOG_LEVEL.lower()
    )

