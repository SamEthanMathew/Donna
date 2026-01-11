"""
Vision monitoring thread for proactive conversation triggers.
Lightweight face detection without Flask/web server.
"""

import sys
import cv2
import numpy as np
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Set
from collections import deque

# Import config
import config

# Add vision module to path
sys.path.insert(0, str(config.VISION_BASE_DIR))
from vision.opencv_detector import OpenCVFaceDetector
from vision.arcface_embedder import ArcFaceEmbedder
from vision.face_align import align_face_112
from vision.face_store import FaceStore

# Import proactive policy
if str(config.BASE_DIR) not in sys.path:
    sys.path.insert(0, str(config.BASE_DIR))
from llm.core.proactive_policy import ProactiveContext, should_start_conversation, make_proactive_brief


class VisionMonitor(threading.Thread):
    """
    Background thread that monitors camera for face detection
    and triggers proactive conversations when appropriate.
    """
    
    def __init__(self, assistant, enabled: bool = True):
        """
        Initialize vision monitor.
        
        Args:
            assistant: Assistant instance to trigger conversations
            enabled: Whether vision monitoring is enabled
        """
        super().__init__(daemon=True)
        self.assistant = assistant
        self.enabled = enabled and config.VISION_ENABLED
        self.running = False
        
        # Vision components
        self.camera = None
        self.detector = None
        self.embedder = None
        self.face_store = None
        
        # State tracking
        self.detected_persons: Set[str] = set()
        self.last_detection_time: Optional[datetime] = None
        self.last_proactive_time: Optional[datetime] = None
        self.recognition_history = {}  # For smoothing
        
        # Thread safety
        self.lock = threading.Lock()
        
    def _init_camera(self) -> bool:
        """Initialize camera. Returns True if successful."""
        try:
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            if not self.camera.isOpened():
                print(f"[Vision] Error: Could not open camera {config.CAMERA_INDEX}")
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            print("[Vision] Camera initialized")
            return True
        except Exception as e:
            print(f"[Vision] Camera initialization error: {e}")
            return False
    
    def _init_models(self) -> bool:
        """Initialize face detection models. Returns True if successful."""
        try:
            arcface_path = config.VISION_MODELS_DIR / "arcface_r100_ms1mv3.onnx"
            
            if not arcface_path.exists():
                print(f"[Vision] ArcFace model not found at {arcface_path}")
                print("[Vision] Run: python vision/scripts/download_models.py")
                return False
            
            self.detector = OpenCVFaceDetector(method='haar')
            self.embedder = ArcFaceEmbedder(str(arcface_path))
            self.face_store = FaceStore(str(config.VISION_BASE_DIR / "data" / "db" / "faces.db"))
            print("[Vision] Models initialized")
            return True
        except Exception as e:
            print(f"[Vision] Model initialization error: {e}")
            return False
    
    def _smooth_recognition(self, face_id: int, name: str) -> str:
        """Smooth recognition results over multiple frames."""
        if face_id not in self.recognition_history:
            self.recognition_history[face_id] = deque(maxlen=config.SMOOTHING_WINDOW)
        
        self.recognition_history[face_id].append(name)
        
        # Return most common name in recent history
        if len(self.recognition_history[face_id]) >= 3:
            names = list(self.recognition_history[face_id])
            return max(set(names), key=names.count)
        return name
    
    def _detect_faces(self) -> List[str]:
        """
        Detect and recognize faces in current frame.
        
        Returns:
            List of recognized person names (empty if none or unknown)
        """
        if not self.camera or not self.detector or not self.embedder:
            return []
        
        try:
            success, frame = self.camera.read()
            if not success:
                return []
            
            # Detect faces
            detections = self.detector.detect(frame, score_thresh=0.5)
            
            recognized = []
            for idx, det in enumerate(detections):
                try:
                    kps = det['kps']
                    
                    # Align face
                    aligned_face = align_face_112(frame, kps)
                    
                    # Extract embedding
                    embedding = self.embedder.embed_from_aligned(aligned_face)
                    
                    # Find match in database
                    match = self.face_store.find_match(
                        embedding,
                        threshold=config.RECOGNITION_THRESHOLD
                    )
                    
                    if match:
                        name, similarity = match
                        # Smooth recognition
                        name = self._smooth_recognition(idx, name)
                        if name != "Unknown":
                            recognized.append(name)
                except Exception as e:
                    # Skip this face if processing fails
                    continue
            
            return list(set(recognized))  # Remove duplicates
            
        except Exception as e:
            if config.LOG_LEVEL == "DEBUG":
                print(f"[Vision] Detection error: {e}")
            return []
    
    def _update_vision_context(self, persons: List[str]):
        """Update assistant's vision context."""
        if persons:
            context = f"Currently visible: {', '.join(persons)}"
        else:
            context = "No recognized persons currently visible"
        
        self.assistant.set_vision_context(context)
    
    def _check_proactive(self) -> bool:
        """Check if proactive conversation should be triggered."""
        if not config.PROACTIVE_ENABLED:
            return False
        
        if not self.detected_persons:
            return False
        
        # Check cooldown
        if self.last_proactive_time:
            delta = datetime.now() - self.last_proactive_time
            if delta.total_seconds() < config.PROACTIVE_COOLDOWN_HOURS * 3600:
                return False
        
        # Check with proactive policy
        ctx = ProactiveContext(
            user_detected=True,
            now_local=datetime.now(),
            last_spoke_at=self.assistant.last_interaction,
            quiet_hours=config.QUIET_HOURS
        )
        
        return should_start_conversation(ctx)
    
    def _trigger_proactive(self):
        """Trigger a proactive conversation."""
        try:
            brief = make_proactive_brief(datetime.now())
            response = self.assistant.process_proactive(brief)
            
            # Speak the response (import TTS)
            sys.path.insert(0, str(config.BASE_DIR))
            from tts.core import say
            say(response)
            
            self.last_proactive_time = datetime.now()
            
            if config.LOG_INTERACTIONS:
                print(f"[Vision] Proactive conversation triggered: {brief[:50]}...")
        except Exception as e:
            print(f"[Vision] Error triggering proactive: {e}")
    
    def run(self):
        """Main monitoring loop."""
        if not self.enabled:
            print("[Vision] Vision monitoring disabled")
            return
        
        # Initialize camera and models
        if not self._init_camera():
            print("[Vision] Failed to initialize camera, disabling vision")
            return
        
        if not self._init_models():
            print("[Vision] Failed to initialize models, disabling vision")
            return
        
        print("[Vision] Vision monitoring started")
        self.running = True
        
        while self.running:
            try:
                # Detect faces
                persons = self._detect_faces()
                
                # Update detected persons
                with self.lock:
                    self.detected_persons = set(persons)
                    if persons:
                        self.last_detection_time = datetime.now()
                
                # Update vision context
                self._update_vision_context(persons)
                
                # Check for proactive trigger
                if self._check_proactive():
                    self._trigger_proactive()
                
                # Sleep between checks
                time.sleep(config.VISION_CHECK_INTERVAL)
                
            except Exception as e:
                print(f"[Vision] Error in monitoring loop: {e}")
                time.sleep(config.VISION_CHECK_INTERVAL)
    
    def stop(self):
        """Stop the vision monitor."""
        self.running = False
        if self.camera:
            self.camera.release()
        print("[Vision] Vision monitoring stopped")
    
    def get_detected_persons(self) -> List[str]:
        """Get currently detected persons (thread-safe)."""
        with self.lock:
            return list(self.detected_persons)

