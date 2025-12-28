"""
OpenCV-based face detector fallback.
Uses Haar Cascade or DNN face detector as an alternative to SCRFD.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict


class OpenCVFaceDetector:
    """
    OpenCV face detector with landmark estimation.
    This is a fallback detector that works reliably but with lower accuracy than SCRFD.
    """
    
    def __init__(self, method='haar'):
        """
        Initialize OpenCV face detector.
        
        Args:
            method: 'haar' for Haar Cascade (faster) or 'dnn' for DNN detector (more accurate)
        """
        self.method = method
        
        if method == 'haar':
            # Load Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            print(f"Loaded Haar Cascade face detector")
        elif method == 'dnn':
            # Load DNN face detector (Caffe model)
            try:
                model_file = cv2.data.haarcascades.replace('haarcascades', 'dnn') + 'opencv_face_detector_uint8.pb'
                config_file = model_file.replace('.pb', '.pbtxt')
                self.detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                print(f"Loaded DNN face detector")
            except:
                # Fallback to Haar if DNN not available
                print("DNN detector not available, using Haar Cascade")
                self.method = 'haar'
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
        
        # Load facial landmark detector if available
        try:
            self.landmark_detector = cv2.face.createFacemarkLBF()
            landmark_model = cv2.data.haarcascades.replace('haarcascades', 'lbfmodel.yaml')
            if Path(landmark_model).exists():
                self.landmark_detector.loadModel(landmark_model)
                self.has_landmarks = True
            else:
                self.has_landmarks = False
        except:
            self.has_landmarks = False
        
        print(f"  Landmarks: {'Enabled' if self.has_landmarks else 'Using estimated positions'}")
    
    def _estimate_landmarks(self, bbox: np.ndarray) -> np.ndarray:
        """
        Estimate 5-point landmarks from bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2]
        
        Returns:
            Landmarks as (5, 2) array: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Rough estimates based on typical face proportions
        landmarks = np.array([
            [x1 + w * 0.3, y1 + h * 0.4],  # left eye
            [x1 + w * 0.7, y1 + h * 0.4],  # right eye
            [x1 + w * 0.5, y1 + h * 0.6],  # nose
            [x1 + w * 0.35, y1 + h * 0.8], # left mouth
            [x1 + w * 0.65, y1 + h * 0.8], # right mouth
        ], dtype=np.float32)
        
        return landmarks
    
    def detect(self, frame_bgr: np.ndarray, score_thresh: float = 0.5, 
               iou_thresh: float = 0.4) -> List[Dict]:
        """
        Detect faces in frame.
        
        Args:
            frame_bgr: Input BGR frame
            score_thresh: Minimum confidence (used for DNN only)
            iou_thresh: NMS threshold (used for DNN only)
        
        Returns:
            List of dicts with keys:
                - bbox: [x1, y1, x2, y2]
                - score: confidence score
                - kps: 5 landmarks as array of shape (5, 2)
        """
        detections = []
        
        if self.method == 'haar':
            # Haar Cascade detection
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
                landmarks = self._estimate_landmarks(bbox)
                
                detections.append({
                    'bbox': bbox,
                    'score': 1.0,  # Haar doesn't provide confidence
                    'kps': landmarks
                })
        
        elif self.method == 'dnn':
            # DNN detection
            h, w = frame_bgr.shape[:2]
            blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), [104, 117, 123], False, False)
            
            self.detector.setInput(blob)
            detections_dnn = self.detector.forward()
            
            for i in range(detections_dnn.shape[2]):
                confidence = detections_dnn[0, 0, i, 2]
                
                if confidence > score_thresh:
                    x1 = int(detections_dnn[0, 0, i, 3] * w)
                    y1 = int(detections_dnn[0, 0, i, 4] * h)
                    x2 = int(detections_dnn[0, 0, i, 5] * w)
                    y2 = int(detections_dnn[0, 0, i, 6] * h)
                    
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                    landmarks = self._estimate_landmarks(bbox)
                    
                    detections.append({
                        'bbox': bbox,
                        'score': float(confidence),
                        'kps': landmarks
                    })
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict],
                       draw_kps: bool = True) -> np.ndarray:
        """Draw detections on frame."""
        vis = frame.copy()
        
        for det in detections:
            bbox = det['bbox'].astype(int)
            score = det['score']
            kps = det['kps']
            
            # Draw bounding box
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 255, 0), 2)
            
            # Draw score
            label = f"{score:.2f}"
            cv2.putText(vis, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw keypoints
            if draw_kps:
                for i, (x, y) in enumerate(kps):
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        return vis

