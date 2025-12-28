"""
SCRFD ONNX face detector wrapper with landmark detection.
Supports SCRFD models with 5-point landmark output (BNKPS variants).
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Tuple


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.4) -> List[int]:
    """
    Non-Maximum Suppression (NMS).
    
    Args:
        boxes: Array of shape (N, 4) with boxes as [x1, y1, x2, y2]
        scores: Array of shape (N,) with confidence scores
        iou_thresh: IoU threshold for suppression
    
    Returns:
        List of indices to keep
    """
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    
    return keep


def distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape=None) -> np.ndarray:
    """
    Decode bounding boxes from anchor points and distances.
    
    Args:
        points: Anchor points, shape (N, 2)
        distance: Predicted distances, shape (N, 4)
        max_shape: Optional image shape (H, W) for clipping
    
    Returns:
        Boxes in format [x1, y1, x2, y2], shape (N, 4)
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray, max_shape=None) -> np.ndarray:
    """
    Decode keypoints from anchor points and distances.
    
    Args:
        points: Anchor points, shape (N, 2)
        distance: Predicted keypoint distances, shape (N, 10) for 5 points
        max_shape: Optional image shape (H, W) for clipping
    
    Returns:
        Keypoints in format [[x1,y1], [x2,y2], ...], shape (N, 5, 2)
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        
        preds.append(np.stack([px, py], axis=-1))
    
    return np.stack(preds, axis=1)


class SCRFDFaceDetector:
    """
    SCRFD face detector with 5-point landmarks.
    
    Supports SCRFD BNKPS models (e.g., scrfd_2.5g_bnkps).
    """
    
    def __init__(self, model_path: str, providers=None, input_size=(640, 640)):
        """
        Initialize SCRFD detector.
        
        Args:
            model_path: Path to SCRFD ONNX model
            providers: ONNX Runtime execution providers
            input_size: Model input size (width, height)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Default to GPU if available, fallback to CPU
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        # Create session options to avoid threading issues
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        print(f"Loading SCRFD model: {self.model_path.name}")
        self.sess = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        
        active_provider = self.sess.get_providers()[0]
        print(f"  Active provider: {active_provider}")
        print(f"  Input size: {input_size}")
        print(f"  Outputs: {len(self.output_names)}")
        
        # Detect number of feature map levels from model outputs
        # Check model outputs to determine fmc
        num_outputs = len(self.output_names)
        self.fmc = num_outputs // 3  # score, bbox, kps for each level
        
        print(f"  Detected {self.fmc} feature map levels")
        
        # Common stride configurations
        if self.fmc == 3:
            self._feat_stride_fpn = [8, 16, 32]
        elif self.fmc == 5:
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        else:
            # Fallback
            self._feat_stride_fpn = [8, 16, 32]
            
        self._num_anchors = 1  # Most SCRFD models use 1 anchor per location
        self.use_kps = True  # BNKPS models have keypoints
        
        # Pre-generate anchor centers
        self.center_cache = {}
        self._init_anchors()
    
    def _init_anchors(self):
        """Pre-generate anchor centers for each stride level."""
        # Get actual output shapes from model if available
        try:
            # Try to get dynamic shapes from first inference
            # For now, use computed sizes based on input
            for idx, stride in enumerate(self._feat_stride_fpn):
                feat_h = self.input_size[1] // stride
                feat_w = self.input_size[0] // stride
                
                anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1)
                anchor_centers = anchor_centers.astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1)
                    anchor_centers = anchor_centers.reshape((-1, 2))
                
                self.center_cache[stride] = anchor_centers
        except Exception as e:
            print(f"  Warning: Could not pre-generate anchors: {e}")
            # Will generate dynamically during inference
    
    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame for inference.
        
        Args:
            frame_bgr: Input BGR frame
        
        Returns:
            (preprocessed_blob, scale, padding_offset)
        """
        img_h, img_w = frame_bgr.shape[:2]
        
        # Resize maintaining aspect ratio
        scale = min(self.input_size[0] / img_w, self.input_size[1] / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        resized = cv2.resize(frame_bgr, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to float and normalize
        img = padded.astype(np.float32)
        img = (img - 127.5) / 128.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img, scale, (0, 0)
    
    def _postprocess(self, outputs: List[np.ndarray], scale: float, 
                     score_thresh: float, iou_thresh: float) -> List[Dict]:
        """
        Decode SCRFD outputs and apply NMS.
        
        Args:
            outputs: Raw model outputs
            scale: Resize scale factor
            score_thresh: Score threshold for filtering
            iou_thresh: IoU threshold for NMS
        
        Returns:
            List of detections with bbox, score, and keypoints
        """
        all_boxes = []
        all_scores = []
        all_kps = []
        
        # Decode outputs for each stride level
        for idx, stride in enumerate(self._feat_stride_fpn):
            # Output indices for this stride
            score_idx = idx
            bbox_idx = idx + self.fmc
            kps_idx = idx + self.fmc * 2 if self.use_kps else None
            
            # Check if indices are valid
            if score_idx >= len(outputs) or bbox_idx >= len(outputs):
                continue
            
            scores = outputs[score_idx]
            bbox_preds = outputs[bbox_idx]
            
            # Reshape - handle different output formats
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            
            # Generate anchor centers for this level based on actual output size
            num_anchors = len(scores)
            if stride not in self.center_cache or len(self.center_cache[stride]) != num_anchors:
                # Dynamically generate anchors based on output size
                feat_h = int(np.sqrt(num_anchors))
                feat_w = num_anchors // feat_h
                if feat_h * feat_w != num_anchors:
                    # Not a perfect square, calculate dimensions
                    feat_size = int(np.sqrt(num_anchors * self.input_size[0] / self.input_size[1]))
                    feat_h = feat_size
                    feat_w = num_anchors // feat_h
                
                anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1)
                anchor_centers = anchor_centers.astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                self.center_cache[stride] = anchor_centers[:num_anchors]  # Trim to exact size
            
            anchor_centers = self.center_cache[stride]
            
            # Apply stride scaling to bbox predictions
            bbox_preds = bbox_preds * stride
            
            # Filter by score
            valid_mask = scores > score_thresh
            if not valid_mask.any():
                continue
            
            scores = scores[valid_mask]
            bbox_preds = bbox_preds[valid_mask]
            anchor_centers = anchor_centers[valid_mask]
            
            # Decode boxes
            boxes = distance2bbox(anchor_centers, bbox_preds)
            
            # Decode keypoints if available
            if self.use_kps and kps_idx is not None and kps_idx < len(outputs):
                kps_preds = outputs[kps_idx]
                kps_preds = kps_preds.reshape(-1, 10) * stride
                kps_preds = kps_preds[valid_mask]
                kps = distance2kps(anchor_centers, kps_preds)
            else:
                kps = np.zeros((len(boxes), 5, 2), dtype=np.float32)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_kps.append(kps)
        
        if not all_boxes:
            return []
        
        # Concatenate all levels
        all_boxes = np.vstack(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_kps = np.vstack(all_kps)
        
        # Scale back to original image coordinates
        all_boxes /= scale
        all_kps /= scale
        
        # Apply NMS
        keep_indices = nms(all_boxes, all_scores, iou_thresh)
        
        # Build final detections
        detections = []
        for idx in keep_indices:
            detections.append({
                'bbox': all_boxes[idx],
                'score': float(all_scores[idx]),
                'kps': all_kps[idx]
            })
        
        return detections
    
    def detect(self, frame_bgr: np.ndarray, score_thresh: float = 0.5, 
               iou_thresh: float = 0.4) -> List[Dict]:
        """
        Detect faces in frame.
        
        Args:
            frame_bgr: Input BGR frame
            score_thresh: Minimum confidence score
            iou_thresh: IoU threshold for NMS
        
        Returns:
            List of dicts with keys:
                - bbox: [x1, y1, x2, y2]
                - score: confidence score
                - kps: 5 landmarks as array of shape (5, 2)
        """
        # Preprocess
        blob, scale, offset = self._preprocess(frame_bgr)
        
        # Run inference
        outputs = self.sess.run(self.output_names, {self.input_name: blob})
        
        # Postprocess
        detections = self._postprocess(outputs, scale, score_thresh, iou_thresh)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       draw_kps: bool = True) -> np.ndarray:
        """
        Draw detections on frame (for visualization/debugging).
        
        Args:
            frame: Input frame (will be copied)
            detections: List of detection dicts
            draw_kps: Whether to draw keypoints
        
        Returns:
            Frame with annotations
        """
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

