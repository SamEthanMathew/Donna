import cv2
import numpy as np

class YuNetFaceDetector:
    """
    Wraps OpenCV's YuNet face detector.

    Output: an array of faces, each row contains:
      [x, y, w, h, score, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y]
    """

    def __init__(self, model_path: str, input_size=(320, 320), score_thresh=0.9, nms_thresh=0.3, top_k=5000):
        self.model_path = model_path
        self.input_size = input_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

        # Input size gets set each frame (based on actual camera frame size)
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",                 # config (unused for ONNX)
            input_size,         # fixed input size
            score_threshold=score_thresh,
            nms_threshold=nms_thresh,
            top_k=top_k
        )

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        # Resize image to the fixed input size expected by the model
        # This avoids shape mismatch errors in some OpenCV versions
        input_w, input_h = self.input_size
        frame_resized = cv2.resize(frame_bgr, (input_w, input_h))
        
        self.detector.setInputSize((input_w, input_h))

        _, faces = self.detector.detect(frame_resized)

        if faces is None:
            return np.zeros((0, 15), dtype=np.float32)

        # Scale detection results back to original image size
        scale_x = w / input_w
        scale_y = h / input_h

        # Scale bounding box (x, y, w, h)
        faces[:, 0:4] *= [scale_x, scale_y, scale_x, scale_y]
        
        # Scale landmarks (5 points, x and y)
        # Landmarks are at indices 5-14: x1, y1, x2, y2, ...
        faces[:, 5:15:2] *= scale_x
        faces[:, 6:15:2] *= scale_y

        return faces
