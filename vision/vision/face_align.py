"""
Face alignment utility for ArcFace preprocessing.
Aligns face to canonical 112x112 size using 5-point landmarks.
"""

import numpy as np
import cv2


# Standard ArcFace 112x112 alignment template
# 5 landmark points: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_DST = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)


def align_face_112(frame_bgr: np.ndarray, kps5: np.ndarray) -> np.ndarray:
    """
    Align face to canonical 112x112 size using 5-point landmarks.
    
    Args:
        frame_bgr: Input BGR image (from camera/video)
        kps5: 5 landmark points in image coordinates, shape (5, 2)
              Order: left_eye, right_eye, nose, left_mouth, right_mouth
    
    Returns:
        aligned_face: 112x112 BGR aligned face image
    
    Raises:
        RuntimeError: If alignment transform cannot be estimated
    """
    if kps5.shape != (5, 2):
        raise ValueError(f"Expected kps5 shape (5, 2), got {kps5.shape}")
    
    src = kps5.astype(np.float32)
    dst = ARCFACE_DST
    
    # Estimate similarity transform (rotation + scale + translation)
    # Using LMEDS for robustness
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    
    if M is None:
        raise RuntimeError("Failed to estimate alignment transform from landmarks")
    
    # Apply affine transformation
    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderValue=0
    )
    
    return aligned


def visualize_landmarks(image: np.ndarray, kps5: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """
    Draw landmarks on image for debugging/visualization.
    
    Args:
        image: Input image (will be copied, not modified)
        kps5: 5 landmark points, shape (5, 2)
        color: BGR color tuple
    
    Returns:
        Image with landmarks drawn
    """
    vis = image.copy()
    
    for i, (x, y) in enumerate(kps5):
        cv2.circle(vis, (int(x), int(y)), 2, color, -1)
        cv2.putText(vis, str(i), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    return vis

