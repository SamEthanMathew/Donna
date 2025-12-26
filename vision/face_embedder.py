import cv2
import numpy as np
from typing import Tuple


class FaceEmbedder:
    """
    Face embedding generator using OpenCV's SFace model.
    
    Generates 128-dimensional face embeddings for recognition.
    """
    
    def __init__(self, model_path: str = "data/models/sface.onnx"):
        """
        Initialize face embedder.
        
        Args:
            model_path: Path to SFace ONNX model
        """
        self.model_path = model_path
        self.input_size = (112, 112)  # SFace expects 112x112 input
        
        # Load SFace model
        self.recognizer = cv2.FaceRecognizerSF.create(
            model_path,
            ""  # config (unused for ONNX)
        )
        
    def align_face(self, frame: np.ndarray, face_box: np.ndarray) -> np.ndarray:
        """
        Align and crop face from frame using detection box and landmarks.
        
        Args:
            frame: Original frame (BGR)
            face_box: Face detection [x, y, w, h, score, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y]
            
        Returns:
            Aligned face image (112x112)
        """
        # Use OpenCV's alignCrop which handles landmark-based alignment
        aligned_face = self.recognizer.alignCrop(frame, face_box)
        return aligned_face
    
    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Generate embedding from aligned face image.
        
        Args:
            aligned_face: Aligned face image (112x112)
            
        Returns:
            128-dimensional embedding vector (L2 normalized)
        """
        # Extract feature
        embedding = self.recognizer.feature(aligned_face)
        
        # Flatten to 1D array
        embedding = embedding.flatten()
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def embed_from_detection(self, frame: np.ndarray, face_box: np.ndarray) -> np.ndarray:
        """
        Extract and generate embedding from face detection in one step.
        
        Args:
            frame: Original frame (BGR)
            face_box: Face detection [x, y, w, h, score, lmk1_x, lmk1_y, ..., lmk5_x, lmk5_y]
            
        Returns:
            128-dimensional embedding vector
        """
        aligned = self.align_face(frame, face_box)
        return self.embed(aligned)
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        # Both should already be L2 normalized, so dot product = cosine similarity
        return np.dot(embedding1, embedding2)
    
    @staticmethod
    def match_threshold(similarity: float, threshold: float = 0.45) -> bool:
        """
        Check if similarity score exceeds threshold.
        
        Args:
            similarity: Cosine similarity score
            threshold: Matching threshold (default: 0.45)
            
        Returns:
            True if match, False otherwise
        """
        return similarity >= threshold


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    from vision.face_detector import YuNetFaceDetector
    
    print("Testing FaceEmbedder...")
    
    # Initialize
    detector = YuNetFaceDetector("data/models/yunet.onnx")
    embedder = FaceEmbedder("data/models/sface.onnx")
    
    # Open camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    print("Camera opened. Press 'q' to quit, SPACE to generate embedding.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = detector.detect(frame)
        
        # Draw faces
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Show aligned face
            try:
                aligned = embedder.align_face(frame, face)
                # Show in corner
                h_aligned, w_aligned = aligned.shape[:2]
                frame[10:10+h_aligned, 10:10+w_aligned] = aligned
            except Exception as e:
                print(f"Alignment error: {e}")
        
        cv2.imshow("Face Embedder Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and len(faces) > 0:
            # Generate embedding for first face
            try:
                embedding = embedder.embed_from_detection(frame, faces[0])
                print(f"Generated embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
                print(f"First 5 values: {embedding[:5]}")
            except Exception as e:
                print(f"Embedding error: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")

