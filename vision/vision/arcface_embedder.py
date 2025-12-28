"""
ArcFace ONNX embedder wrapper for face recognition.
Produces 512-D L2-normalized embeddings from aligned 112x112 face images.
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path


class ArcFaceEmbedder:
    """
    ArcFace face embedding extractor using ONNX Runtime.
    
    Expects aligned 112x112 face images (use face_align.py first).
    Outputs 512-D L2-normalized embedding vectors.
    """
    
    def __init__(self, model_path: str, providers=None):
        """
        Initialize ArcFace embedder.
        
        Args:
            model_path: Path to ArcFace ONNX model
            providers: ONNX Runtime execution providers
                      Default: ["CUDAExecutionProvider", "CPUExecutionProvider"]
                      For CPU-only: ["CPUExecutionProvider"]
        """
        self.model_path = Path(model_path)
        self.input_size = (112, 112)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Default to GPU if available, fallback to CPU
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        # Create session options to avoid threading issues on Jetson
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        print(f"Loading ArcFace model: {self.model_path.name}")
        self.sess = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        
        # Print active provider
        active_provider = self.sess.get_providers()[0]
        print(f"  Active provider: {active_provider}")
        
        # Get expected input shape
        input_shape = self.sess.get_inputs()[0].shape
        print(f"  Input shape: {input_shape}")
        
        output_shape = self.sess.get_outputs()[0].shape
        print(f"  Output shape: {output_shape}")
    
    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for ArcFace model.
        
        Args:
            aligned_bgr: Aligned 112x112 BGR face image
        
        Returns:
            Preprocessed blob ready for inference (1, 3, 112, 112)
        """
        # Ensure correct size
        if aligned_bgr.shape[:2] != self.input_size:
            img = cv2.resize(aligned_bgr, self.input_size)
        else:
            img = aligned_bgr
        
        # Convert BGR to RGB (most ArcFace models expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float32
        img = img.astype(np.float32)
        
        # Normalize: common ArcFace normalization
        # (pixel - 127.5) / 128.0
        img = (img - 127.5) / 128.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension: CHW to NCHW
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def embed_from_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Extract embedding from aligned face image.
        
        Args:
            aligned_bgr: Aligned 112x112 BGR face image
        
        Returns:
            512-D L2-normalized embedding vector (float32)
        """
        # Preprocess
        blob = self._preprocess(aligned_bgr)
        
        # Run inference
        feat = self.sess.run([self.output_name], {self.input_name: blob})[0]
        
        # Flatten to 1D
        feat = feat.reshape(-1).astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        
        return feat
    
    def embed_batch(self, aligned_faces: list) -> np.ndarray:
        """
        Extract embeddings from multiple aligned faces.
        
        Args:
            aligned_faces: List of aligned 112x112 BGR face images
        
        Returns:
            Array of shape (N, 512) with L2-normalized embeddings
        """
        embeddings = []
        for face in aligned_faces:
            emb = self.embed_from_aligned(face)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding (512-D)
            emb2: Second embedding (512-D)
        
        Returns:
            Cosine similarity in range [-1, 1]
            Higher values indicate more similar faces
        """
        # Both should already be L2-normalized
        return np.dot(emb1, emb2)

