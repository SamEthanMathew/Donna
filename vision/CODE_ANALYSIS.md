# Vision Code Analysis

**Date:** December 28, 2025  
**Location:** `/home/sam/jetson_agent/vision/`

## 📋 Executive Summary

This is a **face recognition system** designed for Jetson devices (NVIDIA embedded systems). The system implements a complete pipeline from camera input to face identification using deep learning models (SCRFD for detection, ArcFace for recognition).

**Status:** ~95% complete - Core implementation done, but missing SCRFD model file (~3-4 MB)

---

## 🏗️ Architecture Overview

### System Pipeline
```
Camera Feed → Face Detection (SCRFD) → Face Alignment → Embedding (ArcFace) → Database Lookup → Recognition Result
```

### Key Components

1. **Face Detection** - SCRFD ONNX model (2.5G BNKPS variant)
   - Detects faces with bounding boxes
   - Extracts 5-point facial landmarks
   - Alternative: OpenCV fallback detector

2. **Face Alignment** - Geometric transformation
   - Aligns faces to canonical 112x112 size
   - Uses 5-point landmarks for affine transformation

3. **Face Embedding** - ArcFace ResNet50 ONNX model
   - Converts aligned faces to 512-D feature vectors
   - L2-normalized embeddings for cosine similarity

4. **Database Storage** - SQLite
   - Stores person names and embeddings
   - Supports multiple embeddings per person
   - Fast similarity search

5. **Web Interface** - Flask
   - Registration app (Port 5000)
   - Recognition app (Port 5001)
   - Camera streaming

---

## 📁 Code Structure

### Main Directory (`vision/`)

#### Core Modules (Empty Stubs)
- `pipeline.py` - Empty, intended for full pipeline
- `face_embedder.py` - Empty, wrapper should be here
- `face_id.py` - Empty, recognition logic
- `face_store.py` - Empty, database wrapper
- `__init__.py` - Empty

#### Camera Modules
- **`camera.py`** - YuNet face detector demo
  - Uses OpenCV's YuNet detector (different from SCRFD)
  - Real-time face detection on camera feed
  - Draws bounding boxes and confidence scores

- **`camera_feed.py`** - Simple camera viewer
  - Basic OpenCV camera capture and display
  - 640x480 resolution
  - Press 'q' or ESC to exit

- **`camera_stream_web.py`** - Web-based camera streaming
  - Flask server on port 5000
  - MJPEG streaming over HTTP
  - Accessible from browser/network

#### Face Detection
- **`face_detector.py`** - YuNet wrapper
  - Wraps OpenCV's `FaceDetectorYN`
  - Handles input resizing and coordinate scaling
  - Returns faces with landmarks

#### Documentation
- **`STATUS.md`** - Comprehensive setup status and instructions
- **`DOWNLOAD_SCRFD.md`** - Model download guide
- **`requirements.txt`** - Python dependencies

#### Setup & Testing
- **`setup.sh`** - Automated setup script
- **`test_setup.py`** - System verification script
  - Checks Python version, dependencies, models, camera, directory structure

### Nested Directory (`vision/vision/`)

This contains the **actual implementation** (the main directory has mostly empty stubs):

#### **`scrfd_detector.py`** (404 lines) - SCRFD Face Detector
- **Purpose:** High-accuracy face detection using SCRFD ONNX model
- **Features:**
  - ONNX Runtime inference (GPU/CPU)
  - Multi-scale feature pyramid detection
  - Non-Maximum Suppression (NMS)
  - 5-point landmark extraction
  - Anchor-based detection decoding
- **Key Classes:**
  - `SCRFDFaceDetector` - Main detector class
- **Methods:**
  - `detect()` - Detect faces in frame
  - `draw_detections()` - Visualization helper
  - `_preprocess()` - Image preprocessing
  - `_postprocess()` - Decode model outputs
- **Technical Details:**
  - Input: 640x640 BGR image
  - Output: List of detections with bbox, score, landmarks
  - Supports dynamic anchor generation
  - Handles multiple feature map levels (3 or 5)

#### **`opencv_detector.py`** (177 lines) - OpenCV Fallback
- **Purpose:** Alternative face detector when SCRFD unavailable
- **Features:**
  - Haar Cascade detector (fast, less accurate)
  - DNN detector option (more accurate)
  - Estimated 5-point landmarks
- **Key Classes:**
  - `OpenCVFaceDetector` - Fallback detector
- **Use Case:** Works without ONNX models, lower accuracy

#### **`arcface_embedder.py`** (160 lines) - ArcFace Embedder
- **Purpose:** Extract 512-D face embeddings from aligned faces
- **Features:**
  - ONNX Runtime inference (GPU/CPU)
  - L2-normalized embeddings
  - Batch processing support
  - Cosine similarity computation
- **Key Classes:**
  - `ArcFaceEmbedder` - Embedding extractor
- **Methods:**
  - `embed_from_aligned()` - Single face embedding
  - `embed_batch()` - Multiple faces
  - `cosine_similarity()` - Compare embeddings
- **Technical Details:**
  - Input: 112x112 aligned BGR face
  - Output: 512-D float32 vector (L2-normalized)
  - Normalization: (pixel - 127.5) / 128.0

#### **`face_align.py`** (81 lines) - Face Alignment
- **Purpose:** Align faces to canonical 112x112 for ArcFace
- **Features:**
  - 5-point landmark-based alignment
  - Affine transformation (rotation, scale, translation)
  - Standard ArcFace template coordinates
- **Key Functions:**
  - `align_face_112()` - Main alignment function
  - `visualize_landmarks()` - Debug visualization
- **Technical Details:**
  - Uses `cv2.estimateAffinePartial2D()` with LMEDS
  - Template: Standard ArcFace 112x112 landmark positions

#### **`face_store.py`** (277 lines) - Database Wrapper
- **Purpose:** SQLite database for face recognition
- **Features:**
  - Person management (add, get, delete)
  - Embedding storage (BLOB format)
  - Similarity search with thresholding
  - Multiple embeddings per person (averaged)
  - Statistics and reporting
- **Key Classes:**
  - `FaceStore` - Database manager
- **Database Schema:**
  ```sql
  persons(id, name, created_at)
  embeddings(id, person_id, embedding_blob, image_path, created_at)
  ```
- **Key Methods:**
  - `add_person()` - Register new person
  - `add_embedding()` - Store face embedding
  - `find_match()` - Search for matching person
  - `get_person_stats()` - Get statistics
- **Technical Details:**
  - Embeddings stored as BLOB (512 * 4 bytes = 2048 bytes)
  - Cosine similarity for matching
  - Default threshold: 0.4
  - Averages multiple embeddings per person

#### **`__init__.py`** - Module metadata
- Version: 1.0.0

### Scripts Directory (`vision/scripts/`)

#### **`download_models.py`** (225 lines) - Model Downloader
- **Purpose:** Download SCRFD and ArcFace ONNX models
- **Features:**
  - Automatic download from GitHub releases
  - Progress reporting
  - ZIP extraction for ArcFace
  - Manual download instructions fallback
- **Models:**
  1. SCRFD: `scrfd_2.5g_bnkps.onnx` (~3-4 MB)
  2. ArcFace: `arcface_r100_ms1mv3.onnx` (~166 MB)

---

## 🔧 Dependencies

### Python Packages
- `opencv-python>=4.8.0` - Computer vision
- `numpy>=1.24.0` - Numerical operations
- `Flask>=2.3.0` - Web server
- `Pillow>=10.0.0` - Image processing
- `onnxruntime>=1.15.0` - ONNX model inference
- `requests>=2.31.0` - HTTP requests

### Optional
- `onnxruntime-gpu` - GPU acceleration (NVIDIA Jetson)

### Hardware
- USB Camera (640x480)
- NVIDIA Jetson device (JetPack 5.1.2 / L4T R35.4.1)
- Optional: GPU for faster inference

---

## 🎯 Functionality

### Current Capabilities

1. **Face Detection**
   - ✅ SCRFD detector (if model available)
   - ✅ OpenCV fallback (always available)
   - ✅ 5-point landmark extraction
   - ✅ Real-time camera processing

2. **Face Recognition**
   - ✅ Face alignment to 112x112
   - ✅ 512-D embedding extraction
   - ✅ Database storage and retrieval
   - ✅ Similarity matching with threshold

3. **Web Interface**
   - ✅ Camera streaming (port 5000)
   - ⚠️ Registration app (mentioned but not in codebase)
   - ⚠️ Recognition app (mentioned but not in codebase)

4. **Utilities**
   - ✅ Setup verification script
   - ✅ Model downloader
   - ✅ Database management

### Missing Components

1. **Main Application Scripts** (referenced in STATUS.md but not present):
   - `register_person.py` - Registration web app
   - `recognize_person.py` - Recognition web app

2. **Model File:**
   - `scrfd_2.5g_bnkps.onnx` - SCRFD detector model (~3-4 MB)

3. **Empty Module Stubs:**
   - `pipeline.py` - Should contain full recognition pipeline
   - `face_embedder.py` - Should wrap ArcFaceEmbedder
   - `face_id.py` - Should contain recognition logic
   - `face_store.py` - Should wrap FaceStore

---

## 🔍 Code Quality Analysis

### Strengths

1. **Well-Structured**
   - Clear separation of concerns
   - Modular design (detector, embedder, aligner, database)
   - Good documentation in STATUS.md

2. **Robust Implementation**
   - Error handling in model loading
   - Fallback detectors (OpenCV)
   - Thread-safe database operations
   - GPU/CPU provider fallback

3. **Jetson-Optimized**
   - ONNX Runtime with CUDA support
   - Thread configuration for embedded systems
   - Efficient preprocessing

4. **Production-Ready Features**
   - Multiple embeddings per person (robustness)
   - Database indexing
   - Comprehensive test script

### Weaknesses

1. **Incomplete Integration**
   - Main directory has empty stubs
   - Actual code in nested `vision/vision/` directory
   - Missing main application scripts

2. **Inconsistent Detector Usage**
   - `camera.py` uses YuNet (OpenCV)
   - `face_detector.py` uses YuNet
   - But system designed for SCRFD
   - No unified detector interface

3. **Missing Web Apps**
   - STATUS.md references `register_person.py` and `recognize_person.py`
   - These files don't exist in the codebase
   - Only `camera_stream_web.py` exists

4. **No Error Recovery**
   - Limited error handling in camera modules
   - No retry logic for model loading
   - No graceful degradation

---

## 🚀 Usage Workflow

### Setup (Current State)
```bash
cd /home/sam/jetson_agent/vision
python3 test_setup.py  # Verify setup
python3 scripts/download_models.py  # Download models
```

### Intended Usage (Per STATUS.md)
```bash
# Register faces
python3 register_person.py  # Not present in codebase
# Open: http://<jetson-ip>:5000

# Run recognition
python3 recognize_person.py  # Not present in codebase
# Open: http://<jetson-ip>:5001
```

### Available Now
```bash
# Camera streaming
python3 camera_stream_web.py
# Open: http://<jetson-ip>:5000

# Face detection demo
python3 camera.py

# Simple camera viewer
python3 camera_feed.py
```

---

## 📊 Performance Characteristics

### Inference Speed (Estimated)
- **CPU Mode:**
  - SCRFD detection: ~50-80ms per frame
  - ArcFace embedding: ~30-50ms per face
  - Total: ~120-150ms per frame (6-8 FPS)

- **GPU Mode (with onnxruntime-gpu):**
  - SCRFD detection: ~15-25ms per frame
  - ArcFace embedding: ~10-15ms per face
  - Total: ~30-40ms per frame (25-30 FPS)

### Model Sizes
- SCRFD: ~3.4 MB
- ArcFace: ~166.3 MB
- Total: ~170 MB

### Database
- Embedding size: 2048 bytes per face (512-D float32)
- Typical database: <10 MB for hundreds of people

---

## 🐛 Known Issues

1. **Missing SCRFD Model**
   - System designed for SCRFD but model not downloaded
   - Falls back to OpenCV (lower accuracy)

2. **Incomplete Main Scripts**
   - Registration and recognition web apps missing
   - Only camera streaming works

3. **Code Duplication**
   - Detector implementations in multiple places
   - No unified interface

4. **Empty Module Stubs**
   - Main directory has empty files
   - Actual code in nested directory

---

## 💡 Recommendations

### Immediate Actions
1. **Download SCRFD Model**
   - Follow `DOWNLOAD_SCRFD.md` instructions
   - Or use `scripts/download_models.py`

2. **Create Missing Web Apps**
   - Implement `register_person.py` using Flask
   - Implement `recognize_person.py` using Flask
   - Integrate with `vision/vision/` modules

3. **Consolidate Code Structure**
   - Move code from `vision/vision/` to main `vision/` directory
   - Remove empty stubs or implement them
   - Create unified detector interface

### Long-term Improvements
1. **Add Error Handling**
   - Retry logic for model loading
   - Graceful degradation when models missing
   - Better camera error handling

2. **Performance Optimization**
   - Batch processing for multiple faces
   - Async web requests
   - Caching for repeated queries

3. **Testing**
   - Unit tests for each module
   - Integration tests for full pipeline
   - Performance benchmarks

4. **Documentation**
   - API documentation
   - Usage examples
   - Architecture diagrams

---

## 📝 Summary

**What Works:**
- ✅ Core face detection and recognition algorithms
- ✅ Database storage and retrieval
- ✅ Camera streaming
- ✅ Model download utilities
- ✅ Setup verification

**What's Missing:**
- ❌ SCRFD model file (~3-4 MB)
- ❌ Registration web app
- ❌ Recognition web app
- ❌ Complete pipeline integration

**Overall Assessment:**
The codebase has a solid foundation with well-implemented core components. The main gap is the missing application layer (web apps) and the SCRFD model file. Once these are added, the system should be fully functional.

**Code Quality:** 7/10
- Good: Structure, algorithms, documentation
- Needs Work: Integration, error handling, completeness


