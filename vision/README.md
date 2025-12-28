# Face Recognition System

A complete face recognition system for Jetson devices using SCRFD detection and ArcFace embeddings. Runs entirely on-device with ONNX Runtime and provides web-based interfaces for both registration and recognition.

## Features

- 🎯 **High-Accuracy Detection**: SCRFD 2.5G with 5-point landmark detection
- 🔍 **Robust Recognition**: ArcFace embeddings with 512-D face vectors
- 🌐 **Web Interface**: Browser-based UI (works perfectly over SSH)
- ⚡ **GPU Accelerated**: ONNX Runtime with CUDA/TensorRT support
- 💾 **SQLite Database**: Efficient storage with multiple embeddings per person
- 📊 **Real-time Processing**: Live camera feed with instant recognition

## System Requirements

- **Hardware**: NVIDIA Jetson device (Nano, Xavier NX, AGX Xavier, Orin, etc.)
- **JetPack**: 5.x (tested on 5.1.2 / L4T R35.4.1)
- **Python**: 3.8+
- **Camera**: USB webcam or CSI camera module

## Architecture

```
┌─────────────┐
│ USB Camera  │
└──────┬──────┘
       │
       ├─────────────────────┬──────────────────────┐
       │                     │                      │
       ▼                     ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   SCRFD      │      │  Face Align  │      │   ArcFace    │
│  Detector    │─────▶│  (112x112)   │─────▶│  Embedder    │
└──────────────┘      └──────────────┘      └──────┬───────┘
                                                    │
                                                    ▼
                                             ┌─────────────┐
                                             │   SQLite    │
                                             │  Database   │
                                             └─────────────┘
```

## Installation

### 1. Install Dependencies

```bash
cd /home/sam/vision
pip install -r requirements.txt
```

**For GPU acceleration** (recommended):
```bash
pip install --extra-index-url https://pypi.nvidia.com onnxruntime-gpu
```

### 2. Download Models

```bash
python scripts/download_models.py
```

This will download:
- `scrfd_2.5g_bnkps.onnx` (~3 MB) - Face detector with landmarks
- `arcface_r100_ms1mv3.onnx` (~166 MB) - Face embedding model

Models are saved to `data/models/`.

## Usage

### Program 1: Register Faces

Register people in the database by capturing their face embeddings.

```bash
python register_person.py
```

Then open your browser:
- **From the Jetson**: `http://localhost:5000`
- **From another device**: `http://<jetson-ip>:5000`

**Steps:**
1. Position your face in front of the camera
2. Enter your name in the text field
3. Click "Capture Face"
4. Repeat 3-5 times from different angles (optional, but improves accuracy)

### Program 2: Recognize Faces

Run real-time face recognition on the camera feed.

```bash
python recognize_person.py
```

Then open your browser:
- **From the Jetson**: `http://localhost:5001`
- **From another device**: `http://<jetson-ip>:5001`

The system will:
- Detect faces in real-time
- Draw bounding boxes around detected faces
- Display recognized names with confidence scores
- Show "Unknown" for unrecognized faces

**Color Legend:**
- 🟢 **Green**: Recognized person
- 🟠 **Orange**: Unknown person
- 🔴 **Red**: Processing error

## Configuration

### Recognition Threshold

Edit `recognize_person.py`:

```python
RECOGNITION_THRESHOLD = 0.4  # Default: 0.4
```

- **Lower values (0.3-0.35)**: More lenient, may have false positives
- **Medium values (0.4-0.45)**: Balanced (recommended)
- **Higher values (0.5+)**: Strict, may miss some matches

### Camera Settings

Edit `CAMERA_INDEX` in both programs:

```python
CAMERA_INDEX = 0  # Default: first camera
```

For CSI camera on Jetson:
```python
CAMERA_INDEX = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

## Directory Structure

```
/home/sam/vision/
├── register_person.py          # Registration web app
├── recognize_person.py         # Recognition web app
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── scripts/
│   └── download_models.py      # Model downloader
├── vision/                     # Core library
│   ├── __init__.py
│   ├── scrfd_detector.py       # SCRFD face detector
│   ├── arcface_embedder.py     # ArcFace embedder
│   ├── face_align.py           # Face alignment utility
│   └── face_store.py           # SQLite database wrapper
├── data/
│   ├── models/                 # ONNX models
│   │   ├── scrfd_2.5g_bnkps.onnx
│   │   └── arcface_r100_ms1mv3.onnx
│   ├── db/                     # Database
│   │   └── faces.db
│   └── faces/                  # Captured face images
│       └── <person_name>/
│           └── capture_*.jpg
├── camera_feed.py              # Original simple camera viewer
└── camera_stream_web.py        # Original web camera stream
```

## Technical Details

### Models

**SCRFD (Sample and Computation Redistribution for Face Detection)**
- Input: 640×640 RGB image
- Output: Bounding boxes, confidence scores, 5 facial landmarks
- Stride levels: 8, 16, 32
- Landmarks: left eye, right eye, nose, left mouth, right mouth

**ArcFace**
- Input: 112×112 aligned RGB face
- Output: 512-D L2-normalized embedding vector
- Normalization: (pixel - 127.5) / 128.0
- Similarity metric: Cosine similarity

### Database Schema

**persons table:**
```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
```

**embeddings table:**
```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,      -- 512 float32 values
    image_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);
```

### Performance

**On Jetson Xavier NX (GPU mode):**
- SCRFD inference: ~15-20 ms
- ArcFace inference: ~5-8 ms
- Total pipeline: ~30-40 ms (25-30 FPS)

**On Jetson Nano (CPU mode):**
- SCRFD inference: ~80-100 ms
- ArcFace inference: ~30-40 ms
- Total pipeline: ~120-150 ms (6-8 FPS)

## Troubleshooting

### Camera not opening

```bash
# Check available cameras
ls /dev/video*

# Check permissions
sudo chmod 666 /dev/video0
```

### Models not found

```bash
# Re-download models
python scripts/download_models.py

# Check if models exist
ls -lh data/models/
```

### ONNX Runtime GPU not working

```bash
# Check CUDA availability
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']

# If only CPU, install GPU version:
pip install --extra-index-url https://pypi.nvidia.com onnxruntime-gpu
```

### Low FPS / Slow inference

1. **Enable GPU**: Make sure `onnxruntime-gpu` is installed
2. **Lower resolution**: Reduce camera resolution in code
3. **Reduce detection frequency**: Process every N frames instead of all frames
4. **Use TensorRT**: Convert models to TensorRT for maximum performance

### Recognition accuracy issues

1. **Register more embeddings**: Capture 5-10 faces per person from different angles
2. **Adjust threshold**: Lower threshold for more lenient matching
3. **Improve lighting**: Ensure good lighting conditions
4. **Check alignment**: Verify face landmarks are detected correctly

## Advanced Usage

### Running as a Service

Create `/etc/systemd/system/face-recognition.service`:

```ini
[Unit]
Description=Face Recognition Service
After=network.target

[Service]
Type=simple
User=sam
WorkingDirectory=/home/sam/vision
ExecStart=/usr/bin/python3 recognize_person.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable face-recognition
sudo systemctl start face-recognition
```

### Database Management

```python
from vision.face_store import FaceStore

# Open database
db = FaceStore("data/db/faces.db")

# Get all registered people
stats = db.get_person_stats()
for person in stats:
    print(f"{person['name']}: {person['embedding_count']} embeddings")

# Delete a person
person_id = db.get_person_id("John Doe")
if person_id:
    db.delete_person(person_id)

db.close()
```

### Export/Import Embeddings

```python
import pickle
import numpy as np
from vision.face_store import FaceStore

# Export
db = FaceStore("data/db/faces.db")
embeddings = db.get_all_embeddings()

with open("embeddings_backup.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# Import
with open("embeddings_backup.pkl", "rb") as f:
    embeddings = pickle.load(f)

for person_id, name, embedding in embeddings:
    pid = db.get_or_create_person(name)
    db.add_embedding(pid, embedding)

db.close()
```

## API Reference

### SCRFDFaceDetector

```python
from vision.scrfd_detector import SCRFDFaceDetector

detector = SCRFDFaceDetector(
    model_path="data/models/scrfd_2.5g_bnkps.onnx",
    input_size=(640, 640)
)

detections = detector.detect(frame_bgr, score_thresh=0.5, iou_thresh=0.4)
# Returns: [{"bbox": [x1,y1,x2,y2], "score": float, "kps": np.array(5,2)}]
```

### ArcFaceEmbedder

```python
from vision.arcface_embedder import ArcFaceEmbedder

embedder = ArcFaceEmbedder(model_path="data/models/arcface_r100_ms1mv3.onnx")

embedding = embedder.embed_from_aligned(aligned_face_112x112)
# Returns: np.array(512,) float32, L2-normalized

similarity = embedder.cosine_similarity(emb1, emb2)
# Returns: float in [-1, 1], higher = more similar
```

### FaceStore

```python
from vision.face_store import FaceStore

db = FaceStore("data/db/faces.db")

# Add person and embedding
person_id = db.add_person("John Doe")
db.add_embedding(person_id, embedding_512d, image_path="path/to/face.jpg")

# Find match
match = db.find_match(query_embedding, threshold=0.4)
if match:
    name, similarity = match
    print(f"Matched: {name} (confidence: {similarity:.2f})")
```

## License

This project is for educational and research purposes. Model weights are from InsightFace research and subject to their respective licenses.

## Credits

- **SCRFD**: Sample and Computation Redistribution for Face Detection
- **ArcFace**: Additive Angular Margin Loss for Deep Face Recognition  
- **InsightFace**: Open-source face analysis library
- **ONNX Runtime**: Cross-platform ML inference accelerator

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your JetPack version and Python version
3. Test with CPU-only mode first (`onnxruntime` instead of `onnxruntime-gpu`)
4. Check that models downloaded correctly

---

**Built for NVIDIA Jetson** | Optimized for edge deployment | SSH-friendly web interface
