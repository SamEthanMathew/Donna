# Face Recognition System

A complete face registration and recognition system for the Jetson Agent.

## Components

### Core Modules

- **`face_detector.py`** - YuNet face detector (existing)
- **`face_embedder.py`** - SFace embedding generator (NEW)
- **`face_store.py`** - SQLite database for storing face data (NEW)

### Scripts

- **`register_person.py`** - Register new people
- **`recognize_person.py`** - Live face recognition

## Quick Start

### 1. Register a Person

Register yourself or someone else in the system:

```bash
cd /home/sam/jetson_agent
python vision/register_person.py
```

**What it does:**
- Prompts for the person's name
- Opens camera feed
- Guides you to capture 5-10 face images
- Stores embeddings in database
- Saves sample images to `data/faces/{name}/`

**Tips:**
- Position face clearly in frame
- Vary angles slightly (front, slight left/right)
- Ensure good lighting
- Press SPACE to capture
- Press 'q' or ESC when done (minimum 5 captures)

### 2. Run Recognition

Recognize registered people from live camera:

```bash
cd /home/sam/jetson_agent
python vision/recognize_person.py
```

**What it does:**
- Opens camera feed
- Detects faces in real-time
- Matches against registered people
- Displays name and confidence percentage
- Shows FPS and registered person count

**Display:**
- 🟢 Green box = Recognized person with name
- 🟠 Orange box = Unknown person

### 3. Test Components

Test the face embedder:

```bash
python vision/face_embedder.py
```

Test the database:

```bash
python vision/face_store.py
```

## Database

Face data is stored in: `data/db/faces.db`

**Schema:**
- `persons` - Person names and metadata
- `face_embeddings` - Face embedding vectors

**Management:**

```python
from vision.face_store import FaceStore

store = FaceStore()

# List all persons
persons = store.list_persons()
for p in persons:
    print(f"{p['name']}: {p['embedding_count']} embeddings")

# Delete a person
store.delete_person("John")

# Get person info
info = store.get_person("Jane")
```

## Technical Details

### Face Detection
- **Model:** YuNet (ONNX)
- **Input:** 320x320
- **Output:** Bounding boxes + 5 facial landmarks

### Face Recognition
- **Model:** SFace (ONNX)
- **Embedding:** 128-dimensional vector
- **Similarity:** Cosine similarity
- **Threshold:** 0.45 (configurable)

### Performance
- **Detection:** ~30 FPS on Jetson
- **Recognition:** ~25 FPS with 1 face
- **Storage:** ~512 bytes per embedding

## File Structure

```
vision/
├── face_detector.py      # Face detection
├── face_embedder.py      # Embedding generation
├── face_store.py         # Database storage
├── register_person.py    # Registration script
├── recognize_person.py   # Recognition demo
└── README.md            # This file

data/
├── db/
│   └── faces.db         # SQLite database
├── faces/
│   └── {name}/          # Sample face images
└── models/
    ├── yunet.onnx       # Face detector
    └── sface.onnx       # Face recognizer
```

## Troubleshooting

### Camera not found
- Try changing `camera_index` from 1 to 0 in the scripts
- Check: `ls /dev/video*`

### Person not recognized
- Register with more captures (8-10 recommended)
- Ensure good lighting during registration and recognition
- Lower threshold in `recognize_person.py` (line ~149)

### Database errors
- Check database exists: `ls data/db/faces.db`
- Recreate if corrupted: `rm data/db/faces.db && python vision/register_person.py`

## Next Steps

- Integrate with voice assistant for personalized greetings
- Add confidence threshold configuration
- Support multiple faces simultaneously
- Add face tracking for smoother recognition
- Export/import person database

