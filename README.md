# Donna - Voice Assistant

A fully offline, multi-modal voice assistant for NVIDIA Jetson devices with speech recognition, text-to-speech, language model integration, face recognition, and a beautiful web interface with animated eyes.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Module Documentation](#module-documentation)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

### Core Capabilities

- **🎤 Voice Interaction**: Wake word detection ("Donna") with offline speech-to-text using Whisper.cpp
- **🔊 Text-to-Speech**: Natural voice synthesis using Piper TTS with Amy medium voice
- **🧠 Language Model**: Integration with Ollama for intelligent conversations
- **👁️ Face Recognition**: Real-time person recognition with automatic greetings
- **💾 Memory System**: Persistent memory storage with semantic retrieval
- **🌐 Web Interface**: Modern web UI with animated eyes, emotion detection, and real-time streaming
- **😊 Emotion Detection**: Automatic emotion detection from responses with animated eye expressions
- **🔄 Proactive Conversations**: Intelligent proactive interactions based on person detection and time

### Web UI Features

- **Animated Eyes**: [Web-Eye-Animation](https://github.com/CyberAgentAILab/Web-Eye-Animation) library integration for expressive eye animations
- **Real-time Streaming**: Token-by-token response streaming via WebSocket
- **Person Registration**: Easy person registration through web interface
- **Face Tracking**: Eyes follow detected faces in real-time
- **Black & Gold Theme**: Professional, bold color scheme
- **Responsive Design**: Adapts to screen size automatically

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Donna Voice Assistant                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  STT Module  │      │  TTS Module │      │  LLM Module  │
│ (Whisper.cpp)│      │   (Piper)    │      │   (Ollama)  │
└──────┬───────┘      └──────┬──────┘      └──────┬──────┘
       │                      │                     │
       └──────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Core Assistant  │
                    │  (Orchestrator)  │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Vision Module│      │ Memory Store │      │  Web Server  │
│ (Face Recog) │      │   (SQLite)   │      │   (FastAPI)  │
└──────────────┘      └──────────────┘      └──────────────┘
```

### Module Breakdown

1. **STT (Speech-to-Text)**: Uses Whisper.cpp for offline speech recognition with wake word detection
2. **TTS (Text-to-Speech)**: Uses Piper TTS with Amy voice for natural speech synthesis
3. **LLM (Language Model)**: Integrates with Ollama for conversational AI
4. **Vision**: Face recognition using SCRFD detection and ArcFace embeddings
5. **Core**: Orchestrates all modules, manages memory, and handles conversation state
6. **Web**: FastAPI server with WebSocket support for real-time interactions

## Prerequisites

### Hardware Requirements

- **NVIDIA Jetson Device**: AGX Orin, Xavier NX, AGX Xavier, or similar
- **USB Microphone**: For voice input
- **USB Speaker**: For audio output (or use HDMI audio)
- **USB Camera**: For face recognition (optional but recommended)
- **Network Connection**: For initial setup and model downloads

### Software Requirements

- **JetPack**: 5.x (tested on 5.1.2 / L4T R35.4.1)
- **Python**: 3.8 or higher
- **Ollama**: Running locally with a compatible model (e.g., `llama3.2:1b`)
- **System Packages**: Build tools, audio utilities, and multimedia libraries

### OS Compatibility

- **Primary**: Ubuntu 20.04/22.04 (JetPack)
- **Architecture**: aarch64 (ARM64)

## Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd jetson_agent

# Install Python dependencies
pip install -r llm/requirements.txt
pip install -r vision/requirements.txt
pip install -r web/requirements.txt

# Setup STT module
cd stt
./setup.sh

# Setup TTS module
cd ../tts
./setup.sh

# Setup Vision module
cd ../vision
./setup.sh

# Install and start Ollama
# Follow instructions at https://ollama.ai
ollama pull llama3.2:1b

# Return to project root
cd ..
```

### Detailed Installation

#### 1. System Dependencies

The setup scripts will install most dependencies automatically, but you may need:

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake make g++ \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    alsa-utils wget \
    python3-pip
```

#### 2. Python Dependencies

Install Python packages for each module:

```bash
# LLM module
pip install -r llm/requirements.txt

# Vision module (CPU version)
pip install -r vision/requirements.txt

# For GPU acceleration (recommended on Jetson):
pip install --extra-index-url https://pypi.nvidia.com onnxruntime-gpu

# Web server
pip install -r web/requirements.txt
```

#### 3. STT Module Setup

The STT module uses Whisper.cpp for speech recognition:

```bash
cd stt
./setup.sh
```

This will:
- Install build dependencies
- Clone and build whisper.cpp from source
- Download the `ggml-base.en.bin` model (~142 MB)
- Build the `whisper-cli` binary

**Note**: The build process may take 10-20 minutes.

#### 4. TTS Module Setup

The TTS module uses Piper TTS:

```bash
cd tts
./setup.sh
```

This will:
- Install `alsa-utils` and `wget`
- Download the Piper aarch64 binary
- Download the Amy medium voice model (~15 MB)

#### 5. Vision Module Setup

The Vision module requires ONNX models:

```bash
cd vision
./setup.sh
python3 scripts/download_models.py
```

This will:
- Install Python dependencies
- Optionally install ONNX Runtime GPU
- Download SCRFD and ArcFace models

#### 6. Ollama Setup

Install and configure Ollama:

```bash
# Install Ollama (follow instructions at https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, pull a model
ollama pull llama3.2:1b
```

#### 7. Verify Installation

Test each module:

```bash
# Test STT
cd stt && python3 test_stt.py

# Test TTS
cd ../tts && python3 test_tts.py

# Test Vision (if camera available)
cd ../vision && python3 test_setup.py
```

## Configuration

All configuration is centralized in `config.py` at the project root.

### Key Configuration Options

#### LLM Configuration

```python
OLLAMA_URL = "http://localhost:11434"  # Ollama server URL
MODEL_NAME = "llama3.2:1b"              # Model to use
VERBATIM_TURNS = 4                      # Keep last N turns verbatim
SUMMARY_UPDATE_THRESHOLD = 6            # Update summary after N turns
MAX_MEMORY_CONTEXT = 8                  # Max memories to inject
```

#### STT Configuration

```python
WAKE_WORD = "Donna"                      # Wake word to trigger recording
SAMPLE_RATE = 16000                      # Audio sample rate
AUDIO_INPUT_DEVICE = None                # None = auto-detect
```

#### TTS Configuration

```python
AUDIO_OUTPUT_DEVICE = None               # None = auto-detect USB speaker
```

#### Vision Configuration

```python
CAMERA_INDEX = 0                         # Camera device index
RECOGNITION_THRESHOLD = 0.4             # Face recognition threshold
VISION_ENABLED = True                   # Enable/disable vision
VISION_CHECK_INTERVAL = 1.0             # Seconds between checks
```

#### Proactive Policy

```python
QUIET_HOURS = "23:00-07:00"              # Hours to disable proactive
PROACTIVE_COOLDOWN_HOURS = 2            # Min hours between proactive
PROACTIVE_ENABLED = True                # Enable proactive conversations
```

#### Web Server

```python
WEB_HOST = "127.0.0.1"                  # Server host
WEB_PORT = 8000                         # Server port
WEB_ENABLED = True                      # Enable web UI
```

#### Emotion Detection

```python
EMOTION_DETECTION_METHOD = "sentiment"   # Detection method
EMOTION_UPDATE_INTERVAL = 0.5           # Seconds between updates
```

### Audio Device Configuration

The TTS module automatically detects USB audio devices. To manually configure:

1. List audio devices:
   ```bash
   arecord -l  # Input devices
   aplay -l    # Output devices
   ```

2. Update `config.py`:
   ```python
   AUDIO_INPUT_DEVICE = "plughw:2,0"   # Example USB microphone
   AUDIO_OUTPUT_DEVICE = "plughw:2,0"  # Example USB speaker
   ```

## Usage

### Command-Line Modes

#### Voice Mode (Default)

Run the voice assistant with wake word detection:

```bash
python3 donna.py
```

The assistant will:
1. Listen continuously for the wake word "Donna"
2. When detected, record your speech
3. Transcribe using Whisper.cpp
4. Process with the LLM
5. Speak the response using Piper TTS

**Press Ctrl+C to exit.**

#### Web Mode

Start the web server only:

```bash
python3 donna.py --web
# or
python3 donna.py --mode web
```

Then open `http://127.0.0.1:8000` in your browser.

#### Both Mode

Run voice assistant and web server simultaneously:

```bash
python3 donna.py --mode both
```

This starts:
- Voice loop in the main thread
- Web server in a background thread

### Web UI Usage

1. **Start the web server**:
   ```bash
   python3 donna.py --web
   ```

2. **Open in browser**: Navigate to `http://127.0.0.1:8000`

3. **Interact with Donna**:
   - Type messages in the text input
   - Click the microphone button for voice input (uses browser Web Speech API)
   - Watch the animated eyes respond to emotions
   - See real-time streaming responses

4. **Register a Person**:
   - Click "Add Person" button in the top right
   - Enter the person's name
   - Click "Capture & Register"
   - The system will capture a face from the camera and register it

5. **Automatic Greetings**:
   - When a registered person is detected by the camera, Donna will automatically greet them
   - Greetings respect cooldown periods and quiet hours

### Person Registration

#### Via Web UI

1. Click "Add Person" button
2. Enter the person's name
3. Face the camera
4. Click "Capture & Register"

#### Via Command Line

```bash
cd vision
python3 register_person.py
```

Then open `http://localhost:5000` in your browser.

## API Documentation

### REST API Endpoints

#### POST `/api/chat`

Send a text message and receive a response.

**Request:**
```json
{
  "message": "Hello, how are you?"
}
```

**Response:**
```json
{
  "response": "I'm doing well, thank you for asking!",
  "emotion": "joy"
}
```

#### POST `/api/voice`

Upload an audio file for transcription and response.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Audio file (WAV format recommended)

**Response:**
```json
{
  "transcription": "Hello, how are you?",
  "response": "I'm doing well, thank you!",
  "emotion": "joy"
}
```

#### POST `/api/person/register`

Register a new person in the face recognition database.

**Request:**
```json
{
  "name": "John Doe"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Person 'John Doe' registered successfully"
}
```

**Note**: Requires a face to be visible in the camera feed.

#### GET `/api/person/list`

List all registered persons.

**Response:**
```json
{
  "persons": [
    {
      "id": 1,
      "name": "John Doe",
      "embeddings": 3,
      "created_at": "2024-01-15T10:30:00"
    }
  ]
}
```

#### GET `/api/webcam/stream`

Stream webcam feed as MJPEG.

**Response:**
- Content-Type: `multipart/x-mixed-replace; boundary=frame`
- MJPEG video stream

#### GET `/api/webcam/face-position`

Get face position in webcam feed for eye tracking.

**Response:**
```json
{
  "detected": true,
  "x": 0.5,
  "y": 0.5
}
```

Coordinates are normalized (0-1), where (0.5, 0.5) is center.

### WebSocket API

#### Endpoint: `/ws`

Real-time bidirectional communication for streaming responses.

**Client → Server Messages:**

```json
{
  "type": "message",
  "content": "Hello, Donna!"
}
```

**Server → Client Messages:**

1. **Token Stream**:
   ```json
   {
     "type": "token",
     "content": "Hello"
   }
   ```

2. **Emotion Update**:
   ```json
   {
     "type": "emotion",
     "emotion": "joy"
   }
   ```

3. **Complete Response**:
   ```json
   {
     "type": "complete",
     "content": "Hello! How can I help you today?",
     "emotion": "joy"
   }
   ```

4. **Person Detected**:
   ```json
   {
     "type": "person_detected",
     "person": "John Doe",
     "greeting": "Hello John Doe! How can I help you today?"
   }
   ```

5. **Error**:
   ```json
   {
     "type": "error",
     "content": "Error message"
   }
   ```

## Module Documentation

### STT Module (Speech-to-Text)

**Location**: `stt/`

**Technology**: Whisper.cpp

**Features**:
- Wake word detection ("Donna")
- Offline speech recognition
- High accuracy with base English model
- Low latency optimized for Jetson

**Key Files**:
- `stt_whisper.py`: Main STT interface
- `setup.sh`: Installation script
- `bin/whisper.cpp/`: Whisper.cpp source and binaries
- `models/ggml-base.en.bin`: Whisper model

**Usage**:
```python
from stt.stt_whisper import detect_wake_word, listen_and_transcribe

# Detect wake word
if detect_wake_word():
    # Record and transcribe
    text = listen_and_transcribe()
    print(text)
```

### TTS Module (Text-to-Speech)

**Location**: `tts/`

**Technology**: Piper TTS

**Features**:
- Natural voice synthesis (Amy medium voice)
- Offline processing
- Automatic USB speaker detection
- Low latency

**Key Files**:
- `core/tts_piper.py`: Main TTS interface
- `setup.sh`: Installation script
- `bin/piper/`: Piper binary
- `models/en_US-amy-medium.onnx`: Voice model

**Usage**:
```python
from tts.core import say

say("Hello, this is a test.")
```

### LLM Module (Language Model)

**Location**: `llm/`

**Technology**: Ollama

**Features**:
- Conversational AI
- Memory integration
- Conversation summarization
- Streaming responses
- Memory suggestion extraction

**Key Files**:
- `core/ollama_client.py`: Ollama API client
- `core/memory_store.py`: Memory database
- `core/conversation_summary.py`: Conversation management
- `core/memory_policy.py`: Memory extraction and validation
- `donna_prompt.txt`: System prompt

**Usage**:
```python
from core.assistant import Assistant

assistant = Assistant()
response = assistant.process("Hello!")
print(response)
```

### Vision Module (Face Recognition)

**Location**: `vision/`

**Technology**: SCRFD + ArcFace + ONNX Runtime

**Features**:
- Real-time face detection
- Person recognition with embeddings
- Multiple embeddings per person
- SQLite database storage
- Web interfaces for registration and recognition

**Key Files**:
- `recognize_person.py`: Recognition web server
- `register_person.py`: Registration web server
- `vision/face_store.py`: Database interface
- `vision/opencv_detector.py`: Face detection
- `vision/arcface_embedder.py`: Face embeddings

**Usage**:
```python
from vision.face_store import FaceStore
from vision.opencv_detector import OpenCVFaceDetector
from vision.arcface_embedder import ArcFaceEmbedder

# Initialize
detector = OpenCVFaceDetector()
embedder = ArcFaceEmbedder("path/to/model.onnx")
face_store = FaceStore("path/to/faces.db")

# Detect and recognize
detections = detector.detect(frame)
# ... process and recognize
```

### Core Modules

**Location**: `core/`

#### Assistant (`core/assistant.py`)

Main orchestrator that:
- Manages LLM interactions
- Handles memory retrieval and storage
- Manages conversation state
- Integrates vision context
- Processes proactive conversations

#### VisionMonitor (`core/vision_monitor.py`)

Background thread that:
- Monitors camera for face detection
- Recognizes persons using face recognition
- Triggers proactive conversations
- Updates vision context

#### EmotionDetector (`core/emotion_detector.py`)

Detects emotions from text:
- Keyword-based sentiment analysis
- Maps to Web-Eye-Animation emotions
- Supports: joy, sadness, surprise, anger, fear, disgust, confusion, love, sleepy, excitement

### Web Module

**Location**: `web/`

**Technology**: FastAPI + WebSocket

**Features**:
- REST API endpoints
- WebSocket streaming
- Static file serving
- Person recognition integration
- Real-time emotion updates

**Key Files**:
- `app.py`: FastAPI application
- `static/index.html`: Web UI
- `static/app.js`: Frontend JavaScript
- `static/web-eye-animation.js`: Eye animation library
- `static/person_modal.js`: Person registration modal
- `static/style.css`: Styling

## Project Structure

```
jetson_agent/
├── config.py                 # Unified configuration
├── donna.py                  # Main entry point
├── README.md                 # This file
│
├── core/                     # Core orchestration modules
│   ├── __init__.py
│   ├── assistant.py         # Main assistant orchestrator
│   ├── emotion_detector.py  # Emotion detection
│   └── vision_monitor.py    # Vision monitoring thread
│
├── stt/                      # Speech-to-Text module
│   ├── stt_whisper.py       # Main STT interface
│   ├── setup.sh             # Installation script
│   ├── requirements.txt     # Dependencies
│   ├── README.md            # STT documentation
│   ├── bin/                 # Binaries
│   │   └── whisper.cpp/    # Whisper.cpp source and build
│   └── models/              # Whisper models
│       └── ggml-base.en.bin
│
├── tts/                      # Text-to-Speech module
│   ├── core/
│   │   └── tts_piper.py     # Main TTS interface
│   ├── setup.sh             # Installation script
│   ├── requirements.txt     # Dependencies
│   ├── README.md            # TTS documentation
│   ├── bin/                 # Binaries
│   │   └── piper/           # Piper binary
│   └── models/              # Voice models
│       └── en_US-amy-medium.onnx
│
├── llm/                      # Language Model module
│   ├── app.py               # Standalone LLM app
│   ├── config.py            # LLM config (legacy)
│   ├── requirements.txt     # Dependencies
│   ├── donna_prompt.txt     # System prompt
│   ├── core/
│   │   ├── ollama_client.py # Ollama API client
│   │   ├── memory_store.py # Memory database
│   │   ├── memory_policy.py # Memory extraction
│   │   ├── conversation_summary.py # Conversation management
│   │   └── utils.py         # Utilities
│   └── data/                # Data directory
│       └── donna.db         # SQLite memory database
│
├── vision/                   # Vision module
│   ├── recognize_person.py  # Recognition web server
│   ├── register_person.py   # Registration web server
│   ├── setup.sh             # Installation script
│   ├── requirements.txt     # Dependencies
│   ├── README.md            # Vision documentation
│   ├── scripts/             # Utility scripts
│   │   └── download_models.py
│   ├── vision/              # Vision library
│   │   ├── opencv_detector.py
│   │   ├── arcface_embedder.py
│   │   ├── face_align.py
│   │   └── face_store.py
│   └── data/                # Data directory
│       ├── models/          # ONNX models
│       └── db/              # Face database
│           └── faces.db
│
└── web/                      # Web server module
    ├── app.py               # FastAPI application
    ├── requirements.txt     # Dependencies
    └── static/              # Static files
        ├── index.html       # Web UI
        ├── app.js           # Frontend logic
        ├── style.css        # Styling
        ├── web-eye-animation.js # Eye animation library
        └── person_modal.js # Registration modal
```

### Data Storage

- **Memory Database**: `llm/data/donna.db` - SQLite database for conversation memories
- **Face Database**: `vision/data/db/faces.db` - SQLite database for face embeddings
- **Models**: Stored in respective module `models/` directories
- **Binaries**: Stored in respective module `bin/` directories

## Development

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd jetson_agent
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r llm/requirements.txt
   pip install -r vision/requirements.txt
   pip install -r web/requirements.txt
   ```

3. **Set up modules** (see Installation section)

4. **Configure for development**:
   - Edit `config.py` for your environment
   - Set `LOG_LEVEL = "DEBUG"` for verbose logging

### Code Structure

- **Modular Design**: Each module (STT, TTS, LLM, Vision) is self-contained
- **Unified Config**: All configuration in `config.py`
- **Core Orchestration**: `core/` modules coordinate between modules
- **Web Integration**: FastAPI server provides REST and WebSocket APIs

### Adding New Features

1. **New Module**: Create a new directory with its own `requirements.txt` and setup
2. **Core Integration**: Add to `core/assistant.py` if needed
3. **Web API**: Add endpoints to `web/app.py`
4. **Configuration**: Add settings to `config.py`

### Running Tests

```bash
# Test STT
cd stt && python3 test_stt.py

# Test TTS
cd ../tts && python3 test_tts.py

# Test Vision
cd ../vision && python3 test_setup.py
```

## Troubleshooting

### Audio Issues

**Problem**: No audio playback
- **Solution**: Check audio device with `aplay -l` and update `AUDIO_OUTPUT_DEVICE` in `config.py`
- **Solution**: Ensure USB speaker is connected and recognized
- **Solution**: Try `aplay -D plughw:2,0 test.wav` to test specific device

**Problem**: Microphone not working
- **Solution**: Check microphone with `arecord -l`
- **Solution**: Test recording: `arecord -d 5 test.wav && aplay test.wav`
- **Solution**: Update `AUDIO_INPUT_DEVICE` in `config.py`

### Camera Issues

**Problem**: Camera not detected
- **Solution**: Check camera with `lsusb` or `v4l2-ctl --list-devices`
- **Solution**: Update `CAMERA_INDEX` in `config.py` (try 0, 1, 2)
- **Solution**: Ensure camera is not being used by another process

**Problem**: Face recognition not working
- **Solution**: Ensure models are downloaded: `python3 vision/scripts/download_models.py`
- **Solution**: Check database exists: `ls vision/data/db/faces.db`
- **Solution**: Register at least one person first

### Model Loading Issues

**Problem**: Ollama connection error
- **Solution**: Ensure Ollama is running: `ollama serve`
- **Solution**: Check model is available: `ollama list`
- **Solution**: Pull model if missing: `ollama pull llama3.2:1b`

**Problem**: Whisper model not found
- **Solution**: Run STT setup: `cd stt && ./setup.sh`
- **Solution**: Check model exists: `ls stt/models/ggml-base.en.bin`

**Problem**: Piper model not found
- **Solution**: Run TTS setup: `cd tts && ./setup.sh`
- **Solution**: Check model exists: `ls tts/models/en_US-amy-medium.onnx`

### Web Server Issues

**Problem**: Web server won't start
- **Solution**: Check port is available: `netstat -tuln | grep 8000`
- **Solution**: Change port in `config.py` if needed
- **Solution**: Ensure FastAPI is installed: `pip install fastapi uvicorn`

**Problem**: WebSocket connection fails
- **Solution**: Check browser console for errors
- **Solution**: Ensure WebSocket support in browser
- **Solution**: Check CORS settings in `web/app.py`

### Vision Issues

**Problem**: Vision monitor not starting
- **Solution**: Set `VISION_ENABLED = True` in `config.py`
- **Solution**: Ensure camera is available
- **Solution**: Check models are downloaded
- **Solution**: Review logs for specific error messages

**Problem**: Person recognition inaccurate
- **Solution**: Register multiple embeddings per person (capture from different angles)
- **Solution**: Adjust `RECOGNITION_THRESHOLD` in `config.py` (lower = more strict)
- **Solution**: Ensure good lighting and face visibility

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing

- Test new features before submitting
- Ensure existing functionality still works
- Update documentation if needed

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Acknowledgments

### Libraries and Tools

- **[Whisper.cpp](https://github.com/ggerganov/whisper.cpp)**: Speech-to-text engine
- **[Piper TTS](https://github.com/rhasspy/piper)**: Text-to-speech synthesis
- **[Ollama](https://ollama.ai)**: Local LLM inference
- **[FastAPI](https://fastapi.tiangolo.com)**: Web framework
- **[Web-Eye-Animation](https://github.com/CyberAgentAILab/Web-Eye-Animation)**: Eye animation library
- **[OpenCV](https://opencv.org)**: Computer vision
- **[ONNX Runtime](https://onnxruntime.ai)**: Model inference
- **[SCRFD](https://github.com/deepinsight/insightface)**: Face detection
- **[ArcFace](https://github.com/deepinsight/insightface)**: Face recognition

### Models

- **Whisper Base English**: OpenAI Whisper model
- **Piper Amy Medium**: Rhasspy voice model
- **SCRFD 2.5G**: InsightFace face detection model
- **ArcFace R100**: InsightFace face recognition model

---

**Donna Voice Assistant** - Bringing AI assistance to edge devices.

