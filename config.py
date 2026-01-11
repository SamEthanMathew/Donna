"""
Unified configuration for Donna voice assistant.
Consolidates settings from all modules (LLM, STT, TTS, Vision).
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================================
# LLM Configuration
# ============================================================================
LLM_BASE_DIR = BASE_DIR / "llm"
LLM_DATA_DIR = LLM_BASE_DIR / "data"
LLM_DATA_DIR.mkdir(exist_ok=True)

DB_PATH = LLM_DATA_DIR / "donna.db"
PROMPT_PATH = LLM_BASE_DIR / "donna_prompt.txt"

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"

# Conversation management
VERBATIM_TURNS = 4  # Keep last 4 turns verbatim
SUMMARY_UPDATE_THRESHOLD = 6  # Update summary after N turns

# Memory context limits
MAX_MEMORY_CONTEXT = 8  # Max memories to inject
MIN_MEMORY_CONTEXT = 3  # Min memories to inject

# Guardrails
MAX_USER_CHARS = 4000
MAX_ASSISTANT_CHARS = 4000

# ============================================================================
# STT Configuration
# ============================================================================
STT_BASE_DIR = BASE_DIR / "stt"

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_FORMAT = "S16_LE"  # 16-bit signed little-endian

# Wake word detection settings
WAKE_WORD = "Donna"
WAKE_WORD_CHUNK_DURATION = 2  # seconds to record for wake word detection
MAIN_RECORDING_DURATION = 5  # seconds to record after wake word detected

# Audio input device (None = auto-detect or use default)
AUDIO_INPUT_DEVICE = None  # Can be set to "plughw:2,0" or "default"

# ============================================================================
# TTS Configuration
# ============================================================================
TTS_BASE_DIR = BASE_DIR / "tts"

# Audio output device (None = auto-detect USB speaker)
AUDIO_OUTPUT_DEVICE = None  # Auto-detected by TTS module

# ============================================================================
# Vision Configuration
# ============================================================================
VISION_BASE_DIR = BASE_DIR / "vision"
VISION_DATA_DIR = VISION_BASE_DIR / "data"
VISION_DATA_DIR.mkdir(exist_ok=True)

# Camera settings
CAMERA_INDEX = 0
VISION_MODELS_DIR = VISION_DATA_DIR / "models"

# Face recognition settings
RECOGNITION_THRESHOLD = 0.4  # Cosine similarity threshold for face recognition
SMOOTHING_WINDOW = 5  # Frames to smooth recognition results

# Vision monitoring settings
VISION_CHECK_INTERVAL = 1.0  # Seconds between vision checks
VISION_ENABLED = True  # Set to False to disable vision module

# ============================================================================
# Proactive Policy Configuration
# ============================================================================
QUIET_HOURS = "23:00-07:00"  # Hours when proactive conversations are disabled
PROACTIVE_COOLDOWN_HOURS = 2  # Minimum hours between proactive conversations
PROACTIVE_ENABLED = True  # Set to False to disable proactive conversations

# ============================================================================
# Assistant Behavior
# ============================================================================
ASSISTANT_NAME = "Donna"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_INTERACTIONS = True  # Log all user interactions

# ============================================================================
# Web Server Configuration
# ============================================================================
WEB_HOST = "127.0.0.1"  # Localhost only
WEB_PORT = 8000  # Web server port
WEB_ENABLED = True  # Enable/disable web UI

# ============================================================================
# Emotion Detection Configuration
# ============================================================================
EMOTION_DETECTION_METHOD = "sentiment"  # Detection method: "sentiment" or "keyword"
EMOTION_UPDATE_INTERVAL = 0.5  # Seconds between emotion updates during streaming

