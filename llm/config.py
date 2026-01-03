from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "donna.db"
PROMPT_PATH = BASE_DIR / "donna_prompt.txt"

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"

# Keep last N conversation turns (user+assistant pairs).
WORKING_MEMORY_TURNS = 5

# Conversation management
VERBATIM_TURNS = 4  # Keep last 4 turns verbatim
SUMMARY_UPDATE_THRESHOLD = 6  # Update summary after N turns

# Memory context limits
MAX_MEMORY_CONTEXT = 8  # Max memories to inject
MIN_MEMORY_CONTEXT = 3  # Min memories to inject

# Guardrails
MAX_USER_CHARS = 4000
MAX_ASSISTANT_CHARS = 4000


