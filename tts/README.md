# Piper TTS Module

Text-to-speech module using Piper TTS with the Amy medium voice, optimized for NVIDIA Jetson devices.

## Overview

This module provides a simple Python interface to Piper TTS, a fast, local neural text-to-speech system. It uses the Amy medium voice model for natural-sounding speech synthesis.

## Installation

Run the setup script to install all dependencies and download required files:

```bash
cd /home/sam/jetson_agent/tts
./setup.sh
```

This will:
1. Install system dependencies (`alsa-utils`, `wget`)
2. Download the Piper aarch64 binary (~10 MB)
3. Download the Amy medium voice model (~15 MB)
4. Test the installation

## Usage

### Basic Usage

```python
from tts.core import say

say("Hello, this is a test of the text-to-speech system.")
```

### Command Line Test

```bash
cd /home/sam/jetson_agent/tts
python3 -m core.tts_piper
```

## Module Structure

```
tts/
├── setup.sh              # Installation script
├── requirements.txt      # Python dependencies (none required)
├── README.md            # This file
├── bin/                  # Piper binary location
│   └── piper/           # Extracted binary directory
│       └── piper        # Executable
├── models/              # Voice model files
│   ├── en_US-amy-medium.onnx
│   └── en_US-amy-medium.onnx.json
└── core/                # Python module
    ├── __init__.py
    └── tts_piper.py      # Main TTS wrapper
```

## API Reference

### `say(text: str) -> None`

Convert text to speech and play it using the default audio output.

**Parameters:**
- `text` (str): The text to speak. Empty strings are ignored.

**Raises:**
- `FileNotFoundError`: If Piper binary or model files are missing (run `setup.sh` first)
- `RuntimeError`: If Piper TTS fails to generate or play audio

**Example:**
```python
from tts.core import say

say("Hello, world!")
```

## Integration

This module is designed to be standalone and can be imported from anywhere in the project:

```python
# From llm module
from tts.core import say

# From vision module
from tts.core import say

# From project root
import sys
sys.path.insert(0, '/home/sam/jetson_agent')
from tts.core import say
```

## Troubleshooting

### "Piper binary not found"
Run `./setup.sh` to install the Piper binary.

### "Model file not found"
Run `./setup.sh` to download the voice model.

### "aplay: command not found"
Install ALSA utilities:
```bash
sudo apt-get install alsa-utils
```

### No audio output
1. Check that audio hardware is connected and working:
   ```bash
   aplay /usr/share/sounds/alsa/Front_Left.wav
   ```
2. Verify audio permissions (may need to add user to audio group):
   ```bash
   sudo usermod -a -G audio $USER
   ```
   (Log out and back in for changes to take effect)

### Permission denied on setup.sh
Make the script executable:
```bash
chmod +x setup.sh
```

## Technical Details

- **Engine**: Piper TTS v1.2.0
- **Voice**: Amy (medium quality, US English)
- **Model Format**: ONNX
- **Audio Output**: ALSA (via `aplay`)
- **Platform**: aarch64 (Jetson AGX Orin / JetPack 5)

## Resources

- [Piper TTS GitHub](https://github.com/rhasspy/piper)
- [Piper Voices](https://huggingface.co/rhasspy/piper-voices)
- [ALSA Documentation](https://www.alsa-project.org/)

## License

Piper TTS is licensed under the Apache 2.0 license. See the [Piper repository](https://github.com/rhasspy/piper) for details.
