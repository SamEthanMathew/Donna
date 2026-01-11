# Whisper.cpp Speech-to-Text Module

Offline speech-to-text module using whisper.cpp with wake word detection, optimized for NVIDIA Jetson devices.

## Overview

This module provides speech-to-text functionality using whisper.cpp, a high-performance C++ implementation of OpenAI's Whisper. It continuously listens for the wake word "Donna", then records and transcribes speech. All processing runs fully offline on the Jetson device.

## Features

- **Wake Word Detection**: Continuously monitors audio for "Donna" to trigger recording
- **Offline Processing**: No internet connection required
- **High Accuracy**: Uses Whisper base English model
- **Low Latency**: Optimized for Jetson AGX Orin
- **Simple API**: Easy to integrate

## Installation

### Prerequisites

- Jetson AGX Orin with JetPack 5
- USB microphone connected
- Build tools (installed by setup script)

### Setup Steps

Run the setup script to build whisper.cpp and download the model:

```bash
cd /home/sam/jetson_agent/stt
./setup.sh
```

This will:
1. Install build dependencies (cmake, make, g++, ffmpeg)
2. Clone and build whisper.cpp from source
3. Download the ggml-base.en.bin model (~142 MB)
4. Verify the installation

**Note**: The build process may take 10-20 minutes depending on your system.

## Usage

### Basic Usage

Run the module directly:

```bash
cd /home/sam/jetson_agent/stt
python3 stt_whisper.py
```

The module will:
1. Start listening for the wake word "Donna"
2. When detected, record audio for 5 seconds
3. Transcribe the recorded audio
4. Print the transcription

### Python API

Import and use in your code:

```python
from stt_whisper import listen_and_transcribe

# Listen for wake word, record, and transcribe
transcription = listen_and_transcribe()
print(f"You said: {transcription}")
```

### Custom Functions

```python
from stt_whisper import detect_wake_word, record_audio, transcribe
from pathlib import Path

# Check for wake word
if detect_wake_word():
    print("Wake word detected!")

# Record audio manually
audio_file = Path("/tmp/recording.wav")
record_audio(audio_file, duration=5.0)

# Transcribe a file
text = transcribe(audio_file)
print(text)
```

## Module Structure

```
stt/
├── setup.sh              # Installation script
├── requirements.txt      # Python dependencies (none required)
├── README.md            # This file
├── bin/                  # whisper.cpp binary location
│   └── whisper.cpp/     # Source and build directory
│       └── build/
│           └── bin/
│               └── main  # Executable
├── models/              # Whisper model files
│   └── ggml-base.en.bin
└── stt_whisper.py       # Main STT module
```

## API Reference

### `listen_and_transcribe(device: str = "default") -> str`

Main function that listens for wake word, records audio, and transcribes.

**Parameters:**
- `device` (str): ALSA device name (default: "default")

**Returns:**
- `str`: Transcribed text

**Example:**
```python
text = listen_and_transcribe()
```

### `detect_wake_word(device: str = "default") -> bool`

Check if the wake word "Donna" is present in a short audio recording.

**Parameters:**
- `device` (str): ALSA device name

**Returns:**
- `bool`: True if wake word detected

### `record_audio(output_file: Path, duration: float, device: str = "default") -> None`

Record audio using arecord.

**Parameters:**
- `output_file` (Path): Path to save WAV file
- `duration` (float): Recording duration in seconds
- `device` (str): ALSA device name

### `transcribe(audio_file: Path) -> str`

Transcribe an audio file using whisper.cpp.

**Parameters:**
- `audio_file` (Path): Path to WAV audio file

**Returns:**
- `str`: Transcribed text (no timestamps)

## Configuration

### Adjust Recording Duration

Edit `stt_whisper.py`:

```python
WAKE_WORD_CHUNK_DURATION = 2  # seconds for wake word detection
MAIN_RECORDING_DURATION = 5   # seconds to record after wake word
```

### Change Wake Word

Edit `stt_whisper.py`:

```python
WAKE_WORD = "Donna"  # Change to your preferred wake word
```

### Use Different Model

Download a different model and update the path:

```python
MODEL = _MODULE_DIR / "models" / "ggml-tiny.en.bin"  # Faster, less accurate
# or
MODEL = _MODULE_DIR / "models" / "ggml-small.en.bin"  # Slower, more accurate
```

## Integration

This module integrates with the voice assistant pipeline:

### With TTS Module

```python
from stt_whisper import listen_and_transcribe
from tts.core import say

# Listen and transcribe
text = listen_and_transcribe()

# Speak response
say(f"You said: {text}")
```

### With LLM Module

```python
from stt_whisper import listen_and_transcribe
from llm.app import process_user_input  # Hypothetical function

# Voice input
user_input = listen_and_transcribe()

# Process with LLM
response = process_user_input(user_input)
```

## Troubleshooting

### "Whisper binary not found"

Run the setup script:
```bash
./setup.sh
```

### "Model file not found"

The model should download automatically during setup. If it fails:
```bash
cd models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

### "Audio recording failed"

1. Check microphone is connected:
   ```bash
   arecord -l
   ```

2. Test microphone:
   ```bash
   arecord -f S16_LE -r 16000 -c 1 -d 3 /tmp/test.wav
   aplay /tmp/test.wav
   ```

3. Check ALSA device:
   ```bash
   cat /proc/asound/cards
   ```

### Wake word not detected

- Speak clearly: "Donna"
- Ensure microphone is working
- Try increasing `WAKE_WORD_CHUNK_DURATION` in the code
- Check microphone volume: `alsamixer`

### Slow transcription

- Use a smaller model (tiny instead of base)
- Reduce `MAIN_RECORDING_DURATION`
- Ensure CPU is not throttled

### Build fails

1. Check build dependencies:
   ```bash
   sudo apt-get install cmake make g++ ffmpeg libavcodec-dev libavformat-dev libavutil-dev
   ```

2. Check available memory (build needs ~2GB):
   ```bash
   free -h
   ```

3. Try building with fewer cores:
   ```bash
   cmake --build build -j2  # Instead of -j$(nproc)
   ```

## Performance

**On Jetson AGX Orin:**
- Wake word detection: ~1-2 seconds per chunk
- Transcription (5 seconds audio): ~2-5 seconds
- Model size: ~142 MB (base.en)

**Model Options:**
- `ggml-tiny.en.bin` (~75 MB): Fastest, lower accuracy
- `ggml-base.en.bin` (~142 MB): Balanced (default)
- `ggml-small.en.bin` (~466 MB): Slower, higher accuracy

## Technical Details

- **Engine**: whisper.cpp (C++ implementation)
- **Model**: Whisper base English (ggml format)
- **Audio Format**: 16-bit PCM, 16kHz, mono
- **Recording**: ALSA arecord
- **Platform**: aarch64 (Jetson AGX Orin)

## Resources

- [whisper.cpp GitHub](https://github.com/ggerganov/whisper.cpp)
- [Whisper Models](https://huggingface.co/ggerganov/whisper.cpp)
- [ALSA Documentation](https://www.alsa-project.org/)

## License

whisper.cpp is licensed under the MIT license. See the [whisper.cpp repository](https://github.com/ggerganov/whisper.cpp) for details.

