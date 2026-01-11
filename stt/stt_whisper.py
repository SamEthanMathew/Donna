#!/usr/bin/env python3
"""
Whisper.cpp Speech-to-Text module with wake word detection.

Continuously listens for the wake word "Donna", then records and transcribes speech.
All processing runs fully offline using whisper.cpp.
"""

import subprocess
import tempfile
import signal
import sys
from pathlib import Path
from typing import Optional


# Get paths relative to this file's location
_MODULE_DIR = Path(__file__).parent
# Use whisper-cli (recommended) or fall back to main (deprecated)
_WHISPER_CLI = _MODULE_DIR / "bin" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
_WHISPER_MAIN = _MODULE_DIR / "bin" / "whisper.cpp" / "build" / "bin" / "main"
WHISPER_BIN = _WHISPER_CLI if _WHISPER_CLI.exists() else _WHISPER_MAIN
MODEL = _MODULE_DIR / "models" / "ggml-base.en.bin"

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = "S16_LE"  # 16-bit signed little-endian

# Wake word detection settings
WAKE_WORD = "Donna"
WAKE_WORD_CHUNK_DURATION = 2  # seconds to record for wake word detection
MAIN_RECORDING_DURATION = 5  # seconds to record after wake word detected


def check_dependencies() -> None:
    """Verify that whisper.cpp binary and model exist."""
    if not WHISPER_BIN.exists():
        raise FileNotFoundError(
            f"Whisper binary not found at {WHISPER_BIN}. "
            "Run setup.sh to build whisper.cpp."
        )
    
    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL}. "
            "Run setup.sh to download the model."
        )


def record_audio(output_file: Path, duration: float, device: str = "default") -> None:
    """
    Record audio using arecord.
    
    Args:
        output_file: Path to save WAV file
        duration: Recording duration in seconds
        device: ALSA device (default: "default")
    """
    cmd = [
        "arecord",
        "-D", device,
        "-f", FORMAT,
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-d", str(int(duration)),
        str(output_file)
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"Audio recording failed: {error_msg}")


def transcribe(audio_file: Path) -> str:
    """
    Transcribe audio file using whisper.cpp.
    
    Args:
        audio_file: Path to WAV audio file
        
    Returns:
        Transcribed text (no timestamps)
    """
    cmd = [
        str(WHISPER_BIN),
        "-m", str(MODEL),
        "-f", str(audio_file),
        "--no-timestamps",
        "--language", "en"
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    
    if result.returncode != 0:
        error_msg = result.stderr
        raise RuntimeError(f"Transcription failed: {error_msg}")
    
    # Parse output to extract only the transcription text
    # whisper-cli outputs the transcription directly, possibly with leading/trailing whitespace
    lines = result.stdout.strip().split('\n')
    
    # Filter out empty lines and extract text
    transcription_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are just metadata markers (but keep [inaudible] as it's useful info)
        if line.lower().startswith('whisper') or 'loading model' in line.lower():
            continue
        transcription_lines.append(line)
    
    # Join all transcription lines
    transcription = ' '.join(transcription_lines).strip()
    
    return transcription


def detect_wake_word(device: str = "default") -> bool:
    """
    Record a short audio chunk and check if it contains the wake word.
    
    Args:
        device: ALSA device to use
        
    Returns:
        True if wake word detected, False otherwise
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_audio = Path(f.name)
    
    try:
        # Record short chunk for wake word detection
        record_audio(temp_audio, WAKE_WORD_CHUNK_DURATION, device)
        
        # Transcribe the chunk
        text = transcribe(temp_audio)
        
        # Check if wake word is in transcription (case-insensitive)
        return WAKE_WORD.lower() in text.lower()
    except Exception as e:
        # If transcription fails, assume no wake word
        return False
    finally:
        # Clean up temp file
        try:
            temp_audio.unlink()
        except OSError:
            pass


def listen_for_wake_word(device: str = "default") -> None:
    """
    Continuously listen for the wake word "Donna".
    
    Args:
        device: ALSA device to use
    """
    print(f"Listening for wake word '{WAKE_WORD}'...")
    print("(Press Ctrl+C to exit)")
    
    try:
        while True:
            if detect_wake_word(device):
                print(f"\n✓ Wake word '{WAKE_WORD}' detected!")
                return
            # Small delay to avoid excessive CPU usage
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


def listen_and_transcribe(device: str = "default") -> str:
    """
    Main function: Listen for wake word, then record and transcribe.
    
    Args:
        device: ALSA device to use
        
    Returns:
        Transcribed text
    """
    check_dependencies()
    
    # Step 1: Listen for wake word
    listen_for_wake_word(device)
    
    # Step 2: Record main audio
    print(f"Recording for {MAIN_RECORDING_DURATION} seconds...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_file = Path(f.name)
    
    try:
        record_audio(audio_file, MAIN_RECORDING_DURATION, device)
        
        # Step 3: Transcribe
        print("Transcribing...")
        transcription = transcribe(audio_file)
        
        return transcription
    finally:
        # Clean up temp file
        try:
            audio_file.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        transcription = listen_and_transcribe()
        if transcription:
            print("\n" + "="*50)
            print("TRANSCRIPTION:")
            print("="*50)
            print(transcription)
            print("="*50)
        else:
            print("\nNo transcription generated.")
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

