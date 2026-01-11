"""
Piper TTS wrapper for Jetson devices.

Provides text-to-speech functionality using the Piper TTS engine
with the Amy medium voice model.
"""

import subprocess
import tempfile
import re
import sys
import time
from pathlib import Path
from typing import Optional


# Get paths relative to this file's location
_MODULE_DIR = Path(__file__).parent
_PROJECT_DIR = _MODULE_DIR.parent
PIPER_BIN = _PROJECT_DIR / "bin" / "piper" / "piper"
MODEL = _PROJECT_DIR / "models" / "en_US-amy-medium.onnx"


def find_usb_audio_device() -> str:
    """
    Auto-detect USB audio device by looking for USB Audio or P10S devices.
    Falls back to 'default' if no USB device found.
    
    Returns:
        ALSA device string (e.g., 'plughw:2,0' or 'default')
    """
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return "default"
        
        # Look for USB audio devices (P10S, USB Audio, etc.)
        lines = result.stdout.split('\n')
        for line in lines:
            # Match lines like: "card 2: P10S [P10S], device 0: USB Audio [USB Audio]"
            match = re.search(r'card (\d+):.*?(?:P10S|USB Audio|USB-Audio)', line, re.IGNORECASE)
            if match:
                card_num = match.group(1)
                # Extract device number (usually 0 for USB devices)
                device_match = re.search(r'device (\d+):', line)
                device_num = device_match.group(1) if device_match else "0"
                return f"plughw:{card_num},{device_num}"
        
        # If no USB device found, try to find any non-HDMI device
        for line in lines:
            match = re.search(r'card (\d+):', line)
            if match:
                card_num = match.group(1)
                # Skip card 0 (usually HDMI) and card 1 (usually APE/internal)
                if card_num not in ["0", "1"]:
                    device_match = re.search(r'device (\d+):', line)
                    device_num = device_match.group(1) if device_match else "0"
                    return f"plughw:{card_num},{device_num}"
        
    except Exception:
        pass
    
    # Fallback to default ALSA device
    return "default"


def say(text: str) -> None:
    """
    Convert text to speech and play it using Piper TTS.
    
    Args:
        text: The text to speak. Empty strings are ignored.
        
    Raises:
        RuntimeError: If Piper TTS fails to generate or play audio.
        FileNotFoundError: If Piper binary or model files are missing.
    """
    text = (text or "").strip()
    if not text:
        return
    
    # Check if binary exists
    if not PIPER_BIN.exists():
        raise FileNotFoundError(
            f"Piper binary not found at {PIPER_BIN}. "
            "Run setup.sh to install Piper TTS."
        )
    
    # Check if model exists
    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL}. "
            "Run setup.sh to download the model."
        )
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        # Generate WAV file using Piper
        p = subprocess.run(
            [str(PIPER_BIN), "--model", str(MODEL), "--output_file", wav_path],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        
        if p.returncode != 0:
            error_msg = p.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"Piper TTS failed: {error_msg}")
        
        # Ensure file is fully written before playing
        if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
            raise RuntimeError("Generated WAV file is empty or missing")
        
        # Small delay to ensure file is ready
        time.sleep(0.1)

        # Auto-detect USB audio device (handles card number changes)
        audio_device = find_usb_audio_device()
        
        # Play WAV file using aplay
        # Use plughw to allow ALSA to handle format conversion (mono->stereo, sample rate)
        # Wait for completion and capture errors
        play_result = subprocess.run(
            ["aplay", "-D", audio_device, wav_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30  # Timeout after 30 seconds
        )
        
        if play_result.returncode != 0:
            error_msg = play_result.stderr.decode('utf-8', errors='ignore')
            # Try fallback to default device if specific device fails
            if audio_device != "default":
                print(f"[TTS Warning] Device {audio_device} failed, trying default: {error_msg}", file=sys.stderr)
                play_result = subprocess.run(
                    ["aplay", "-D", "default", wav_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=30
                )
                if play_result.returncode != 0:
                    error_msg2 = play_result.stderr.decode('utf-8', errors='ignore')
                    print(f"[TTS Error] Default device also failed: {error_msg2}", file=sys.stderr)
            else:
                print(f"[TTS Error] aplay failed: {error_msg}", file=sys.stderr)
        
        # Small delay to ensure device is ready for next call
        time.sleep(0.2)
    finally:
        # Clean up temporary file
        try:
            Path(wav_path).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    # Test the TTS module
    say("If you can hear this, your Piper T T S integration works.")

