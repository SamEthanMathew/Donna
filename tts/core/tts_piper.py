"""
Piper TTS wrapper for Jetson devices.

Provides text-to-speech functionality using the Piper TTS engine
with the Amy medium voice model.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# Get paths relative to this file's location
_MODULE_DIR = Path(__file__).parent
_PROJECT_DIR = _MODULE_DIR.parent
PIPER_BIN = _PROJECT_DIR / "bin" / "piper" / "piper"
MODEL = _PROJECT_DIR / "models" / "en_US-amy-medium.onnx"


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

        # Play WAV file using aplay on P10S (card 0, device 0)
        # Use plughw:0,0 to allow ALSA to handle format conversion (mono->stereo, sample rate)
        subprocess.run(
            ["aplay", "-D", "plughw:0,0", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
    finally:
        # Clean up temporary file
        try:
            Path(wav_path).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    # Test the TTS module
    say("If you can hear this, your Piper T T S integration works.")

