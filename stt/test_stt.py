#!/usr/bin/env python3
"""
Quick test script for the STT module.
Tests basic functionality and provides feedback.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stt.stt_whisper import (
    check_dependencies,
    record_audio,
    transcribe,
    detect_wake_word,
    listen_and_transcribe
)


def test_stt():
    """Run STT tests."""
    print("=" * 60)
    print("STT Module Test")
    print("=" * 60)
    
    # Test 1: Check dependencies
    print("\n[Test 1] Checking dependencies...")
    try:
        check_dependencies()
        print("✓ Test 1 passed - all dependencies found")
    except FileNotFoundError as e:
        print(f"✗ Test 1 failed - missing dependencies: {e}")
        print("\nRun setup.sh to install whisper.cpp and download the model.")
        return False
    except Exception as e:
        print(f"✗ Test 1 failed - unexpected error: {e}")
        return False
    
    # Test 2: Audio recording
    print("\n[Test 2] Testing audio recording...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_file = Path(f.name)
        
        print("Recording 2 seconds of audio... (please speak)")
        record_audio(audio_file, duration=2.0)
        
        if audio_file.exists() and audio_file.stat().st_size > 0:
            print(f"✓ Test 2 passed - recorded {audio_file.stat().st_size} bytes")
        else:
            print("✗ Test 2 failed - audio file is empty or missing")
            return False
        
        # Test 3: Transcription
        print("\n[Test 3] Testing transcription...")
        print("Transcribing audio (this may take a few seconds)...")
        transcription = transcribe(audio_file)
        
        if transcription:
            print(f"✓ Test 3 passed - transcription: '{transcription}'")
        else:
            print("✗ Test 3 failed - no transcription generated")
            return False
        
        # Clean up
        try:
            audio_file.unlink()
        except:
            pass
        
    except Exception as e:
        print(f"✗ Test 2/3 failed: {e}")
        return False
    
    # Test 4: Wake word detection (optional, requires speaking "Donna")
    print("\n[Test 4] Testing wake word detection...")
    print("This will record 2 seconds and check for 'Donna'")
    print("Say 'Donna' when prompted...")
    try:
        import time
        print("Starting in 2 seconds...")
        time.sleep(2)
        detected = detect_wake_word()
        if detected:
            print("✓ Test 4 passed - wake word detected")
        else:
            print("⚠ Test 4 - wake word not detected (this is OK if you didn't say 'Donna')")
    except Exception as e:
        print(f"⚠ Test 4 - error: {e} (this is OK)")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nTo test the full workflow, run:")
    print("  python3 stt/stt_whisper.py")
    print("\nThis will listen for 'Donna', then record and transcribe.")
    return True


if __name__ == "__main__":
    try:
        success = test_stt()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

