#!/usr/bin/env python3
"""
Quick test script for the TTS module.
Tests basic functionality and provides feedback.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts.core import say


def test_tts():
    """Run TTS tests."""
    print("=" * 60)
    print("TTS Module Test")
    print("=" * 60)
    
    # Test 1: Basic functionality
    print("\n[Test 1] Basic TTS test...")
    try:
        say("Test one. If you can hear this, the text to speech system is working.")
        print("✓ Test 1 passed - audio should have played")
    except FileNotFoundError as e:
        print(f"✗ Test 1 failed - missing files: {e}")
        return False
    except RuntimeError as e:
        print(f"✗ Test 1 failed - runtime error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test 1 failed - unexpected error: {e}")
        return False
    
    # Test 2: Empty string (should be ignored)
    print("\n[Test 2] Empty string test...")
    try:
        say("")
        say("   ")  # Whitespace only
        print("✓ Test 2 passed - empty strings ignored")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return False
    
    # Test 3: Longer text
    print("\n[Test 3] Longer text test...")
    try:
        say("This is a longer test sentence to verify that the text to speech system can handle multiple words and longer phrases correctly.")
        print("✓ Test 3 passed - longer text processed")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return False
    
    # Test 4: Special characters
    print("\n[Test 4] Special characters test...")
    try:
        say("Testing numbers: one, two, three. Testing punctuation: Hello, world! How are you?")
        print("✓ Test 4 passed - special characters handled")
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nIf you heard all the test messages, the TTS module is working correctly.")
    return True


if __name__ == "__main__":
    try:
        success = test_tts()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)

