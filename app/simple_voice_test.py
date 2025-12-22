#!/usr/bin/env python3
"""
Simplified voice test - loads components one at a time to isolate issues.
"""

import sys
import os

# Fix for static TLS issue on Jetson
os.environ['LD_PRELOAD'] = '/home/sam/jetson_agent/.venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_stt():
    print("\n[TEST 1] Loading Speech-to-Text...")
    try:
        from audio.stt import SpeechToText
        stt = SpeechToText(model_size="base.en", use_cuda=False)
        print("✓ STT loaded successfully")
        return stt
    except Exception as e:
        print(f"✗ STT failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_llm():
    print("\n[TEST 2] Loading LLM...")
    try:
        from llm import QwenModel
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "llm/qwen2.5-7b-instruct-f16.gguf")
        llm = QwenModel(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False
        )
        print("✓ LLM loaded successfully")
        
        # Test a simple generation
        print("Testing LLM generation...")
        response = llm.generate("Hello", max_tokens=20)
        print(f"✓ LLM response: {response[:50]}...")
        return llm
    except Exception as e:
        print(f"✗ LLM failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_tts():
    print("\n[TEST 3] Loading Text-to-Speech...")
    try:
        from audio.tts import TextToSpeech
        tts = TextToSpeech(use_cuda=False)
        print("✓ TTS loaded successfully")
        return tts
    except Exception as e:
        print(f"✗ TTS failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 60)
    print("Jetson Agent - Component Test")
    print("=" * 60)
    
    # Test each component
    stt = test_stt()
    llm = test_llm()
    tts = test_tts()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  STT: {'✓ OK' if stt else '✗ FAILED'}")
    print(f"  LLM: {'✓ OK' if llm else '✗ FAILED'}")
    print(f"  TTS: {'✓ OK' if tts else '✗ FAILED'}")
    print("=" * 60)
    
    # If all loaded, try a simple conversation
    if stt and llm and tts:
        print("\nAll components loaded! Ready for voice interaction.")
        print("Try: python app/voice_test.py")
    else:
        print("\nSome components failed to load. Please fix errors above.")

if __name__ == "__main__":
    main()

