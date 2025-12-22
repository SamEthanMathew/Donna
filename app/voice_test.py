#!/usr/bin/env python3
"""
Voice interaction test for Jetson Agent.

This script demonstrates the full audio-to-LLM-to-audio pipeline:
1. Listen to user speech
2. Transcribe speech to text (STT)
3. Send to LLM for response
4. Speak the response (TTS)
"""

import sys
import os

# Fix for static TLS issue on Jetson
import glob
gomp_libs = glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"))
if gomp_libs:
    existing_preload = os.environ.get('LD_PRELOAD', '')
    os.environ['LD_PRELOAD'] = gomp_libs[0] + (':' + existing_preload if existing_preload else '')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.stt import SpeechToText
from audio.tts import TextToSpeech
from llm import QwenModel


def main():
    print("=" * 60)
    print("Jetson Agent - Voice Interaction Test")
    print("=" * 60)
    print()
    
    # Initialize components
    print("Initializing components...")
    print()
    
    try:
        # Initialize Speech-to-Text
        print("[1/3] Loading Speech-to-Text...")
        stt = SpeechToText(model_size="base.en")
        print()
        
        # Initialize Text-to-Speech
        print("[2/3] Loading Text-to-Speech...")
        tts = TextToSpeech()
        print()
        
        # Initialize LLM
        print("[3/3] Loading Language Model...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "llm/qwen2.5-7b-instruct-f16.gguf")
        llm = QwenModel(
            model_path=model_path,
            temperature=0.7,
            n_ctx=2048,  # Reduced context to avoid memory issues
            n_gpu_layers=0  # Explicitly use CPU
        )
        print()
        
        print("=" * 60)
        print("All systems ready! Voice interaction starting...")
        print("=" * 60)
        print()
        print("Instructions:")
        print("- You will be recorded for 5 seconds after each prompt")
        print("- Speak clearly into your microphone")
        print("- Say 'exit', 'quit', or 'stop' to end the conversation")
        print("- Press Ctrl+C at any time to exit")
        print()
        
        # Set system prompt for the agent
        system_prompt = (
            "You are a helpful AI assistant running on a Jetson device. "
            "Keep your responses concise and conversational, ideally 1-2 sentences. "
            "Be friendly and helpful."
        )
        
        # Conversation loop
        conversation_count = 0
        
        while True:
            try:
                print("-" * 60)
                print(f"\n[Turn {conversation_count + 1}]")
                print()
                
                # Listen and transcribe
                print("🎤 Listening... (speak now, 5 seconds)")
                user_text = stt.listen_and_transcribe(duration=5.0)
                
                if not user_text or len(user_text.strip()) < 2:
                    print("❌ No speech detected or too short. Please try again.")
                    continue
                
                print(f"📝 You said: \"{user_text}\"")
                print()
                
                # Check for exit commands
                exit_words = ['exit', 'quit', 'stop', 'goodbye', 'bye']
                if any(word in user_text.lower() for word in exit_words):
                    farewell = "Goodbye! It was nice talking to you."
                    print(f"🤖 Agent: {farewell}")
                    tts.speak(farewell)
                    break
                
                # Generate LLM response
                print("🤔 Thinking...")
                if conversation_count == 0:
                    # Use system prompt on first turn
                    response = llm.chat(user_text, max_tokens=256, system_prompt=system_prompt)
                else:
                    response = llm.chat(user_text, max_tokens=256)
                
                print(f"🤖 Agent: {response}")
                print()
                
                # Speak the response
                print("🔊 Speaking...")
                tts.speak(response)
                
                conversation_count += 1
                print()
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error during conversation: {e}")
                import traceback
                traceback.print_exc()
                print("\nTrying to continue...")
                continue
        
        print()
        print("=" * 60)
        print(f"Conversation ended. Total turns: {conversation_count}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


