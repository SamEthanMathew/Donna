#!/usr/bin/env python3
"""
Full voice chat - Speech to LLM to Speech.

Uses tiny.en Whisper model to minimize memory usage.
"""

import sys
import os

# Fix for static TLS issue
os.environ['LD_PRELOAD'] = '/home/sam/jetson_agent/.venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.stt import SpeechToText
from audio.tts import TextToSpeech
from llm import QwenModel


def main():
    print("=" * 60)
    print("Jetson Agent - Full Voice Chat")
    print("=" * 60)
    print()
    
    print("Initializing components (this may take a minute)...")
    print()
    
    try:
        # Initialize Speech-to-Text with smallest model
        print("[1/3] Loading Speech-to-Text (tiny model)...")
        stt = SpeechToText(model_size="tiny.en", use_cuda=False)
        print()
        
        # Initialize Text-to-Speech
        print("[2/3] Loading Text-to-Speech...")
        tts = TextToSpeech(use_cuda=False)
        print()
        
        # Initialize LLM
        print("[3/3] Loading Language Model...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "llm/qwen2.5-7b-instruct-f16.gguf")
        llm = QwenModel(
            model_path=model_path,
            temperature=0.7,
            n_ctx=2048,
            n_gpu_layers=0
        )
        print()
        
        print("=" * 60)
        print("All systems ready! Voice chat starting...")
        print("=" * 60)
        print()
        print("Instructions:")
        print("- Press ENTER when ready to speak")
        print("- Speak for up to 5 seconds after the beep")
        print("- The agent will transcribe, think, and respond via voice")
        print("- Say 'exit', 'quit', or 'stop' to end")
        print("- Press Ctrl+C at any time to exit")
        print()
        
        # Welcome message
        tts.speak("Hello! I'm your voice assistant. How can I help you today?")
        
        # System prompt
        system_prompt = (
            "You are a helpful AI assistant running on a Jetson device. "
            "Keep your responses brief and conversational, maximum 2-3 sentences. "
            "Be friendly and helpful."
        )
        
        conversation_count = 0
        
        while True:
            try:
                print("-" * 60)
                print(f"\n[Turn {conversation_count + 1}]")
                print()
                
                # Wait for user to be ready
                input("Press ENTER when ready to speak...")
                print()
                
                # Listen and transcribe
                print("🎤 Listening... (speak now, 5 seconds)")
                user_text = stt.listen_and_transcribe(duration=5.0)
                
                if not user_text or len(user_text.strip()) < 2:
                    print("❌ No speech detected or too short. Please try again.")
                    tts.speak("I didn't catch that. Please try again.")
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
                    response = llm.chat(user_text, max_tokens=128, system_prompt=system_prompt)
                else:
                    response = llm.chat(user_text, max_tokens=128)
                
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
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

